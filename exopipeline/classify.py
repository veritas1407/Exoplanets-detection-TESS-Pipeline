"""Stage 5 — Classifier + calibrated confidence.

A LightGBM multiclass model on the vetting-feature table, wrapped in isotonic calibration
so the reported "confidence" is a meaningful probability (the PS explicitly requires a
confidence level). Reports per-class precision/recall/F1, PR-AUC, and a confusion matrix.

Until a model is trained, :func:`predict` transparently falls back to the rule-based
:func:`exopipeline.vetting.verdict_heuristic`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .vetting import verdict_heuristic


# --------------------------------------------------------------------------------------
# Feature-table helpers
# --------------------------------------------------------------------------------------
def features_to_row(features: dict, label: str | None = None,
                    target: str | None = None) -> dict:
    """Project a vetting-feature dict onto the classifier columns (+ optional label)."""
    row = {c: features.get(c, np.nan) for c in config.FEATURE_COLUMNS}
    if label is not None:
        row["label"] = label
    if target is not None:
        row["target"] = target
    return row


def append_feature_row(row: dict, path=None):
    """Append one row to the on-disk feature table (parquet), creating it if needed."""
    path = path or config.FEATURE_TABLE
    df_new = pd.DataFrame([row])
    if path.exists():
        df = pd.read_parquet(path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_parquet(path, index=False)
    return df


def load_feature_table(path=None) -> pd.DataFrame:
    path = path or config.FEATURE_TABLE
    if not path.exists():
        raise FileNotFoundError(f"No feature table at {path}. Build it first "
                                f"(see classifier_training.ipynb).")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train(df: pd.DataFrame, calibrate=True, test_size=0.25, random_state=42,
          use_smote=True):
    """Train + (isotonic) calibrate a LightGBM multiclass classifier.

    Returns a dict with the fitted model, the held-out split, predictions, and a metrics
    bundle (confusion matrix, per-class report, PR-AUC).
    """
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 average_precision_score)
    from sklearn.preprocessing import label_binarize

    X = df[config.FEATURE_COLUMNS].astype(float).fillna(-99.0).values
    y = df["label"].values
    classes = sorted(np.unique(y).tolist())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            kmin = min(np.bincount([classes.index(v) for v in y_tr]))
            if kmin > 1:
                X_tr, y_tr = SMOTE(random_state=random_state,
                                   k_neighbors=min(5, kmin - 1)).fit_resample(X_tr, y_tr)
        except Exception as e:
            print(f"[classify] SMOTE skipped: {e}")

    base = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=400, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=random_state, verbose=-1)

    if calibrate:
        # cv='prefit' would need a holdout; use internal CV calibration instead.
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
        model.fit(X_tr, y_tr)
    else:
        base.fit(X_tr, y_tr)
        model = base

    proba = model.predict_proba(X_te)
    model_classes = list(model.classes_)
    y_pred = np.array(model_classes)[np.argmax(proba, axis=1)]

    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=model_classes)
    # PR-AUC (macro, one-vs-rest)
    try:
        Y_te = label_binarize(y_te, classes=model_classes)
        pr_auc = average_precision_score(Y_te, proba, average="macro")
    except Exception:
        pr_auc = np.nan

    return dict(model=model, classes=model_classes,
                X_test=X_te, y_test=y_te, y_pred=y_pred, proba=proba,
                report=report, confusion_matrix=cm, pr_auc=pr_auc)


def save_model(model, path=None):
    import joblib
    path = path or config.MODEL_PATH
    joblib.dump(model, path)
    return path


def load_model(path=None):
    import joblib
    path = path or config.MODEL_PATH
    if not path.exists():
        return None
    return joblib.load(path)


# --------------------------------------------------------------------------------------
# Prediction (with heuristic fallback)
# --------------------------------------------------------------------------------------
def predict(features: dict, model=None) -> tuple[str, float]:
    """Return (class, calibrated_confidence). Falls back to the rule-based heuristic if no
    trained model is available."""
    if model is None:
        model = load_model()
    if model is None:
        return verdict_heuristic(features)

    x = np.array([[features.get(c, np.nan) for c in config.FEATURE_COLUMNS]],
                 dtype=float)
    x = np.nan_to_num(x, nan=-99.0)
    proba = model.predict_proba(x)[0]
    classes = list(model.classes_)
    i = int(np.argmax(proba))
    return classes[i], float(proba[i])
