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
          use_smote=True, params=None):
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

    base_params = dict(n_estimators=400, learning_rate=0.05, num_leaves=31)
    if params:
        base_params.update(params)
    base = lgb.LGBMClassifier(
        objective="multiclass", subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=random_state, verbose=-1, **base_params)

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


def cross_validate(df: pd.DataFrame, n_splits=5, params=None, random_state=42,
                   save_fold_models=False):
    """Stratified k-fold CV macro-F1 for the tabular model (the honest headline metric).

    When ``save_fold_models=True``, each fold's fitted ``LGBMClassifier`` is saved to
    ``data/features/classifier_fold_{i}.joblib`` so :func:`predict` can *bag* them at
    inference (averaging probabilities across folds reduces variance; Schanche+2020).

    Returns dict(mean, std, per_fold, oof_pred, oof_true, fold_models).
    """
    import lightgbm as lgb
    import joblib
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    X = df[config.FEATURE_COLUMNS].astype(float).fillna(-99.0).values
    y = df["label"].values
    params = params or dict(n_estimators=400, learning_rate=0.05, num_leaves=31)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores, oof_p, oof_t, fold_models = [], [], [], []
    for i, (tr, te) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(objective="multiclass", class_weight="balanced",
                               random_state=random_state, verbose=-1, **params)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        scores.append(f1_score(y[te], p, average="macro"))
        oof_p.extend(p.tolist()); oof_t.extend(y[te].tolist())
        fold_models.append(m)
        if save_fold_models:
            joblib.dump(m, config.MODEL_PATH.parent / f"classifier_fold_{i}.joblib")
    return dict(mean=float(np.mean(scores)), std=float(np.std(scores)),
                per_fold=scores, oof_pred=oof_p, oof_true=oof_t, fold_models=fold_models)


def tune_lightgbm(df: pd.DataFrame, n_splits=4, random_state=42, verbose=True):
    """Small grid search on CV macro-F1. Returns (best_params, best_score, all_results)."""
    from itertools import product
    grid = dict(
        num_leaves=[15, 31, 63, 127],
        n_estimators=[600, 1000],
        learning_rate=[0.03, 0.05],
        min_child_samples=[3, 5, 20],
    )
    keys = list(grid)
    best, best_score, results = None, -1.0, []
    for combo in product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        cv = cross_validate(df, n_splits=n_splits, params=params, random_state=random_state)
        results.append((params, cv["mean"]))
        if cv["mean"] > best_score:
            best, best_score = params, cv["mean"]
    if verbose:
        print(f"[classify] best CV macro-F1 = {best_score:.3f} with {best}")
    return best, best_score, results


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


def load_fold_models():
    """Load all saved k-fold LightGBM models (``classifier_fold_{i}.joblib``).

    Returns a list (empty if no fold models exist -> caller uses the single model)."""
    import joblib
    models = []
    base = config.MODEL_PATH.parent
    for i in range(10):
        p = base / f"classifier_fold_{i}.joblib"
        if p.exists():
            models.append(joblib.load(p))
        else:
            break
    return models


def _bagged_proba(fold_models, x):
    """Average predicted probabilities across fold models onto a shared class axis."""
    all_classes = sorted(set().union(*[set(m.classes_) for m in fold_models]))
    proba = np.zeros(len(all_classes))
    for m in fold_models:
        p = m.predict_proba(x)[0]
        for j, cls in enumerate(m.classes_):
            proba[all_classes.index(cls)] += p[j]
    proba /= len(fold_models)
    return all_classes, proba


# --------------------------------------------------------------------------------------
# Prediction (with heuristic fallback)
# --------------------------------------------------------------------------------------
def predict(features: dict, model=None) -> tuple[str, float]:
    """Return (class, calibrated_confidence). Falls back to the rule-based heuristic if no
    trained model is available.

    If k-fold models exist (``classifier_fold_*.joblib``) they are *bagged* — probabilities
    averaged across folds — which reduces variance; otherwise the single calibrated model is
    used (backward compatible)."""
    x = np.array([[features.get(c, np.nan) for c in config.FEATURE_COLUMNS]], dtype=float)
    x = np.nan_to_num(x, nan=-99.0)

    if model is None:
        fold_models = load_fold_models()
        if fold_models:
            try:
                classes, proba = _bagged_proba(fold_models, x)
                i = int(np.argmax(proba))
                return classes[i], float(proba[i])
            except Exception:
                pass                         # stale fold models -> fall through to single/heuristic
        model = load_model()
    if model is None:
        return verdict_heuristic(features)

    try:
        proba = model.predict_proba(x)[0]
    except Exception:
        return verdict_heuristic(features)   # stale model (feature mismatch) -> heuristic
    classes = list(model.classes_)
    i = int(np.argmax(proba))
    return classes[i], float(proba[i])
