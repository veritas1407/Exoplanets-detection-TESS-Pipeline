"""Track B — dual-view 1D-CNN classifier (Astronet / Yu et al. 2019 style).

Complements the tabular LightGBM (``classify.py``) with a convolutional model that reads
the *shape* of the phase-folded transit directly. Two input branches:

* **global view** — the whole phase-folded light curve binned to ``N_GLOBAL`` points
  (captures period, secondary eclipses, out-of-transit variability);
* **local view** — a zoom on the transit binned to ``N_LOCAL`` points (captures
  ingress/egress slope, U-vs-V shape, depth).

Each branch is a small 1D-CNN; their features are concatenated into a dense head that
predicts the four PS7 classes. We also provide a **late-fusion ensemble** that averages the
CNN and the calibrated LightGBM probabilities (per the ultraplan — typically beats either
alone).

PyTorch is imported lazily so the view-building helpers work without a DL stack installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import config

# View geometry (TESS Astronet-Vetting defaults; smaller than Kepler's 2001/201)
N_GLOBAL = 201
N_LOCAL = 61
LOCAL_DURATIONS = 2.5     # local view spans +/- this many transit durations
VIEW_DATASET = config.FEATURES_DIR / "cnn_views.npz"
CNN_MODEL_PATH = config.FEATURES_DIR / "cnn_model.pt"
CNN_FOLD_PATH_TEMPLATE = str(config.FEATURES_DIR / "cnn_fold_{i}.pt")


# --------------------------------------------------------------------------------------
# View construction (no torch dependency)
# --------------------------------------------------------------------------------------
def _binned(phase, flux, lo, hi, nbins):
    """Median-bin ``flux`` over ``phase`` in [lo, hi]; empty bins linearly interpolated."""
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.digitize(phase, edges) - 1
    out = np.full(nbins, np.nan)
    for b in range(nbins):
        sel = flux[idx == b]
        if sel.size:
            out[b] = np.median(sel)
    # fill empty bins
    bad = ~np.isfinite(out)
    if bad.all():
        return np.zeros(nbins)
    if bad.any():
        good = ~bad
        out[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(good), out[good])
    return out


def _normalize(view):
    """Astronet normalisation: median -> 0, transit minimum -> -1."""
    med = np.median(view)
    v = view - med
    depth = -np.min(v)
    if depth > 0:
        v = v / depth
    return v.astype("float32")


def make_views(time, flat_flux, period, t0, duration,
               n_global=N_GLOBAL, n_local=N_LOCAL, local_durations=LOCAL_DURATIONS):
    """Build (global_view, local_view) from a detrended light curve and an ephemeris.

    ``flat_flux`` is the detrended flux (~1.0 baseline). ``duration`` and ``period`` in days.
    Returns two float32 arrays of length ``n_global`` and ``n_local``.
    """
    time = np.asarray(time, float)
    flux = np.asarray(flat_flux, float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]

    phase = ((time - t0 + 0.5 * period) % period) / period - 0.5     # in [-0.5, 0.5]

    g = _normalize(_binned(phase, flux, -0.5, 0.5, n_global))

    half = min(0.5, local_durations * duration / period) if period > 0 else 0.1
    half = max(half, 1.5 / n_local)          # keep at least a few bins of width
    m = np.abs(phase) <= half
    if m.sum() < n_local:                    # fall back to a wider window if too sparse
        half = min(0.5, 3 * half)
        m = np.abs(phase) <= half
    if m.sum() >= 3:
        l = _normalize(_binned(phase[m], flux[m], -half, half, n_local))
    else:
        l = np.zeros(n_local, dtype="float32")
    return g, l


# --------------------------------------------------------------------------------------
# Dataset builder (no torch dependency)
# --------------------------------------------------------------------------------------
def build_view_dataset(sample, max_sectors=4, save=True, out_path=None, verbose=True):
    """Build the CNN training tensor set from a labelled sample.

    ``sample`` is a DataFrame with columns ``tic``, ``label``, ``pl_orbper`` (as produced by
    ``labels.balanced_sample``). For each target: ingest -> detrend -> focused search at the
    known period (for t0/duration) -> make_views. Caches an ``.npz`` with arrays
    ``Xg`` (N, n_global), ``Xl`` (N, n_local), ``y`` (N,), ``tics`` (N,).
    """
    from . import ingest, detrend, search

    out_path = Path(out_path or VIEW_DATASET)
    Xg, Xl, y, tics = [], [], [], []
    rows = sample.reset_index(drop=True)
    for i, r in rows.iterrows():
        tic, label, per = r["tic"], r["label"], r.get("pl_orbper")
        try:
            star = ingest.clean(ingest.load_star(tic, max_sectors=max_sectors))
            flat = detrend.detrend(star.time, star.flux)
            if per and np.isfinite(per):
                cand = search.search_single(flat.time, flat.flux,
                                            period_min=max(per * 0.98, 0.4),
                                            period_max=per * 1.02, refine=False)
            else:
                cands = search.find_planets(flat.time, flat.flux, max_planets=1,
                                            refine=False, verbose=False)
                cand = cands[0] if cands else None
            if cand is None:
                continue
            g, l = make_views(flat.time, flat.flux, cand.period, cand.T0, cand.duration)
            Xg.append(g); Xl.append(l); y.append(label); tics.append(tic)
        except Exception as e:
            if verbose:
                print(f"[cnn] {tic} failed: {type(e).__name__}: {e}")
        if verbose and (i + 1) % 10 == 0:
            print(f"[cnn] {i+1}/{len(rows)} processed, {len(y)} good")

    Xg = np.array(Xg, dtype="float32"); Xl = np.array(Xl, dtype="float32")
    y = np.array(y); tics = np.array(tics)
    if save and len(y):
        np.savez_compressed(out_path, Xg=Xg, Xl=Xl, y=y, tics=tics)
        if verbose:
            print(f"[cnn] saved {len(y)} views -> {out_path}")
    return Xg, Xl, y, tics


def load_view_dataset(path=None):
    path = Path(path or VIEW_DATASET)
    if not path.exists():
        raise FileNotFoundError(f"No CNN view dataset at {path}. Build it first "
                                f"(see cnn_training.ipynb / build_view_dataset).")
    d = np.load(path, allow_pickle=True)
    return d["Xg"], d["Xl"], d["y"], d["tics"]


# --------------------------------------------------------------------------------------
# Model (PyTorch, imported lazily)
# --------------------------------------------------------------------------------------
def _build_model(n_classes, n_global=N_GLOBAL, n_local=N_LOCAL):
    """Deeper dual-view 1D-CNN with residual blocks (Schanche+2020-inspired).

    Global branch: 5 conv layers (1->16->32->ResBlock->64->ResBlock->128) -> 512 feats.
    Local branch : 4 conv layers (1->16->32->ResBlock->64)               -> 256 feats.
    BatchNorm on the deeper layers stabilises training; skip connections ease optimisation.
    """
    import torch
    import torch.nn as nn

    class ResBlock1d(nn.Module):
        """Conv-BN-ReLU-Conv-BN + skip (identity) connection."""
        def __init__(self, ch, k=5):
            super().__init__()
            self.conv1 = nn.Conv1d(ch, ch, k, padding=k // 2)
            self.bn1 = nn.BatchNorm1d(ch)
            self.conv2 = nn.Conv1d(ch, ch, k, padding=k // 2)
            self.bn2 = nn.BatchNorm1d(ch)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            r = self.relu(self.bn1(self.conv1(x)))
            return self.relu(self.bn2(self.conv2(r)) + x)

    def global_branch():
        return nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            ResBlock1d(32),
            nn.Conv1d(32, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            ResBlock1d(64),
            nn.Conv1d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(4), nn.Flatten())          # -> 128*4 = 512

    def local_branch():
        return nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            ResBlock1d(32),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(4), nn.Flatten())          # -> 64*4 = 256

    class DualViewCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = global_branch()
            self.l = local_branch()
            feat = 128 * 4 + 64 * 4                          # 512 + 256 = 768
            self.head = nn.Sequential(
                nn.Linear(feat, 256), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, n_classes))

        def forward(self, xg, xl):
            return self.head(torch.cat([self.g(xg), self.l(xl)], dim=1))

    return DualViewCNN()


def _augment_batch(xg, xl, rng, noise=0.02, max_roll=8, depth_jitter=0.1,
                   mirror_prob=0.5):
    """On-the-fly augmentation for the dual views (combats small-data overfitting):
    random phase-roll, phase mirroring, Gaussian noise, and depth (amplitude) jitter.
    ``xg``/``xl`` are (B,1,L) torch tensors. Returns new augmented tensors."""
    import torch
    # phase-roll the (periodic) global view
    shift = int(rng.integers(-max_roll, max_roll + 1))
    xg = torch.roll(xg, shifts=shift, dims=-1)
    # phase mirroring: horizontal flip of both views (Schanche+2020 — removing it cost
    # 3.1 AP points). Transit/EB shapes are time-symmetric, so a flip is a valid sample.
    if rng.random() < mirror_prob:
        xg = torch.flip(xg, dims=[-1])
        xl = torch.flip(xl, dims=[-1])
    # depth/amplitude jitter (both views scaled together)
    scale = float(rng.uniform(1 - depth_jitter, 1 + depth_jitter))
    xg = xg * scale
    xl = xl * scale
    # additive Gaussian noise
    xg = xg + torch.randn_like(xg) * noise
    xl = xl + torch.randn_like(xl) * noise
    return xg, xl


def train_cnn(Xg, Xl, y, n_epochs=60, batch_size=32, lr=1e-3, test_size=0.25,
              random_state=42, save=True, augment=True, verbose=True):
    """Train the dual-view CNN. Returns a bundle dict (model, classes, metrics, split).

    ``augment`` applies on-the-fly phase-roll / noise / depth-jitter to each training batch.
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix

    classes = sorted(np.unique(y).tolist())
    cls_idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([cls_idx[v] for v in y])

    Xg_tr, Xg_te, Xl_tr, Xl_te, y_tr, y_te = train_test_split(
        Xg, Xl, yi, test_size=test_size, random_state=random_state, stratify=yi)

    def _t(a):
        return torch.tensor(a, dtype=torch.float32).unsqueeze(1)   # (N,1,L)

    Xg_tr_t, Xl_tr_t = _t(Xg_tr), _t(Xl_tr)
    Xg_te_t, Xl_te_t = _t(Xg_te), _t(Xl_te)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)

    # class weights for imbalance
    counts = np.bincount(y_tr, minlength=len(classes)).astype(float)
    w = torch.tensor((counts.sum() / np.maximum(counts, 1)), dtype=torch.float32)

    model = _build_model(len(classes))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=w)

    n = Xg_tr_t.shape[0]
    rng = np.random.default_rng(random_state)
    for ep in range(n_epochs):
        model.train()
        perm = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch_size):
            b = perm[s:s + batch_size]
            if len(b) < 2:                     # BatchNorm needs >=2 samples in train mode
                continue
            xg_b, xl_b = Xg_tr_t[b], Xl_tr_t[b]
            if augment:
                xg_b, xl_b = _augment_batch(xg_b, xl_b, rng)
            opt.zero_grad()
            out = model(xg_b, xl_b)
            loss = loss_fn(out, y_tr_t[b])
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if verbose and (ep + 1) % 10 == 0:
            print(f"[cnn] epoch {ep+1}/{n_epochs}  loss={tot/n:.4f}")

    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(Xg_te_t, Xl_te_t), dim=1).numpy()
    y_pred = proba.argmax(1)
    report = classification_report(y_te, y_pred, labels=list(range(len(classes))),
                                   target_names=classes, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=list(range(len(classes))))

    if save:
        save_cnn(model, classes)
    return dict(model=model, classes=classes, proba=proba, y_test=y_te, y_pred=y_pred,
                report=report, confusion_matrix=cm)


def save_cnn(model, classes, path=None):
    import torch
    path = Path(path or CNN_MODEL_PATH)
    torch.save({"state_dict": model.state_dict(), "classes": classes}, path)
    return path


def load_cnn(path=None):
    import torch
    path = Path(path or CNN_MODEL_PATH)
    if not path.exists():
        return None, None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    classes = ckpt["classes"]
    model = _build_model(len(classes))
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError:
        # architecture changed since this checkpoint was saved -> signal "no model"
        print(f"[cnn] {path.name}: architecture mismatch (old checkpoint); retrain the CNN.")
        return None, None
    model.eval()
    return model, classes


def save_cnn_fold(model, classes, fold_idx):
    """Save one k-fold CNN model to ``cnn_fold_{i}.pt`` for inference-time bagging."""
    import torch
    path = Path(CNN_FOLD_PATH_TEMPLATE.format(i=fold_idx))
    torch.save({"state_dict": model.state_dict(), "classes": classes}, path)
    return path


def load_cnn_folds():
    """Load all saved k-fold CNN models. Returns list of (model, classes); empty if none."""
    models = []
    for i in range(10):
        p = Path(CNN_FOLD_PATH_TEMPLATE.format(i=i))
        if not p.exists():
            break
        m, classes = load_cnn(path=p)
        if m is not None:
            models.append((m, classes))
    return models


def train_cnn_cv(Xg, Xl, y, n_splits=5, n_epochs=200, batch_size=32, lr=1e-3,
                 random_state=42, augment=True, verbose=True):
    """k-fold CNN training (Schanche+2020 bagging). Saves each fold to ``cnn_fold_{i}.pt``
    so :func:`predict_cnn` / :func:`predict_ensemble` can average them at inference.

    Returns a list of per-fold info dicts."""
    import torch
    import torch.nn as nn
    from sklearn.model_selection import StratifiedKFold

    classes = sorted(np.unique(y).tolist())
    cls_idx = {c: i for i, c in enumerate(classes)}
    yi = np.array([cls_idx[v] for v in y])

    def _t(a):
        return torch.tensor(a, dtype=torch.float32).unsqueeze(1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_results = []
    for fold, (tr, _te) in enumerate(skf.split(Xg, yi)):
        if verbose:
            print(f"[cnn] fold {fold + 1}/{n_splits} ({len(tr)} train)")
        Xg_tr_t, Xl_tr_t = _t(Xg[tr]), _t(Xl[tr])
        y_tr_t = torch.tensor(yi[tr], dtype=torch.long)
        counts = np.bincount(yi[tr], minlength=len(classes)).astype(float)
        w = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
        model = _build_model(len(classes))
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = nn.CrossEntropyLoss(weight=w)
        rng = np.random.default_rng(random_state + fold)
        n = Xg_tr_t.shape[0]
        for ep in range(n_epochs):
            model.train()
            perm = rng.permutation(n)
            for s in range(0, n, batch_size):
                b = perm[s:s + batch_size]
                if len(b) < 2:                 # BatchNorm needs >=2 samples in train mode
                    continue
                xg_b, xl_b = Xg_tr_t[b], Xl_tr_t[b]
                if augment:
                    xg_b, xl_b = _augment_batch(xg_b, xl_b, rng)
                opt.zero_grad()
                loss_fn(model(xg_b, xl_b), y_tr_t[b]).backward()
                opt.step()
            if verbose and (ep + 1) % 50 == 0:
                print(f"  epoch {ep + 1}/{n_epochs}")
        save_cnn_fold(model, classes, fold)
        fold_results.append({"fold": fold, "n_train": int(len(tr))})
    return fold_results


def _cnn_tta_proba(model, gv, xl_t, n_tta=8):
    """Average softmax probabilities over ``n_tta`` phase-rolls of the global view."""
    import torch
    rolls = [0] if n_tta <= 1 else np.linspace(0, len(gv), n_tta, endpoint=False).astype(int)
    probs = []
    with torch.no_grad():
        for r in rolls:
            xg = torch.tensor(np.roll(gv, r), dtype=torch.float32).view(1, 1, -1)
            probs.append(torch.softmax(model(xg, xl_t), dim=1).numpy()[0])
    return np.mean(probs, axis=0)


# --------------------------------------------------------------------------------------
# Prediction + late-fusion ensemble
# --------------------------------------------------------------------------------------
def predict_cnn(global_view, local_view, model=None, classes=None, n_tta=8):
    """Return (class, confidence) from the CNN for one pair of views.

    Uses k-fold **bagging** (all ``cnn_fold_*.pt`` models) when available, each with
    ``n_tta`` phase-roll test-time augmentations averaged; else the single model.
    """
    import torch
    gv = np.asarray(global_view, dtype="float32")
    xl = torch.tensor(local_view, dtype=torch.float32).view(1, 1, -1)

    if model is None:
        folds = load_cnn_folds()
        if folds:
            probs = [_cnn_tta_proba(m, gv, xl, n_tta) for (m, _c) in folds]
            p = np.mean(probs, axis=0)
            classes = folds[0][1]
            i = int(p.argmax())
            return classes[i], float(p[i])
        model, classes = load_cnn()
    if model is None:
        raise RuntimeError("No trained CNN found; train it first.")
    p = _cnn_tta_proba(model, gv, xl, n_tta)
    i = int(p.argmax())
    return classes[i], float(p[i])


def predict_ensemble(features, global_view, local_view,
                     tab_model=None, cnn_model=None, cnn_classes=None, w_cnn=0.5):
    """Late-fusion of the tabular LightGBM and the dual-view CNN (averaged probabilities).

    Returns (class, confidence). Falls back gracefully to whichever model is available.
    """
    from . import classify

    x = np.nan_to_num(
        np.array([[features.get(c, np.nan) for c in config.FEATURE_COLUMNS]], float),
        nan=-99.0)

    # tabular probabilities (bagged across k folds when available, else single model)
    tab_p = None
    if tab_model is None:
        fold_models = classify.load_fold_models()
        if fold_models:
            try:
                cls, proba = classify._bagged_proba(fold_models, x)
                tab_p = dict(zip(cls, proba))
            except Exception:
                tab_p = None
        if tab_p is None:
            tab_model = classify.load_model()
    if tab_p is None and tab_model is not None:
        try:
            tab_p = dict(zip(list(tab_model.classes_), tab_model.predict_proba(x)[0]))
        except Exception:
            tab_p = None        # stale model (feature-count mismatch) -> skip tabular

    # cnn probabilities (bagged across k folds when available; each with TTA)
    cnn_p = None
    if cnn_model is None:
        folds = load_cnn_folds()
        if folds:
            import torch
            gv = np.asarray(global_view, dtype="float32")
            xl_t = torch.tensor(local_view, dtype=torch.float32).view(1, 1, -1)
            probs = [_cnn_tta_proba(m, gv, xl_t, 8) for (m, _c) in folds]
            cnn_p = dict(zip(folds[0][1], np.mean(probs, axis=0)))
        else:
            cnn_model, cnn_classes = load_cnn()
    if cnn_p is None and cnn_model is not None:
        import torch
        gv = np.asarray(global_view, dtype="float32")
        xl_t = torch.tensor(local_view, dtype=torch.float32).view(1, 1, -1)
        cnn_p = dict(zip(cnn_classes, _cnn_tta_proba(cnn_model, gv, xl_t, 8)))

    if tab_p is None and cnn_p is None:
        return classify.predict(features)
    if cnn_p is None:
        i = max(tab_p, key=tab_p.get); return i, float(tab_p[i])
    if tab_p is None:
        i = max(cnn_p, key=cnn_p.get); return i, float(cnn_p[i])

    classes = sorted(set(tab_p) | set(cnn_p))
    fused = {c: (1 - w_cnn) * tab_p.get(c, 0.0) + w_cnn * cnn_p.get(c, 0.0)
             for c in classes}
    i = max(fused, key=fused.get)
    return i, float(fused[i])
