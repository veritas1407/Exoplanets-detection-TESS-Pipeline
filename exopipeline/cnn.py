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
    import torch.nn as nn

    def branch(out_len):
        return nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.AdaptiveMaxPool1d(out_len),
            nn.Flatten())

    class DualViewCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.g = branch(4)
            self.l = branch(4)
            feat = 64 * 4 + 64 * 4
            self.head = nn.Sequential(
                nn.Linear(feat, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, n_classes))

        def forward(self, xg, xl):
            import torch
            return self.head(torch.cat([self.g(xg), self.l(xl)], dim=1))

    return DualViewCNN()


def train_cnn(Xg, Xl, y, n_epochs=60, batch_size=32, lr=1e-3, test_size=0.25,
              random_state=42, save=True, verbose=True):
    """Train the dual-view CNN. Returns a bundle dict (model, classes, metrics, split)."""
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
            opt.zero_grad()
            out = model(Xg_tr_t[b], Xl_tr_t[b])
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
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes


# --------------------------------------------------------------------------------------
# Prediction + late-fusion ensemble
# --------------------------------------------------------------------------------------
def predict_cnn(global_view, local_view, model=None, classes=None):
    """Return (class, confidence) from the CNN for one pair of views."""
    import torch
    if model is None:
        model, classes = load_cnn()
    if model is None:
        raise RuntimeError("No trained CNN found; train it first.")
    xg = torch.tensor(global_view, dtype=torch.float32).view(1, 1, -1)
    xl = torch.tensor(local_view, dtype=torch.float32).view(1, 1, -1)
    with torch.no_grad():
        p = torch.softmax(model(xg, xl), dim=1).numpy()[0]
    i = int(p.argmax())
    return classes[i], float(p[i])


def predict_ensemble(features, global_view, local_view,
                     tab_model=None, cnn_model=None, cnn_classes=None, w_cnn=0.5):
    """Late-fusion of the tabular LightGBM and the dual-view CNN (averaged probabilities).

    Returns (class, confidence). Falls back gracefully to whichever model is available.
    """
    from . import classify

    # tabular probabilities
    if tab_model is None:
        tab_model = classify.load_model()
    tab_p = None
    if tab_model is not None:
        x = np.nan_to_num(
            np.array([[features.get(c, np.nan) for c in config.FEATURE_COLUMNS]], float),
            nan=-99.0)
        tab_p = dict(zip(list(tab_model.classes_), tab_model.predict_proba(x)[0]))

    # cnn probabilities
    cnn_p = None
    if cnn_model is None:
        cnn_model, cnn_classes = load_cnn()
    if cnn_model is not None:
        import torch
        xg = torch.tensor(global_view, dtype=torch.float32).view(1, 1, -1)
        xl = torch.tensor(local_view, dtype=torch.float32).view(1, 1, -1)
        with torch.no_grad():
            p = torch.softmax(cnn_model(xg, xl), dim=1).numpy()[0]
        cnn_p = dict(zip(cnn_classes, p))

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
