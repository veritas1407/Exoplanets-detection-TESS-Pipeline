"""Parallel shared builder — one pass per target produces BOTH a vetting-feature row and
the dual-view CNN tensors, for the large training set.

Reuses the ProcessPoolExecutor pattern from ``scan.scan_slice`` (the per-target cost is the
CPU-bound BLS search). Resumable: checkpoints features to parquet and views to npz.

Per target:
  * known-period (transit / EB)  -> focused ``search.search_single`` at the catalog period;
  * field star 'other' (no period) -> strongest *blind* BLS peak (``find_planets``, low
    threshold) — the test-time false-positive scenario.
Then ``vetting.compute_features`` (+ stellar density) and ``cnn.make_views``.
"""
from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from . import config, ingest, detrend, search, vetting, classify, cnn


def _featurize_one(row: dict) -> dict | None:
    tic, label, per = row.get("tic"), row.get("label"), row.get("pl_orbper")
    try:
        star = ingest.clean(ingest.load_star(tic, max_sectors=4))
        if star.time.size < 200:
            return None
        flat = detrend.detrend(star.time, star.flux)
        if per is not None and np.isfinite(per) and per > 0:
            cand = search.search_single(flat.time, flat.flux,
                                        max(per * 0.98, 0.4), per * 1.02, refine=False)
        else:
            baseline = float(flat.time.max() - flat.time.min())
            pmax = min(config.SCAN_PERIOD_MAX_CAP,
                       max(2.0, baseline * config.SCAN_PERIOD_MAX_FRAC))
            cands = search.find_planets(flat.time, flat.flux, max_planets=1, refine=False,
                                        verbose=False, period_max=pmax,
                                        n_periods=config.SCAN_N_PERIODS, sde_threshold=-1.0)
            cand = cands[0] if cands else None
        if cand is None:
            return None
        rstar, mstar = ingest.fetch_stellar(tic)
        feats = vetting.compute_features(flat.time, flat.flux, cand,
                                         crowdsap=star.crowdsap,
                                         rstar_sun=rstar, mstar_sun=mstar)
        g, l = cnn.make_views(flat.time, flat.flux, cand.period, cand.T0, cand.duration)
        out = classify.features_to_row(feats, label=label, target=tic)
        out["tid"] = row.get("tid")
        out["_g"] = g.tolist()
        out["_l"] = l.tolist()
        return out
    except Exception as e:
        print(f"[featurize] {tic} failed: {type(e).__name__}: {e}")
        return None


def build_training_set(sample: pd.DataFrame, feat_path=None, view_path=None,
                       n_workers: int | None = None, resume: bool = True,
                       checkpoint_every: int = 50, verbose: bool = True):
    """Build features + views for ``sample`` (cols: tic, label, pl_orbper[, tid]) in parallel.

    Caches features -> ``feat_path`` (parquet) and views -> ``view_path`` (npz). Resumable.
    Returns ``(feat_df, Xg, Xl, y, tics)``.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    feat_path = feat_path or (config.FEATURES_DIR / "train_features.parquet")
    view_path = view_path or (config.FEATURES_DIR / "train_views.npz")
    if n_workers is None:
        n_workers = config.SCAN_WORKERS
    if not n_workers:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    sample = sample.drop_duplicates("tic").reset_index(drop=True)
    done = set()
    feat_rows: list[dict] = []
    views_g: list = []
    views_l: list = []
    if resume and feat_path.exists() and view_path.exists():
        try:
            prev = pd.read_parquet(feat_path)
            d = np.load(view_path, allow_pickle=True)
            feat_rows = prev.to_dict("records")
            views_g = list(d["Xg"]); views_l = list(d["Xl"])
            done = set(prev["target"].tolist())
            if verbose:
                print(f"[featurize] resuming: {len(done)} already built")
        except Exception:
            feat_rows, views_g, views_l, done = [], [], [], set()

    todo = sample[~sample["tic"].isin(done)]
    rows = todo.to_dict("records")
    if verbose:
        print(f"[featurize] {len(sample)} targets, {len(rows)} to build, {n_workers} workers")

    # Prefetch: warm data/cache/ with a large I/O-bound thread pool BEFORE the CPU-bound
    # BLS stage starts, so the ProcessPoolExecutor below hits an already-cached FITS per
    # target instead of each worker separately serializing a MAST search+download.
    if rows:
        ingest.prefetch_targets(todo["tic"].tolist(), max_sectors=4, verbose=verbose)

    def _save():
        """Write the checkpoint, retrying on transient filesystem errors (e.g. an
        external/secondary drive briefly dropping out) instead of crashing the whole
        build. ``feat_rows``/``views_g``/``views_l`` keep accumulating in memory across
        calls, so a skipped checkpoint just means the next one carries everything again.
        """
        import time as _t

        df = pd.DataFrame([{k: v for k, v in r.items()} for r in feat_rows])
        last_err = None
        for attempt in range(5):
            try:
                feat_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(feat_path, index=False)
                np.savez_compressed(view_path,
                                    Xg=np.array(views_g, "float32"),
                                    Xl=np.array(views_l, "float32"),
                                    y=df["label"].values, tics=df["target"].values)
                return
            except OSError as e:
                last_err = e
                print(f"[featurize] checkpoint write failed (attempt {attempt+1}/5): "
                     f"{e}; retrying...")
                _t.sleep(5 * (attempt + 1))
        print(f"[featurize] checkpoint SKIPPED after 5 retries ({last_err}); "
             f"will retry at the next checkpoint interval.")

    t0, n_new = _time.time(), 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for res in pool.map(_featurize_one, rows, chunksize=4):
            if res is not None:
                g = res.pop("_g"); l = res.pop("_l")
                feat_rows.append(res); views_g.append(g); views_l.append(l)
            n_new += 1
            if verbose and n_new % 20 == 0:
                rate = n_new / max(_time.time() - t0, 1e-9)
                print(f"  {n_new}/{len(rows)} ({rate:.2f}/s, ETA {(len(rows)-n_new)/max(rate,1e-9)/60:.0f} min)"
                      f"  good={len(feat_rows)}")
            if n_new % checkpoint_every == 0 and feat_rows:
                _save()
    if feat_rows:
        _save()

    df = pd.read_parquet(feat_path)
    d = np.load(view_path, allow_pickle=True)
    return df, d["Xg"], d["Xl"], d["y"], d["tics"]
