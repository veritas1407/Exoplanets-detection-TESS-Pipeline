"""Batch sector scan — the PS7 O2->O6 driver over a real TESS sector.

This is the data-layer the problem statement asks for: instead of pulling hand-picked
targets, take one sector's 2-minute light curves (parsed from the MAST bulk-download
script) and run the pipeline **blind** across the whole slice.

Two tiers (standard SPOC/ExoMiner design — you cannot/should not MCMC 20k stars):

* **Tier 1** (:func:`scan_target` / :func:`scan_slice`) runs on EVERY star: detect
  (blind BLS), compute vetting *features*, classify the **type** (transit / EB / blend /
  other), and report **significance** (SDE, SNR, FAP).  -> O2, O4, O5, O6-significance.
* **Tier 2** (:func:`characterize_top`) runs only on the detected events: full
  ``batman``+``emcee`` shape fit + blend test + vetting sheet.  -> O3, O6-parameters.

Everything is checkpointed to a CSV so a long run is resumable.
"""
from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from . import config, ingest, detrend, search, vetting, classify


# --------------------------------------------------------------------------------------
# Slice construction
# --------------------------------------------------------------------------------------
def build_slice(sector: int | None = None, n: int | None = None,
                include_tois: bool = True) -> pd.DataFrame:
    """Build the list of targets to scan for one sector.

    Takes the first ``n`` manifest rows (an unbiased slice of the ~20k, *not* a cherry
    pick) and, if ``include_tois``, unions in every TOI-labelled TIC that falls in this
    sector so the known planets/EBs are present for training-validation.

    Returns a DataFrame with columns: tic, tid, lc_file, url, sector, is_known_toi,
    true_label, known_period.
    """
    sector = int(sector if sector is not None else config.DEFAULT_SECTOR)
    n = config.SLICE_SIZE if n is None else n

    manifest = ingest.sector_lc_manifest(sector)
    slice_df = manifest.head(n).copy() if n else manifest.copy()

    # Attach TOI labels (known dataset) where available.
    toi = _toi_lookup()
    if include_tois and toi is not None:
        in_sector = manifest[manifest["tid"].isin(toi.index)]
        slice_df = pd.concat([slice_df, in_sector], ignore_index=True)
    slice_df = slice_df.drop_duplicates(subset="tid").reset_index(drop=True)

    if toi is not None:
        slice_df["true_label"] = slice_df["tid"].map(toi["label"])
        slice_df["known_period"] = slice_df["tid"].map(toi["pl_orbper"])
    else:
        slice_df["true_label"] = np.nan
        slice_df["known_period"] = np.nan
    slice_df["is_known_toi"] = slice_df["true_label"].notna()
    return slice_df


def _toi_lookup():
    """TOI catalog indexed by TIC id (tid) -> label / pl_orbper, or None if unavailable."""
    try:
        from . import labels
        toi = labels.fetch_toi()
        toi = toi.dropna(subset=["tid"]).copy()
        toi["tid"] = toi["tid"].astype(int)
        return toi.drop_duplicates(subset="tid").set_index("tid")
    except Exception as e:
        print(f"[scan] TOI lookup unavailable ({e}); proceeding without labels.")
        return None


# --------------------------------------------------------------------------------------
# Tier 1 — detect + classify + significance (cheap, runs on EVERY star)
# --------------------------------------------------------------------------------------
# Stable result schema so the candidate CSV always has the same columns, regardless of
# whether a given star yields a detection.
_RESULT_FIELDS = [
    "tic", "tid", "sector", "is_known_toi", "true_label", "status", "n_points",
    "crowdsap", "period", "t0", "depth_ppm", "duration_hr", "sde", "snr", "fap",
    "rp_rs", "n_transits", "pred_class", "confidence",
    "odd_even_sigma", "secondary_ppm", "flatness",
]


def _empty_result(row: dict) -> dict:
    r = {k: np.nan for k in _RESULT_FIELDS}
    r.update(tic=row.get("tic"), tid=row.get("tid"), sector=row.get("sector"),
             is_known_toi=bool(row.get("is_known_toi", False)),
             true_label=row.get("true_label"))
    return r


def scan_target(row: dict, keep_fits: bool = True, model=None) -> dict:
    """Run the cheap detection+classification tier on one target.

    Always returns a full-schema result row (``status`` records the outcome).
    """
    res = _empty_result(row)
    tic = row.get("tic")
    try:
        star = ingest.load_lc_from_url(row["url"], lc_file=row.get("lc_file"),
                                       target=tic, keep_fits=keep_fits)
        star = ingest.clean(star)
        res["n_points"] = int(star.time.size)
        res["crowdsap"] = star.crowdsap
        if star.time.size < 200:
            res["status"] = "too_few_points"
            return res

        flat = detrend.detrend(star.time, star.flux)
        cands = search.find_planets(flat.time, flat.flux, max_planets=1,
                                    refine=False, verbose=False)
        if not cands:
            res["status"] = "no_detection"
            return res

        cand = cands[0]
        feats = vetting.compute_features(flat.time, flat.flux, cand,
                                         crowdsap=star.crowdsap)
        pred_class, conf = classify.predict(feats, model=model)
        res.update(
            status="ok", period=cand.period, t0=cand.T0,
            depth_ppm=cand.depth_ppm, duration_hr=cand.duration_hr,
            sde=cand.SDE, snr=cand.snr, fap=cand.FAP, rp_rs=cand.rp_rs,
            n_transits=cand.distinct_transit_count,
            pred_class=pred_class, confidence=conf,
            odd_even_sigma=feats.get("odd_even_sigma"),
            secondary_ppm=feats.get("secondary_ppm"),
            flatness=feats.get("flatness"))
        return res
    except Exception as e:
        res["status"] = f"error: {type(e).__name__}: {e}"
        return res


def scan_slice(sector: int | None = None, n: int | None = None,
               out_csv=None, keep_fits: bool = True, resume: bool = True,
               model=None, limit: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Scan a sector slice (Tier 1), checkpointing to CSV. Resumable.

    Parameters
    ----------
    n : int, optional
        Slice size (``config.SLICE_SIZE`` default; ``None`` via build_slice = full sector).
    out_csv : path, optional
        Candidate CSV (default ``data/features/sector_{sector}_candidates.csv``).
    keep_fits : bool
        Keep downloaded FITS in ``data/cache`` (False = streaming, deletes after load).
    resume : bool
        Skip TICs already present in ``out_csv``.
    limit : int, optional
        Hard cap on number of targets processed this call (handy for smoke tests).
    """
    sector = int(sector if sector is not None else config.DEFAULT_SECTOR)
    out_csv = out_csv or (config.FEATURES_DIR / f"sector_{sector}_candidates.csv")

    targets = build_slice(sector, n=n)
    done = set()
    if resume and out_csv.exists():
        try:
            prev = pd.read_csv(out_csv)
            done = set(prev["tid"].dropna().astype(int).tolist())
            if verbose:
                print(f"[scan] resuming: {len(done)} targets already done")
        except Exception:
            pass

    todo = targets[~targets["tid"].isin(done)]
    if limit:
        todo = todo.head(limit)

    if verbose:
        print(f"[scan] sector {sector}: {len(targets)} in slice, "
              f"{len(todo)} to process this run")

    results, t_start, ck = [], _time.time(), config.SCAN_CHECKPOINT_EVERY
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        res = scan_target(row.to_dict(), keep_fits=keep_fits, model=model)
        if res is not None:
            results.append(res)
        if verbose and (i % 10 == 0 or i == len(todo)):
            rate = i / max(_time.time() - t_start, 1e-9)
            print(f"  {i}/{len(todo)}  ({rate:.2f} targets/s)  last={res.get('status')}")
        if results and (i % ck == 0):
            _checkpoint(results, out_csv)
    if results:
        _checkpoint(results, out_csv)

    return pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(results)


def _checkpoint(new_rows: list[dict], out_csv):
    """Append new rows to the candidate CSV (merging with any existing, dedup on tid)."""
    df_new = pd.DataFrame(new_rows)
    if out_csv.exists():
        df = pd.concat([pd.read_csv(out_csv), df_new], ignore_index=True)
        df = df.drop_duplicates(subset="tid", keep="last")
    else:
        df = df_new
    df.to_csv(out_csv, index=False)


# --------------------------------------------------------------------------------------
# Tier 2 — full shape characterisation (expensive, detected events only)
# --------------------------------------------------------------------------------------
def rank_candidates(df: pd.DataFrame, only_transit: bool = True,
                    min_sde: float | None = None) -> pd.DataFrame:
    """Rank scan rows for follow-up: predicted transits first, by SDE."""
    out = df[df["status"] == "ok"].copy() if "status" in df else df.copy()
    if "sde" not in out.columns or out.empty:
        return out.iloc[0:0]
    if only_transit and "pred_class" in out:
        out = out[out["pred_class"] == "transit"]
    if min_sde is None:
        min_sde = config.SDE_THRESHOLD
    out = out[out["sde"] >= min_sde]
    return out.sort_values("sde", ascending=False).reset_index(drop=True)


def characterize_top(df: pd.DataFrame, k: int = 3, save_dir=None,
                     stellar=None, nsteps: int = 1500, nburn: int = 500,
                     verbose: bool = True) -> list[dict]:
    """Tier-2 follow-up on the top ``k`` ranked detections: full MCMC fit + vetting sheet.

    Parameters
    ----------
    stellar : dict, optional
        ``{tid: (rstar_sun, mstar_sun)}`` to supply density priors for specific targets.
    save_dir : path, optional
        Where to write each ``vetting_sheet_TIC*.png`` (default ``data/features``).
    """
    from . import fit, report

    save_dir = save_dir or config.FEATURES_DIR
    stellar = stellar or {}
    ranked = rank_candidates(df)
    out = []
    for _, r in ranked.head(k).iterrows():
        tic, tid = r["tic"], int(r["tid"])
        if verbose:
            print(f"[characterize] {tic}  P={r['period']:.4f} d  SDE={r['sde']:.1f}")
        try:
            star = ingest.clean(ingest.load_star(tic, max_sectors=None))
            flat = detrend.to_flattened(star)
            cand = search.search_single(
                flat.time, flat.flux,
                period_min=r["period"] * 0.98, period_max=r["period"] * 1.02,
                refine=False)
            feats = vetting.compute_features(flat.time, flat.flux, cand,
                                             crowdsap=star.crowdsap)
            verdict, conf = classify.predict(feats)
            rstar, mstar = stellar.get(tid, (config.DEFAULT_RSTAR_SUN, None))
            fr = fit.fit_transit(flat.time, flat.flux, cand, crowdsap=star.crowdsap,
                                 rstar_sun=rstar,
                                 mstar_sun=mstar if mstar else rstar,
                                 nsteps=nsteps, nburn=nburn)
            sheet = save_dir / f"vetting_sheet_{tic.replace(' ', '_')}.png"
            report.vetting_sheet(star, flat, cand, feats, fr, verdict, conf,
                                 title=f"{tic} — Vetting Sheet (Sector {r['sector']})",
                                 save_path=str(sheet))
            out.append({"tic": tic, "verdict": verdict, "confidence": conf,
                        "Rp": fr.medians.get("Rp", [np.nan])[0],
                        "period": fr.medians.get("period", [np.nan])[0],
                        "sheet": str(sheet)})
        except Exception as e:
            print(f"[characterize] {tic} failed: {type(e).__name__}: {e}")
    return out
