"""Label assembly — build a training set from public catalogs.

Pulls the TOI catalog (NASA Exoplanet Archive) and maps ExoFOP dispositions to the PS's
four classes, then (optionally) runs the pipeline on a sample of targets to build the
feature table used by ``classify.train``.

Class mapping (from ``tfopwg_disp``):
    CP, KP, PC  -> transit            (confirmed / known / candidate planet)
    FP          -> eclipsing_binary   (most FPs are EBs; refined by centroid test)
    FA          -> other              (false alarm / noise)
    APC         -> dropped (ambiguous)
The dedicated TESS-EB catalog can be merged in for cleaner EB labels, and quiet stars for
the 'other' class. Kept deliberately simple and cache-backed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

TOI_TAP = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
           "query=select+toi,tid,toipfx,tfopwg_disp,pl_orbper,pl_trandurh,"
           "pl_trandep,st_tmag+from+toi&format=csv")

DISP_MAP = {
    "CP": "transit", "KP": "transit", "PC": "transit",
    "FP": "eclipsing_binary",
    "FA": "other",
}

# Clean mapping (preferred): only *confirmed* planets are "transit"; EB labels come from the
# dedicated TESS EB catalog (not FP, which is a noisy grab-bag of blends/systematics);
# FA (false alarm) -> "other". This makes the classes physically separable in transit-shape
# space, which the noisy FP->EB mapping does not (transit and FP overlap almost completely).
CLEAN_DISP_MAP = {
    "CP": "transit", "KP": "transit",     # confirmed / known planets only
    "FA": "other",                        # false alarms / noise / systematics
}

# Prsa et al. 2022, "TESS Eclipsing Binary Stars. I." (ApJS 258, 16) via VizieR.
TESS_EB_VIZIER = "J/ApJS/258/16"


def fetch_toi(force=False) -> pd.DataFrame:
    """Download (and cache) the TOI catalog with mapped labels (noisy FP->EB mapping)."""
    path = config.LABELS_DIR / "toi_catalog.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    df = pd.read_csv(TOI_TAP)
    df["label"] = df["tfopwg_disp"].map(DISP_MAP)
    df = df.dropna(subset=["label", "tid", "pl_orbper"])
    df["tic"] = "TIC " + df["tid"].astype(int).astype(str)
    df.to_csv(path, index=False)
    return df


def fetch_tess_eb(force=False) -> pd.DataFrame:
    """Download (and cache) the Prsa+ 2022 TESS Eclipsing Binary catalog as clean EB labels.

    Returns a DataFrame with columns ``tic, tid, pl_orbper, st_tmag, label`` where
    ``label == 'eclipsing_binary'``.
    """
    path = config.LABELS_DIR / "tess_eb_catalog.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    from astroquery.vizier import Vizier
    v = Vizier(columns=["TIC", "Per", "Tmag", "Morph"])
    v.ROW_LIMIT = -1
    tbl = v.get_catalogs(TESS_EB_VIZIER)[0].to_pandas()
    tbl = tbl.rename(columns={"TIC": "tid", "Per": "pl_orbper", "Tmag": "st_tmag"})
    tbl = tbl.dropna(subset=["tid", "pl_orbper"])
    tbl["tid"] = tbl["tid"].astype(int)
    tbl["tic"] = "TIC " + tbl["tid"].astype(str)
    tbl["label"] = "eclipsing_binary"
    tbl = tbl[(tbl["pl_orbper"] > 0)].drop_duplicates("tid")
    tbl.to_csv(path, index=False)
    return tbl


def fetch_toi_clean(force=False) -> pd.DataFrame:
    """TOI catalog with the *clean* mapping (CP/KP -> transit, FA -> other; FP dropped)."""
    df = fetch_toi(force=force).copy()
    df["label"] = df["tfopwg_disp"].map(CLEAN_DISP_MAP)
    return df.dropna(subset=["label", "tid", "pl_orbper"])


def build_clean_sample(per_class=80, tmag_max=12.5, seed=42) -> pd.DataFrame:
    """Class-balanced sample with CLEAN labels: confirmed planets (transit), real EBs from
    the TESS EB catalog (eclipsing_binary), and TOI false alarms (other).

    Columns: tic, tid, label, pl_orbper, st_tmag.
    """
    cols = ["tic", "tid", "label", "pl_orbper", "st_tmag"]
    toi = fetch_toi_clean()
    eb = fetch_tess_eb()
    pool = pd.concat([toi[toi["label"].isin(["transit", "other"])][cols], eb[cols]],
                     ignore_index=True)
    return balanced_sample(pool, per_class=per_class, tmag_max=tmag_max, seed=seed)


def balanced_sample(df: pd.DataFrame, per_class=60, tmag_max=12.5,
                    seed=42) -> pd.DataFrame:
    """Pick a bright, roughly class-balanced development sample (small on purpose)."""
    rng = np.random.default_rng(seed)
    bright = df[df["st_tmag"] <= tmag_max] if "st_tmag" in df else df
    parts = []
    for cls, g in bright.groupby("label"):
        g = g.drop_duplicates(subset="tic")
        take = min(per_class, len(g))
        parts.append(g.sample(take, random_state=int(rng.integers(1e6))))
    return pd.concat(parts, ignore_index=True)


def build_feature_row(tic, label, known_period=None, max_sectors=4):
    """Run ingest -> detrend -> search -> vetting on one target; return a feature row.

    If ``known_period`` is given, a focused search around it is used (fast + reliable for
    labelled training data). Returns ``None`` on any failure (caller skips it).
    """
    from . import ingest, detrend, search, vetting, classify
    try:
        star = ingest.clean(ingest.load_star(tic, max_sectors=max_sectors))
        flat = detrend.detrend(star.time, star.flux)
        if known_period and np.isfinite(known_period):
            cand = search.search_single(flat.time, flat.flux,
                                        period_min=max(known_period * 0.98, 0.4),
                                        period_max=known_period * 1.02, refine=False)
        else:
            cands = search.find_planets(flat.time, flat.flux, max_planets=1,
                                        refine=False, verbose=False)
            cand = cands[0] if cands else None
        if cand is None:
            return None
        feats = vetting.compute_features(flat.time, flat.flux, cand,
                                         crowdsap=star.crowdsap)
        return classify.features_to_row(feats, label=label, target=tic)
    except Exception as e:
        print(f"[labels] {tic} failed: {e}")
        return None


def build_feature_table(sample: pd.DataFrame, max_sectors=4, save=True) -> pd.DataFrame:
    """Build the full feature table from a labelled sample dataframe (tic, label,
    pl_orbper). Caches to ``config.FEATURE_TABLE``."""
    rows = []
    for i, r in sample.reset_index(drop=True).iterrows():
        row = build_feature_row(r["tic"], r["label"],
                                known_period=r.get("pl_orbper"), max_sectors=max_sectors)
        if row is not None:
            rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"[labels] {i+1}/{len(sample)} processed, {len(rows)} good")
    df = pd.DataFrame(rows)
    if save and len(df):
        df.to_parquet(config.FEATURE_TABLE, index=False)
        print(f"[labels] saved {len(df)} rows -> {config.FEATURE_TABLE}")
    return df
