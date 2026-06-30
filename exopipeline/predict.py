"""Explainable inference entrypoint — the single API the (unknown) test harness calls.

A raw light curve goes in; a structured, *explained* verdict comes out:
  detect -> classify (transit / eclipsing_binary / other) -> significance -> (optional)
  parameters -> (optional) blend flag, plus a human-readable paragraph justifying the call.

    from exopipeline import predict
    v = predict.predict_target("TIC 259377017", fit=False)
    print(v.summary())

Robust to: (time, flux) arrays OR a FITS path OR a TIC id; single/multi-sector; gaps/NaNs;
and the no-detection case (-> 'other').
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config, ingest, detrend, search, vetting, cnn, classify


@dataclass
class Verdict:
    target: str = ""
    event_detected: bool = False
    classification: str = "other"
    confidence: float = 0.0
    class_probs: dict = field(default_factory=dict)
    # event / significance
    period: float = np.nan
    t0: float = np.nan
    depth_ppm: float = np.nan
    duration_hr: float = np.nan
    sde: float = np.nan
    snr: float = np.nan
    fap: float = np.nan
    n_transits: int = 0
    # physical parameters (filled if fit=True)
    rp_earth: float = np.nan
    rp_rs: float = np.nan
    # blend
    is_blend: bool = False
    centroid_offset_arcsec: float = np.nan
    # explanation
    reasons: list = field(default_factory=list)
    explanation: str = ""

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k not in ("reasons",)}
        return d

    def summary(self) -> str:
        return self.explanation


# Friendly names for the report
_NAME = {"transit": "Planet candidate (transiting)",
         "eclipsing_binary": "Eclipsing binary",
         "other": "No planet / variable or noise"}


def _explain(cls, conf, feats, v: Verdict) -> tuple[str, list]:
    """Build human-readable reasons + paragraph from the features."""
    r = []
    depth = feats.get("depth_ppm", np.nan)
    sec = feats.get("secondary_snr", 0) or 0
    oe = feats.get("odd_even_sigma", 0) or 0
    vsh = feats.get("vshape_ratio", np.nan)
    rho = feats.get("rho_ratio", np.nan)
    sde = feats.get("sde", np.nan)
    snr = feats.get("snr", np.nan)

    if v.event_detected:
        r.append(f"a periodic {depth:.0f} ppm dip at P={v.period:.4f} d "
                 f"({v.n_transits} transits, SDE={sde:.1f}, SNR={snr:.1f})")
    if np.isfinite(rho) and abs(rho) > 0.5:
        r.append(f"transit-implied stellar density {10**rho:.1f}x the catalog value "
                 f"(|log rho ratio|={abs(rho):.1f})")
    if sec > 3:
        r.append(f"a secondary eclipse at phase 0.5 ({sec:.1f} sigma)")
    if oe > 3:
        r.append(f"odd/even depth difference ({oe:.1f} sigma)")
    if np.isfinite(vsh) and vsh > 1.3:
        r.append(f"a V-shaped (grazing) profile (wing/core depth {vsh:.1f})")
    if v.is_blend:
        r.append(f"an off-target photocenter ({v.centroid_offset_arcsec:.1f}\" -> blend)")

    head = f"{_NAME.get(cls, cls)} ({conf*100:.0f}% confidence)"
    if not v.event_detected:
        para = (f"{head}: no significant periodic transit-like dip was found "
                f"(strongest signal below the detection threshold) -> classified 'other'.")
        return para, r
    why = "; ".join(r) if r else "transit-like shape with no eclipsing-binary signatures"
    para = f"{head}. Evidence: {why}."
    if cls == "transit" and np.isfinite(v.rp_earth):
        para += f" Fitted radius Rp ~ {v.rp_earth:.2f} R_earth."
    return para, r


def predict_lightcurve(time, flux, flux_err=None, target="", tpf=None,
                       stellar=None, fit=False, sheet_path=None,
                       sde_threshold=None) -> Verdict:
    """Classify one raw light curve end-to-end and return an explained :class:`Verdict`."""
    time = np.asarray(time, dtype="float64")
    flux = np.asarray(flux, dtype="float64")
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    v = Verdict(target=str(target))
    if time.size < 200:
        v.explanation = "Too few valid points to analyse."
        return v

    # normalise + detrend
    med = np.nanmedian(flux)
    if np.isfinite(med) and med != 0:
        flux = flux / med
    flat = detrend.detrend(time, flux)

    # blind detection (single best candidate, single-sector-sized grid)
    baseline = float(flat.time.max() - flat.time.min())
    pmax = min(config.PERIOD_MAX, max(2.0, baseline * config.SCAN_PERIOD_MAX_FRAC)) \
        if baseline < 60 else config.PERIOD_MAX
    sde_thr = config.SDE_THRESHOLD if sde_threshold is None else sde_threshold
    cands = search.find_planets(flat.time, flat.flux, max_planets=1, refine=False,
                                verbose=False, period_max=pmax,
                                n_periods=config.SCAN_N_PERIODS, sde_threshold=sde_thr)

    # stellar params for the density feature
    if stellar is None and target:
        stellar = ingest.fetch_stellar(target)
    rstar, mstar = (stellar or (np.nan, np.nan))

    if not cands:
        # no significant event -> 'other'
        v.classification, v.confidence = "other", 0.7
        v.explanation, v.reasons = _explain("other", 0.7, {}, v)
        return v

    cand = cands[0]
    v.event_detected = True
    v.period, v.t0 = cand.period, cand.T0
    v.depth_ppm, v.duration_hr = cand.depth_ppm, cand.duration_hr
    v.sde, v.snr, v.fap = cand.SDE, cand.snr, cand.FAP
    v.n_transits = int(cand.distinct_transit_count)
    v.rp_rs = cand.rp_rs

    feats = vetting.compute_features(flat.time, flat.flux, cand, crowdsap=np.nan,
                                     rstar_sun=rstar, mstar_sun=mstar)
    g, l = cnn.make_views(flat.time, flat.flux, cand.period, cand.T0, cand.duration)
    cls, conf = cnn.predict_ensemble(feats, g, l)
    v.classification, v.confidence = cls, float(conf)

    # blend / centroid test
    if tpf is not None:
        from . import blend
        bt = blend.centroid_test(tpf, cand.period, cand.T0, cand.duration)
        v.is_blend = bool(bt.get("is_blend", False))
        v.centroid_offset_arcsec = float(bt.get("offset_arcsec", np.nan))

    # optional physical parameters (MCMC)
    if fit and cls == "transit":
        try:
            from . import fit as _fit
            fr = _fit.fit_transit(flat.time, flat.flux, cand, crowdsap=np.nan,
                                  rstar_sun=(rstar if np.isfinite(rstar) else config.DEFAULT_RSTAR_SUN),
                                  mstar_sun=(mstar if np.isfinite(mstar) else config.DEFAULT_RSTAR_SUN),
                                  nsteps=1500, nburn=500)
            v.rp_earth = float(fr.medians.get("Rp", [np.nan])[0])
        except Exception:
            pass

    v.explanation, v.reasons = _explain(cls, conf, feats, v)

    if sheet_path:
        try:
            from . import report, fit as _fit
            fr = _fit.fit_transit(flat.time, flat.flux, cand, crowdsap=np.nan,
                                  rstar_sun=(rstar if np.isfinite(rstar) else config.DEFAULT_RSTAR_SUN),
                                  mstar_sun=(mstar if np.isfinite(mstar) else config.DEFAULT_RSTAR_SUN),
                                  nsteps=1000, nburn=300)
            star_ns = type("S", (), {"target": target, "raw_lc": None,
                                     "crowdsap": np.nan, "time": time, "flux": flux})
            report.vetting_sheet(star_ns, flat, cand, feats, fr, cls, conf,
                                 title=f"{target} — Vetting Sheet", save_path=sheet_path)
        except Exception as e:
            print(f"[predict] vetting sheet skipped: {e}")
    return v


def predict_fits(path, target="", **kw) -> Verdict:
    """Classify a single SPOC light-curve FITS file on disk."""
    star = ingest.clean(ingest.load_lc_from_file(path, target=target or None))
    return predict_lightcurve(star.time, star.flux, target=star.target, **kw)


def predict_target(tic, max_sectors=4, **kw) -> Verdict:
    """Classify a TIC by downloading its SPOC light curve(s)."""
    star = ingest.clean(ingest.load_star(tic, max_sectors=max_sectors))
    return predict_lightcurve(star.time, star.flux, target=tic, **kw)


def predict_batch(items, kind="auto", **kw):
    """Classify a list of FITS paths / TICs / (time,flux) tuples -> DataFrame of verdicts."""
    import pandas as pd
    rows = []
    for it in items:
        try:
            if isinstance(it, (tuple, list)) and len(it) == 2 and hasattr(it[0], "__len__"):
                v = predict_lightcurve(it[0], it[1], **kw)
            elif isinstance(it, str) and it.lower().endswith(".fits"):
                v = predict_fits(it, **kw)
            else:
                v = predict_target(it, **kw)
            rows.append(v.to_dict())
        except Exception as e:
            rows.append({"target": str(it), "classification": "error",
                         "explanation": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(rows)
