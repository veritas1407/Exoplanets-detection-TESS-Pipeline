"""Stage 4 — Vetting features (physics-aware false-positive diagnostics).

One function, ``compute_features``, turns a detrended light curve + a search Candidate into
a flat feature dict. Each feature targets a specific impostor:

    odd-even depth difference   -> eclipsing binaries (2x-period trap)
    secondary eclipse depth     -> EBs / blended EBs
    flatness (U vs V shape)     -> grazing EBs
    transit count + per-tr SNR  -> single-event / noise false alarms
    duration / period           -> stellar-density self-consistency

The same dict feeds the LightGBM classifier (one row per candidate) and the vetting sheet.
"""
from __future__ import annotations

import numpy as np


def fold_phase(t, P, T0):
    return (t - T0 + 0.5 * P) % P - 0.5 * P


_G_CGS = 6.674e-8        # gravitational constant, cgs
_RHO_SUN = 1.408         # mean solar density, g/cc


def _density_ratio(period_days, t14_days, depth_frac, rstar_sun, mstar_sun):
    """log10( transit-derived stellar density / catalog stellar density ).

    A core false-positive diagnostic: the transit duration implies a stellar density
    (assuming a central, circular planetary orbit); if it disagrees with the catalog
    density (from R*, M*) the event is likely an EB / blend / wrong period. Computed in
    consistent cgs units so the ratio is physical (≈0 in log for a real planet).

    a/R* ~ (P / (pi*T14)) * (1 + sqrt(depth))  [small-planet central-transit approx];
    rho_transit = 3*pi/(G * P_sec^2) * (a/R*)^3.  Returns NaN if inputs are unusable.
    """
    if not (np.isfinite(rstar_sun) and np.isfinite(mstar_sun) and rstar_sun > 0
            and mstar_sun > 0 and t14_days and t14_days > 0 and period_days > 0):
        return np.nan
    depth = max(depth_frac, 0.0)
    a_rs = (period_days / (np.pi * t14_days)) * (1.0 + np.sqrt(depth))
    if not np.isfinite(a_rs) or a_rs <= 1:
        return np.nan
    p_sec = period_days * 86400.0
    rho_transit = 3.0 * np.pi / (_G_CGS * p_sec ** 2) * a_rs ** 3      # g/cc
    rho_catalog = _RHO_SUN * mstar_sun / rstar_sun ** 3               # g/cc
    if rho_catalog <= 0 or rho_transit <= 0:
        return np.nan
    return float(np.log10(rho_transit / rho_catalog))


def compute_features(time, flat_flux, candidate, crowdsap=np.nan,
                     rstar_sun=np.nan, mstar_sun=np.nan) -> dict:
    """Compute the full vetting-feature vector for one candidate.

    Parameters
    ----------
    time, flat_flux : arrays
        Detrended (~1.0) light curve.
    candidate : object with .period, .T0, .duration, .SDE, .snr, .FAP, .rp_rs,
        .distinct_transit_count  (an exopipeline.search.Candidate, or any namespace).
    crowdsap : float
        Dilution keyword, carried into the feature row for the classifier.
    rstar_sun, mstar_sun : float
        Stellar radius / mass (solar units) for the stellar-density consistency feature.
        Auxiliary stellar info only (not transit/disposition parameters); NaN -> feature NaN.
    """
    time = np.asarray(time, dtype="float64")
    flat_flux = np.asarray(flat_flux, dtype="float64")
    P = float(candidate.period)
    t0 = float(candidate.T0)
    dur = float(candidate.duration)

    ph = fold_phase(time, P, t0)
    in_tr = np.abs(ph) < 0.5 * dur
    oot = (np.abs(ph) > 1.0 * dur) & (np.abs(ph) < 3.0 * dur)
    base = np.median(flat_flux[oot]) if oot.any() else 1.0
    scatter = np.std(flat_flux[oot]) if oot.sum() > 2 else np.std(flat_flux)

    def _depth(mask):
        return (base - np.median(flat_flux[mask])) * 1e6 if mask.sum() else np.nan

    # --- Odd vs even transits ---------------------------------------------------------
    epoch = np.round((time - t0) / P).astype(int)
    odd_in = in_tr & (epoch % 2 == 1)
    even_in = in_tr & (epoch % 2 == 0)
    depth_odd = _depth(odd_in)
    depth_even = _depth(even_in)
    odd_even_diff = abs(depth_odd - depth_even) if np.isfinite(depth_odd) and \
        np.isfinite(depth_even) else np.nan
    denom = scatter * np.sqrt(1 / max(odd_in.sum(), 1) + 1 / max(even_in.sum(), 1))
    odd_even_sigma = (odd_even_diff * 1e-6 / denom) if denom > 0 else np.nan

    # --- Secondary eclipse at phase ~0.5 ----------------------------------------------
    ph_sec = fold_phase(time, P, t0 + 0.5 * P)
    sec_in = np.abs(ph_sec) < 0.5 * dur
    depth_secondary = _depth(sec_in)
    # secondary significance (how many sigma is the phase-0.5 dip)
    n_sec = max(int(sec_in.sum()), 1)
    secondary_snr = (depth_secondary * 1e-6) / (scatter / np.sqrt(n_sec)) \
        if scatter > 0 and np.isfinite(depth_secondary) else np.nan

    # --- Primary depth + U/V shape ----------------------------------------------------
    depth_primary = _depth(in_tr)
    core = np.abs(ph) < 0.25 * dur
    depth_core = _depth(core)
    flatness = (depth_core / depth_primary) if depth_primary and depth_primary > 0 \
        else np.nan
    # explicit V-shape: wing depth (0.25-0.5 dur) vs core depth; EBs are V (wings deep)
    wing = (np.abs(ph) >= 0.25 * dur) & (np.abs(ph) < 0.5 * dur)
    depth_wing = _depth(wing)
    vshape_ratio = (depth_wing / depth_core) if depth_core and depth_core > 0 else np.nan

    # --- Counting / SNR ---------------------------------------------------------------
    n_transits = int(getattr(candidate, "distinct_transit_count", 0)) or \
        int(len(np.unique(epoch[in_tr])))
    pts_per_transit = in_tr.sum() / max(n_transits, 1)
    snr_per_transit = (depth_primary * 1e-6) / (scatter / np.sqrt(max(pts_per_transit, 1))) \
        if scatter > 0 and np.isfinite(depth_primary) else np.nan
    dur_over_period = dur / P
    # total transit SNR (MES-like): depth * sqrt(N_in) / scatter
    transit_snr = (depth_primary * 1e-6) * np.sqrt(max(int(in_tr.sum()), 1)) / scatter \
        if scatter > 0 and np.isfinite(depth_primary) else np.nan

    # --- Per-transit depth consistency (real transits repeat; noise/variables don't) --
    per_depths = []
    for e in np.unique(epoch[in_tr]):
        m = in_tr & (epoch == e)
        if m.sum() >= 2:
            per_depths.append((base - np.median(flat_flux[m])) * 1e6)
    if len(per_depths) >= 2 and depth_primary and abs(depth_primary) > 0:
        depth_consistency = float(np.std(per_depths) / abs(depth_primary))
    else:
        depth_consistency = np.nan

    # --- Phase coverage (data completeness across the orbit) --------------------------
    nb = 50
    bins = np.floor((ph / P + 0.5) * nb).astype(int)
    phase_coverage = float(len(np.unique(np.clip(bins, 0, nb - 1))) / nb)

    # --- Stellar-density consistency --------------------------------------------------
    depth_frac = (depth_primary * 1e-6) if np.isfinite(depth_primary) else np.nan
    rho_ratio = _density_ratio(P, dur, depth_frac, rstar_sun, mstar_sun)

    fap = float(getattr(candidate, "FAP", np.nan))
    log_fap = np.log10(fap) if (np.isfinite(fap) and fap > 0) else -12.0

    return {
        "period": P,
        "depth_ppm": depth_primary,
        "duration_hr": dur * 24.0,
        "sde": float(getattr(candidate, "SDE", np.nan)),
        "snr": float(getattr(candidate, "snr", np.nan)),
        "log_fap": log_fap,
        "odd_even_diff_ppm": odd_even_diff,
        "odd_even_sigma": odd_even_sigma,
        "secondary_ppm": depth_secondary,
        "secondary_snr": secondary_snr,
        "flatness": flatness,
        "vshape_ratio": vshape_ratio,
        "n_transits": n_transits,
        "snr_per_transit": snr_per_transit,
        "transit_snr": transit_snr,
        "depth_consistency": depth_consistency,
        "phase_coverage": phase_coverage,
        "dur_over_period": dur_over_period,
        "rho_ratio": rho_ratio,
        "rp_rs": float(getattr(candidate, "rp_rs", np.nan)),
        "crowdsap": float(crowdsap),
        # extras kept for plotting / reporting (not classifier columns)
        "_depth_odd": depth_odd,
        "_depth_even": depth_even,
        "_base": base,
        "_scatter": scatter,
    }


def verdict_heuristic(features: dict) -> tuple[str, float]:
    """A transparent rule-based fallback verdict (used before the trained classifier
    exists, and as a sanity cross-check). Returns (class, pseudo-confidence)."""
    oe = features.get("odd_even_sigma", 0) or 0
    sec = features.get("secondary_ppm", 0) or 0
    flat = features.get("flatness", 1) or 1
    depth = features.get("depth_ppm", 0) or 0
    snr = features.get("snr", 0) or 0

    if snr < 7:
        return "other", 0.6
    if oe > 3.0:
        return "eclipsing_binary", 0.7
    if sec > 0.3 * max(depth, 1):
        return "eclipsing_binary", 0.65
    if flat < 0.5:
        return "eclipsing_binary", 0.6      # V-shaped / grazing
    return "transit", 0.75
