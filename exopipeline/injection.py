"""Injection-recovery test — the single most credible evidence of pipeline quality.

Inject synthetic batman transits into a real (quiet) light curve over a grid of period and
depth/SNR, run the full detection path (detrend -> search), and record the recovery
fraction as a function of injected SNR. Produces the pipeline-completeness curve.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, detrend as _detrend, search as _search


@dataclass
class InjectionResult:
    records: list          # list of dicts: injected params + recovered flag
    snr_bins: np.ndarray
    recovery_fraction: np.ndarray
    snr50: float           # ~50% completeness SNR


def _inject(time, flux, period, t0, rp, a=20.0, inc=89.5, ld=None):
    import batman
    ld = ld or config.LD_QUADRATIC
    bp = batman.TransitParams()
    bp.t0, bp.per, bp.rp, bp.a, bp.inc = t0, period, rp, a, inc
    bp.ecc, bp.w, bp.limb_dark, bp.u = 0.0, 90.0, "quadratic", list(ld)
    model = batman.TransitModel(bp, time).light_curve(bp)
    return flux * model


def _expected_snr(rp, time, flux, period, duration):
    """Crude expected SNR: depth / (point_noise / sqrt(n_in_transit))."""
    depth = rp ** 2
    noise = np.std(np.diff(flux)) / np.sqrt(2)      # robust white-noise estimate
    cadence = np.median(np.diff(time))
    n_in = max((duration / cadence) * (time[-1] - time[0]) / period, 1)
    return depth / (noise / np.sqrt(n_in))


def run_injection_recovery(time, base_flux, n_inject=120, period_range=(2.0, 25.0),
                           rp_range=(0.01, 0.05), a=20.0, seed=0,
                           tol_days=0.02, verbose=True) -> InjectionResult:
    """Inject ``n_inject`` synthetic transits one at a time into ``base_flux`` and test
    recovery with the blind search.

    ``base_flux`` should be a real, quiet, *already-flattened* light curve (so the injected
    transit is what we try to recover).
    """
    rng = np.random.default_rng(seed)
    time = np.ascontiguousarray(time, dtype="float64")
    base_flux = np.ascontiguousarray(base_flux, dtype="float64")

    records = []
    for k in range(n_inject):
        period = rng.uniform(*period_range)
        t0 = time[0] + rng.uniform(0, period)
        rp = rng.uniform(*rp_range)
        injected = _inject(time, base_flux, period, t0, rp, a=a)

        duration = period / np.pi * np.arcsin(1.0 / a) * 2  # rough total duration (days)
        duration = max(duration, 0.03)
        snr = _expected_snr(rp, time, injected, period, duration)

        # Detrend + single strongest-candidate search (fast).
        flat = _detrend.detrend(time, injected)
        cand = _search.search_single(flat.time, flat.flux,
                                     period_min=max(period_range[0] - 0.5, 0.5),
                                     period_max=period_range[1] + 0.5, refine=False)
        recovered = bool(cand is not None and
                         (abs(cand.period - period) < tol_days or
                          abs(cand.period - 2 * period) < tol_days or
                          abs(cand.period - 0.5 * period) < tol_days))
        records.append(dict(period=period, t0=t0, rp=rp, snr=snr,
                            recovered_period=(cand.period if cand else np.nan),
                            recovered=recovered))
        if verbose and (k + 1) % 20 == 0:
            print(f"[injection] {k+1}/{n_inject} done")

    snr_arr = np.array([r["snr"] for r in records])
    rec_arr = np.array([r["recovered"] for r in records], dtype=float)

    # Recovery fraction in SNR bins
    edges = np.linspace(np.percentile(snr_arr, 2), np.percentile(snr_arr, 98), 9)
    centers = 0.5 * (edges[1:] + edges[:-1])
    frac = np.full(centers.size, np.nan)
    for i in range(centers.size):
        m = (snr_arr >= edges[i]) & (snr_arr < edges[i + 1])
        if m.sum() >= 2:
            frac[i] = rec_arr[m].mean()

    # ~50% completeness SNR (first bin crossing 0.5)
    snr50 = np.nan
    valid = np.isfinite(frac)
    if valid.any():
        for c, f in zip(centers[valid], frac[valid]):
            if f >= 0.5:
                snr50 = float(c)
                break

    return InjectionResult(records=records, snr_bins=centers,
                           recovery_fraction=frac, snr50=snr50)


def plot_recovery(result: InjectionResult, save_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    snr = np.array([r["snr"] for r in result.records])
    rec = np.array([r["recovered"] for r in result.records])
    ax.scatter(snr[rec], np.ones(rec.sum()) * 1.02, marker="|", color="C2",
               label="recovered", alpha=0.5)
    ax.scatter(snr[~rec], np.zeros((~rec).sum()) - 0.02, marker="|", color="C3",
               label="missed", alpha=0.5)
    good = np.isfinite(result.recovery_fraction)
    ax.plot(result.snr_bins[good], result.recovery_fraction[good], "o-", color="C0",
            lw=2, label="recovery fraction")
    ax.axhline(0.5, color="0.6", ls="--", lw=1)
    if np.isfinite(result.snr50):
        ax.axvline(result.snr50, color="0.6", ls=":", lw=1)
        ax.text(result.snr50, 0.05, f" 50% @ SNR≈{result.snr50:.1f}", fontsize=9)
    ax.set(xlabel="Injected SNR", ylabel="Recovery fraction",
           title="Injection-recovery completeness", ylim=(-0.1, 1.1))
    ax.legend(fontsize=9)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
