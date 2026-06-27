"""Stage 8 — Visualization & report.

Assemble the one-page "vetting sheet" (DV-report style): raw + detrended light curves,
the BLS periodogram, the phase-fold with the best-fit batman model, an odd-vs-even panel,
a secondary-eclipse zoom, an optional difference-image / centroid panel, and a header box
carrying the verdict + fitted parameters.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .vetting import fold_phase
from . import fit as _fit


def _binned(phase, flux, lo, hi, nbins):
    bins = np.linspace(lo, hi, nbins + 1)
    idx = np.digitize(phase, bins)
    cen = 0.5 * (bins[1:] + bins[:-1])
    mean = np.array([flux[idx == k].mean() if np.any(idx == k) else np.nan
                     for k in range(1, nbins + 1)])
    return cen, mean


def vetting_sheet(star, flat, candidate, features, fit_result,
                  verdict, confidence, blend=None, periodogram=None,
                  title=None, save_path=None, dpi=300):
    """Build the vetting sheet figure.

    Parameters
    ----------
    star    : ingest.Star (for raw LC + CROWDSAP)
    flat    : detrend.Flattened (detrended LC)
    candidate : search.Candidate
    features : dict from vetting.compute_features
    fit_result : fit.FitResult
    verdict, confidence : str, float  (from the classifier)
    blend   : optional dict from blend.centroid_test (adds a difference-image panel)
    periodogram : optional (periods, power) tuple for the BLS panel
    """
    P = candidate.period
    t0 = candidate.T0
    dur = candidate.duration
    ph = fold_phase(flat.time, P, t0)

    phase_t, model = _fit.model_curve(fit_result)

    has_blend = blend is not None and blend.get("diff_image") is not None
    fig = plt.figure(figsize=(15, 11))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.85, 1, 1, 1], hspace=0.55, wspace=0.22)

    # --- Header ----------------------------------------------------------------------
    axh = fig.add_subplot(gs[0, :]); axh.axis("off")
    P_m, P_lo, P_hi = fit_result.medians["period"]
    D_m, D_lo, D_hi = fit_result.medians["Depth"]
    Rp_m, Rp_lo, Rp_hi = fit_result.medians["Rp"]
    dP = max(P_lo, P_hi); dD = max(D_lo, D_hi); dRp = max(Rp_lo, Rp_hi)
    dur_fit = fit_result.medians.get("Duration_hr", (dur * 24, 0, 0))[0]
    fap = candidate.FAP
    fap_s = f"{fap:.1e}" if np.isfinite(fap) else "n/a"
    cent = ""
    if blend is not None and np.isfinite(blend.get("offset_arcsec", np.nan)):
        cent = (f"   Centroid offset = {blend['offset_arcsec']:.2f}\" "
                f"({'BLEND' if blend.get('is_blend') else 'on-target'})")
    header = (
        f"{star.target}\n"
        f"PREDICTED CLASS:  {verdict.upper():18s}  Confidence: {confidence*100:.0f}%\n"
        f"SDE = {candidate.SDE:.1f}    SNR = {candidate.snr:.1f}    FAP = {fap_s}    "
        f"Transits = {candidate.distinct_transit_count}    CROWDSAP = {star.crowdsap:.3f}\n"
        f"Period   = {P_m:.5f} +/- {dP:.5f} d\n"
        f"Depth    = {D_m:.0f} +/- {dD:.0f} ppm (dilution-corrected)    "
        f"Rp = {Rp_m:.2f} +/- {dRp:.2f} R_Earth\n"
        f"Duration = {dur_fit:.2f} h    Odd-Even = {features['odd_even_diff_ppm']:.0f} ppm "
        f"({features['odd_even_sigma']:.1f} sigma)    "
        f"Secondary = {features['secondary_ppm']:.0f} ppm{cent}"
    )
    color = {"transit": "#eef5ff", "eclipsing_binary": "#fff0f0",
             "blend": "#fff7e6", "other": "#f0f0f0"}.get(verdict, "#eef5ff")
    axh.text(0.01, 0.95, header, va="top", ha="left", family="monospace", fontsize=11,
             bbox=dict(boxstyle="round", fc=color, ec="C0"))

    # --- Raw LC ----------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[1, 0])
    if star.raw_lc is not None:
        ax1.plot(star.raw_lc.time.value, star.raw_lc.flux.value, ".", ms=0.6, color="0.5")
    else:
        ax1.plot(star.time, star.flux, ".", ms=0.6, color="0.5")
    ax1.set(title="Raw PDCSAP light curve", xlabel="BTJD", ylabel="flux")

    # --- Detrended LC ----------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.plot(flat.time, flat.flux, ".", ms=0.6, color="C2")
    ax2.axhline(1.0, color="0.5", lw=0.7)
    ax2.set(title="Detrended light curve", xlabel="BTJD", ylabel="flux")

    # --- Periodogram -----------------------------------------------------------------
    ax3 = fig.add_subplot(gs[2, 0])
    if periodogram is not None:
        per, pw = periodogram
        ax3.plot(per, pw, color="C0", lw=0.8)
        ax3.axvline(P, color="C3", alpha=0.6, lw=2)
        ax3.set(title=f"BLS periodogram (peak {P:.4f} d)", xlabel="Period [d]",
                ylabel="power")
    else:
        ax3.text(0.5, 0.5, "periodogram not supplied", ha="center", va="center")
        ax3.set_axis_off()

    # --- Phase fold + model ----------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 1])
    fw = fit_result.fit_window
    sel = np.abs(ph) < fw
    bcen, bmean = _binned(ph[sel], flat.flux[sel], -fw, fw, 40)
    ax4.plot(ph[sel] * 24, flat.flux[sel], ".", ms=1.5, color="0.7", label="folded")
    ax4.plot(bcen * 24, bmean, "o", color="C0", ms=4, label="binned")
    ax4.plot(phase_t * 24, model, "-", color="C3", lw=2, label="batman fit")
    ax4.set(title="Phase-folded + best-fit model", xlabel="Phase [hours]", ylabel="flux")
    ax4.legend(fontsize=8)

    # --- Odd vs even -----------------------------------------------------------------
    ax5 = fig.add_subplot(gs[3, 0])
    epoch = np.round((flat.time - t0) / P).astype(int)
    for parity, lab, col in [(1, "odd", "C0"), (0, "even", "C1")]:
        m = (np.abs(ph) < fw) & (epoch % 2 == parity)
        bc, bm = _binned(ph[m], flat.flux[m], -fw, fw, 30)
        ax5.plot(bc * 24, bm, "o-", ms=3, color=col, label=lab)
    ax5.axhline(1.0, color="0.5", lw=0.7)
    ax5.set(title=f"Odd vs even ({features['odd_even_sigma']:.1f} sigma)",
            xlabel="Phase [hours]", ylabel="flux")
    ax5.legend(fontsize=8)

    # --- Secondary zoom OR difference image ------------------------------------------
    ax6 = fig.add_subplot(gs[3, 1])
    if has_blend:
        im = ax6.imshow(blend["diff_image"], origin="lower", cmap="viridis")
        if blend.get("target_xy") is not None:
            ax6.plot(*blend["target_xy"], "x", color="white", ms=10, mew=2, label="target")
        if blend.get("centroid_xy") is not None:
            ax6.plot(*blend["centroid_xy"], "+", color="red", ms=12, mew=2,
                     label="dip centroid")
        ax6.set(title="Difference image (in - out of transit)")
        ax6.legend(fontsize=8, loc="upper right")
        fig.colorbar(im, ax=ax6, fraction=0.046)
    else:
        ph_sec = fold_phase(flat.time, P, t0 + 0.5 * P)
        ssel = np.abs(ph_sec) < fw
        bc, bm = _binned(ph_sec[ssel], flat.flux[ssel], -fw, fw, 30)
        ax6.plot(ph_sec[ssel] * 24, flat.flux[ssel], ".", ms=1.5, color="0.7")
        ax6.plot(bc * 24, bm, "o", color="C4", ms=4)
        ax6.axhline(1.0, color="0.5", lw=0.7)
        ax6.set(title=f"Secondary-eclipse zoom ({features['secondary_ppm']:.0f} ppm)",
                xlabel="Phase from sec. [hours]", ylabel="flux")

    fig.suptitle(title or "Exoplanet Vetting Sheet  —  PS7 Pipeline", fontsize=15, y=0.98)
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    return fig
