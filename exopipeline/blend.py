"""Stage 6 — Crowded-field / blend disambiguation (THE DIFFERENTIATOR).

Two complementary tests the PS calls out:

1. **Difference imaging / centroid test** — from a Target Pixel File, build mean in-transit
   and out-of-transit images, difference them, and measure the photocenter of the dip. If
   it sits significantly off the target's catalog pixel, the transit comes from a
   *neighbour* -> blend / background eclipsing binary.

2. **Dilution correction** — the true depth is depth_obs / CROWDSAP. Apply before parameter
   estimation so radii are not underestimated in crowded fields.
"""
from __future__ import annotations

import numpy as np

# TESS pixel scale (arcsec/pixel)
TESS_PIXEL_ARCSEC = 21.0


def dilution_correct_depth(depth_ppm: float, crowdsap: float) -> float:
    """True transit depth corrected for aperture dilution."""
    if crowdsap and np.isfinite(crowdsap) and crowdsap > 0:
        return depth_ppm / crowdsap
    return depth_ppm


def _photocenter(image):
    """Flux-weighted centroid (col, row) of a 2-D image (NaNs/negatives -> 0)."""
    img = np.nan_to_num(image, nan=0.0)
    img = np.clip(img, 0, None)
    total = img.sum()
    if total <= 0:
        return None
    ys, xs = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return float((xs * img).sum() / total), float((ys * img).sum() / total)


def centroid_test(tpf, period, t0, duration, offset_threshold_arcsec=2.0) -> dict:
    """Difference-image centroid test on a Target Pixel File.

    Parameters
    ----------
    tpf : lightkurve TargetPixelFile
    period, t0, duration : float
        Transit ephemeris (days).
    offset_threshold_arcsec : float
        Centroid offset above which we flag a blend.

    Returns a dict with the difference image, target & centroid pixel coords, the offset
    in arcsec, and an ``is_blend`` flag. Returns a dict with ``diff_image=None`` if the
    test cannot be run.
    """
    null = dict(diff_image=None, target_xy=None, centroid_xy=None,
                offset_pix=np.nan, offset_arcsec=np.nan, is_blend=False)
    if tpf is None:
        return null

    try:
        t = np.asarray(tpf.time.value, dtype="float64")
        cube = np.asarray(tpf.flux.value, dtype="float64")     # (cadence, row, col)
    except Exception:
        return null

    ph = (t - t0 + 0.5 * period) % period - 0.5 * period
    in_tr = np.abs(ph) < 0.5 * duration
    oot = (np.abs(ph) > 1.0 * duration) & (np.abs(ph) < 2.5 * duration)
    if in_tr.sum() < 3 or oot.sum() < 3:
        return null

    img_in = np.nanmean(cube[in_tr], axis=0)
    img_oot = np.nanmean(cube[oot], axis=0)
    diff = img_oot - img_in        # positive where flux dropped during transit (the source)

    centroid = _photocenter(diff)
    if centroid is None:
        return null

    # Target's catalog position in pixel coords (column, row).
    target_xy = None
    try:
        col0, row0 = tpf.column, tpf.row
        # lightkurve stores the catalog position via WCS; fall back to aperture centroid.
        if hasattr(tpf, "wcs") and tpf.wcs is not None and hasattr(tpf, "ra"):
            x, y = tpf.wcs.all_world2pix([[tpf.ra, tpf.dec]], 0)[0]
            target_xy = (float(x), float(y))
    except Exception:
        target_xy = None
    if target_xy is None:
        # Fall back to the brightest pixel of the out-of-transit image.
        yb, xb = np.unravel_index(np.nanargmax(img_oot), img_oot.shape)
        target_xy = (float(xb), float(yb))

    offset_pix = float(np.hypot(centroid[0] - target_xy[0],
                                centroid[1] - target_xy[1]))
    offset_arcsec = offset_pix * TESS_PIXEL_ARCSEC

    return dict(
        diff_image=diff,
        target_xy=target_xy,
        centroid_xy=centroid,
        offset_pix=offset_pix,
        offset_arcsec=offset_arcsec,
        is_blend=bool(offset_arcsec > offset_threshold_arcsec),
    )
