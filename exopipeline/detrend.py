"""Stage 2 — Detrending.

Remove slow stellar variability with a robust biweight filter (`wotan`), optionally
iterating with known transits masked so deep/long transits cannot bias the trend.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config


@dataclass
class Flattened:
    time: np.ndarray
    flux: np.ndarray          # flattened (≈1.0) flux
    trend: np.ndarray         # the fitted trend that was divided out
    raw_flux: np.ndarray      # pre-detrend (cleaned) flux, aligned to ``time``


def _flatten_once(time, flux, window, method):
    from wotan import flatten
    flat, trend = flatten(time, flux, window_length=window, method=method,
                          return_trend=True)
    return flat, trend


def to_flattened(star) -> "Flattened":
    """Return a :class:`Flattened` for a Star, skipping detrending if it was already
    flattened per-sector at load time (avoids double-detrending)."""
    if getattr(star, "pre_flattened", False):
        return Flattened(time=np.ascontiguousarray(star.time, dtype="float64"),
                         flux=np.ascontiguousarray(star.flux, dtype="float64"),
                         trend=np.ones_like(star.flux),
                         raw_flux=np.ascontiguousarray(star.flux, dtype="float64"))
    return detrend(star.time, star.flux)


def detrend(time, flux, window=None, method=None, mask=None) -> Flattened:
    """Single-pass biweight detrend.

    ``mask`` (bool array, True = ignore/in-transit) is currently advisory; wotan's
    biweight is already robust to the in-transit points, so a single pass is clean for
    shallow planets. Use :func:`detrend_iterative` to mask + re-fit explicitly.
    """
    window = window or config.DETREND_WINDOW
    method = method or config.DETREND_METHOD
    time = np.ascontiguousarray(time, dtype="float64")
    flux = np.ascontiguousarray(flux, dtype="float64")

    flat, trend = _flatten_once(time, flux, window, method)
    good = np.isfinite(flat)
    return Flattened(time=time[good], flux=flat[good], trend=trend[good],
                     raw_flux=flux[good])


def detrend_iterative(time, flux, ephemerides=None, window=None, method=None,
                      n_iter=2) -> Flattened:
    """Iterative detrend: mask the in-transit points of known ephemerides, fit the trend
    on the rest, then divide the *full* series by that trend. Repeats ``n_iter`` times.

    Parameters
    ----------
    ephemerides : list of (period, t0, duration) tuples, optional
        Known transits to protect. If ``None`` this reduces to a single robust pass.
    """
    window = window or config.DETREND_WINDOW
    method = method or config.DETREND_METHOD
    time = np.ascontiguousarray(time, dtype="float64")
    flux = np.ascontiguousarray(flux, dtype="float64")

    if not ephemerides:
        return detrend(time, flux, window, method)

    from wotan import flatten

    in_transit = np.zeros_like(time, dtype=bool)
    for period, t0, dur in ephemerides:
        ph = (time - t0 + 0.5 * period) % period - 0.5 * period
        in_transit |= np.abs(ph) < (0.6 * dur)   # slightly wider than the transit

    trend = np.ones_like(flux)
    work_flux = flux.copy()
    for _ in range(max(1, n_iter)):
        masked = work_flux.copy()
        masked[in_transit] = np.nan            # hide transits from the trend fit
        _, trend = flatten(time, masked, window_length=window, method=method,
                           return_trend=True)
        # Interpolate trend across the masked gaps so we can divide the full series.
        bad = ~np.isfinite(trend)
        if bad.any():
            trend[bad] = np.interp(time[bad], time[~bad], trend[~bad])
        work_flux = flux / trend

    flat = flux / trend
    good = np.isfinite(flat)
    return Flattened(time=time[good], flux=flat[good], trend=trend[good],
                     raw_flux=flux[good])
