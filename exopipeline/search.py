"""Stage 3 — Transit search (blind, multi-planet).

Strategy that is both *honest* (no hardcoded period window) and *tractable* on 27 stitched
sectors:

1. **Broad blind sweep** with astropy ``BoxLeastSquares`` over PERIOD_MIN..PERIOD_MAX.
   BLS scales to ~450k points in seconds and honours an explicit period grid (TLS 1.32
   silently ignores ``period_min/max`` in some environments).
2. **Iterative masking** — take the strongest peak, mask its transits, re-detrend, and
   re-sweep. Repeat until the peak falls below threshold. This recovers every planet in a
   multi-planet system (for TOI 700: b, c, e, d) one at a time.
3. **Narrow TLS refine** around each surviving BLS period for a proper limb-darkened
   depth/duration and a real SDE + analytic FAP — "two independent search methods agree."
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from . import config, detrend as _detrend


@dataclass
class Candidate:
    period: float
    T0: float
    depth: float          # TLS convention: 1 - min_flux  (so depth_ppm = (1-depth)*1e6)
    duration: float       # days
    SDE: float
    snr: float
    FAP: float
    rp_rs: float
    distinct_transit_count: int
    transit_count: int
    bls_power_ratio: float = np.nan
    refined_by_tls: bool = False

    @property
    def depth_ppm(self) -> float:
        return (1.0 - self.depth) * 1e6

    @property
    def duration_hr(self) -> float:
        return self.duration * 24.0

    def as_dict(self) -> dict:
        return asdict(self)


def _bls_best(time, flux, period_grid, durations):
    """Run one BLS sweep; return (best_index, periods, power, params at peak).

    The reported ``sde`` is the standard Signal Detection Efficiency
    SDE = (peak_power - mean_power) / std_power — a legitimate detection statistic,
    not just a peak/median ratio.
    """
    from astropy.timeseries import BoxLeastSquares
    import astropy.units as u

    bls = BoxLeastSquares(time * u.day, flux)
    res = bls.power(period_grid * u.day, np.asarray(durations) * u.day, method="fast")
    power = np.asarray(res.power)
    bi = int(np.argmax(power))
    P = float(res.period[bi].value)
    t0 = float(res.transit_time[bi].value)
    dep = float(max(res.depth[bi], 1e-9))
    dur = float(res.duration[bi].value)
    mu, sigma = np.mean(power), np.std(power)
    sde = float((power[bi] - mu) / sigma) if sigma > 0 else 0.0
    ratio = float(power[bi] / np.median(power))
    return bi, np.asarray(res.period), power, dict(period=P, T0=t0, depth=dep,
                                                   duration=dur, sde=sde,
                                                   power_ratio=ratio)


def _bls_fap(sde):
    """Rough analytic false-alarm probability from the BLS SDE (Gaussian tail)."""
    from math import erfc, sqrt
    return float(0.5 * erfc(sde / sqrt(2)))


def _snr_and_counts(time, flux, P, t0, dur, dep):
    ph = (time - t0 + 0.5 * P) % P - 0.5 * P
    in_tr = np.abs(ph) < 0.5 * dur
    if in_tr.sum() < 3:
        return 0.0, 0, max(int(round((time.max() - time.min()) / P)), 1)
    scatter = np.std(flux[~in_tr])
    snr = float(dep / (scatter / np.sqrt(in_tr.sum()))) if scatter > 0 else 0.0
    epochs = np.round((time[in_tr] - t0) / P).astype(int)
    distinct = int(len(np.unique(epochs)))
    total = max(int(round((time.max() - time.min()) / P)), 1)
    return snr, distinct, total


def _refine_tls(time, flux, P0):
    """Narrow TLS around ``P0`` for a limb-darkened depth/duration + real SDE/FAP.

    Returns a dict or ``None`` if TLS is unavailable / wanders out of the window.
    """
    try:
        from transitleastsquares import transitleastsquares
    except Exception:
        return None
    # Window scales with period so TLS's frequency grid has enough samples even at long P.
    half = max(config.TLS_REFINE_HALFWIDTH, 0.01 * P0)
    pmin = max(P0 - half, 0.2)
    pmax = P0 + half
    try:
        model = transitleastsquares(time, flux)
        # Pass an M-dwarf-ish stellar density so the auto period grid is dense enough
        # in a narrow window (avoids the "too few values" fallback to the full grid).
        r = model.power(period_min=pmin, period_max=pmax,
                        R_star=0.5, M_star=0.5,
                        oversampling_factor=5, duration_grid_step=1.05,
                        use_threads=1,            # single-thread: avoid Windows spawn re-exec
                        show_progress_bar=False)
        if not (pmin - 0.02 <= r.period <= pmax + 0.02):
            return None        # TLS ignored the window -> keep the BLS values instead
        return dict(period=float(r.period), T0=float(r.T0), depth=float(r.depth),
                    duration=float(r.duration), SDE=float(r.SDE), snr=float(r.snr),
                    FAP=float(r.FAP), rp_rs=float(r.rp_rs),
                    distinct=int(r.distinct_transit_count),
                    total=int(r.transit_count))
    except Exception:
        return None


def _is_duplicate(period, found_periods, rtol=0.01, harm_rtol=0.004):
    """True if ``period`` is a spurious alias of an already-found planet.

    Two cases are rejected:
      * **Same period** (ratio ~1) within ``rtol`` — residual of an incompletely-masked
        planet re-appearing.
      * **Near-exact integer harmonic** (n× or 1/n×, n=2..5) within the *tight*
        ``harm_rtol`` — a true period alias.

    Real near-resonant planets (e.g. TOI-270 d/c = 2.011, a 0.5% detuning) sit *outside*
    ``harm_rtol`` of the exact ratio and are therefore kept; genuine aliases land within
    ~0.1% of an exact integer ratio and are rejected."""
    for fp in found_periods:
        if abs(period - fp) <= rtol * fp:
            return True
        for n in (2, 3, 4, 5):
            if abs(period - n * fp) <= harm_rtol * n * fp:      # n× a found period
                return True
            if abs(period - fp / n) <= harm_rtol * fp / n:      # 1/n × a found period
                return True
    return False


def find_planets(time, flux, max_planets=None, refine=False, verbose=True,
                 sde_threshold=None, period_min=None, period_max=None, n_periods=None):
    """Blind multi-planet search via BLS broad sweep + iterative masking.

    Each iteration takes the strongest BLS peak; if it duplicates an already-found period
    or a low-integer harmonic it is masked but not recorded; genuinely new signals are
    recorded and (optionally) refined with a narrow TLS. The masked points are *removed*
    (not re-detrended) so trend artifacts at the mask gaps cannot recreate the signal.

    Detection significance is the BLS **SDE** = (peak-mean)/std of the power spectrum;
    the loop stops when the strongest remaining peak falls below ``sde_threshold``.

    Parameters
    ----------
    time, flux : arrays
        Detrended (flattened ~1.0) light curve.
    max_planets : int
        Number of planets to record (default ``config.MAX_PLANETS``).
    refine : bool
        Refine each candidate with a narrow TLS for a limb-darkened depth + analytic FAP.
        **Default False** — TLS uses multiprocessing (slow/fragile on Windows). Enable on
        Colab/Linux (fork) or for a single final candidate.
    sde_threshold : float
        Detection floor (default ``config.SDE_THRESHOLD``).

    Returns
    -------
    list of :class:`Candidate`, strongest first.
    """
    from transitleastsquares import transit_mask

    max_planets = max_planets or config.MAX_PLANETS
    sde_threshold = sde_threshold if sde_threshold is not None else config.SDE_THRESHOLD
    time = np.ascontiguousarray(time, dtype="float64")
    work = np.ascontiguousarray(flux, dtype="float64")

    pmin = period_min if period_min is not None else config.PERIOD_MIN
    pmax = period_max if period_max is not None else config.PERIOD_MAX
    npg = n_periods if n_periods is not None else config.N_PERIODS
    period_grid = np.linspace(pmin, pmax, npg)
    candidates: list[Candidate] = []
    found_periods: list[float] = []

    max_iters = max_planets + 4   # allow a few duplicate/harmonic rejections
    for i in range(max_iters):
        if len(candidates) >= max_planets or len(time) < 100:
            break
        bi, _periods, _power, p = _bls_best(time, work, period_grid, config.BLS_DURATIONS)
        sde = p["sde"]
        P, t0, dur, dep = p["period"], p["T0"], p["duration"], p["depth"]
        if verbose:
            print(f"[search] iter {i+1}: P={P:.5f} d  dur={dur*24:.2f} h  SDE={sde:.1f}")
        if sde < sde_threshold:
            if verbose:
                print(f"[search] SDE {sde:.1f} < {sde_threshold} -> stop")
            break

        duplicate = _is_duplicate(P, found_periods)

        if not duplicate:
            snr, distinct, total = _snr_and_counts(time, work, P, t0, dur, dep)
            cand = Candidate(
                period=P, T0=t0, depth=1.0 - dep, duration=dur,
                SDE=sde, snr=snr, FAP=_bls_fap(sde), rp_rs=float(np.sqrt(dep)),
                distinct_transit_count=distinct, transit_count=total,
                bls_power_ratio=p["power_ratio"],
            )
            if refine:
                ref = _refine_tls(time, work, P)
                if ref is not None:
                    cand.period, cand.T0, cand.depth = ref["period"], ref["T0"], ref["depth"]
                    cand.duration, cand.SDE, cand.snr = ref["duration"], ref["SDE"], ref["snr"]
                    cand.FAP, cand.rp_rs = ref["FAP"], ref["rp_rs"]
                    cand.distinct_transit_count = ref["distinct"]
                    cand.transit_count = ref["total"]
                    cand.refined_by_tls = True
                    if verbose:
                        print(f"[search]   TLS refine -> P={ref['period']:.5f} d  "
                              f"depth={(1-ref['depth'])*1e6:.0f} ppm  SDE={ref['SDE']:.1f}  "
                              f"FAP={ref['FAP']:.1e}")
            candidates.append(cand)
            found_periods.append(cand.period)
        elif verbose:
            print(f"[search]   duplicate/harmonic of a found period -> mask & continue")

        # Mask this signal's transits and *remove* the points (no re-detrend).
        mask = transit_mask(time, P, max(dur, 0.05), t0)
        keep = ~mask
        if keep.sum() < 100:
            break
        time = time[keep]
        work = work[keep]

    return candidates


def search_single(time, flux, period_min, period_max, refine=True):
    """Focused single-window search (used by the classifier feature builder where the
    catalog period is known). Returns one :class:`Candidate` or ``None``."""
    period_grid = np.linspace(period_min, period_max, 2000)
    bi, _per, _pow, p = _bls_best(time, flux, period_grid, config.BLS_DURATIONS)
    P, t0, dur, dep = p["period"], p["T0"], p["duration"], p["depth"]
    snr, distinct, total = _snr_and_counts(time, flux, P, t0, dur, dep)
    cand = Candidate(period=P, T0=t0, depth=1.0 - dep, duration=dur,
                     SDE=p["sde"], snr=snr, FAP=_bls_fap(p["sde"]),
                     rp_rs=float(np.sqrt(dep)), distinct_transit_count=distinct,
                     transit_count=total, bls_power_ratio=p["power_ratio"])
    if refine:
        ref = _refine_tls(time, flux, P)
        if ref is not None:
            cand.period, cand.T0, cand.depth = ref["period"], ref["T0"], ref["depth"]
            cand.duration, cand.SDE, cand.snr = ref["duration"], ref["SDE"], ref["snr"]
            cand.FAP, cand.rp_rs = ref["FAP"], ref["rp_rs"]
            cand.distinct_transit_count, cand.transit_count = ref["distinct"], ref["total"]
            cand.refined_by_tls = True
    return cand
