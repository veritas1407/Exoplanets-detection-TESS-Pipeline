"""Stage 7 — Parameter estimation with uncertainties.

batman (Mandel-Agol limb-darkened transit) + a fast scipy least-squares seed + emcee for
honest posterior credible intervals. Applies CROWDSAP dilution correction to the depth
before reporting physical parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config
from .vetting import fold_phase


@dataclass
class FitResult:
    labels: list
    medians: dict          # name -> (median, minus_1sigma, plus_1sigma)
    flat_samples: np.ndarray
    t_fit: np.ndarray
    f_fit: np.ndarray
    best: np.ndarray       # [t0, period, rp_rs, a_rs, inc]
    crowdsap: float
    fit_window: float
    acceptance: float = np.nan
    extra: dict = field(default_factory=dict)


def _make_model(t_fit, ld=None):
    import batman
    ld = ld or config.LD_QUADRATIC
    bp = batman.TransitParams()
    bp.t0, bp.per, bp.rp, bp.a, bp.inc = 0.0, 1.0, 0.02, 80.0, 89.9
    bp.ecc, bp.w = 0.0, 90.0
    bp.limb_dark, bp.u = "quadratic", list(ld)
    bm = batman.TransitModel(bp, t_fit)
    return bp, bm


def _density_a_rs(period_days, rstar_sun, mstar_sun):
    """Scaled semi-major axis a/R* from the orbital period and stellar mean density
    (Seager & Mallen-Ornelas 2003):  a/R* = 4.2098 * rho_star^(1/3) * P_days^(2/3),
    with rho_star in solar units = (M/Msun)/(R/Rsun)^3."""
    rho = mstar_sun / rstar_sun ** 3
    return 4.2098 * rho ** (1.0 / 3.0) * period_days ** (2.0 / 3.0)


def fit_transit(time, flat_flux, candidate, crowdsap=np.nan,
                rstar_sun=None, mstar_sun=None, ld=None,
                nwalkers=None, nsteps=None, nburn=None, fit_window=None,
                a_rs_prior_frac=0.15, progress=False) -> FitResult:
    """Fit a single transit candidate and return posterior summaries.

    Reports median +/- asymmetric 1-sigma (16/84th percentiles) for
    t0, period, Rp/R*, a/R*, inc, plus derived depth (ppm), Rp (R_Earth), duration, and
    impact parameter. Depth is dilution-corrected:  depth_true = depth_obs / CROWDSAP.

    If ``mstar_sun`` is given, a Gaussian **prior on a/R\*** centred on the stellar-density
    value (fractional width ``a_rs_prior_frac``) is applied. This uses the catalog stellar
    density to pin the transit geometry and break the a/R*–impact-parameter–depth
    degeneracy that otherwise lets a shallow transit masquerade as a grazing one.
    """
    import emcee
    from scipy.optimize import least_squares

    rstar_sun = rstar_sun if rstar_sun is not None else config.DEFAULT_RSTAR_SUN
    mstar_sun = mstar_sun if mstar_sun is not None else rstar_sun  # M-dwarf M~R in Rsun
    nwalkers = nwalkers or config.MCMC_NWALKERS
    nsteps = nsteps or config.MCMC_NSTEPS
    nburn = nburn or config.MCMC_NBURN
    fit_window = fit_window or config.FIT_WINDOW
    a_rs_density = _density_a_rs(float(candidate.period), rstar_sun, mstar_sun)

    time = np.asarray(time, dtype="float64")
    flat_flux = np.asarray(flat_flux, dtype="float64")
    P0 = float(candidate.period)
    t0_0 = float(candidate.T0)
    rp0 = float(getattr(candidate, "rp_rs", np.sqrt(max((1 - candidate.depth), 1e-6))))

    ph = fold_phase(time, P0, t0_0)
    near = np.abs(ph) < fit_window
    t_fit = time[near]
    f_fit = flat_flux[near]
    oot = (np.abs(ph) > 1.0 * candidate.duration) & (np.abs(ph) < 3.0 * candidate.duration)
    err_scalar = np.std(flat_flux[oot]) if oot.sum() > 2 else np.std(flat_flux)
    err = np.full_like(f_fit, err_scalar)

    bp, bm = _make_model(t_fit, ld=ld)

    def model(theta):
        t0_, per_, rp_, a_, inc_ = theta
        bp.t0, bp.per, bp.rp, bp.a, bp.inc = t0_, per_, rp_, a_, inc_
        return bm.light_curve(bp)

    # Seed a/R* from the stellar-density estimate (not the depth-biased BLS duration), and
    # seed a near-central inclination (low impact parameter) so the walkers start in the
    # physical transit mode rather than the BLS-seeded narrow/grazing one.
    a_seed = float(np.clip(a_rs_density, 11.0, 199.0))
    inc_seed = float(np.degrees(np.arccos(0.15 / a_seed)))   # impact parameter b ~ 0.15
    seed = np.array([t0_0, P0, rp0, a_seed, inc_seed])

    # --- scipy least-squares seed (fast point estimate before MCMC) -------------------
    def resid(theta):
        return (f_fit - model(theta)) / err
    lo = np.array([t0_0 - 0.1, P0 - 0.05, 1e-4, 10.0, 80.0])
    hi = np.array([t0_0 + 0.1, P0 + 0.05, 0.1, 200.0, 90.0])
    try:
        ls = least_squares(resid, np.clip(seed, lo + 1e-6, hi - 1e-6),
                           bounds=(lo, hi), max_nfev=2000)
        seed = ls.x
    except Exception:
        pass

    # --- priors + likelihood ----------------------------------------------------------
    a_sigma = a_rs_prior_frac * a_rs_density
    def log_prior(theta):
        t0_, per_, rp_, a_, inc_ = theta
        if not (t0_0 - 0.1 < t0_ < t0_0 + 0.1):       return -np.inf
        if not (P0 - 0.05 < per_ < P0 + 0.05):        return -np.inf
        if not (0.0 < rp_ < 0.1):                     return -np.inf
        if not (10.0 < a_ < 200.0):                   return -np.inf
        if not (80.0 < inc_ <= 90.0):                 return -np.inf
        # Gaussian prior on a/R* from the catalog stellar density (breaks the
        # a/R*-impact-depth degeneracy that mimics grazing for shallow transits).
        return -0.5 * ((a_ - a_rs_density) / a_sigma) ** 2

    def log_prob(theta):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        m = model(theta)
        return lp - 0.5 * np.sum((f_fit - m) ** 2 / err ** 2)

    ndim = 5
    scale = np.array([1e-3, 1e-4, 1e-3, 1.0, 0.05])
    p0 = seed + scale * np.random.randn(nwalkers, ndim)
    p0[:, 2] = np.abs(p0[:, 2])
    p0[:, 4] = np.clip(p0[:, 4], 80.1, 90.0)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, nsteps, progress=progress)
    flat_samples = sampler.get_chain(discard=nburn, flat=True)

    labels = ["t0", "period", "Rp/R*", "a/R*", "inc"]
    medians = {}
    best = np.zeros(ndim)
    for i, lab in enumerate(labels):
        loq, med, hiq = np.percentile(flat_samples[:, i], [16, 50, 84])
        medians[lab] = (med, med - loq, hiq - med)
        best[i] = med

    # --- derived, dilution-corrected quantities --------------------------------------
    cf = crowdsap if (np.isfinite(crowdsap) and crowdsap > 0) else 1.0
    per_samp = flat_samples[:, 1]
    rp_samp = flat_samples[:, 2]
    a_samp = flat_samples[:, 3]
    inc_samp = np.radians(flat_samples[:, 4])
    depth_obs = rp_samp ** 2 * 1e6
    depth_true = depth_obs / cf
    rp_true = rp_samp / np.sqrt(cf)                    # corrected radius ratio
    rpe_samp = rp_true * rstar_sun * config.RSUN_REARTH

    # Transit duration (T14) in hours from the fitted geometry:
    #   T14 = (P/pi) * arcsin[ (1/a) * sqrt((1+rp)^2 - b^2) / sin(i) ],  b = a cos(i)
    b_samp = a_samp * np.cos(inc_samp)
    arg = (1.0 + rp_samp) ** 2 - b_samp ** 2
    arg = np.clip(arg, 0, None)
    inner = np.sqrt(arg) / (a_samp * np.sin(inc_samp))
    inner = np.clip(inner, -1, 1)
    dur_samp = (per_samp / np.pi) * np.arcsin(inner) * 24.0   # hours

    for name, s in [("Depth_obs", depth_obs), ("Depth", depth_true), ("Rp", rpe_samp),
                    ("Duration_hr", dur_samp), ("impact_b", b_samp)]:
        loq, med, hiq = np.percentile(s, [16, 50, 84])
        medians[name] = (med, med - loq, hiq - med)

    return FitResult(
        labels=labels, medians=medians, flat_samples=flat_samples,
        t_fit=t_fit, f_fit=f_fit, best=best, crowdsap=cf, fit_window=fit_window,
        acceptance=float(np.mean(sampler.acceptance_fraction)),
    )


def model_curve(fit: FitResult, n=400, ld=None):
    """Best-fit transit curve over the fit window, for the phase-fold plot."""
    import batman
    t0 = fit.best[0]
    phase_t = np.linspace(-fit.fit_window, fit.fit_window, n)
    bp, bm = _make_model(t0 + phase_t, ld=ld)
    bp.t0, bp.per, bp.rp, bp.a, bp.inc = fit.best
    return phase_t, bm.light_curve(bp)
