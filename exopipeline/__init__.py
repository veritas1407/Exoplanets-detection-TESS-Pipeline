"""exopipeline — a transit-vetting pipeline for noisy TESS light curves (NASA PS7).

Stages (each module is a thin, testable wrapper around one scientific step):

    ingest    -> download SPOC 2-min light curves, stitch, read CROWDSAP/FLFRCSAP, TPFs
    detrend   -> wotan biweight detrending with iterative transit masking
    search    -> blind BLS broad sweep + iterative masking + narrow TLS refine
    vetting   -> physics-aware false-positive features (one row per candidate)
    classify  -> calibrated LightGBM classifier (transit / EB / blend / other)
    blend     -> difference-imaging centroid test + CROWDSAP dilution
    fit       -> batman + scipy seed + emcee posterior parameters with uncertainties
    report    -> the multi-panel vetting sheet
    injection -> injection-recovery completeness test

Typical use:
    from exopipeline import ingest, detrend, search, vetting, fit, report
    star = ingest.load_star("TIC 307210830")
    flat = detrend.detrend(star.time, star.flux)
    candidates = search.find_planets(flat.time, flat.flux)
"""
from __future__ import annotations

from . import config  # noqa: F401

__all__ = [
    "config",
    "ingest",
    "detrend",
    "search",
    "vetting",
    "classify",
    "blend",
    "fit",
    "report",
    "injection",
]

__version__ = "0.1.0"
