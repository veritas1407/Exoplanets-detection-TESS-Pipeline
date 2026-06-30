"""exopipeline — a transit-vetting pipeline for noisy TESS light curves (NASA PS7).

Stages (each module is a thin, testable wrapper around one scientific step):

    ingest    -> SPOC 2-min light curves (single target + bulk per-sector manifest), TPFs
    detrend   -> wotan biweight detrending with iterative transit masking
    search    -> blind BLS broad sweep + iterative masking + narrow TLS refine
    vetting   -> physics-aware false-positive features (one row per candidate)
    classify  -> calibrated LightGBM classifier (transit / EB / blend / other)
    cnn       -> dual-view 1D-CNN (Astronet-style) + late-fusion ensemble with classify
    blend     -> difference-imaging centroid test + CROWDSAP dilution
    fit       -> batman + scipy seed + emcee posterior parameters with uncertainties
    report    -> the multi-panel vetting sheet
    scan      -> blind batch scan of a whole sector slice (the PS7 O2->O6 driver)
    injection -> injection-recovery completeness test
    labels    -> TOI-catalog label assembly + feature-table builder

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
    "cnn",
    "blend",
    "fit",
    "report",
    "scan",
    "featurize",
    "predict",
    "injection",
    "labels",
]

__version__ = "0.1.0"
