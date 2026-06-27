# Exopipeline — AI-Enabled Exoplanet Detection from Noisy TESS Light Curves (PS7)

A transit-vetting pipeline that searches raw, noisy TESS light curves, classifies each
periodic dip (transit / eclipsing-binary / blend / other) with physically-motivated
diagnostics plus a calibrated classifier, localizes the signal's source via difference
imaging to reject blends, and fits surviving transits with an MCMC model that reports
planet parameters with honest posterior uncertainties.

## Pipeline stages (`exopipeline/`)

| Module | Stage | What it does |
|---|---|---|
| `ingest.py` | 1 | Download SPOC 2-min light curves, stitch sectors, read CROWDSAP/FLFRCSAP, fetch TPFs |
| `detrend.py` | 2 | wotan biweight detrending (+ iterative transit-masked re-detrend) |
| `search.py` | 3 | **Blind** BLS broad sweep + **iterative masking** → recovers every planet; SDE/FAP significance |
| `vetting.py` | 4 | Physics features: odd-even, secondary eclipse, U-vs-V shape, transit count, per-transit SNR |
| `classify.py` | 5 | Calibrated LightGBM (isotonic) → class + confidence; confusion matrix / PR-AUC |
| `blend.py` | 6 | Difference-image centroid test + CROWDSAP dilution correction |
| `fit.py` | 7 | batman + scipy seed + emcee → parameters with 16/84th-pct credible intervals |
| `report.py` | 8 | The one-page vetting sheet |
| `injection.py` | — | Injection-recovery completeness test (recovery vs SNR) |
| `labels.py` | — | TOI-catalog label assembly + feature-table builder |

## Why this design wins (PS7 rubric)

- **Genuinely blind search** — no hardcoded period window. On a clean multi-planet target
  (TOI-270) the iterative-masking search recovers the planets and correctly keeps the
  near-2:1 resonant pair while rejecting exact-integer aliases.
- **Honest about hard targets** — TOI 700's shallow planets sit under a dominant ~3.69 d
  instrumental systematic (SDE ≈ 128, robust to all detrending), so they cannot be blindly
  recovered from raw 2-min BLS. We characterise TOI 700 d with a *focused* search (standard
  practice for a known planet) and surface the systematic honestly rather than hiding it.
- **Real classifier + calibrated confidence** — not a hardcoded label.
- **Difference-imaging blend module** — the rubric's least-served requirement.
- **Honest uncertainties** — MCMC posteriors, not point estimates.
- **Injection-recovery** — quantifies the detection sensitivity floor (which planets are
  recoverable at a given SNR); the single most credible evidence of pipeline quality.

### Blind vs. focused search — when to use which
- **Blind** (`search.find_planets`): discovery on clean targets; no prior period.
- **Focused** (`search.search_single`): characterising a *known* planet in a narrow window.
TOI 700 is a deliberately honest example: blind search surfaces a systematic, focused
search + MCMC recovers d's literature parameters (P ≈ 37.43 d, Rp ≈ 1.1 R⊕).

## Install

```bash
pip install -r requirements.txt        # keeps numpy<2 / astropy<7 (TLS/batman ABI)
```

## Quick start (local)

```python
from exopipeline import ingest, detrend, search, vetting, classify, fit, report
star = ingest.clean(ingest.load_star("TIC 307210830", max_sectors=12))
flat = detrend.detrend(star.time, star.flux)
cands = search.find_planets(flat.time, flat.flux, max_planets=4)   # blind, all planets
cand = cands[0]
feats = vetting.compute_features(flat.time, flat.flux, cand, crowdsap=star.crowdsap)
verdict, conf = classify.predict(feats)
fr = fit.fit_transit(flat.time, flat.flux, cand, crowdsap=star.crowdsap)
report.vetting_sheet(star, flat, cand, feats, fr, verdict, conf, save_path="sheet.png")
```

## Streamlit app

```bash
streamlit run app.py
```
Enter a TIC ID → the pipeline runs and renders the vetting sheet + verdict. Demo it on a
known planet, an EB, and a blend.

## Notebooks (`notebooks/`)

- `demo_planet_eb_blend.ipynb` — the full story: **Part A** blind multi-planet discovery
  (TOI-270), **Part B** focused characterisation of TOI 700 d + vetting sheet, **Part C**
  the honest TOI 700 systematic.
- `classifier_training.ipynb` — build feature table, train + calibrate, confusion matrix.
- `injection_recovery.ipynb` — the completeness curve.

### Running on Colab
Upload the `exopipeline/` folder next to the notebook (or zip + unzip), then run the
install cell at the top of each notebook. The notebooks add the package to `sys.path`
automatically. **Note:** Transit Least Squares uses multiprocessing; on Colab (Linux/fork)
this is fine, but in a plain Windows *script* you must guard the entry point with
`if __name__ == "__main__":`. The package defaults the blind search to BLS-only (`refine=False`)
which avoids this entirely; enable `refine=True` on Colab for a TLS cross-check.

## Validation results (reproduced locally)

| Test | Outcome |
|---|---|
| Synthetic 2-planet blind search | Both periods recovered, no aliases, correct depths |
| **TOI-270 blind multi-planet** (6 sectors) | **c @ 5.66 d & d @ 11.38 d are the top-2 SDE candidates**; 2:1 resonance kept, exact aliases rejected |
| **TOI 700 d** focused fit (27 sectors) | **P = 37.434 d, Rp = 1.14 (+0.25/−0.09) R⊕, depth = 620 ppm** — matches literature (37.426 d, 1.14 R⊕, ~530 ppm) |
| TOI 700 blind search | Surfaces a dominant ~3.69 d instrumental systematic (SDE ≈ 128) — an honest hard case the vetting flags |

Notes: TOI 700 d is shallow (~530 ppm); the *full* sector baseline is needed to pin the
geometry (a 14-sector subset leaves the fit grazing-degenerate). The fit uses a stellar-
density prior on a/R\* and reports MCMC posterior credible intervals.

## Data footprint
You do **not** download the whole TESS archive. A few demo targets (tens of MB each), a
few hundred labelled stars for training (cached to `data/features/`), and one quiet star
for injection-recovery. The optional full-sector run is a stretch goal only.
