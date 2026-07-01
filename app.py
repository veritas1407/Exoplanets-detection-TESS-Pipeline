"""Streamlit vetting app for the PS7 exoplanet pipeline.

Enter a TIC ID -> the pipeline downloads SPOC 2-min photometry, runs the blind multi-planet
search, vets + classifies the strongest candidate with the **dual-track ensemble** (calibrated
LightGBM + dual-view CNN), fits it with MCMC, optionally runs the difference-image blend test,
and renders an **explainable verdict** + the vetting sheet.

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from exopipeline import (ingest, detrend, search, vetting, classify, fit, report,
                         blend, config, cnn, predict)

st.set_page_config(page_title="TESS Exoplanet Vetting Pipeline",
                   page_icon="🪐", layout="wide")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
.big-verdict {font-size:2.0rem; font-weight:800; margin:0.2rem 0;}
.reason-box {background:#0e1117; border-left:4px solid #ff6b00; padding:0.9rem 1.1rem;
             border-radius:6px; font-size:1.05rem; line-height:1.5;}
.stMetric {background:#0e1117; border-radius:8px; padding:0.4rem 0.6rem;}
</style>
""", unsafe_allow_html=True)

st.title("🪐 TESS Exoplanet Vetting Pipeline")
st.caption("Blind transit search · physics-aware vetting · **dual-track AI ensemble** · "
           "MCMC parameters · difference-imaging blend test · explainable verdicts — PS7")

# Verified demo targets (ground truth from NASA)
EXAMPLES = {
    "TOI-270 (3-planet system)": ("TIC 259377017", "planet", 6),
    "TOI 700 (habitable-zone)":  ("TIC 307210830", "planet", 12),
    "Eclipsing binary":          ("TIC 441075486", "EB", 2),
    "Quiet field star":          ("TIC 25132999", "noise", 2),
}

CLASS_STYLE = {
    "transit":          ("🟢", "green",  "Planet candidate (transiting)"),
    "eclipsing_binary": ("🔴", "red",    "Eclipsing binary"),
    "blend":            ("🟠", "orange", "Blended / background eclipsing binary"),
    "other":            ("⚪", "gray",   "No planet — variable star or noise"),
}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🎯 Target")
    if "tic" not in st.session_state:
        st.session_state.tic = config.DEMO_BLIND
        st.session_state.sectors = 6

    st.markdown("**Quick demos** (verified vs. NASA):")
    for label, (tic, _kind, secs) in EXAMPLES.items():
        if st.button(label, use_container_width=True):
            st.session_state.tic = tic
            st.session_state.sectors = secs

    target = st.text_input("TIC ID", key="tic")
    max_sectors = st.slider("Max sectors (baseline vs. speed)", 1, 30,
                            st.session_state.get("sectors", 6))
    max_planets = st.slider("Planets to search", 1, 5, 3)
    do_fit = st.checkbox("MCMC parameter fit (slower)", value=True)
    run_blend = st.checkbox("Difference-image blend test (downloads TPF)", value=False)
    run_btn = st.button("🚀 Run pipeline", type="primary", use_container_width=True)


@st.cache_data(show_spinner=False)
def _load(target, max_sectors):
    return ingest.clean(ingest.load_star(target, max_sectors=max_sectors))


# ------------------------------------------------------------------ pipeline
def run(target, max_sectors, max_planets, do_fit, run_blend):
    with st.status("Downloading + processing…", expanded=True) as status:
        st.write("Downloading SPOC light curves from MAST…")
        try:
            star = _load(target, max_sectors)
        except Exception as e:
            status.update(label="Download failed.", state="error")
            st.error(f"Could not load {target}: {e}")
            return
        st.write(f"Stitched {star.n_sectors} sector(s), baseline {star.baseline_days:.0f} d, "
                 f"CROWDSAP={star.crowdsap:.3f}")

        st.write("Detrending (wotan biweight)…")
        flat = detrend.detrend(star.time, star.flux)

        st.write("Blind multi-planet search (BLS + iterative masking)…")
        cands = search.find_planets(flat.time, flat.flux, max_planets=max_planets,
                                    verbose=False)
        if not cands:
            status.update(label="No transit-like signal found → 'other'.", state="complete")
            st.markdown('<div class="big-verdict">⚪ :gray[NO PLANET — variable star or '
                        'noise]</div>', unsafe_allow_html=True)
            st.info("No periodic dip crossed the detection threshold. For an unknown target "
                    "this is the correct **'other'** classification.")
            return
        st.write(f"Found {len(cands)} candidate(s): "
                 + ", ".join(f"{c.period:.3f} d" for c in cands))

        cand = cands[0]
        st.write("Computing 21 physics vetting features…")
        stellar = ingest.fetch_stellar(target)
        rstar, mstar = stellar or (np.nan, np.nan)
        feats = vetting.compute_features(flat.time, flat.flux, cand,
                                         crowdsap=star.crowdsap,
                                         rstar_sun=rstar, mstar_sun=mstar)

        st.write("Classifying with dual-track ensemble (LightGBM + CNN + TTA)…")
        g, l = cnn.make_views(flat.time, flat.flux, cand.period, cand.T0, cand.duration)
        fused, tab_p, cnn_p = cnn.ensemble_proba(feats, g, l)
        if fused:
            verdict = max(fused, key=fused.get)
            conf = float(fused[verdict])
        else:
            verdict, conf = classify.predict(feats)
            fused = {verdict: conf}

        fr = None
        if do_fit:
            st.write("Fitting transit model (batman + emcee MCMC)…")
            try:
                fr = fit.fit_transit(flat.time, flat.flux, cand, crowdsap=star.crowdsap,
                                     nsteps=1500, nburn=500)
            except Exception as e:
                st.write(f"(fit skipped: {e})")

        blend_res = None
        if run_blend:
            st.write("Difference-image blend test…")
            try:
                tpf = ingest.load_tpf(target)
                blend_res = blend.centroid_test(tpf, cand.period, cand.T0, cand.duration)
            except Exception as e:
                st.write(f"(blend test skipped: {e})")

        status.update(label="Done.", state="complete")

    _render(target, star, flat, cand, cands, feats, fr, blend_res,
            verdict, conf, fused, tab_p, cnn_p)


# ------------------------------------------------------------------ rendering
def _phase_plot(flat, cand):
    """Simple phase-folded scatter (fallback when no MCMC fit is available)."""
    import matplotlib.pyplot as plt
    phase = ((flat.time - cand.T0 + 0.5 * cand.period) % cand.period) / cand.period - 0.5
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.scatter(phase, flat.flux, s=2, alpha=0.35, color="#9ecbff")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("phase"); ax.set_ylabel("normalized flux")
    ax.set_title(f"Phase-folded at P = {cand.period:.4f} d")
    st.pyplot(fig)


def _render(target, star, flat, cand, cands, feats, fr, blend_res,
            verdict, conf, fused, tab_p, cnn_p):
    icon, color, nice = CLASS_STYLE.get(verdict, ("🔵", "blue", verdict))

    # --- Verdict banner ---
    st.markdown(f'<div class="big-verdict">{icon} :{color}[{nice.upper()}] '
                f'— {conf*100:.0f}% confidence</div>', unsafe_allow_html=True)

    # --- Explainable justification (the USP) ---
    v = predict.Verdict(target=str(target), event_detected=True,
                        classification=verdict, confidence=conf,
                        period=cand.period, t0=cand.T0, depth_ppm=cand.depth_ppm,
                        duration_hr=cand.duration_hr, sde=cand.SDE, snr=cand.snr,
                        fap=cand.FAP, n_transits=int(cand.distinct_transit_count),
                        rp_rs=cand.rp_rs)
    if fr is not None:
        try:
            v.rp_earth = float(fr.medians["Rp"][0])
        except Exception:
            pass
    if blend_res is not None:
        v.is_blend = bool(blend_res.get("is_blend", False))
        v.centroid_offset_arcsec = float(blend_res.get("offset_arcsec", np.nan))
    para, _reasons = predict._explain(verdict, conf, feats, v)
    st.markdown(f'<div class="reason-box">🔎 {para}</div>', unsafe_allow_html=True)
    st.write("")

    # --- Significance metrics ---
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Period (d)", f"{cand.period:.4f}")
    c2.metric("Depth (ppm)", f"{cand.depth_ppm:.0f}")
    c3.metric("SDE", f"{cand.SDE:.1f}")
    c4.metric("SNR", f"{cand.snr:.1f}")
    c5.metric("Transits", f"{int(cand.distinct_transit_count)}")
    if fr is not None:
        try:
            Rp_m, Rp_lo, Rp_hi = fr.medians["Rp"]
            st.metric("Planet radius Rp (R⊕)", f"{Rp_m:.2f}",
                      f"+{Rp_hi:.2f} / -{Rp_lo:.2f}")
        except Exception:
            pass

    # --- Class probability bars (ensemble + per-track) ---
    st.subheader("Classification confidence")
    order = ["transit", "eclipsing_binary", "other"]
    import pandas as pd
    rows = {}
    if fused:
        rows["Ensemble"] = {k: fused.get(k, 0.0) for k in order}
    if tab_p:
        rows["LightGBM (features)"] = {k: tab_p.get(k, 0.0) for k in order}
    if cnn_p:
        rows["CNN (shape)"] = {k: cnn_p.get(k, 0.0) for k in order}
    if rows:
        df = pd.DataFrame(rows).T[order]
        df.columns = ["🟢 transit", "🔴 eclipsing binary", "⚪ other"]
        st.bar_chart(df.T)

    # --- Vetting sheet (full sheet if we have an MCMC fit, else a phase-fold) ---
    st.subheader("Vetting sheet")
    if fr is not None:
        try:
            fig = report.vetting_sheet(star, flat, cand, feats, fr, verdict, conf,
                                       blend=blend_res, periodogram=None,
                                       title=f"{target} — Vetting Sheet")
            st.pyplot(fig)
        except Exception as e:
            st.write(f"(full sheet unavailable: {e})")
            _phase_plot(flat, cand)
    else:
        _phase_plot(flat, cand)

    # --- All candidates ---
    if len(cands) > 1:
        st.subheader("All recovered candidates")
        st.table([{"period_d": round(c.period, 4),
                   "depth_ppm": round(c.depth_ppm, 0),
                   "duration_h": round(c.duration_hr, 2),
                   "SDE": round(c.SDE, 1), "SNR": round(c.snr, 1)} for c in cands])


# ------------------------------------------------------------------ entry
if run_btn:
    run(target, max_sectors, max_planets, do_fit, run_blend)
else:
    st.info("👈 Pick a **quick demo** or enter a TIC ID, then click **Run pipeline**. "
            "The pipeline blindly searches the light curve, classifies the strongest "
            "signal with the AI ensemble, and explains *why* — planet, eclipsing binary, "
            "or noise — with the significance and physical parameters that justify the call.")
    cols = st.columns(4)
    facts = [("Ensemble macro-F1", "0.93"), ("Sector-5 blind scan", "4,157 stars"),
             ("Detections", "570 in 66 min"), ("Vetting features", "21 physics")]
    for col, (k, val) in zip(cols, facts):
        col.metric(k, val)
