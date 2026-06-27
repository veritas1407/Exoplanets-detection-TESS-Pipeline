"""Streamlit vetting app for the PS7 exoplanet pipeline.

Enter a TIC ID -> the pipeline downloads SPOC 2-min photometry, runs the blind multi-planet
search, vets + classifies the strongest candidate, fits it with MCMC, optionally runs the
difference-image blend test, and renders the vetting sheet + verdict.

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from exopipeline import ingest, detrend, search, vetting, classify, fit, report, blend, config

st.set_page_config(page_title="TESS Exoplanet Vetting Pipeline", layout="wide")
st.title("🪐 TESS Exoplanet Vetting Pipeline — PS7")
st.caption("Blind transit search · physics-aware vetting · calibrated classifier · "
           "MCMC parameters · difference-imaging blend test")

with st.sidebar:
    st.header("Target")
    target = st.text_input("TIC ID", value=config.DEMO_PLANET)
    max_sectors = st.slider("Max sectors (speed vs. baseline)", 1, 30, 8)
    max_planets = st.slider("Planets to search", 1, 5, 3)
    run_blend = st.checkbox("Run difference-image blend test (downloads TPF)", value=False)
    run_btn = st.button("Run pipeline", type="primary")


@st.cache_data(show_spinner=False)
def _load(target, max_sectors):
    star = ingest.load_star(target, max_sectors=max_sectors)
    star = ingest.clean(star)
    return star


def run(target, max_sectors, max_planets, run_blend):
    with st.status("Downloading + processing…", expanded=True) as status:
        st.write("Downloading SPOC light curves from MAST…")
        star = _load(target, max_sectors)
        st.write(f"Stitched {star.n_sectors} sectors, baseline {star.baseline_days:.0f} d, "
                 f"CROWDSAP={star.crowdsap:.3f}")

        st.write("Detrending (wotan biweight)…")
        flat = detrend.detrend(star.time, star.flux)

        st.write("Blind multi-planet search (BLS + masking + TLS refine)…")
        cands = search.find_planets(flat.time, flat.flux, max_planets=max_planets,
                                    verbose=False)
        if not cands:
            status.update(label="No transit signals found.", state="error")
            st.warning("No candidate above threshold.")
            return
        st.write(f"Found {len(cands)} candidate(s): "
                 + ", ".join(f"{c.period:.3f} d" for c in cands))

        cand = cands[0]
        st.write("Computing vetting features…")
        feats = vetting.compute_features(flat.time, flat.flux, cand, crowdsap=star.crowdsap)

        st.write("Classifying…")
        verdict, conf = classify.predict(feats)

        st.write("Fitting transit model (batman + emcee)…")
        fr = fit.fit_transit(flat.time, flat.flux, cand, crowdsap=star.crowdsap,
                             nsteps=1500, nburn=500)

        blend_res = None
        if run_blend:
            st.write("Difference-image blend test…")
            tpf = ingest.load_tpf(target)
            blend_res = blend.centroid_test(tpf, cand.period, cand.T0, cand.duration)

        status.update(label="Done.", state="complete")

    # --- Verdict banner ---
    color = {"transit": "green", "eclipsing_binary": "red",
             "blend": "orange", "other": "gray"}.get(verdict, "blue")
    st.markdown(f"### Verdict: :{color}[{verdict.upper()}] — confidence {conf*100:.0f}%")

    c1, c2, c3, c4 = st.columns(4)
    P_m, P_lo, P_hi = fr.medians["period"]
    D_m, _, _ = fr.medians["Depth"]
    Rp_m, Rp_lo, Rp_hi = fr.medians["Rp"]
    c1.metric("Period (d)", f"{P_m:.4f}", f"±{max(P_lo,P_hi):.4f}")
    c2.metric("Depth (ppm)", f"{D_m:.0f}")
    c3.metric("Rp (R⊕)", f"{Rp_m:.2f}", f"±{max(Rp_lo,Rp_hi):.2f}")
    c4.metric("SNR / SDE", f"{cand.snr:.1f} / {cand.SDE:.1f}")

    st.write("Building vetting sheet…")
    fig = report.vetting_sheet(star, flat, cand, feats, fr, verdict, conf,
                               blend=blend_res,
                               periodogram=None,
                               title=f"{target} — Vetting Sheet")
    st.pyplot(fig)

    if len(cands) > 1:
        st.subheader("All recovered candidates")
        st.table([{"period_d": round(c.period, 4),
                   "depth_ppm": round(c.depth_ppm, 0),
                   "duration_h": round(c.duration_hr, 2),
                   "SDE": round(c.SDE, 1), "SNR": round(c.snr, 1)} for c in cands])


if run_btn:
    run(target, max_sectors, max_planets, run_blend)
else:
    st.info("Enter a TIC ID in the sidebar and click **Run pipeline**. "
            "Try a known planet (TIC 307210830 = TOI 700), an eclipsing binary, "
            "or a known blend to see the pipeline separate them.")
