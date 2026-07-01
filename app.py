"""Streamlit vetting app for the PS7 exoplanet pipeline.

Enter a TIC ID -> the pipeline downloads SPOC 2-min photometry, runs the blind multi-planet
search, vets + classifies the strongest candidate with the **dual-track ensemble** (calibrated
LightGBM + dual-view CNN), fits it with MCMC, optionally runs the difference-image blend test,
and renders an **explainable verdict** + interactive graphs + the vetting sheet.

Run:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st
import plotly.graph_objects as go

from exopipeline import (ingest, detrend, search, vetting, classify, fit, report,
                         blend, config, cnn, predict)

st.set_page_config(page_title="TESS Exoplanet Vetting Pipeline",
                   page_icon="🪐", layout="wide",
                   initial_sidebar_state="expanded")

# ------------------------------------------------------------------ theme / CSS
st.markdown("""
<style>
.stApp { background: radial-gradient(1200px 600px at 80% -10%, #16203a 0%, #0b0e17 45%); }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.5rem; }

.hero {
  background: linear-gradient(100deg, #1a2440 0%, #0e1526 60%);
  border: 1px solid #243049; border-radius: 16px;
  padding: 1.4rem 1.8rem; margin-bottom: 1.2rem;
  box-shadow: 0 8px 30px rgba(0,0,0,.35);
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 800;
  background: linear-gradient(90deg,#ff8a3d,#ffd18c); -webkit-background-clip: text;
  -webkit-text-fill-color: transparent; }
.hero p { margin: .35rem 0 0; color: #9fb0c8; font-size: .98rem; }

.verdict-card {
  border-radius: 16px; padding: 1.2rem 1.5rem; margin: .3rem 0 1rem;
  border: 1px solid rgba(255,255,255,.08);
  box-shadow: 0 8px 30px rgba(0,0,0,.35);
}
.verdict-title { font-size: 1.9rem; font-weight: 800; margin: 0; letter-spacing: .3px; }
.verdict-sub { color: #cbd6e6; font-size: 1.02rem; margin-top: .5rem; line-height: 1.55; }

.pill { display:inline-block; padding:.18rem .7rem; border-radius:999px;
  font-size:.8rem; font-weight:700; margin-right:.4rem; }

div[data-testid="stMetric"] {
  background: #121a2c; border: 1px solid #223049; border-radius: 12px;
  padding: .7rem .9rem;
}
div[data-testid="stMetricValue"] { font-size: 1.4rem; color:#ffd18c; }
section[data-testid="stSidebar"] { background: #0d1320; border-right: 1px solid #1b2436; }
.stButton>button { border-radius: 10px; border:1px solid #2a3550; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ constants
EXAMPLES = {
    "🪐 TOI-270  (3-planet system)": ("TIC 259377017", 6),
    "🌍 TOI 700  (habitable zone)":  ("TIC 307210830", 12),
    "🔴 Eclipsing binary":           ("TIC 441075486", 2),
    "⚪ Quiet field star":           ("TIC 25132999", 2),
}
CLASS_STYLE = {
    "transit":          ("🟢", "#27c93f", "Planet candidate (transiting)"),
    "eclipsing_binary": ("🔴", "#ff5f56", "Eclipsing binary"),
    "blend":            ("🟠", "#ff9f43", "Blended / background eclipsing binary"),
    "other":            ("⚪", "#8a97a8", "No planet — variable star or noise"),
}
PLOT_BG = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0e1424",
               font=dict(color="#cbd6e6"), margin=dict(l=50, r=20, t=40, b=40))

# ------------------------------------------------------------------ session
if "tic" not in st.session_state:
    st.session_state.tic = config.DEMO_BLIND
if "sectors" not in st.session_state:
    st.session_state.sectors = 6


def _set_target(tic, secs):
    st.session_state.tic = tic
    st.session_state.sectors = secs


# ------------------------------------------------------------------ hero
st.markdown(
    '<div class="hero"><h1>🪐 TESS Exoplanet Vetting Pipeline</h1>'
    '<p>Blind transit search · physics-aware vetting · <b>dual-track AI ensemble</b> '
    '(LightGBM + CNN) · MCMC parameters · difference-imaging blend test · '
    '<b>explainable verdicts</b> — ISRO BAH 2026 · PS7</p></div>',
    unsafe_allow_html=True)

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### 🎯 Target")
    st.caption("Quick demos — verified vs. NASA ground truth:")
    for label, (tic, secs) in EXAMPLES.items():
        st.button(label, use_container_width=True, on_click=_set_target, args=(tic, secs))

    st.divider()
    target = st.text_input("TIC ID", key="tic")
    max_sectors = st.slider("Max sectors (baseline vs. speed)", 1, 30, key="sectors")
    max_planets = st.slider("Planets to search", 1, 5, 3)
    do_fit = st.checkbox("MCMC parameter fit (slower)", value=True)
    run_blend = st.checkbox("Difference-image blend test (downloads TPF)", value=False)
    run_btn = st.button("🚀 Run pipeline", type="primary", use_container_width=True)


@st.cache_resource(show_spinner=False)
def _load(target, max_sectors):
    # cache_resource (not cache_data): Star holds an unpicklable lightkurve object
    return ingest.clean(ingest.load_star(target, max_sectors=max_sectors))


# ------------------------------------------------------------------ graphs
def _fig_lightcurve(star, flat):
    fig = go.Figure()
    fig.add_scatter(x=star.time, y=star.flux, mode="markers",
                    marker=dict(size=2, color="#3d5a80", opacity=.5), name="raw")
    fig.add_scatter(x=flat.time, y=flat.flux, mode="markers",
                    marker=dict(size=2, color="#9ecbff", opacity=.7), name="detrended")
    fig.update_layout(title="Light curve (raw → detrended)", height=280,
                      xaxis_title="time (BTJD)", yaxis_title="normalized flux",
                      legend=dict(orientation="h", y=1.15), **PLOT_BG)
    return fig


def _binned(phase, flux, nbins=120):
    edges = np.linspace(-0.5, 0.5, nbins + 1)
    idx = np.clip(np.digitize(phase, edges) - 1, 0, nbins - 1)
    xs, ys = [], []
    for b in range(nbins):
        sel = flux[idx == b]
        if sel.size:
            xs.append(0.5 * (edges[b] + edges[b + 1])); ys.append(np.median(sel))
    return np.array(xs), np.array(ys)


def _fig_phasefold(flat, cand, color):
    phase = ((flat.time - cand.T0 + 0.5 * cand.period) % cand.period) / cand.period - 0.5
    bx, by = _binned(phase, flat.flux)
    fig = go.Figure()
    fig.add_scatter(x=phase, y=flat.flux, mode="markers",
                    marker=dict(size=3, color="#2f4a6b", opacity=.35), name="folded")
    fig.add_scatter(x=bx, y=by, mode="markers",
                    marker=dict(size=6, color=color), name="binned")
    fig.update_layout(title=f"Phase-folded at P = {cand.period:.4f} d", height=320,
                      xaxis_title="phase", yaxis_title="normalized flux",
                      xaxis_range=[-0.5, 0.5], legend=dict(orientation="h", y=1.15),
                      **PLOT_BG)
    return fig


def _fig_gauge(conf, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=conf * 100,
        number=dict(suffix="%", font=dict(size=34, color=color)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor="#5b6b82"),
                   bar=dict(color=color, thickness=.28),
                   bgcolor="#0e1424", borderwidth=0,
                   steps=[dict(range=[0, 50], color="#141d30"),
                          dict(range=[50, 80], color="#182238"),
                          dict(range=[80, 100], color="#1d2b45")])))
    fig.update_layout(title="Confidence", height=250, **PLOT_BG)
    return fig


def _fig_probs(fused, tab_p, cnn_p):
    order = ["transit", "eclipsing_binary", "other"]
    labels = ["🟢 transit", "🔴 eclipsing binary", "⚪ other"]
    fig = go.Figure()
    tracks = [("Ensemble", fused, "#ff8a3d"), ("LightGBM", tab_p, "#54a0ff"),
              ("CNN", cnn_p, "#5f27cd")]
    for name, p, col in tracks:
        if p:
            fig.add_bar(name=name, x=labels, y=[p.get(k, 0) * 100 for k in order],
                        marker_color=col)
    fig.update_layout(title="Per-track class probability", barmode="group", height=300,
                      yaxis_title="probability (%)", yaxis_range=[0, 100],
                      legend=dict(orientation="h", y=1.18), **PLOT_BG)
    return fig


# ------------------------------------------------------------------ pipeline
def run(target, max_sectors, max_planets, do_fit, run_blend):
    prog = st.progress(0.0, text="Starting…")
    try:
        prog.progress(0.1, text="Downloading SPOC light curves from MAST…")
        star = _load(target, max_sectors)
        prog.progress(0.3, text="Detrending (wotan biweight)…")
        flat = detrend.detrend(star.time, star.flux)
        prog.progress(0.5, text="Blind multi-planet search (BLS + masking)…")
        cands = search.find_planets(flat.time, flat.flux, max_planets=max_planets,
                                    verbose=False)
    except Exception as e:
        prog.empty(); st.error(f"Could not process {target}: {e}"); return

    if not cands:
        prog.empty()
        st.markdown(
            '<div class="verdict-card" style="background:linear-gradient(100deg,#1c2333,#121826)">'
            '<p class="verdict-title">⚪ No planet — variable star or noise</p>'
            '<p class="verdict-sub">No periodic dip crossed the detection threshold. '
            'For an unknown target this is the correct <b>"other"</b> classification.</p></div>',
            unsafe_allow_html=True)
        st.plotly_chart(_fig_lightcurve(star, flat), use_container_width=True)
        return

    cand = cands[0]
    prog.progress(0.65, text="Computing 21 physics vetting features…")
    stellar = ingest.fetch_stellar(target)
    rstar, mstar = stellar or (np.nan, np.nan)
    feats = vetting.compute_features(flat.time, flat.flux, cand, crowdsap=star.crowdsap,
                                     rstar_sun=rstar, mstar_sun=mstar)

    prog.progress(0.78, text="Classifying (LightGBM + CNN + TTA ensemble)…")
    g, l = cnn.make_views(flat.time, flat.flux, cand.period, cand.T0, cand.duration)
    fused, tab_p, cnn_p = cnn.ensemble_proba(feats, g, l)
    if fused:
        verdict = max(fused, key=fused.get); conf = float(fused[verdict])
    else:
        verdict, conf = classify.predict(feats); fused = {verdict: conf}

    fr = None
    if do_fit:
        prog.progress(0.88, text="Fitting transit model (batman + emcee MCMC)…")
        try:
            fr = fit.fit_transit(flat.time, flat.flux, cand, crowdsap=star.crowdsap,
                                 nsteps=1500, nburn=500)
        except Exception:
            fr = None

    blend_res = None
    if run_blend:
        prog.progress(0.95, text="Difference-image blend test…")
        try:
            tpf = ingest.load_tpf(target)
            blend_res = blend.centroid_test(tpf, cand.period, cand.T0, cand.duration)
        except Exception:
            blend_res = None

    prog.progress(1.0, text="Done."); prog.empty()
    _render(target, star, flat, cand, cands, feats, fr, blend_res,
            verdict, conf, fused, tab_p, cnn_p)


# ------------------------------------------------------------------ rendering
def _render(target, star, flat, cand, cands, feats, fr, blend_res,
            verdict, conf, fused, tab_p, cnn_p):
    icon, color, nice = CLASS_STYLE.get(verdict, ("🔵", "#54a0ff", verdict))

    # explanation paragraph (reuse predict._explain)
    v = predict.Verdict(target=str(target), event_detected=True, classification=verdict,
                        confidence=conf, period=cand.period, t0=cand.T0,
                        depth_ppm=cand.depth_ppm, duration_hr=cand.duration_hr,
                        sde=cand.SDE, snr=cand.snr, fap=cand.FAP,
                        n_transits=int(cand.distinct_transit_count), rp_rs=cand.rp_rs)
    if fr is not None:
        try: v.rp_earth = float(fr.medians["Rp"][0])
        except Exception: pass
    if blend_res is not None:
        v.is_blend = bool(blend_res.get("is_blend", False))
        v.centroid_offset_arcsec = float(blend_res.get("offset_arcsec", np.nan))
    para, _ = predict._explain(verdict, conf, feats, v)

    # verdict card + gauge
    left, right = st.columns([2.4, 1])
    with left:
        st.markdown(
            f'<div class="verdict-card" style="background:linear-gradient(100deg,'
            f'{color}22,#12182699); border-left:5px solid {color}">'
            f'<p class="verdict-title" style="color:{color}">{icon} {nice}</p>'
            f'<p class="verdict-sub">🔎 {para}</p></div>', unsafe_allow_html=True)
        m = st.columns(5)
        m[0].metric("Period (d)", f"{cand.period:.4f}")
        m[1].metric("Depth (ppm)", f"{cand.depth_ppm:.0f}")
        m[2].metric("SDE", f"{cand.SDE:.1f}")
        m[3].metric("SNR", f"{cand.snr:.1f}")
        m[4].metric("Transits", f"{int(cand.distinct_transit_count)}")
        if fr is not None:
            try:
                Rp_m, Rp_lo, Rp_hi = fr.medians["Rp"]
                st.metric("Planet radius Rp (R⊕)", f"{Rp_m:.2f}",
                          f"+{Rp_hi:.2f} / -{Rp_lo:.2f}")
            except Exception:
                pass
    with right:
        st.plotly_chart(_fig_gauge(conf, color), use_container_width=True)

    # graphs
    g1, g2 = st.columns(2)
    g1.plotly_chart(_fig_phasefold(flat, cand, color), use_container_width=True)
    g2.plotly_chart(_fig_probs(fused, tab_p, cnn_p), use_container_width=True)
    st.plotly_chart(_fig_lightcurve(star, flat), use_container_width=True)

    # extras
    if len(cands) > 1:
        st.subheader("All recovered candidates")
        st.dataframe([{"period_d": round(c.period, 4), "depth_ppm": round(c.depth_ppm, 0),
                       "duration_h": round(c.duration_hr, 2), "SDE": round(c.SDE, 1),
                       "SNR": round(c.snr, 1)} for c in cands], use_container_width=True)

    if fr is not None:
        with st.expander("📄 Full MCMC vetting sheet"):
            try:
                fig = report.vetting_sheet(star, flat, cand, feats, fr, verdict, conf,
                                           blend=blend_res, periodogram=None,
                                           title=f"{target} — Vetting Sheet")
                st.pyplot(fig)
            except Exception as e:
                st.write(f"(sheet unavailable: {e})")


# ------------------------------------------------------------------ entry
if run_btn:
    run(target, max_sectors, max_planets, do_fit, run_blend)
else:
    st.info("👈 Pick a **quick demo** or enter a TIC ID, then click **🚀 Run pipeline**. "
            "The pipeline blindly searches the light curve, classifies the strongest signal "
            "with the AI ensemble, and explains *why* — with the significance and physical "
            "parameters that justify the call.")
    cols = st.columns(4)
    facts = [("Ensemble macro-F1", "0.93"), ("Sector-5 blind scan", "4,157 stars"),
             ("Detections", "570 in 66 min"), ("Vetting features", "21 physics")]
    for col, (k, val) in zip(cols, facts):
        col.metric(k, val)
