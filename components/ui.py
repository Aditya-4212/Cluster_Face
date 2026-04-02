# ============================================================
# components/ui.py
# Reusable UI components — headers, callouts, metric strips,
# pipeline stepper, progress tracker, sidebar
# Edit the look of any UI element here
# ============================================================

import streamlit as st
from config.settings import PIPELINE_STEPS, SESSION_DEFAULTS


# ── Section divider ─────────────────────────────────────────

def section(label: str):
    st.markdown(f'<div class="sec">{label}</div>', unsafe_allow_html=True)


# ── Contextual learn / warn / info boxes ────────────────────

def explain(title: str, body: str, kind: str = "learn"):
    """
    kind: 'learn' | 'warn' | 'info' | 'success'
    Respects the experience level — hides learn boxes for Advanced users.
    """
    xp = st.session_state.get("xp", "🟢 Beginner")
    if xp == "🔴 Advanced" and kind == "learn":
        return

    class_map = {
        "learn":   ("learn-box",    "learn-title", "learn-body"),
        "warn":    ("warn-box",     "",            ""),
        "info":    ("insight",      "",            ""),
        "success": ("success-box",  "",            ""),
    }
    box_cls, title_cls, body_cls = class_map.get(kind, ("learn-box","learn-title","learn-body"))
    st.markdown(f"""
    <div class="{box_cls}">
      <div class="{title_cls}">{title}</div>
      <div class="{body_cls}">{body}</div>
    </div>""", unsafe_allow_html=True)


# ── Hero banner ──────────────────────────────────────────────

def hero():
    st.markdown("""
    <div class="hero">
      <div class="hero-title">ClusterForge Pro</div>
      <div class="hero-sub">End-to-End ML Clustering Pipeline · From Raw Data to Expert Insights</div>
    </div>""", unsafe_allow_html=True)


# ── Pipeline stepper bar ─────────────────────────────────────

def pipeline_stepper():
    current = st.session_state.get("step", 0)
    html = '<div class="pipeline-nav">'
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        active = "active" if current == i else ""
        done   = "done"   if current > i  else ""
        html += f"""
        <div class="step-btn {active} {done}">
          <span class="step-num">{i+1}</span>
          <span class="step-icon">{icon}</span>
          <span class="step-label">{label}</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # Back / Next navigation
    _, back_col, _, next_col, _ = st.columns([1, 1, 5, 1, 1])
    with back_col:
        if st.button("◀ Back") and current > 0:
            st.session_state.step -= 1
            st.rerun()
    with next_col:
        if st.button("Next ▶") and current < len(PIPELINE_STEPS) - 1:
            st.session_state.step += 1
            st.rerun()


# ── Metric strip (top of Results) ───────────────────────────

def metric_strip(metrics: dict, model_name: str):
    """Renders coloured metric tiles for all clustering scores."""
    color_map = {
        "Clusters":          "#a78bfa",
        "Silhouette ↑":      "#34d399",
        "Davies-Bouldin ↓":  "#fbbf24",
        "Calinski-Harabasz ↑": "#fb7185",
        "Noise pts":         "#6b7090",
    }
    tiles = [("Model", model_name.split("·")[-1].strip(), "#22d3ee")]
    for k, v in metrics.items():
        tiles.append((k, v, color_map.get(k, "#e2e4f0")))

    cols = st.columns(len(tiles))
    for i, (lbl, val, clr) in enumerate(tiles):
        cols[i].markdown(f"""
        <div class="metric-tile">
          <span class="val" style="color:{clr}">{val}</span>
          <span class="lbl">{lbl}</span>
        </div>""", unsafe_allow_html=True)


# ── Pipeline progress tracker (used in Learn step) ──────────

def progress_tracker():
    section("Your Pipeline Progress")
    steps_state = [
        ("📥 Load",     st.session_state.df_raw is not None),
        ("🔍 EDA",      st.session_state.get("eda_done", False)),
        ("🧹 Clean",    st.session_state.get("preprocessing_done", False)),
        ("⚙️ Features", st.session_state.get("engineering_done", False)),
        ("🤖 Cluster",  st.session_state.get("clustering_done", False)),
        ("📈 Results",  st.session_state.get("clustering_done", False)),
    ]
    cols = st.columns(len(steps_state))
    for i, (name, done) in enumerate(steps_state):
        clr = "#34d399" if done else "#2e3050"
        sym = "✓" if done else "○"
        cols[i].markdown(f"""
        <div style="text-align:center;padding:0.8rem 0.4rem;background:#111225;
        border:1px solid {'#34d399' if done else '#1e2035'};border-radius:8px;">
          <div style="font-size:1.2rem;color:{clr};margin-bottom:0.2rem">{sym}</div>
          <div style="font-family:IBM Plex Mono;font-size:0.6rem;color:{clr};
          letter-spacing:0.08em;text-transform:uppercase">{name}</div>
        </div>""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────

def sidebar() -> tuple:
    """
    Renders the sidebar and returns (uploaded_file, xp_level).
    """
    with st.sidebar:
        st.markdown("""
        <div style="font-family:IBM Plex Mono;font-size:0.6rem;letter-spacing:0.18em;
        text-transform:uppercase;color:#2e3050;padding:0.5rem 0 0.8rem;">▸ ClusterForge Pro</div>
        """, unsafe_allow_html=True)

        st.markdown("**Upload Dataset**")
        uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

        if uploaded:
            st.markdown(f"""
            <div style="background:#111225;border:1px solid #1e2035;border-radius:6px;
            padding:0.6rem 0.8rem;margin:0.5rem 0;font-family:IBM Plex Mono;
            font-size:0.7rem;color:#22d3ee;">✓ {uploaded.name}</div>""",
            unsafe_allow_html=True)

        st.divider()
        st.markdown("**Experience Level**")
        xp = st.radio("", ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"],
                      label_visibility="collapsed")

        st.divider()
        st.markdown("**Quick Jump**")
        step_labels = ["📥 Load", "🔍 EDA", "🧹 Clean", "⚙️ Features",
                       "🤖 Cluster", "📈 Results", "🎓 Learn"]
        for i, s in enumerate(step_labels):
            if st.button(s, key=f"nav_{i}", use_container_width=True):
                st.session_state.step = i
                st.rerun()

        st.divider()
        st.caption("ClusterForge Pro · ML Pipeline for Everyone")

    return uploaded, xp
