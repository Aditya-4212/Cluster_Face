# ============================================================
# components/ui.py
# Reliable navigation using session_state + unique keys
# ============================================================

import streamlit as st
from config.settings import PIPELINE_STEPS, SESSION_DEFAULTS


def section(label: str):
    st.markdown(f'<div class="sec">◆ {label}</div>', unsafe_allow_html=True)


def explain(title: str, body: str, kind: str = "learn"):
    xp = st.session_state.get("xp", "🟢 Beginner")
    if xp == "🔴 Advanced" and kind == "learn":
        return

    class_map = {
        "learn":   ("learn-box",    "learn-title", "learn-body"),
        "warn":    ("warn-box",     "",            ""),
        "info":    ("insight",      "",            ""),
        "success": ("success-box",  "",            ""),
    }
    box_cls, title_cls, body_cls = class_map.get(kind, ("learn-box", "learn-title", "learn-body"))

    st.markdown(f"""
    <div class="{box_cls}">
      {f'<div class="{title_cls}">{title}</div>' if title_cls else ''}
      <div class="{body_cls if body_cls else ''}">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def hero():
    st.markdown("""
    <div class="hero">
      <div class="hero-title">ClusterForge Pro</div>
      <div class="hero-sub">
        End-to-End Unsupervised Clustering Pipeline<br>
        <span style="opacity:0.85;">Load • Explore • Clean • Engineer • Cluster • Interpret • Learn</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def pipeline_stepper():
    current = st.session_state.get("step", 0)
    total = len(PIPELINE_STEPS)

    st.progress((current + 1) / total)

    html = '<div class="pipeline-nav">'
    for i, (icon, label) in enumerate(PIPELINE_STEPS):
        active = "active" if current == i else ""
        done   = "done"   if current > i else ""
        html += f"""
        <div class="step-btn {active} {done}">
          <span class="step-num">{i+1}</span>
          <span class="step-icon">{icon}</span>
          <span class="step-label">{label}</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    icon, name = PIPELINE_STEPS[current]
    st.caption(f"**Current Step:** {icon} {name}")


def metric_strip(metrics: dict, model_name: str):
    if not metrics:
        return
    color_map = {
        "Clusters": "#a78bfa",
        "Silhouette ↑": "#34d399",
        "Davies-Bouldin ↓": "#fbbf24",
        "Calinski-Harabasz ↑": "#fb7185",
        "Noise pts": "#6b7090",
    }
    tiles = [("Model", model_name.split("·")[-1].strip() if "·" in model_name else model_name, "#22d3ee")]
    for k, v in metrics.items():
        tiles.append((k, v, color_map.get(k, "#e2e4f0")))

    cols = st.columns(len(tiles))
    for i, (lbl, val, clr) in enumerate(tiles):
        display_val = f"{val:.3f}" if isinstance(val, float) else str(val)
        cols[i].markdown(f"""
        <div class="metric-tile">
          <span class="val" style="color:{clr}">{display_val}</span>
          <span class="lbl">{lbl}</span>
        </div>
        """, unsafe_allow_html=True)


def progress_tracker():
    section("Pipeline Progress")
    steps_state = [
        ("📥 Load Data",     st.session_state.get("df_raw") is not None),
        ("🔍 EDA",           st.session_state.get("eda_done", False)),
        ("🧹 Clean",         st.session_state.get("preprocessing_done", False)),
        ("⚙️ Features",      st.session_state.get("engineering_done", False)),
        ("🤖 Cluster",       st.session_state.get("clustering_done", False)),
        ("📈 Results",       st.session_state.get("clustering_done", False)),
    ]
    cols = st.columns(len(steps_state))
    for i, (name, done) in enumerate(steps_state):
        clr = "#34d399" if done else "#2e3050"
        sym = "✅" if done else "○"
        cols[i].markdown(f"""
        <div style="text-align:center;padding:0.9rem 0.5rem;background:#111225;
        border:1px solid {'#34d399' if done else '#1e2035'};border-radius:8px;">
          <div style="font-size:1.4rem;color:{clr};margin-bottom:0.3rem">{sym}</div>
          <div style="font-family:IBM Plex Mono;font-size:0.62rem;color:{clr};
          letter-spacing:0.1em;text-transform:uppercase">{name}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Main Sidebar with Reliable Navigation ───────────────────
def sidebar() -> tuple:
    with st.sidebar:
        st.markdown("""
        <div style="font-family:IBM Plex Mono;font-size:0.65rem;letter-spacing:0.15em;
        text-transform:uppercase;color:#22d3ee;margin-bottom:1rem;">▸ CLUSTERFORGE PRO</div>
        """, unsafe_allow_html=True)

        st.markdown("**📤 Dataset Upload**")
        uploaded = st.file_uploader(
            "Choose CSV file",
            type=["csv"],
            label_visibility="collapsed",
            help="Supports numeric and categorical columns"
        )

        if uploaded:
            st.success(f"✓ {uploaded.name}", icon="📄")

        st.divider()

        st.markdown("**🎯 Experience Level**")
        xp = st.radio(
            "Select your level",
            options=["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"],
            label_visibility="collapsed"
        )

        st.divider()

        # Reliable Step Navigation
        st.markdown("**📍 Go to Step**")
        current_step = st.session_state.get("step", 0)

        for i, (icon, label) in enumerate(PIPELINE_STEPS):
            if st.button(
                f"{icon} {label}",
                key=f"step_nav_{i}",           # Unique key is very important
                use_container_width=True,
                type="primary" if i == current_step else "secondary"
            ):
                st.session_state.step = i
                st.rerun()

        st.divider()

        # Reset Button
        if st.button("🔄 Reset All & Start Fresh", use_container_width=True, type="secondary"):
            for k, v in SESSION_DEFAULTS.items():
                st.session_state[k] = v
            st.success("✅ Everything has been reset successfully!", icon="🔄")
            st.rerun()

        st.caption("Click any step to jump")

    return uploaded, xp
