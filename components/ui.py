# ============================================================
# components/ui.py
# CLEAN FIXED VERSION (NO INDENTATION ERROR)
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

    box_cls, title_cls, body_cls = class_map.get(
        kind, ("learn-box", "learn-title", "learn-body")
    )

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
        done   = "done" if current > i else ""
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

    tiles = [(
        "Model",
        model_name.split("·")[-1].strip() if "·" in model_name else model_name,
        "#22d3ee"
    )]

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
        ("📥 Load Data", st.session_state.get("df_raw") is not None),
        ("🔍 EDA", st.session_state.get("eda_done", False)),
        ("🧹 Clean", st.session_state.get("preprocessing_done", False)),
        ("⚙️ Features", st.session_state.get("engineering_done", False)),
        ("🤖 Cluster", st.session_state.get("clustering_done", False)),
        ("📈 Results", st.session_state.get("clustering_done", False)),
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


# ============================================================
# ✅ FIXED SIDEBAR (ONLY ONE CLEAN VERSION)
# ============================================================
def sidebar() -> tuple:
    with st.sidebar:

        st.markdown("### ▸ CLUSTERFORGE PRO")

        uploaded = st.file_uploader("Upload CSV", type=["csv"])

        xp = st.radio(
            "Experience",
            ["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced"],
            key="xp"
        )

        st.divider()
        st.markdown("### 📍 Navigate")

        def set_step(i):
            st.session_state.step = i

        for i, (icon, label) in enumerate(PIPELINE_STEPS):
            st.button(
                f"{icon} {label}",
                key=f"nav_{i}",
                on_click=set_step,
                args=(i,),
                use_container_width=True
            )

        st.divider()

        if st.button("🔄 Reset"):
            for k, v in SESSION_DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()

    return uploaded, xp
