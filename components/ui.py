# ============================================================
# app.py
# CLEAN PRODUCTION VERSION
# ============================================================

import streamlit as st
from config.settings import PIPELINE_STEPS, SESSION_DEFAULTS
from pipeline.steps import (
    step_load,
    step_eda,
    step_clean,
    step_features,
    step_cluster,
    step_results,
    step_learn,
)
from config.settings import SESSION_DEFAULTS


# ============================================================
# ✅ SESSION INITIALIZATION (VERY IMPORTANT)
# ============================================================
def init_session():
    for key, value in SESSION_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# 🚀 MAIN APP
# ============================================================
def main():

    st.set_page_config(
        page_title="ClusterForge Pro",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # ✅ Initialize session safely
    init_session()

    # ── Sidebar ────────────────────────────────────────────
    uploaded, xp = sidebar()

    # ❌ DO NOT DO THIS (causes crash)
    # st.session_state["xp"] = xp

    # ✅ SAFE WAY (read only)
    xp = st.session_state.get("xp", "🟢 Beginner")

    # ── Header ─────────────────────────────────────────────
    hero()
    pipeline_stepper()

    # ── Router ─────────────────────────────────────────────
    step = st.session_state.get("step", 0)

    if step == 0:
        step_load(uploaded)

    elif step == 1:
        step_eda()

    elif step == 2:
        step_clean()

    elif step == 3:
        step_features()

    elif step == 4:
        step_cluster()

    elif step == 5:
        step_results()

    elif step == 6:
        step_learn()

    else:
        st.error("Invalid step state")


# ============================================================
# ▶ RUN
# ============================================================
if __name__ == "__main__":
    main()
