# ============================================================
# app.py  ←  RUN THIS FILE
# streamlit run app.py
# ============================================================

import streamlit as st

# ── Page config MUST be the FIRST Streamlit command ──
st.set_page_config(
    page_title="ClusterForge Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS theme ──
from config.theme import CSS
st.markdown(CSS, unsafe_allow_html=True)

# ── Safe Session State Initialization ──
from config.settings import SESSION_DEFAULTS

# Initialize all defaults safely
for key, val in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Ensure step is valid
if not 0 <= st.session_state.get("step", 0) < 7:
    st.session_state.step = 0

# ── Import UI components ──
from components.ui import hero, pipeline_stepper, sidebar

# ── Sidebar (returns uploaded file + experience level) ──
uploaded, xp = sidebar()
st.session_state["xp"] = xp

# ── Hero banner ──
hero()

# ── Pipeline stepper ──
pipeline_stepper()

# ── Import step functions ──
from pipeline.steps import (
    step_load, step_eda, step_clean, step_features,
    step_cluster, step_results, step_learn,
)

# ── Route to the correct step ──
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
    st.error("Invalid step. Resetting...")
    st.session_state.step = 0
    st.rerun()
