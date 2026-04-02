# ============================================================
# app.py  ←  RUN THIS FILE
# streamlit run app.py
#
# This is the ONLY file that wires everything together.
# It contains NO logic — just imports and calls.
#
# ┌─────────────────────────────────────────────────────┐
# │  MODULE MAP  (what to edit for each change)         │
# ├─────────────────────────────────────────────────────┤
# │  Colors / accent palette   →  config/settings.py   │
# │  CSS / fonts / dark theme  →  config/theme.py      │
# │  Plotly chart styles       →  utils/charts.py      │
# │  Data cleaning & scaling   →  utils/data.py        │
# │  Clustering metrics        →  utils/metrics.py     │
# │  Step 1  Load Data         →  pipeline/steps.py    │
# │  Step 2  EDA               →  pipeline/steps.py    │
# │  Step 3  Clean             →  pipeline/steps.py    │
# │  Step 4  Features          →  pipeline/steps.py    │
# │  Step 5  Cluster           →  pipeline/steps.py    │
# │  Step 6  Results           →  pipeline/steps.py    │
# │  Step 7  Learn             →  pipeline/steps.py    │
# │  Sidebar / hero / stepper  →  components/ui.py     │
# │  Algorithm info / metadata →  config/settings.py   │
# └─────────────────────────────────────────────────────┘
# ============================================================

import streamlit as st

# ── Page config (must be first Streamlit call) ──
st.set_page_config(
    page_title="ClusterForge Pro",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS theme ──
from config.theme    import CSS
st.markdown(CSS, unsafe_allow_html=True)

# ── Initialise session state ──
from config.settings import SESSION_DEFAULTS
for key, val in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Sidebar (returns uploaded file + experience level) ──
from components.ui   import hero, pipeline_stepper, sidebar
uploaded, xp = sidebar()
st.session_state["xp"] = xp

# ── Hero banner ──
hero()

# ── Pipeline stepper (navigation bar + back/next buttons) ──
pipeline_stepper()

# ── Route to the correct step ──
from pipeline.steps import (
    step_load, step_eda, step_clean, step_features,
    step_cluster, step_results, step_learn,
)

step = st.session_state.get("step", 0)

if   step == 0: step_load(uploaded)
elif step == 1: step_eda()
elif step == 2: step_clean()
elif step == 3: step_features()
elif step == 4: step_cluster()
elif step == 5: step_results()
elif step == 6: step_learn()
