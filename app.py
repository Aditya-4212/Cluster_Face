# ── Initialise session state (more robust) ──
from config.settings import SESSION_DEFAULTS
for key, val in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Safety: Ensure step is always valid
if not 0 <= st.session_state.step < len(PIPELINE_STEPS):
    st.session_state.step = 0

# ── Sidebar & Hero ──
uploaded, xp = sidebar()
st.session_state["xp"] = xp

hero()
pipeline_stepper()

# ── Route to step (cleaner & safer) ──
step = st.session_state.step

step_functions = {
    0: lambda: step_load(uploaded),
    1: step_eda,
    2: step_clean,
    3: step_features,
    4: step_cluster,
    5: step_results,
    6: step_learn,
}

if step in step_functions:
    step_functions[step]()
else:
    st.error("Invalid step. Resetting to Load Data.")
    st.session_state.step = 0
    st.rerun()
