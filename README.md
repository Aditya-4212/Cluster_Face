# ClusterForge Pro — Modular ML Clustering Platform

Run with:
```bash
pip install streamlit scikit-learn pandas plotly scipy matplotlib
streamlit run app.py
```

---

## 📁 Project Structure

```
clusterforge/
│
├── app.py                  ← Entry point. Run this. Contains NO logic.
│
├── config/
│   ├── settings.py         ← Colors, pipeline steps, algorithm info, session defaults
│   └── theme.py            ← All CSS (dark theme, fonts, cards, buttons, tabs)
│
├── components/
│   └── ui.py               ← Reusable UI: hero, sidebar, stepper, metric strip, explain boxes
│
├── utils/
│   ├── data.py             ← Load, clean, impute, scale, encode, feature engineering
│   ├── metrics.py          ← Silhouette, Davies-Bouldin, Calinski-Harabasz
│   └── charts.py           ← Every Plotly/Matplotlib figure builder
│
└── pipeline/
    └── steps.py            ← One function per step (step_load → step_learn)
```

---

## 🔧 What to Edit for Each Change

| I want to change…                        | Edit this file               |
|------------------------------------------|------------------------------|
| Colors / accent palette                  | `config/settings.py`         |
| Fonts, dark theme, card styles           | `config/theme.py`            |
| Chart colors, sizes, layouts             | `utils/charts.py`            |
| How data is cleaned / scaled             | `utils/data.py`              |
| Which metrics are computed               | `utils/metrics.py`           |
| Sidebar layout / quick-jump buttons      | `components/ui.py`           |
| The hero title / subtitle                | `components/ui.py → hero()`  |
| Explain / callout box styling            | `components/ui.py → explain()`|
| Step 1 (Load) content                    | `pipeline/steps.py → step_load()` |
| Step 2 (EDA) content                     | `pipeline/steps.py → step_eda()`  |
| Step 3 (Clean) options                   | `pipeline/steps.py → step_clean()`|
| Step 4 (Features) options                | `pipeline/steps.py → step_features()` |
| Step 5 (Cluster) algorithms/params       | `pipeline/steps.py → step_cluster()` |
| Step 6 (Results) tabs/charts             | `pipeline/steps.py → step_results()` |
| Step 7 (Learn) curriculum content        | `pipeline/steps.py → step_learn()`   |
| Add a new algorithm                      | `config/settings.py → ALGO_INFO` + `pipeline/steps.py → _build_model()` |
| Add a new chart                          | `utils/charts.py` → new function, then call from `pipeline/steps.py` |
| Change AutoML configs                    | `pipeline/steps.py → _run_automl()` |
| Change page title / icon                 | `app.py → st.set_page_config()` |

---

## ➕ Adding a New Algorithm (example)

**1. Add metadata** in `config/settings.py → ALGO_INFO`:
```python
"MyAlgo": {
    "icon": "🔮",
    "level": "Advanced",
    "desc": "Description here.",
    "best": "When to use.",
    "worst": "When to avoid.",
}
```

**2. Add parameters UI** in `pipeline/steps.py → _algo_params_ui()`:
```python
elif algo == "MyAlgo":
    params["param1"] = st.slider("Param 1", ...)
```

**3. Instantiate the model** in `pipeline/steps.py → _build_model()`:
```python
"MyAlgo": MyAlgoClass(**params),
```

---

## 🎨 Changing the Color Palette

Open `config/settings.py`, edit `COLORS` list and `CYAN`, `VIOLET`, etc.
Open `config/theme.py`, update the `:root {}` CSS variables.

---

## 📊 Adding a New Chart

1. Add a function in `utils/charts.py`:
```python
def my_new_chart(df):
    fig = px.scatter(...)
    fig.update_layout(**PLOTLY_THEME, ...)
    return fig
```

2. Import it in `pipeline/steps.py`:
```python
from utils.charts import my_new_chart
```

3. Call it inside the relevant step function.
