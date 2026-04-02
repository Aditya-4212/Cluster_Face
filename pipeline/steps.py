# ============================================================
# pipeline/steps.py
# One function per pipeline step — called from app.py
# To edit a step: find the function, change only that block
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
#from scipy.stats import zscore

import numpy as np

def zscore(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering,
    SpectralClustering, Birch, MeanShift,
)

from config.settings    import ALGO_INFO, COLORS
from utils.data         import (
    load_csv, apply_imputation, apply_outlier_removal,
    preprocess_X, reduce_2d,
    get_numeric_cols, get_cat_cols,
    get_low_variance_features, get_pca_explained,
    get_high_corr_pairs, get_auto_remove_cols,
)
from utils.metrics      import compute_all_metrics, safe_silhouette, safe_davies_bouldin
from utils.charts       import (
    scatter_clusters, cluster_bar, cluster_pie,
    feature_histogram, feature_boxplot, eda_scatter,
    correlation_heatmap, outlier_scatter, cat_bar,
    pca_variance_chart, elbow_chart, silhouette_sweep,
    feature_importance_chart, radar_profile,
    cluster_heatmap, scatter_matrix, dendrogram_chart,
    automl_comparison_chart,
)
from components.ui      import section, explain, metric_strip, progress_tracker


# ── Shared helpers ───────────────────────────────────────────

def _need(key, msg="Complete the previous step first."):
    if st.session_state.get(key) is None:
        st.warning(msg)
        st.stop()


# ============================================================
# STEP 0 — LOAD DATA
# ============================================================
def step_load(uploaded):
    section("Step 1 · Load Your Dataset")

    explain("📥 What is this step?",
        "We start by <strong>loading your CSV file</strong>. "
        "Every row is a record (customer, product, event) and every column is a feature. "
        "The goal of clustering is to <strong>find hidden groups</strong> automatically.",
        kind="learn")

    # ── No file uploaded ──
    if not uploaded and st.session_state.df_raw is None:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;background:#111225;
        border:2px dashed #1e2035;border-radius:14px;margin:1rem 0;">
          <div style="font-size:3rem;margin-bottom:0.8rem">🧬</div>
          <div style="font-family:IBM Plex Mono;font-size:0.9rem;color:#6b7090;margin-bottom:0.3rem;">
          Upload a CSV from the sidebar to begin
          </div>
          <div style="font-size:0.75rem;color:#2e3050;">
          Supports numeric + categorical columns · Auto-detects types
          </div>
        </div>""", unsafe_allow_html=True)

        section("Or load a sample dataset")
        c1, c2 = st.columns(2)
        samples = {
            "🛒 Mall Customers": "https://raw.githubusercontent.com/dsrscientist/dataset1/master/Mall_Customers.csv",
            "🌸 Iris Flowers": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
        }
        for idx, (name, url) in enumerate(samples.items()):
            col = c1 if idx == 0 else c2
            with col:
                if st.button(f"Load {name}", use_container_width=True):
                    try:
                        import urllib.request, io as _io
                        with urllib.request.urlopen(url) as r:
                            df = pd.read_csv(_io.StringIO(r.read().decode("utf-8")))
                        st.session_state.df_raw = df
                        st.session_state.step = 1
                        st.rerun()
                    except Exception as e:
                        st.error(f"Could not load sample: {e}")
        st.stop()

    if uploaded:
        df = load_csv(uploaded)
        if df.empty:
            st.error("Empty dataset."); st.stop()
        st.session_state.df_raw = df

    df = st.session_state.df_raw

    # ── Preview ──
    section("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, height=300)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",    df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(get_numeric_cols(df)))
    c4.metric("Missing", int(df.isnull().sum().sum()))

    # ── Column info table ──
    section("Column Overview")
    info_rows = []
    for col in df.columns:
        info_rows.append({
            "Column":        col,
            "Type":          str(df[col].dtype),
            "Missing":       int(df[col].isnull().sum()),
            "Unique Values": int(df[col].nunique()),
            "Sample":        str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "—",
        })
    st.dataframe(pd.DataFrame(info_rows), use_container_width=True, hide_index=True)

    explain("💡 Column Types Matter",
        "<strong>Numeric columns</strong> (int, float) are used directly. "
        "<strong>Object/string columns</strong> are categorical and will be encoded. "
        "Columns like ID or Name should be removed in the Clean step.",
        kind="learn")

    if st.button("Proceed to EDA →", type="primary"):
        st.session_state.step = 1
        st.rerun()


# ============================================================
# STEP 1 — EDA
# ============================================================
def step_eda():
    section("Step 2 · Exploratory Data Analysis")
    _need("df_raw", "Please load a dataset first.")
    df = st.session_state.df_raw

    explain("🔍 What is EDA?",
        "<strong>EDA</strong> is about <em>understanding</em> your data before doing anything else. "
        "Professionals spend 60–80% of their time here — it's the most important step!",
        kind="learn")

    num_cols = get_numeric_cols(df)
    cat_cols = get_cat_cols(df)

    tab_dist, tab_corr, tab_out, tab_cat, tab_stats = st.tabs([
        "📊 Distributions", "🔗 Correlations", "🎯 Outliers", "🏷️ Categorical", "📋 Statistics"
    ])

    # ── Distributions ──
    with tab_dist:
        section("Feature Distributions")
        if num_cols:
            feat = st.selectbox("Select feature", num_cols, key="eda_feat")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(feature_histogram(df, feat), use_container_width=True)
            with c2:
                st.plotly_chart(feature_boxplot(df, feat), use_container_width=True)

            if len(num_cols) >= 2:
                section("Feature vs Feature Scatter")
                c1, c2, c3 = st.columns(3)
                fx = c1.selectbox("X axis", num_cols, key="sc_x")
                fy = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
                color_by = c3.selectbox("Color by", ["None"] + cat_cols, key="sc_c")
                color_col = None if color_by == "None" else color_by
                st.plotly_chart(eda_scatter(df, fx, fy, color_col), use_container_width=True)
        else:
            st.info("No numeric features found.")

    # ── Correlations ──
    with tab_corr:
        section("Correlation Matrix")
        if len(num_cols) >= 2:
            st.plotly_chart(correlation_heatmap(df[num_cols]), use_container_width=True)

            high = get_high_corr_pairs(df)
            if high:
                st.markdown(f"""<div class="warn-box">
                <strong>⚠ High Correlation:</strong> {len(high)} pair(s) with |r| > 0.8.
                Consider removing one from each pair.
                </div>""", unsafe_allow_html=True)
                st.dataframe(pd.DataFrame(high, columns=["Feature A","Feature B","r"]),
                             hide_index=True, use_container_width=True)

            explain("📐 What does correlation mean?",
                "+1 = perfectly in sync · -1 = opposite · 0 = no relationship. "
                "Features with |r| > 0.8 carry redundant information.", kind="learn")
        else:
            st.info("Need at least 2 numeric features.")

    # ── Outliers ──
    with tab_out:
        section("Outlier Detection")
        if num_cols:
            sel = st.selectbox("Feature", num_cols, key="out_feat")
            col_data = df[sel].dropna()
            z = np.abs(zscore(col_data))
            iqr_low  = col_data.quantile(0.25) - 1.5*(col_data.quantile(0.75)-col_data.quantile(0.25))
            iqr_high = col_data.quantile(0.75) + 1.5*(col_data.quantile(0.75)-col_data.quantile(0.25))

            c1, c2, c3 = st.columns(3)
            c1.metric("Z-Score Outliers (|z|>3)", int((z > 3).sum()))
            c2.metric("IQR Outliers", int(((col_data < iqr_low) | (col_data > iqr_high)).sum()))
            c3.metric("Total Rows", len(col_data))

            st.plotly_chart(outlier_scatter(col_data, z, sel), use_container_width=True)

            explain("🎯 What are outliers?",
                "<strong>Z-score</strong>: flags points >3 std from mean. "
                "<strong>IQR</strong>: flags points beyond 1.5×interquartile range. "
                "KMeans is sensitive to them; DBSCAN handles them natively.", kind="learn")

    # ── Categorical ──
    with tab_cat:
        section("Categorical Features")
        if cat_cols:
            sel_c = st.selectbox("Column", cat_cols, key="cat_sel")
            st.plotly_chart(cat_bar(df[sel_c], sel_c), use_container_width=True)
        else:
            st.info("No categorical features found.")

    # ── Statistics ──
    with tab_stats:
        section("Descriptive Statistics")
        st.dataframe(df[num_cols].describe().round(3), use_container_width=True)
        explain("📋 What do these numbers mean?",
            "<strong>mean</strong> = average · <strong>std</strong> = spread · "
            "<strong>25%/50%/75%</strong> = quartiles. "
            "Large std relative to mean = feature may need scaling.", kind="learn")

    st.session_state.eda_done = True
    if st.button("Proceed to Data Cleaning →", type="primary"):
        st.session_state.step = 2
        st.rerun()


# ============================================================
# STEP 2 — CLEAN
# ============================================================
def step_clean():
    section("Step 3 · Data Cleaning")
    _need("df_raw", "Please load a dataset first.")
    df = st.session_state.df_raw

    explain("🧹 Why clean data?",
        "Real-world data is messy. Missing values confuse algorithms. "
        "Useless columns add noise. Outliers drag clusters off-centre.",
        kind="learn")

    # ── Missing values ──
    section("Missing Value Strategy")
    miss_pct = (df.isnull().sum() / len(df) * 100).round(1)
    miss_df  = miss_pct[miss_pct > 0].reset_index()
    if miss_df.empty:
        st.markdown('<div class="success-box"><strong>✓ No missing values!</strong></div>',
                    unsafe_allow_html=True)
    else:
        miss_df.columns = ["Column", "Missing %"]
        st.dataframe(miss_df, hide_index=True, use_container_width=True)

    imputer_choice = st.selectbox(
        "Imputation Strategy", ["Mean", "Median", "KNN", "Drop Rows"],
        help="Mean/Median fill with average. KNN uses similar rows. Drop Rows removes them.")
    st.session_state["imputer"] = imputer_choice

    explain("🔢 Which imputer to choose?",
        "<strong>Mean</strong>: fast, normal data. "
        "<strong>Median</strong>: better with outliers. "
        "<strong>KNN</strong>: smartest — borrows values from similar rows. "
        "<strong>Drop Rows</strong>: safest if missing is random and data is large.",
        kind="learn")

    # ── Column selection ──
    section("Column Management")
    auto_remove = get_auto_remove_cols(df)
    if auto_remove:
        st.markdown(f"""<div class="warn-box"><strong>⚠ Suggested for removal:</strong>
        {', '.join(auto_remove)} — likely IDs or constant values.</div>""",
        unsafe_allow_html=True)

    keep_cols = st.multiselect(
        "Columns to KEEP for clustering",
        df.columns.tolist(),
        default=[c for c in df.columns if c not in auto_remove],
    )

    # ── Outlier removal ──
    section("Outlier Handling")
    outlier_method = st.selectbox(
        "Outlier Removal Method",
        ["None", "Z-Score (|z| > 3)", "IQR (1.5×IQR)", "Clip to 99th Percentile"])
    st.session_state["outlier_method"] = outlier_method

    explain("✂️ When to remove outliers?",
        "Use <strong>Z-Score / IQR</strong> for clear data errors. "
        "Use <strong>Clip</strong> to keep rows but dampen extremes. "
        "Use <strong>None</strong> when using DBSCAN.",
        kind="learn")

    # ── Scaler ──
    section("Scaling Method")
    scaler_choice = st.selectbox("Feature Scaler",
        ["StandardScaler", "MinMaxScaler", "RobustScaler"])
    st.session_state["scaler"] = scaler_choice

    explain("📏 Why scale features?",
        "Age (0–100) vs Salary (0–100,000) — salary unfairly dominates distances. "
        "<strong>StandardScaler</strong>: mean=0, std=1 (best generally). "
        "<strong>MinMaxScaler</strong>: range [0,1]. "
        "<strong>RobustScaler</strong>: uses median/IQR — best when outliers exist.",
        kind="learn")

    # ── Apply ──
    if st.button("✅ Apply Cleaning & Continue →", type="primary"):
        df_c = df[keep_cols].copy() if keep_cols else df.copy()
        df_c = apply_imputation(df_c, imputer_choice)
        df_c = apply_outlier_removal(df_c, outlier_method)
        st.session_state.df_clean = df_c
        st.session_state.preprocessing_done = True
        st.success(f"✓ Cleaned: {df_c.shape[0]} rows × {df_c.shape[1]} columns")
        st.session_state.step = 3
        st.rerun()


# ============================================================
# STEP 3 — FEATURE ENGINEERING
# ============================================================
def step_features():
    section("Step 4 · Feature Engineering & Selection")
    _need("df_clean", "Complete the cleaning step first.")
    df = st.session_state.df_clean

    explain("⚙️ What is Feature Engineering?",
        "Creating or selecting the best inputs for your model. "
        "Combining two features (Spend / Visits = Spend per Visit) can reveal more signal. "
        "Removing noisy features often <em>improves</em> clustering quality.",
        kind="learn")

    num_cols = get_numeric_cols(df)
    df_eng   = df.copy()

    # ── Ratio feature ──
    section("Create New Features")
    if len(num_cols) >= 2:
        with st.expander("➕ Add ratio feature (A ÷ B)"):
            c1, c2, c3 = st.columns(3)
            feat_a = c1.selectbox("Numerator",   num_cols, key="ra")
            feat_b = c2.selectbox("Denominator", num_cols, index=1, key="rb")
            feat_n = c3.text_input("Name", value=f"{feat_a}_per_{feat_b}")
            if st.button("Create Ratio Feature"):
                df_eng[feat_n] = df_eng[feat_a] / (df_eng[feat_b].replace(0, np.nan) + 1e-9)
                st.session_state.df_clean = df_eng
                st.success(f"✓ Created: {feat_n}")
                st.rerun()

        with st.expander("✖️ Add interaction feature (A × B)"):
            c1, c2, c3 = st.columns(3)
            feat_a2 = c1.selectbox("Feature A", num_cols, key="ia")
            feat_b2 = c2.selectbox("Feature B", num_cols, index=min(1, len(num_cols)-1), key="ib")
            feat_n2 = c3.text_input("Name", value=f"{feat_a2}_x_{feat_b2}")
            if st.button("Create Interaction Feature"):
                df_eng[feat_n2] = df_eng[feat_a2] * df_eng[feat_b2]
                st.session_state.df_clean = df_eng
                st.success(f"✓ Created: {feat_n2}")
                st.rerun()
    else:
        st.info("Need ≥ 2 numeric columns.")

    # ── Feature selection ──
    section("Feature Selection")
    num_cols2 = get_numeric_cols(df_eng)
    low_var   = get_low_variance_features(df_eng) if num_cols2 else []
    if low_var:
        st.markdown(f"""<div class="warn-box"><strong>⚠ Low Variance:</strong>
        {', '.join(low_var)} — near-constant, may add noise.</div>""",
        unsafe_allow_html=True)

    selected = st.multiselect(
        "Features to include in clustering",
        df_eng.columns.tolist(),
        default=[c for c in df_eng.columns if c not in low_var],
    )

    # ── PCA chart ──
    section("PCA Variance Explained")
    indiv, cum = get_pca_explained(df_eng)
    if indiv is not None:
        st.plotly_chart(pca_variance_chart(indiv, cum), use_container_width=True)
        explain("🔵 What is PCA?",
            "<strong>PCA</strong> compresses features into fewer dimensions while preserving variance. "
            "The chart shows how many components explain 80% of information.",
            kind="learn")

    if st.button("✅ Lock Features & Continue →", type="primary"):
        df_final = df_eng[selected] if selected else df_eng
        st.session_state.df_engineered = df_final
        st.session_state.engineering_done = True
        st.session_state.step = 4
        st.rerun()


# ============================================================
# STEP 4 — CLUSTERING
# ============================================================
def step_cluster():
    section("Step 5 · Clustering")
    df = st.session_state.df_engineered or st.session_state.df_clean
    if df is None:
        st.warning("Complete previous steps first."); st.stop()

    explain("🤖 What is Clustering?",
        "Clustering is <strong>unsupervised ML</strong> — we don't predict a label, "
        "we <em>discover</em> natural groups. "
        "The algorithm finds them automatically based on similarity.",
        kind="learn")

    mode_tab, auto_tab = st.tabs(["🎓 Manual Mode", "⚡ AutoML Mode"])

    # ── Manual ──
    with mode_tab:
        section("Choose Your Algorithm")
        algo = st.selectbox("Algorithm", list(ALGO_INFO.keys()))
        info = ALGO_INFO[algo]

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div class="card" style="height:100%">
              <div style="font-size:2.5rem;margin-bottom:0.5rem">{info['icon']}</div>
              <div style="font-family:IBM Plex Mono;font-size:0.6rem;letter-spacing:0.12em;
              text-transform:uppercase;color:#fbbf24;margin-bottom:0.4rem">{info['level']}</div>
              <div style="font-size:0.82rem;color:#6b7090;line-height:1.6">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="card">
              <div style="margin-bottom:0.6rem">
                <div style="font-family:IBM Plex Mono;font-size:0.6rem;color:#34d399;
                letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem">✓ Best when</div>
                <div style="font-size:0.82rem;color:#6b7090">{info['best']}</div>
              </div>
              <div>
                <div style="font-family:IBM Plex Mono;font-size:0.6rem;color:#fb7185;
                letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.3rem">✗ Avoid when</div>
                <div style="font-size:0.82rem;color:#6b7090">{info['worst']}</div>
              </div>
            </div>""", unsafe_allow_html=True)

        section("Hyperparameters")
        params = _algo_params_ui(algo)

        scaler_c   = st.selectbox("Scaler", ["StandardScaler","MinMaxScaler","RobustScaler"], key="c_scaler")
        reduction_c = st.selectbox("Visualisation", ["PCA","t-SNE"], key="c_red")
        st.session_state["reduction"] = reduction_c

        if st.button("▶ Train Model", type="primary", key="train_manual"):
            _train_model(df, algo, params, scaler_c)

    # ── AutoML ──
    with auto_tab:
        section("AutoML — Automated Model Selection")
        explain("⚡ What does AutoML do?",
            "Tries <strong>many algorithms and configurations automatically</strong> "
            "and picks the best Silhouette Score. Great when you don't know which algorithm to use!",
            kind="learn")

        scaler_a    = st.selectbox("Scaler", ["StandardScaler","MinMaxScaler","RobustScaler"], key="a_scaler")
        reduction_a = st.selectbox("Visualisation", ["PCA","t-SNE"], key="a_red")
        n_km        = st.slider("KMeans: max K to try", 2, 12, 8)

        if st.button("⚡ Run AutoML", type="primary", key="run_automl"):
            _run_automl(df, scaler_a, reduction_a, n_km)


def _algo_params_ui(algo: str) -> dict:
    """Render hyperparameter widgets and return params dict."""
    params = {}
    if algo == "KMeans":
        c1, c2, c3 = st.columns(3)
        params["n_clusters"] = c1.slider("Clusters (k)", 2, 15, 3)
        params["init"]       = c2.selectbox("Init", ["k-means++","random"])
        params["max_iter"]   = c3.slider("Max iterations", 100, 1000, 300, step=50)
        explain("🔧 KMeans Parameters",
            "<strong>k</strong>: number of clusters — use elbow curve to tune. "
            "<strong>k-means++</strong>: smarter init, almost always better. "
            "<strong>max_iter</strong>: 300 is usually enough.", kind="learn")

    elif algo == "DBSCAN":
        c1, c2 = st.columns(2)
        params["eps"]         = c1.slider("ε (epsilon)", 0.05, 3.0, 0.5, step=0.05)
        params["min_samples"] = c2.slider("Min Samples", 2, 30, 5)
        explain("🔧 DBSCAN Parameters",
            "<strong>ε</strong>: max distance to be a neighbour. Too small = noise. Too large = merged. "
            "<strong>min_samples</strong>: neighbours needed for a core point. Rule: ≈ 2×features.",
            kind="learn")

    elif algo == "Agglomerative":
        c1, c2 = st.columns(2)
        params["n_clusters"] = c1.slider("Clusters", 2, 15, 3)
        params["linkage"]    = c2.selectbox("Linkage", ["ward","complete","average","single"])
        explain("🔧 Agglomerative Parameters",
            "<strong>ward</strong>: minimises within-cluster variance (usually best). "
            "<strong>complete</strong>: max distance. <strong>average</strong>: mean distance.",
            kind="learn")

    elif algo == "Spectral":
        c1, c2 = st.columns(2)
        params["n_clusters"] = c1.slider("Clusters", 2, 10, 3)
        params["affinity"]   = c2.selectbox("Affinity", ["rbf","nearest_neighbors"])

    elif algo == "Birch":
        c1, c2, c3 = st.columns(3)
        params["n_clusters"]      = c1.slider("Clusters", 2, 15, 3)
        params["threshold"]       = c2.slider("Threshold", 0.1, 1.0, 0.5, step=0.05)
        params["branching_factor"] = c3.slider("Branch Factor", 10, 100, 50, step=10)

    elif algo == "MeanShift":
        bw = st.slider("Bandwidth (0 = auto)", 0.0, 5.0, 0.0, step=0.1)
        params["bandwidth"] = bw if bw > 0 else None

    return params


def _build_model(algo: str, params: dict):
    """Instantiate sklearn model from algo name + params."""
    return {
        "KMeans": KMeans(
            n_clusters=params.get("n_clusters", 3),
            init=params.get("init", "k-means++"),
            max_iter=params.get("max_iter", 300),
            random_state=42, n_init="auto"),
        "DBSCAN": DBSCAN(
            eps=params.get("eps", 0.5),
            min_samples=params.get("min_samples", 5)),
        "Agglomerative": AgglomerativeClustering(
            n_clusters=params.get("n_clusters", 3),
            linkage=params.get("linkage", "ward")),
        "Spectral": SpectralClustering(
            n_clusters=params.get("n_clusters", 3),
            affinity=params.get("affinity", "rbf"), random_state=42),
        "Birch": Birch(
            n_clusters=params.get("n_clusters", 3),
            threshold=params.get("threshold", 0.5),
            branching_factor=params.get("branching_factor", 50)),
        "MeanShift": MeanShift(bandwidth=params.get("bandwidth")),
    }[algo]


def _train_model(df, algo, params, scaler):
    try:
        with st.spinner("Training…"):
            X = preprocess_X(df, scaler, st.session_state.get("imputer", "Mean"))
            model = _build_model(algo, params)
            labels = model.fit_predict(X)
            st.session_state.X_processed   = X
            st.session_state.model         = model
            st.session_state.model_name    = algo
            st.session_state.labels        = labels
            st.session_state.metrics       = compute_all_metrics(X, labels)
            st.session_state.clustering_done = True
        n = st.session_state.metrics.get("Clusters", "?")
        st.success(f"✓ Training complete! Found {n} clusters.")
        st.session_state.step = 5
        st.rerun()
    except Exception as e:
        st.error(f"Training failed: {e}")


def _run_automl(df, scaler, reduction, n_km):
    X = preprocess_X(df, scaler, st.session_state.get("imputer", "Mean"))
    st.session_state.X_processed  = X
    st.session_state["reduction"] = reduction

    configs = []
    for k in range(2, n_km + 1):
        configs.append(("KMeans",        {"n_clusters": k, "random_state": 42, "n_init": "auto"}))
    for k in range(2, 6):
        for link in ["ward", "complete", "average"]:
            configs.append(("Agglomerative", {"n_clusters": k, "linkage": link}))
    for eps in [0.2, 0.4, 0.6, 0.8, 1.0]:
        configs.append(("DBSCAN",        {"eps": eps, "min_samples": 5}))
    for k in range(2, 5):
        configs.append(("Birch",          {"n_clusters": k, "threshold": 0.5}))

    prog   = st.progress(0)
    status = st.empty()
    results = []

    for i, (name, cfg) in enumerate(configs):
        prog.progress((i + 1) / len(configs))
        status.markdown(f'<div style="font-family:IBM Plex Mono;font-size:0.7rem;color:#6b7090;">'
                        f'▸ Testing {name} {cfg}…</div>', unsafe_allow_html=True)
        try:
            model  = _build_model(name, cfg)
            labels = model.fit_predict(X)
            arr    = np.array(labels)
            valid  = arr != -1
            n_c    = len(set(arr[valid]))
            if n_c < 2:
                continue
            sil = safe_silhouette(X, labels) or -1
            db  = safe_davies_bouldin(X, labels) or 999
            results.append({
                "Algorithm":       name,
                "Config":          str(cfg),
                "Silhouette ↑":    sil,
                "Davies-Bouldin ↓": db,
                "Clusters":        n_c,
                "_model":          model,
                "_labels":         labels,
            })
        except Exception:
            continue

    prog.empty(); status.empty()

    if not results:
        st.error("No valid configurations found. Try different cleaning settings.")
        return

    results.sort(key=lambda x: x["Silhouette ↑"], reverse=True)
    best = results[0]
    st.session_state.automl_results    = results
    st.session_state.model             = best["_model"]
    st.session_state.model_name        = f"AutoML · {best['Algorithm']}"
    st.session_state.labels            = best["_labels"]
    st.session_state.metrics           = compute_all_metrics(X, best["_labels"])
    st.session_state.clustering_done   = True

    st.markdown(f"""<div class="success-box">
    <strong>✓ AutoML Complete!</strong> Best: <strong>{best['Algorithm']}</strong>
    — Silhouette = <strong>{best['Silhouette ↑']}</strong> ({best['Clusters']} clusters)
    </div>""", unsafe_allow_html=True)

    display = [{k:v for k,v in r.items() if not k.startswith("_")} for r in results[:15]]
    st.dataframe(pd.DataFrame(display), use_container_width=True, hide_index=True)

    if len(results) >= 3:
        st.plotly_chart(automl_comparison_chart(results), use_container_width=True)

    st.session_state.step = 5
    st.rerun()


# ============================================================
# STEP 5 — RESULTS
# ============================================================
def step_results():
    section("Step 6 · Results & Visualisation")

    X      = st.session_state.X_processed
    labels = st.session_state.labels
    metrics= st.session_state.metrics

    if labels is None or X is None:
        st.warning("Run clustering first."); st.stop()

    df_src = st.session_state.df_raw or st.session_state.df_clean
    df_r   = df_src.copy()
    if len(df_r) == len(labels):
        df_r["Cluster"] = labels
    else:
        df_r = (st.session_state.df_clean or st.session_state.df_engineered).copy()
        df_r["Cluster"] = labels

    model_name = st.session_state.model_name

    # ── Score strip ──
    metric_strip(metrics, model_name)

    if metrics.get("Noise pts", 0) > 0:
        st.markdown(f"""<div class="warn-box">
        <strong>⚠ {metrics['Noise pts']} noise points</strong> (DBSCAN label -1)
        — excluded from scoring.
        </div>""", unsafe_allow_html=True)

    explain("📊 How to read the scores?",
        "<strong>Silhouette</strong> (−1→1): >0.5 = strong clusters. "
        "<strong>Davies-Bouldin</strong>: lower = better. "
        "<strong>Calinski-Harabasz</strong>: higher = better.",
        kind="learn")

    reduction = st.session_state.get("reduction", "PCA")

    tab_sc, tab_dist, tab_prof, tab_heat, tab_elbow, tab_dend, tab_exp = st.tabs([
        "🗺️ Scatter", "📊 Distribution", "🧬 Profiles",
        "🌡️ Heatmap", "📐 Elbow / Sweep", "🌳 Dendrogram", "💾 Export"
    ])

    with tab_sc:
        with st.spinner(f"Computing {reduction}…"):
            X2d = reduce_2d(X, reduction)
        num_cols = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
        hover_col = st.selectbox("Hover column", ["None"] + num_cols, key="hover")
        hover_s = df_r[hover_col] if hover_col != "None" and len(df_r) == len(X2d) else None
        st.plotly_chart(scatter_clusters(X2d, labels, f"Cluster Map · {reduction}", hover_s),
                        use_container_width=True)

    with tab_dist:
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(cluster_bar(labels),  use_container_width=True)
        with c2: st.plotly_chart(cluster_pie(labels),  use_container_width=True)

    with tab_prof:
        fig_radar = radar_profile(df_r)
        if fig_radar:
            st.plotly_chart(fig_radar, use_container_width=True)
        fig_imp = feature_importance_chart(df_r)
        if fig_imp:
            st.plotly_chart(fig_imp, use_container_width=True)
        num_cols = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
        if num_cols:
            section("Mean Values by Cluster")
            st.dataframe(df_r.groupby("Cluster")[num_cols].mean().round(3), use_container_width=True)

    with tab_heat:
        fig_heat = cluster_heatmap(df_r)
        if fig_heat:
            st.plotly_chart(fig_heat, use_container_width=True)
        fig_pair = scatter_matrix(df_r)
        if fig_pair:
            st.plotly_chart(fig_pair, use_container_width=True)

    with tab_elbow:
        max_k = st.slider("Max K", 3, 15, 10)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(elbow_chart(X, max_k),     use_container_width=True)
        with c2: st.plotly_chart(silhouette_sweep(X, max_k), use_container_width=True)
        explain("📐 How to use these?",
            "<strong>Elbow</strong>: look for the kink where improvement slows. "
            "<strong>Silhouette sweep</strong>: pick the peak.", kind="learn")

    with tab_dend:
        fig_d = dendrogram_chart(df_r)
        if fig_d:
            st.pyplot(fig_d)
            import matplotlib.pyplot as plt; plt.close()
            explain("🌳 Reading the Dendrogram",
                "Height of a merge = how different the groups are. "
                "Draw a horizontal line → count branches below it = number of clusters.",
                kind="learn")

    with tab_exp:
        section("Export Your Results")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("⬇ Clustered CSV", df_r.to_csv(index=False).encode(),
                "clustered_data.csv", "text/csv", use_container_width=True)
        with c2:
            if st.session_state.model:
                st.download_button("💾 Model (.pkl)", pickle.dumps(st.session_state.model),
                    "model.pkl", use_container_width=True)
        with c3:
            num_cols = [c for c in df_r.select_dtypes(include=np.number).columns if c != "Cluster"]
            if num_cols:
                st.download_button("📄 Stats Summary",
                    df_r.groupby("Cluster")[num_cols].describe().to_csv().encode(),
                    "cluster_summary.csv", "text/csv", use_container_width=True)
        with c4:
            st.download_button("📊 Metrics CSV",
                pd.DataFrame([metrics]).to_csv(index=False).encode(),
                "metrics.csv", "text/csv", use_container_width=True)

    if st.button("📈 View Learning Module →", type="primary"):
        st.session_state.step = 6
        st.rerun()


# ============================================================
# STEP 6 — LEARN
# ============================================================
def step_learn():
    section("Step 7 · Learning Module — Beginner to Pro")

    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(34,211,238,0.06),rgba(167,139,250,0.06));
    border:1px solid rgba(34,211,238,0.15);border-radius:12px;padding:1.5rem 2rem;margin-bottom:1.5rem;">
      <div style="font-family:Syne;font-size:1.3rem;font-weight:800;color:#e2e4f0;margin-bottom:0.4rem;">
        🎓 The ML Clustering Roadmap
      </div>
      <div style="font-size:0.85rem;color:#6b7090;line-height:1.7;">
        Everything you need to understand, apply, and explain clustering — from first model to production.
      </div>
    </div>""", unsafe_allow_html=True)

    t1, t2, t3, t4 = st.tabs(["🟢 Beginner", "🟡 Intermediate", "🔴 Advanced", "📚 Glossary"])

    with t1:
        _beginner_content()
    with t2:
        _intermediate_content()
    with t3:
        _advanced_content()
    with t4:
        _glossary()

    progress_tracker()

    if st.button("🔄 Start New Analysis", type="primary"):
        from config.settings import SESSION_DEFAULTS
        for k, v in SESSION_DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


def _beginner_content():
    topics = [
        ("What is Machine Learning?",
         "ML is teaching computers to find patterns <em>without explicit programming</em>. "
         "Instead of writing rules, you show data and let the computer learn.",
         "Like showing a child 1000 dog photos instead of listing rules — they learn to recognise dogs."),
        ("What is Clustering?",
         "<strong>Unsupervised learning</strong> — no labels. The algorithm discovers natural groups.",
         "Retailer uploads transactions. Clustering finds 'high-value', 'bargain', 'occasional' buyers automatically."),
        ("How Does KMeans Work?",
         "1. Pick K random centres. 2. Assign each point to nearest centre. "
         "3. Move centre to average of its points. 4. Repeat until stable.",
         "Like magnets attracting iron filings — each magnet moves to the centre of its cluster."),
        ("What is a Silhouette Score?",
         "+1 = clearly in right cluster · 0 = on boundary · -1 = possibly wrong cluster.",
         "Score 0.7 = great separation. Score 0.3 = fuzzy/overlapping clusters."),
    ]
    for title, body, example in topics:
        with st.expander(f"📖 {title}"):
            st.markdown(f'<div style="font-size:0.85rem;color:#6b7090;line-height:1.8;margin-bottom:1rem">{body}</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""<div style="background:rgba(34,211,238,0.05);border-left:3px solid #22d3ee;
            border-radius:4px;padding:0.8rem 1rem;font-size:0.82rem;color:#6b7090;line-height:1.7;">
            💡 <strong>Example:</strong> {example}</div>""", unsafe_allow_html=True)


def _intermediate_content():
    topics = [
        ("The Full ML Pipeline",
         ["1️⃣ Load Data", "2️⃣ EDA", "3️⃣ Clean", "4️⃣ Feature Engineering",
          "5️⃣ Preprocessing", "6️⃣ Model Selection", "7️⃣ Evaluation", "8️⃣ Interpret", "9️⃣ Deploy"]),
        ("When to Use Each Algorithm",
         ["<strong>KMeans</strong>: default. Fast. Needs K. Spherical clusters.",
          "<strong>DBSCAN</strong>: noise/outliers. Finds K. Any shape.",
          "<strong>Agglomerative</strong>: hierarchy needed. Small–medium data.",
          "<strong>Spectral</strong>: complex shapes. Expensive.",
          "<strong>Birch</strong>: very large data. Memory efficient.",
          "<strong>MeanShift</strong>: no K needed. Slow."]),
        ("The Curse of Dimensionality",
         ["As features increase, distances become meaningless.",
          "Fix 1: PCA — compress dimensions.",
          "Fix 2: Feature selection — drop redundant features.",
          "Fix 3: Use cosine similarity.",
          "Rule of thumb: for N features, aim for ≤ √N clusters."]),
        ("Choosing K",
         ["<strong>Elbow</strong>: plot inertia vs K — pick the kink.",
          "<strong>Silhouette</strong>: plot score vs K — pick the peak.",
          "<strong>Domain knowledge</strong>: business logic often dictates K.",
          "Always triangulate with multiple methods."]),
    ]
    for title, bullets in topics:
        with st.expander(f"⚙️ {title}"):
            for b in bullets:
                st.markdown(f'<div style="font-size:0.83rem;color:#6b7090;line-height:1.9;padding:0.15rem 0">• {b}</div>',
                            unsafe_allow_html=True)


def _advanced_content():
    topics = [
        ("Evaluation Beyond Silhouette",
         "Use a battery: Calinski-Harabasz, Davies-Bouldin, Dunn Index. "
         "External (with ground truth): ARI, NMI. "
         "Real insight = <em>business validation</em> by domain experts."),
        ("Preprocessing Choices",
         "StandardScaler assumes Gaussian. RobustScaler for outliers. "
         "PCA before clustering can help (removes noise) or hurt (loses cluster structure). "
         "For text: TF-IDF + cosine. For mixed: Gower distance natively handles categorical + numeric."),
        ("DBSCAN Parameter Tuning",
         "ε: use k-distance graph — plot sorted distances to kth neighbour, pick ε at the elbow. "
         "min_samples ≈ 2 × n_features. HDBSCAN removes the need for ε entirely."),
        ("Production Deployment",
         "Save preprocessor + model in an sklearn Pipeline. Version with MLflow or DVC. "
         "Monitor cluster drift over time. Mini-batch KMeans for streaming. "
         "SHAP values for cluster membership explainability."),
    ]
    for title, body in topics:
        with st.expander(f"🔬 {title}"):
            st.markdown(f'<div style="font-size:0.83rem;color:#6b7090;line-height:1.9">{body}</div>',
                        unsafe_allow_html=True)


def _glossary():
    terms = {
        "Clustering":           "Unsupervised grouping of data points by similarity.",
        "Silhouette Score":     "Measures cluster separation quality. Range: −1 to +1.",
        "Davies-Bouldin":       "Lower = better. Avg ratio of scatter to inter-cluster distance.",
        "Calinski-Harabasz":    "Higher = better. Between-cluster vs within-cluster dispersion.",
        "Inertia":              "KMeans objective: sum of squared distances to cluster centres.",
        "Elbow Method":         "Plot inertia vs K — pick K where improvement slows.",
        "PCA":                  "Dimensionality reduction finding axes of maximum variance.",
        "t-SNE":                "Non-linear reduction for visualisation. Preserves local structure.",
        "Imputation":           "Filling missing values (mean, median, KNN).",
        "StandardScaler":       "Normalises to mean=0, std=1.",
        "RobustScaler":         "Uses median and IQR — robust to outliers.",
        "One-Hot Encoding":     "Converts categories into binary columns.",
        "Outlier":              "Data point far from others. Distorts KMeans.",
        "Hyperparameter":       "Setting chosen before training — not learned from data.",
        "Unsupervised Learning":"Learning without labels — finds structure in data.",
        "Feature Engineering":  "Creating/transforming features to improve model performance.",
        "Dendrogram":           "Tree showing hierarchical cluster merges.",
        "DBSCAN":               "Density-based clustering. Groups dense regions, marks sparse as noise.",
        "Variance Threshold":   "Removes near-constant (low variance) features.",
        "AutoML":               "Automated model/hyperparameter search — picks the best config.",
    }
    c1, c2 = st.columns(2)
    for i, (term, defn) in enumerate(terms.items()):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f"""
            <div style="background:#111225;border:1px solid #1e2035;border-radius:6px;
            padding:0.7rem 0.9rem;margin-bottom:0.6rem;">
              <div style="font-family:IBM Plex Mono;font-size:0.72rem;color:#22d3ee;margin-bottom:0.2rem">{term}</div>
              <div style="font-size:0.8rem;color:#6b7090;line-height:1.5">{defn}</div>
            </div>""", unsafe_allow_html=True)
