# ============================================================
# config/settings.py
# All constants, color palette, plotly theme, pipeline steps
# ============================================================

# ── Accent Colors (used across all charts & UI) ──
COLORS = [
    "#22d3ee",  # cyan
    "#a78bfa",  # violet
    "#34d399",  # emerald
    "#fbbf24",  # amber
    "#fb7185",  # rose
    "#38bdf8",  # sky
    "#c084fc",  # purple
    "#6ee7b7",  # teal
]

# ── Named palette references ──
CYAN    = "#22d3ee"
VIOLET  = "#a78bfa"
EMERALD = "#34d399"
AMBER   = "#fbbf24"
ROSE    = "#fb7185"

# ── Plotly dark theme defaults (spread into update_layout) ──
PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="IBM Plex Mono, monospace", color="#6b7090", size=11),
    colorway=COLORS,
)

# ── Pipeline step definitions ──
PIPELINE_STEPS = [
    ("📥", "Load Data"),
    ("🔍", "EDA"),
    ("🧹", "Clean"),
    ("⚙️", "Features"),
    ("🤖", "Cluster"),
    ("📈", "Results"),
    ("🎓", "Learn"),
]

# ── Algorithm metadata ──
ALGO_INFO = {
    "KMeans": {
        "icon": "🎯",
        "level": "Beginner",
        "desc": "Partitions data into K spherical clusters by minimising within-cluster variance. Fast and scalable.",
        "best": "Clean data, roughly equal cluster sizes, you know roughly how many clusters to expect.",
        "worst": "Outliers, non-spherical clusters, varying density.",
    },
    "DBSCAN": {
        "icon": "🌌",
        "level": "Intermediate",
        "desc": "Density-based — clusters are regions of high density separated by low density. Finds outliers naturally.",
        "best": "Arbitrary-shaped clusters, data with noise/outliers, unknown number of clusters.",
        "worst": "Varying density clusters, high-dimensional data.",
    },
    "Agglomerative": {
        "icon": "🌳",
        "level": "Intermediate",
        "desc": "Hierarchical bottom-up: starts with each point as its own cluster, merges closest pairs.",
        "best": "When you want a hierarchy/dendrogram, small-to-medium datasets.",
        "worst": "Large datasets (slow), need to specify K.",
    },
    "Spectral": {
        "icon": "🌊",
        "level": "Advanced",
        "desc": "Uses graph/eigenvalue decomposition to find clusters. Excellent for non-convex shapes.",
        "best": "Complex non-spherical clusters, image segmentation.",
        "worst": "Very large datasets (memory intensive).",
    },
    "Birch": {
        "icon": "🌿",
        "level": "Intermediate",
        "desc": "Builds a tree structure for fast incremental clustering. Memory efficient.",
        "best": "Very large datasets, streaming data.",
        "worst": "Non-spherical clusters, outlier-heavy data.",
    },
    "MeanShift": {
        "icon": "🎱",
        "level": "Intermediate",
        "desc": "Finds cluster centres by shifting towards high-density regions. Auto-finds K.",
        "best": "Unknown K, smooth density distributions.",
        "worst": "Large datasets (slow), choosing bandwidth is tricky.",
    },
}

# ── Session state defaults ──
SESSION_DEFAULTS = {
    "step": 0,
    "df_raw": None,
    "df_clean": None,
    "df_engineered": None,
    "X_processed": None,
    "labels": None,
    "model": None,
    "model_name": "",
    "metrics": {},
    "preprocessing_done": False,
    "eda_done": False,
    "engineering_done": False,
    "clustering_done": False,
    "outlier_method": "none",
    "scaler": "StandardScaler",
    "imputer": "Mean",
    "removed_cols": [],
    "automl_results": [],
    "reduction": "PCA",
    "xp": "🟢 Beginner",
}
