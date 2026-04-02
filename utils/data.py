# ============================================================
# utils/data.py
# All data loading, cleaning, preprocessing, feature engineering
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
from scipy.stats import zscore

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA


# ── Column helpers ──────────────────────────────────────────

def get_numeric_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=np.number).columns.tolist()

def get_cat_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(exclude=np.number).columns.tolist()


# ── Load ────────────────────────────────────────────────────

@st.cache_data
def load_csv(f) -> pd.DataFrame:
    return pd.read_csv(f)


# ── Clean ───────────────────────────────────────────────────

def apply_imputation(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Fill missing values.
    strategy: 'Mean' | 'Median' | 'KNN' | 'Drop Rows'
    """
    df = df.copy()
    if strategy == "Drop Rows":
        return df.dropna()

    for col in get_numeric_cols(df):
        if df[col].isnull().any():
            if strategy == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif strategy == "Median":
                df[col].fillna(df[col].median(), inplace=True)
            elif strategy == "KNN":
                vals = df[col].values.reshape(-1, 1)
                df[col] = KNNImputer(n_neighbors=5).fit_transform(vals)

    for col in get_cat_cols(df):
        df[col].fillna("Missing", inplace=True)

    return df


def apply_outlier_removal(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    method: 'None' | 'Z-Score (|z| > 3)' | 'IQR (1.5×IQR)' | 'Clip to 99th Percentile'
    """
    df = df.copy()
    num_cols = get_numeric_cols(df)
    if not num_cols:
        return df

    if method == "Z-Score (|z| > 3)":
        z = np.abs(zscore(df[num_cols]))
        mask = (z < 3).all(axis=1)
        df = df[mask]

    elif method == "IQR (1.5×IQR)":
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    elif method == "Clip to 99th Percentile":
        for col in num_cols:
            lo = df[col].quantile(0.01)
            hi = df[col].quantile(0.99)
            df[col] = df[col].clip(lo, hi)

    return df


# ── Preprocess (scale + encode) ─────────────────────────────

def preprocess_X(df: pd.DataFrame, scaler_name: str, imputer_name: str) -> np.ndarray:
    """
    Returns a 2-D numpy array ready for clustering.
    Applies imputation → scaling on numeric cols, one-hot on categoricals.
    """
    num = get_numeric_cols(df)
    cat = get_cat_cols(df)

    scaler_map = {
        "StandardScaler": StandardScaler(),
        "MinMaxScaler":   MinMaxScaler(),
        "RobustScaler":   RobustScaler(),
    }

    imp_map = {
        "Mean":   SimpleImputer(strategy="mean"),
        "Median": SimpleImputer(strategy="median"),
        "KNN":    KNNImputer(n_neighbors=5),
    }

    num_pipe = Pipeline([
        ("imp",    imp_map.get(imputer_name, SimpleImputer(strategy="mean"))),
        ("scaler", scaler_map.get(scaler_name, StandardScaler())),
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
    ])

    transformers = []
    if num:
        transformers.append(("num", num_pipe, num))
    if cat:
        transformers.append(("cat", cat_pipe, cat))

    if not transformers:
        return np.zeros((len(df), 1))

    ct = ColumnTransformer(transformers)
    return ct.fit_transform(df)


# ── Dimensionality reduction (for visualisation only) ───────

def reduce_2d(X: np.ndarray, method: str = "PCA") -> np.ndarray:
    """Reduce to 2 components for scatter plot visualisation."""
    if X.shape[1] == 1:
        X = np.hstack([X, np.zeros((X.shape[0], 1))])

    if method == "t-SNE":
        from sklearn.manifold import TSNE
        perp = min(30, max(5, X.shape[0] // 3))
        return TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X)

    return PCA(n_components=2, random_state=42).fit_transform(X)


# ── Feature engineering helpers ─────────────────────────────

def get_low_variance_features(df: pd.DataFrame, threshold: float = 0.01) -> list:
    num_cols = get_numeric_cols(df)
    if not num_cols:
        return []
    vt = VarianceThreshold(threshold=threshold)
    try:
        vt.fit(df[num_cols].fillna(0).values)
        variances = dict(zip(num_cols, vt.variances_))
        return [c for c, v in variances.items() if v < threshold]
    except Exception:
        return []


def get_pca_explained(df: pd.DataFrame):
    """Returns (individual%, cumulative%) arrays for PCA variance chart."""
    num_cols = get_numeric_cols(df)
    if len(num_cols) < 2:
        return None, None
    arr = StandardScaler().fit_transform(df[num_cols].fillna(0))
    pca = PCA().fit(arr)
    individual  = pca.explained_variance_ratio_ * 100
    cumulative  = np.cumsum(individual)
    return individual, cumulative


def get_high_corr_pairs(df: pd.DataFrame, threshold: float = 0.8):
    """Returns list of (colA, colB, r) tuples where |r| > threshold."""
    num_cols = get_numeric_cols(df)
    if len(num_cols) < 2:
        return []
    corr = df[num_cols].corr()
    pairs = []
    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            v = corr.iloc[i, j]
            if abs(v) > threshold:
                pairs.append((corr.columns[i], corr.columns[j], round(v, 3)))
    return pairs


def get_auto_remove_cols(df: pd.DataFrame) -> list:
    """Suggest columns that are unique-per-row (IDs) or fully constant."""
    return [
        c for c in df.columns
        if df[c].nunique() == len(df) or df[c].nunique() <= 1
    ]
