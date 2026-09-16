"""Data ingestion, target encoding and structural cleaning.

Everything here is safe to run before the train/test split: it uses no
information from the target distribution. Imputation deliberately lives in
core/features.py, where it is fitted on the training fold only.
"""

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Read an uploaded CSV/Excel file into a DataFrame.

    Streamlit hands us an in-memory buffer (an UploadedFile, which subclasses
    io.BytesIO), so the file never touches disk.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Recover numeric columns that a CSV read left as strings. A single stray
    # non-numeric cell - the 11 blank " " entries in TotalCharges are the
    # classic example - makes pandas type the whole column as object.
    #
    # pd.to_numeric(errors="ignore") used to do this, but it was deprecated in
    # pandas 2.2 and removed in 3.0. Convert explicitly instead, and keep the
    # original column whenever converting would destroy real text.
    for col in df.columns:
        if df[col].dtype == object:
            converted = pd.to_numeric(df[col], errors="coerce")
            original_na = df[col].isna().sum()
            tolerance = original_na + 0.05 * len(df)
            if converted.notna().sum() > 0 and converted.isna().sum() <= tolerance:
                df[col] = converted

    return df


def binarize_target(series):
    """Map an arbitrary two-class column onto 0/1.

    Returns (encoded_series, human_readable_mapping, ok).
    """
    values = series.dropna()
    n_classes = values.nunique()

    if n_classes != 2:
        return None, f"'{series.name}' has {n_classes} distinct values", False

    if pd.api.types.is_numeric_dtype(series):
        distinct = sorted(values.unique())
        if set(distinct) == {0, 1}:
            return series.astype(int), "already encoded as 0/1", True
        return (
            (series == distinct[-1]).astype(int),
            f"{distinct[-1]} -> 1, {distinct[0]} -> 0",
            True,
        )

    normalised = set(str(v).strip().lower() for v in values.unique())
    if normalised == {"yes", "no"}:
        encoded = series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0})
        return encoded, "Yes -> 1, No -> 0", True

    # Fall back to treating the minority class as positive. Churn is by
    # definition the rarer event, so the rarer label is the one to predict.
    counts = series.value_counts()
    positive = counts.index[-1]
    return (series == positive).astype(int), f"{positive} (minority) -> 1", True


@st.cache_data(show_spinner=False)
def structural_clean(df, target_column):
    """Cleaning that uses no information from the target distribution.

    Dropping identifiers, de-duplicating and discarding rows with no label are
    all safe before the split. Returns (cleaned_df, report).
    """
    report = {}
    df = df.copy()

    # Identifier columns are pure noise: a unique key lets a tree memorise
    # individual rows instead of learning a pattern.
    id_like = [
        col
        for col in df.columns
        if col.lower() in {"customerid", "customer_id", "id", "index"}
        or (df[col].nunique() == len(df) and df[col].dtype == object)
    ]
    id_like = [col for col in id_like if col != target_column]
    if id_like:
        df = df.drop(columns=id_like)
    report["dropped_id_columns"] = id_like

    initial_rows = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = initial_rows - len(df)

    after_dedup = len(df)
    df = df.dropna(subset=[target_column])
    report["rows_without_target_dropped"] = after_dedup - len(df)

    report["missing_cells"] = int(df.isnull().sum().sum())
    report["rows"] = len(df)

    return df, report


def detect_feature_types(X):
    """Split feature columns into numeric and categorical by dtype."""
    numerical = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return numerical, categorical
