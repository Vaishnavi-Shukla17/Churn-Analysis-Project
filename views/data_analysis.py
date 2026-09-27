"""Data Analysis page: upload, choose a target, clean and profile."""

import pandas as pd
import plotly.express as px
import streamlit as st

from core.ingest import (
    binarize_target,
    detect_feature_types,
    load_data,
    structural_clean,
)


def render():
    st.title("Data Analysis & Exploration")

    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Upload a CSV or Excel file to begin.")
        return

    raw = load_data(uploaded_file)

    # ---------------- target selection ----------------
    st.subheader("Select Target Column")

    default_index = 0
    for candidate in ("Churn", "churn", "Exited", "Attrition"):
        if candidate in raw.columns:
            default_index = list(raw.columns).index(candidate)
            break

    target_column = st.selectbox(
        "Choose the column you want to predict",
        list(raw.columns),
        index=default_index,
    )

    # Validate before doing anything else, with a message that says what to do
    # about it. Previously a three-class target failed deep inside .fit() and
    # surfaced as a generic error card.
    n_classes = raw[target_column].nunique(dropna=True)
    if n_classes != 2:
        st.error(
            f"**'{target_column}' has {n_classes} distinct values.** "
            "Churn prediction needs a column with exactly two - pick a yes/no "
            "or 0/1 column."
        )
        with st.expander("What is in this column?"):
            st.write(raw[target_column].value_counts().head(20))
        st.stop()

    _, mapping, _ = binarize_target(raw[target_column])
    st.caption(f"Target encoding: {mapping}")

    # ---------------- cleaning ----------------
    df, report = structural_clean(raw, target_column)

    st.subheader("Data Cleaning")
    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicate rows removed", report["duplicates_removed"])
    c2.metric("Identifier columns dropped", len(report["dropped_id_columns"]))
    c3.metric("Missing cells remaining", report["missing_cells"])

    if report["dropped_id_columns"]:
        st.caption("Dropped as identifiers: " + ", ".join(report["dropped_id_columns"]))
    st.caption(
        "Missing values are filled during training using statistics from the "
        "training split only, which keeps test data out of the model."
    )

    # ---------------- feature typing ----------------
    X = df.drop(columns=[target_column])
    numerical_cols, categorical_cols = detect_feature_types(X)

    st.session_state.df = df
    st.session_state.target_column = target_column
    st.session_state.numerical_cols = numerical_cols
    st.session_state.categorical_cols = categorical_cols

    st.subheader("Detected Feature Types")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Numerical ({len(numerical_cols)})**")
        st.write(numerical_cols or "none")
    with c2:
        st.write(f"**Categorical ({len(categorical_cols)})**")
        st.write(categorical_cols or "none")

    # ---------------- profile ----------------
    y_encoded, _, _ = binarize_target(df[target_column])

    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Rows", f"{len(df):,}")
    c2.metric("Features", len(df.columns) - 1)
    c3.metric("Missing Values", f"{int(df.isnull().sum().sum()):,}")
    c4.metric("Positive Rate", f"{float(y_encoded.mean()) * 100:.1f}%")

    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    _render_charts(df, target_column, y_encoded, numerical_cols, categorical_cols)

    st.success("Data ready. Continue to **Train & Evaluate**.")


def _render_charts(df, target_column, y_encoded, numerical_cols, categorical_cols):
    """Charts driven by the detected schema rather than hardcoded column names.

    The old version special-cased 'tenure', 'MonthlyCharges' and 'TotalCharges',
    so on any dataset without those exact names the charts silently vanished.
    """
    st.subheader("Interactive Visualizations")

    plot_df = df.copy()
    plot_df["_target"] = y_encoded.map({1: "Churned", 0: "Retained"})

    tab1, tab2 = st.tabs(["Target Distribution", "Feature Analysis"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            counts = plot_df["_target"].value_counts()
            st.plotly_chart(
                px.pie(
                    values=counts.values,
                    names=counts.index,
                    title=f"{target_column} Distribution",
                ),
                use_container_width=True,
            )
        with c2:
            if numerical_cols:
                col = st.selectbox("Numeric feature", numerical_cols, key="hist_col")
                st.plotly_chart(
                    px.histogram(
                        plot_df.dropna(subset=[col]),
                        x=col,
                        color="_target",
                        barmode="overlay",
                        title=f"{col} by {target_column}",
                    ),
                    use_container_width=True,
                )

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            if numerical_cols:
                col = st.selectbox("Distribution by class", numerical_cols, key="box_col")
                st.plotly_chart(
                    px.box(
                        plot_df.dropna(subset=[col]),
                        x="_target",
                        y=col,
                        title=f"{col} vs {target_column}",
                    ),
                    use_container_width=True,
                )
        with c2:
            if categorical_cols:
                col = st.selectbox("Categorical breakdown", categorical_cols, key="bar_col")
                rates = (
                    plot_df.groupby(col)["_target"]
                    .apply(lambda s: (s == "Churned").mean() * 100)
                    .reset_index(name="Churn %")
                    .sort_values("Churn %", ascending=False)
                )
                st.plotly_chart(
                    px.bar(rates, x=col, y="Churn %", title=f"Churn rate by {col}"),
                    use_container_width=True,
                )
