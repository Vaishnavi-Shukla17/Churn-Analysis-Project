"""Score Customers page: one customer from a generated form, or a whole file."""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from core.explain import HAS_SHAP, render_waterfall
from core.features import encode_for_scoring
from core.ingest import load_data
from core.model import risk_band
from core.state import require_model


def render():
    st.title("Score Customers")
    require_model()

    df = st.session_state.df
    target_column = st.session_state.target_column
    model = st.session_state.model
    feature_columns = st.session_state.feature_columns
    impute_values = st.session_state.impute_values
    threshold = st.session_state.get("threshold", 0.5)
    signature = st.session_state.get("train_signature", "")

    X = df.drop(columns=[target_column])

    tab_single, tab_batch = st.tabs(["Single Customer", "Batch Scoring"])

    with tab_single:
        _single(X, model, feature_columns, impute_values, threshold, signature)

    with tab_batch:
        _batch(model, feature_columns, impute_values, threshold, target_column)


def _single(X, model, feature_columns, impute_values, threshold, signature):
    """A form built at runtime from the dataset's own schema.

    Numeric columns become spin-boxes pre-filled with the median; categorical
    columns become dropdowns of that column's actual values, so it is
    impossible to enter a category the model has never seen.
    """
    st.caption("The form below is generated from your dataset's own schema.")

    inputs = {}
    with st.form("single_customer"):
        columns = st.columns(3)
        for i, col in enumerate(X.columns):
            with columns[i % 3]:
                if pd.api.types.is_numeric_dtype(X[col]):
                    inputs[col] = st.number_input(
                        col, value=float(X[col].median()), format="%.4f"
                    )
                else:
                    options = sorted(X[col].dropna().astype(str).unique().tolist())
                    inputs[col] = st.selectbox(col, options) if options else ""

        # st.form batches every edit: without it each of the fields above would
        # trigger its own full script rerun while being filled in.
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    encoded = encode_for_scoring(
        pd.DataFrame([inputs]), feature_columns, impute_values
    )
    probability = float(model.predict_proba(encoded)[0, 1])
    band = risk_band(probability)

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn Probability", f"{probability:.2%}")
    c2.metric("Risk Level", band)
    c3.metric(
        "Decision",
        "Churn" if probability >= threshold else "Retain",
        help=f"At the current threshold of {threshold:.2f}",
    )

    st.subheader("Recommended Actions")
    if band == "High":
        st.error("High risk - immediate action needed")
        st.write("- Contact the customer directly")
        st.write("- Offer a targeted discount or plan review")
        st.write("- Check recent support history")
    elif band == "Medium":
        st.warning("Medium risk - monitor and engage")
        st.write("- Add to the next retention campaign")
        st.write("- Encourage use of the features that drive stickiness")
    else:
        st.success("Low risk - standard monitoring")

    if HAS_SHAP:
        st.subheader("Why this prediction?")
        render_waterfall(model, encoded, feature_columns, signature)


def _batch(model, feature_columns, impute_values, threshold, target_column):
    """Score a whole file and hand back a ranked, downloadable CSV."""
    st.caption(
        "Upload a file with the same columns as your training data. The target "
        "column is optional and ignored if present."
    )

    batch_file = st.file_uploader(
        "Upload customers to score", type=["csv", "xlsx"], key="batch"
    )
    if batch_file is None:
        return

    batch_raw = load_data(batch_file)
    encoded = encode_for_scoring(
        batch_raw, feature_columns, impute_values, drop=[target_column]
    )

    probabilities = model.predict_proba(encoded)[:, 1]

    results = batch_raw.copy()
    results["Churn_Probability"] = probabilities.round(4)
    results["Risk_Level"] = [risk_band(p) for p in probabilities]
    results["Prediction"] = np.where(probabilities >= threshold, "Churn", "Retain")
    results = results.sort_values("Churn_Probability", ascending=False)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scored", f"{len(results):,}")
    c2.metric("High risk", int((results["Risk_Level"] == "High").sum()))
    c3.metric("Medium risk", int((results["Risk_Level"] == "Medium").sum()))
    c4.metric("Low risk", int((results["Risk_Level"] == "Low").sum()))

    st.plotly_chart(
        px.histogram(
            results, x="Churn_Probability", nbins=40, title="Risk distribution"
        ),
        use_container_width=True,
    )

    st.subheader("Highest-risk customers first")
    st.dataframe(results.head(100), use_container_width=True)

    st.download_button(
        "Download scored customers (CSV)",
        results.to_csv(index=False).encode("utf-8"),
        file_name="churn_scores.csv",
        mime="text/csv",
    )
