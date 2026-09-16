"""Monitoring page: data health and real drift detection."""

import plotly.express as px
import streamlit as st

from core.drift import drift_scores
from core.ingest import binarize_target, load_data
from core.state import require_data

STATUS_COLOURS = {
    "Stable": "#2E7D4F",
    "Moderate shift": "#B07A0B",
    "Significant shift": "#B4321F",
}


def render():
    st.title("Model Monitoring & Health")
    df, target_column = require_data()

    # ---------------- data health ----------------
    st.subheader("Data Health")
    total_cells = len(df) * len(df.columns)
    completeness = (1 - df.isnull().sum().sum() / max(total_cells, 1)) * 100

    y_encoded, _, _ = binarize_target(df[target_column])

    c1, c2, c3 = st.columns(3)
    c1.metric("Completeness", f"{completeness:.1f}%")
    c2.metric("Rows", f"{len(df):,}")
    c3.metric("Positive cases", f"{int(y_encoded.sum()):,}")

    # ---------------- drift ----------------
    st.subheader("Data Drift Detection")

    if "baseline" not in st.session_state:
        st.info("Train the model first to capture a reference distribution.")
        return

    st.caption(
        "Population Stability Index against the distribution captured at "
        "training time. Below 0.10 is stable, 0.10-0.25 is a moderate shift, "
        "above 0.25 is significant."
    )

    compare_file = st.file_uploader(
        "Upload a newer dataset to compare against the training baseline "
        "(optional - without it the current dataset is compared to itself)",
        type=["csv", "xlsx"],
        key="drift",
    )
    current = load_data(compare_file) if compare_file is not None else df

    scores = drift_scores(st.session_state.baseline, current)
    if scores.empty:
        st.info("No comparable features found between the baseline and this dataset.")
        return

    st.plotly_chart(
        px.bar(
            scores,
            x="Feature",
            y="PSI",
            color="Status",
            color_discrete_map=STATUS_COLOURS,
            title="Feature drift (Population Stability Index)",
        ),
        use_container_width=True,
    )
    st.dataframe(scores.round(4), use_container_width=True)

    shifted = scores[scores["PSI"] >= 0.25]
    if not shifted.empty:
        st.warning(
            "Significant shift in: "
            + ", ".join(shifted["Feature"])
            + ". Consider retraining the model on more recent data."
        )
    else:
        st.success("No significant drift detected.")
