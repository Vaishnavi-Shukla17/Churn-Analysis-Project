"""Model Performance page.

Every number here is computed from the held-out test split. The previous
version of this page displayed hardcoded strings ("87.3%", "84.7%", "0.91")
and an np.random.normal trend line that re-rolled on every interaction.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.model import threshold_sweep
from core.state import require_model


def render():
    st.title("Model Performance Dashboard")
    require_model()

    metrics = st.session_state.metrics
    y_test = st.session_state.y_test
    y_prob = st.session_state.y_prob

    st.caption("Every number on this page comes from the held-out test split.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall", f"{metrics['recall']:.2%}")
    c4.metric("AUC-ROC", f"{metrics['auc']:.3f}")

    # ---------------- threshold trade-off ----------------
    st.subheader("Threshold Trade-off")
    curve = threshold_sweep(y_test, y_prob)

    figure = px.line(
        curve,
        x="Threshold",
        y=["Precision", "Recall", "F1"],
        title="How the metrics move as the decision threshold changes",
    )
    figure.add_vline(
        x=metrics["threshold"],
        line_dash="dash",
        annotation_text=f"current {metrics['threshold']:.2f}",
    )
    st.plotly_chart(figure, use_container_width=True)
    st.caption(
        "Precision and recall move in opposite directions. Where you sit on "
        "this curve is a business decision, not a modelling one."
    )

    # ---------------- run history ----------------
    history = st.session_state.get("metrics_history", [])

    st.subheader("Training Run History")
    if len(history) < 2:
        st.info(
            "Only one training run so far. Retrain with different settings on "
            "**Train & Evaluate** and the comparison will appear here."
        )
    else:
        hist_df = pd.DataFrame(history)
        figure = go.Figure()
        for metric in ("accuracy", "precision", "recall", "f1", "auc"):
            figure.add_trace(
                go.Scatter(
                    x=hist_df["run"],
                    y=hist_df[metric],
                    mode="lines+markers",
                    name=metric.title(),
                )
            )
        figure.update_layout(
            title="Metrics across training runs in this session",
            xaxis_title="Run",
            yaxis_title="Score",
        )
        st.plotly_chart(figure, use_container_width=True)

    if history:
        st.dataframe(
            pd.DataFrame(history)[
                [
                    "run",
                    "timestamp",
                    "rows",
                    "threshold",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "auc",
                ]
            ].round(4),
            use_container_width=True,
        )
