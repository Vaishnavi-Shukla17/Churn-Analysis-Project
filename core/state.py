"""Session-state helpers and page guards.

Streamlit re-executes the whole script on every interaction, so anything that
must survive between pages lives in st.session_state rather than in locals.
"""

import pandas as pd
import streamlit as st


def require_data():
    """Stop the page cleanly if no dataset has been loaded yet."""
    if "df" not in st.session_state:
        st.warning("Upload a dataset on the **Data Analysis** page first.")
        st.stop()
    return st.session_state.df, st.session_state.target_column


def require_model():
    """Stop the page cleanly if the model has not been trained yet."""
    require_data()
    if "model" not in st.session_state:
        st.warning("Train the model on the **Train & Evaluate** page first.")
        st.stop()


def record_run(metrics, rows, target_column, signature):
    """Append a training run to the in-session history.

    The Model Performance trend chart plots these - real runs, not simulated
    numbers. The page reruns on every widget interaction, so a run is only
    appended when the training signature changes; otherwise nudging the
    threshold slider would record a new "run" on every move.
    """
    history = st.session_state.setdefault("metrics_history", [])

    if history and history[-1].get("signature") == signature:
        history[-1].update(
            threshold=metrics["threshold"],
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
        )
        return

    history.append(
        {
            "run": len(history) + 1,
            "signature": signature,
            "timestamp": pd.Timestamp.now(),
            "target": target_column,
            "rows": rows,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "threshold": metrics["threshold"],
        }
    )
