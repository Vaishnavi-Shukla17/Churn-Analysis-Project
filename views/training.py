"""Train & Evaluate page: fit the model and show honest metrics."""

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
import streamlit as st

from core.drift import build_baseline
from core.explain import HAS_SHAP, render_summary_plot
from core.features import prepare_training_data
from core.model import best_f1_threshold, evaluate, train_model
from core.persistence import save_artifacts
from core.state import record_run, require_data


def render():
    st.title("Train & Evaluate")
    df, target_column = require_data()

    with st.sidebar:
        st.subheader("Model settings")
        balance = st.checkbox("Handle class imbalance", value=True)
        n_estimators = st.slider("Trees", 50, 400, 100, step=50)
        max_depth = st.slider("Max depth", 2, 10, 6)
        learning_rate = st.select_slider(
            "Learning rate", [0.01, 0.05, 0.1, 0.2, 0.3], value=0.1
        )
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, step=0.05)

    # Identifies this exact model configuration so downstream caches and the
    # run history can tell a real retrain from a mere widget interaction.
    signature = (
        f"{target_column}|{len(df)}|{test_size}|{balance}|"
        f"{n_estimators}|{max_depth}|{learning_rate}"
    )

    X_train, X_test, y_train, y_test, feature_columns, impute_values = (
        prepare_training_data(df, target_column, test_size=test_size)
    )

    with st.spinner("Training XGBoost..."):
        model = train_model(
            X_train,
            y_train,
            balance_classes=balance,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
        )

    y_prob = model.predict_proba(X_test)[:, 1]

    # ---------------- threshold ----------------
    st.subheader("Decision Threshold")
    suggested = best_f1_threshold(y_test, y_prob)
    c1, c2 = st.columns([3, 1])
    with c1:
        threshold = st.slider(
            "Classify as churn when probability exceeds", 0.05, 0.95, 0.5, 0.01
        )
    with c2:
        st.metric("Best-F1 threshold", f"{suggested:.2f}")
    st.caption(
        "0.5 is a default, not an optimum. Lowering the threshold catches more "
        "churners at the cost of more false alarms."
    )

    metrics = evaluate(y_test, y_prob, threshold)

    st.session_state.update(
        model=model,
        feature_columns=feature_columns,
        impute_values=impute_values,
        X_train=X_train,
        X_test=X_test,
        y_test=y_test,
        y_prob=y_prob,
        metrics=metrics,
        threshold=threshold,
        train_signature=signature,
        baseline=build_baseline(
            df.drop(columns=[target_column]),
            st.session_state.numerical_cols + st.session_state.categorical_cols,
        ),
    )

    # ---------------- metrics ----------------
    st.subheader("Model Performance")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall", f"{metrics['recall']:.2%}")
    c4.metric("F1 Score", f"{metrics['f1']:.2%}")
    c5.metric("AUC-ROC", f"{metrics['auc']:.3f}")

    st.caption(
        f"Trained on {len(X_train):,} rows, evaluated on {len(X_test):,} held-out "
        f"rows, {len(feature_columns)} encoded features."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            metrics["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(metrics["fpr"], metrics["tpr"], label=f"AUC = {metrics['auc']:.3f}")
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    # ---------------- importance ----------------
    st.subheader("Feature Importance")
    importance = (
        pd.DataFrame(
            {"Feature": feature_columns, "Importance": model.feature_importances_}
        )
        .sort_values("Importance", ascending=False)
        .head(15)
    )
    st.plotly_chart(
        px.bar(
            importance.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 features by gain",
        ),
        use_container_width=True,
    )

    # ---------------- explainability ----------------
    st.subheader("Model Explainability (SHAP)")
    if not HAS_SHAP:
        st.info("Install `shap` to enable explainability plots.")
    else:
        render_summary_plot(model, X_test, signature)

    record_run(metrics, len(df), target_column, signature)

    if st.button("Save model to disk"):
        ok, detail = save_artifacts(
            model, feature_columns, impute_values, st.session_state.baseline, metrics
        )
        if ok:
            st.success(f"Saved to `{detail}`")
        else:
            st.warning(f"Could not save: {detail}")
