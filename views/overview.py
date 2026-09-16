"""Overview page: what the platform does and how to start."""

import streamlit as st

CARDS = [
    ("Purpose", "Identify at-risk customers before they churn"),
    ("Technology", "XGBoost + SHAP + Streamlit"),
    ("Impact", "Save revenue through targeted retention"),
]


def render():
    st.markdown(
        '<h1 class="main-header">Customer Churn Prediction Platform</h1>',
        unsafe_allow_html=True,
    )

    for column, (title, body) in zip(st.columns(3), CARDS):
        with column:
            st.markdown(
                f'<div class="metric-card"><h3>{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.subheader("Platform Features")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            ### Analytics & Insights
            - Works with any dataset, no hardcoded columns
            - Automatic cleaning and feature-type detection
            - Interactive data exploration
            - Real population-stability drift monitoring
            """
        )
    with c2:
        st.markdown(
            """
            ### AI-Powered Predictions
            - XGBoost with class-imbalance handling
            - SHAP explanations for every prediction
            - Single-customer and batch scoring
            - Tunable decision threshold
            """
        )

    st.markdown("---")
    st.subheader("Get Started")
    st.info(
        "Go to **Data Analysis**, upload a dataset and choose the column you "
        "want to predict. Then train the model on **Train & Evaluate**."
    )
