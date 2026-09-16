"""
AI-Powered Churn Prediction System
==================================

Entry point for the Streamlit app.

    streamlit run final.py

This file is deliberately thin: it owns page config, styling and routing, and
nothing else. The logic lives in core/ and the page renderers in views/.
"""

import os
import sys
import warnings

import streamlit as st

warnings.filterwarnings("ignore")

# Make the project importable regardless of the working directory the host
# launches us from. Streamlit Cloud runs the script from the repo root, but a
# local `streamlit run path/to/final.py` and the AppTest harness do not
# guarantee it, and without this the core/ and views/ imports below fail.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# set_page_config must be the first Streamlit call in the script - it writes
# page-level metadata that has to reach the browser before the first render.
st.set_page_config(
    page_title="Customer Churn Prediction Platform",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "AI-Powered Churn Prediction System"},
)

from core.config import CUSTOM_CSS, PAGES  # noqa: E402
from views import about, data_analysis, monitoring, overview, performance  # noqa: E402
from views import scoring, training  # noqa: E402

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

ROUTES = {
    "Overview": overview.render,
    "Data Analysis": data_analysis.render,
    "Train & Evaluate": training.render,
    "Score Customers": scoring.render,
    "Model Performance": performance.render,
    "Monitoring": monitoring.render,
    "About": about.render,
}


def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")

    try:
        ROUTES[page]()
    except Exception as exc:
        # st.stop() and st.rerun() signal themselves by raising. Swallowing
        # them here would turn a deliberate halt into an error card, so let
        # Streamlit's control-flow exceptions through untouched.
        if type(exc).__name__ in {"StopException", "RerunException"}:
            raise
        st.error("Something went wrong while rendering this page.")
        st.exception(exc)


if __name__ == "__main__":
    main()
