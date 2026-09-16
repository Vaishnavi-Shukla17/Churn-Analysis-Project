"""SHAP explainability.

feature_importances_ answers "which features matter overall". SHAP answers
"how did each feature move THIS prediction", which is the question a business
user actually asks when they are told to phone a customer.

SHAP is imported softly: if the dependency is missing the rest of the app must
still work, so the explainability sections degrade to a message instead of
taking the whole deployment down with an ImportError.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

try:
    import shap

    HAS_SHAP = True
except Exception:  # pragma: no cover - depends on the deployment environment
    HAS_SHAP = False


# Computing SHAP over a full test set stalls the app on every rerun, so the
# summary plot works from a capped sample.
SUMMARY_SAMPLE_SIZE = 100


@st.cache_resource(show_spinner=False)
def get_explainer(_model, cache_key):
    """Build (and cache) a TreeExplainer.

    Streamlit excludes leading-underscore arguments from the cache key, so
    `cache_key` carries the training signature. Without it the first explainer
    would be served forever, even after the model was retrained.
    """
    return shap.TreeExplainer(_model)


def shap_values_for(explainer, frame):
    """SHAP values as a plain (n_rows, n_features) array.

    Older SHAP returns one array per class for binary classifiers; newer
    versions return a single array. Normalise both to the positive class.
    """
    values = explainer.shap_values(frame)
    if isinstance(values, list):
        values = values[1]
    return np.asarray(values)


def expected_value_of(explainer):
    """The model's average output - where a waterfall plot starts."""
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        ev = np.asarray(ev).ravel()[-1]
    return float(ev)


def render_summary_plot(model, X_test, signature):
    """Beeswarm: one dot per customer per feature across a sample."""
    if not HAS_SHAP:
        st.info("Install `shap` to enable explainability plots.")
        return

    sample = X_test.iloc[: min(SUMMARY_SAMPLE_SIZE, len(X_test))]
    with st.spinner("Computing SHAP values..."):
        explainer = get_explainer(model, signature)
        values = shap_values_for(explainer, sample)

    # SHAP manages its own figure internally, so hand st.pyplot the current
    # figure rather than an empty one created beforehand.
    shap.summary_plot(values, sample, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

    st.caption(
        "One dot per customer per feature. Horizontal position is the push "
        "toward or away from churn; colour is the feature's own value."
    )


def render_waterfall(model, encoded_row, feature_columns, signature):
    """Waterfall: why this one customer got this one score."""
    if not HAS_SHAP:
        return

    explainer = get_explainer(model, signature)
    values = shap_values_for(explainer, encoded_row)

    explanation = shap.Explanation(
        values=values[0],
        base_values=expected_value_of(explainer),
        data=encoded_row.iloc[0],
        feature_names=list(feature_columns),
    )

    try:
        shap.plots.waterfall(explanation, show=False)
    except Exception:          # older SHAP exposes the top-level helper
        shap.waterfall_plot(explanation, show=False)

    st.pyplot(plt.gcf())
    plt.clf()

    st.caption(
        "Starts at the model's average output and stacks each feature's "
        "contribution until it reaches this customer's score."
    )
