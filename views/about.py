"""About page: how the system works, and what it does not do."""

import streamlit as st


def render():
    st.title("About This Project")
    st.markdown(
        """
## Overview
An AI-powered churn prediction platform that works with any customer dataset.
Upload a file, pick the column you want to predict, and the app handles
cleaning, encoding, training, evaluation and explanation.

## How it works
1. **Data Analysis** - upload, choose the target, structural cleaning and profiling.
2. **Train & Evaluate** - encode, split, fit XGBoost, review metrics and SHAP.
3. **Score Customers** - score one customer from a generated form, or a whole file.
4. **Model Performance** - threshold trade-off and history across training runs.
5. **Monitoring** - drift detection using the Population Stability Index.

## Project structure
| Module | Responsibility |
|---|---|
| `core/ingest.py` | Loading, target encoding, structural cleaning |
| `core/features.py` | Encoding, leak-free split, column alignment |
| `core/model.py` | Training, evaluation, threshold selection |
| `core/explain.py` | SHAP explainability |
| `core/drift.py` | Population Stability Index |
| `core/persistence.py` | Saving and reloading artifacts |
| `views/` | One renderer per page |
| `final.py` | Page config and routing only |

## Methodology notes
- Missing values are imputed using **training-split statistics only**, so no
  test information leaks into the model.
- Class imbalance is handled with `scale_pos_weight`. This lowers headline
  accuracy and raises recall, which is the right trade for churn: a missed
  churner is a lost customer, a false alarm costs one voucher.
- Single-row and batch inputs are realigned to the exact training column set
  and order before scoring. Skipping this returns confidently wrong numbers
  with no error raised.
- The decision threshold is adjustable, and the best-F1 threshold is
  suggested rather than assuming 0.5 is correct.
- Drift is measured against a stored baseline, not simulated.

## Known limitations
- Persistence is best-effort. Hosted deployments have an ephemeral
  filesystem, so a saved model does not survive a container restart.
- Training runs are compared within a session; there is no long-term metrics
  store.
- Hyperparameters are set from the sidebar; there is no automated search.
- Drift compares distributions only. It cannot detect a change in the
  relationship between features and the target without fresh labels.
"""
    )
