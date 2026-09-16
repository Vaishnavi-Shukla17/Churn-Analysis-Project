# AI-Powered Churn Prediction System

An interactive platform that predicts customer churn using machine learning,
built with Streamlit and XGBoost. Upload any customer dataset, choose the
column you want to predict, and the app cleans it, encodes it, trains a model
and explains every prediction.

## Overview

Acquiring a new customer costs several times more than retaining an existing
one, so identifying customers who are about to leave — while they are still
paying — is where the value is. This platform is **domain-agnostic**: nothing
is hardcoded to a particular schema, so it works with data from telecom,
banking, e-commerce or subscription services.

## Features

**Data analysis**
- Upload CSV or Excel datasets
- Automatic identifier removal, de-duplication and type recovery
- Feature types detected from the data, not hardcoded
- Charts that adapt to whatever columns you uploaded

**Prediction**
- XGBoost with class-imbalance handling
- Single-customer scoring through a form generated from your schema
- Batch scoring: upload a file, download ranked results
- Adjustable decision threshold, with the best-F1 value suggested

**Explainability**
- SHAP beeswarm across the test set
- Per-prediction waterfall answering "why this customer?"
- Feature importance by gain

**Monitoring**
- Real metrics computed from the held-out test split
- Threshold trade-off curve
- Drift detection via the Population Stability Index

## Project structure

```
final.py                  Entry point: page config, styling, routing only
core/
├── config.py             Risk bands, PSI thresholds, paths, page list
├── ingest.py             Loading, target encoding, structural cleaning
├── features.py           Encoding, leak-free split, column alignment
├── model.py              Training, evaluation, threshold selection
├── explain.py            SHAP explainability
├── drift.py              Population Stability Index
├── persistence.py        Saving and reloading artifacts
└── state.py              Session guards and run history
views/                    One renderer per page
tests/                    Unit tests for the feature contracts
```

`views/` is deliberately not named `pages/` — Streamlit auto-discovers that
directory name and would replace the app's own navigation.

## Getting started

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
git clone <your-repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

### Run

```bash
streamlit run final.py
```

The app opens at `http://localhost:8501`.

### Tests

```bash
python -m pytest tests -q
```

## Using the app

The pages have dependencies, so work through them in order:

1. **Data Analysis** — upload a dataset and pick the target column. Use the
   bundled `cleaned.csv` with `Churn` as the target to try it out.
2. **Train & Evaluate** — fit the model and review metrics, ROC, importance
   and SHAP.
3. **Score Customers** — score one customer, or upload a file for batch
   scoring.
4. **Model Performance** / **Monitoring** — threshold trade-off, run history
   and drift.

## Methodology notes

- Missing values are imputed using **training-split statistics only**. Fitting
  them over the whole dataset first leaks test information into the model.
- Class imbalance is handled with `scale_pos_weight`. This lowers headline
  accuracy and raises recall, which is the right trade for churn: a missed
  churner is a lost customer, a false alarm costs one voucher.
- Inputs are realigned to the exact training column set **and order** before
  scoring. Skipping this returns confidently wrong numbers with no error.
- Drift is measured against a stored baseline, not simulated.

On the bundled dataset (7,043 rows, 26.5% churn): accuracy 75.2%,
precision 52.3%, recall 74.6%, F1 61.5%, AUC 0.828.

## Known limitations

- Model persistence is best-effort — hosted deployments have an ephemeral
  filesystem, so a saved model does not survive a container restart.
- Run history is per-session; there is no long-term metrics store.
- Hyperparameters are set from the sidebar; there is no automated search.
- Drift compares distributions only. Detecting a change in the relationship
  between features and the target needs fresh labels.

## Branches

| Branch | Purpose |
|---|---|
| `main` | Original upstream history. Left untouched. |
| `develop` | Integration branch — **this is what is deployed** |
| `feat/data-pipeline` | Ingestion, target encoding, cleaning |
| `feat/feature-engineering` | Encoding, split, alignment, tests |
| `feat/model-training` | Training, evaluation, performance dashboard |
| `feat/explainability-scoring` | SHAP, single and batch scoring |
| `feat/monitoring-deployment` | Drift, persistence, deployment config |

## Technologies

Streamlit · XGBoost · SHAP · scikit-learn · pandas · NumPy · Plotly ·
Matplotlib · Seaborn
