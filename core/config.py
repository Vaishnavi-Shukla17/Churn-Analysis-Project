"""Shared constants for the churn prediction platform."""

import os

# Risk bands. These are business policy, not learned values - they should be
# re-derived from the cost of a retention offer versus customer lifetime value.
HIGH_RISK = 0.70
MEDIUM_RISK = 0.40

# Population Stability Index interpretation bands (industry standard).
PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25

# Saved model artifacts. Hosted deployments have an ephemeral filesystem, so
# persistence here is best-effort and never assumed.
ARTIFACT_DIR = "artifacts"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.joblib")
SCHEMA_PATH = os.path.join(ARTIFACT_DIR, "schema.json")

RANDOM_STATE = 42

PAGES = [
    "Overview",
    "Data Analysis",
    "Train & Evaluate",
    "Score Customers",
    "Model Performance",
    "Monitoring",
    "About",
]

CUSTOM_CSS = """
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1.25rem;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 0.5rem;
    color: white;
    text-align: center;
}
</style>
"""
