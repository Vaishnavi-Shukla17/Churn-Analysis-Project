"""Saving and reloading trained model artifacts.

A booster without its exact feature-column order is unusable, so the model and
its schema are always written as a pair.

Best-effort by design: hosted Streamlit deployments have an ephemeral and
sometimes read-only filesystem, so every call reports success rather than
assuming it.
"""

import json
import os

import numpy as np

from core.config import ARTIFACT_DIR, MODEL_PATH, SCHEMA_PATH

try:
    import joblib

    HAS_JOBLIB = True
except Exception:  # pragma: no cover
    HAS_JOBLIB = False


_METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "auc", "threshold"}


def save_artifacts(model, feature_columns, impute_values, baseline, metrics):
    """Persist model + schema. Returns (ok, detail)."""
    if not HAS_JOBLIB:
        return False, "joblib is unavailable"

    try:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        payload = {
            "feature_columns": list(feature_columns),
            "impute_values": {k: float(v) for k, v in impute_values.items()},
            "baseline": baseline,
            "metrics": {
                k: float(v)
                for k, v in metrics.items()
                if k in _METRIC_KEYS and isinstance(v, (int, float, np.floating))
            },
        }
        with open(SCHEMA_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

        return True, MODEL_PATH

    except Exception as exc:
        return False, str(exc)


def load_artifacts():
    """Reload a previously saved model and schema. Returns (model, schema) or None."""
    if not HAS_JOBLIB:
        return None
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCHEMA_PATH)):
        return None

    try:
        model = joblib.load(MODEL_PATH)
        with open(SCHEMA_PATH, encoding="utf-8") as fh:
            schema = json.load(fh)
        return model, schema
    except Exception:
        return None
