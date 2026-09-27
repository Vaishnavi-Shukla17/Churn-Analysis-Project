"""Data drift detection using the Population Stability Index.

The previous implementation was `np.random.random(len(features))` - numbers
that changed on every interaction and meant nothing. PSI is the standard
measure: it compares a reference distribution captured at training time
against the distribution seen now.

    PSI < 0.10          stable
    0.10 <= PSI < 0.25  moderate shift
    PSI >= 0.25         significant shift, consider retraining
"""

import numpy as np
import pandas as pd

from core.config import PSI_SIGNIFICANT, PSI_STABLE


def build_baseline(df, columns, bins=10):
    """Snapshot the reference distribution of each feature at training time.

    Stores bin edges and proportions rather than the raw data, so the baseline
    stays small enough to serialise alongside the model.
    """
    baseline = {}

    for col in columns:
        if col not in df.columns:
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        if pd.api.types.is_numeric_dtype(series):
            edges = np.histogram_bin_edges(series, bins=bins)
            counts, _ = np.histogram(series, edges)
            baseline[col] = {
                "kind": "numeric",
                "edges": edges.tolist(),
                "props": (counts / max(counts.sum(), 1)).tolist(),
            }
        else:
            props = series.value_counts(normalize=True)
            baseline[col] = {
                "kind": "categorical",
                "props": {str(k): float(v) for k, v in props.items()},
            }

    return baseline


def population_stability_index(reference_props, current_props, eps=1e-6):
    """PSI between two discrete distributions.

    eps keeps empty bins from producing a division by zero or log(0).
    """
    ref = np.asarray(reference_props, dtype=float) + eps
    cur = np.asarray(current_props, dtype=float) + eps
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def classify(psi):
    if psi < PSI_STABLE:
        return "Stable"
    if psi < PSI_SIGNIFICANT:
        return "Moderate shift"
    return "Significant shift"


def drift_scores(baseline, current_df):
    """PSI per feature against the stored training baseline."""
    rows = []

    for col, reference in baseline.items():
        if col not in current_df.columns:
            continue

        series = current_df[col].dropna()
        if series.empty:
            continue

        if reference["kind"] == "numeric" and pd.api.types.is_numeric_dtype(series):
            # Re-bin the current data using the BASELINE edges, otherwise the
            # two histograms would not be comparable.
            counts, _ = np.histogram(series, np.asarray(reference["edges"]))
            current_props = counts / max(counts.sum(), 1)
            psi = population_stability_index(reference["props"], current_props)

        elif reference["kind"] == "categorical":
            current = series.astype(str).value_counts(normalize=True)
            categories = sorted(set(reference["props"]) | set(current.index))
            reference_props = [reference["props"].get(c, 0.0) for c in categories]
            current_props = [float(current.get(c, 0.0)) for c in categories]
            psi = population_stability_index(reference_props, current_props)

        else:
            continue

        rows.append({"Feature": col, "PSI": psi, "Status": classify(psi)})

    return pd.DataFrame(rows).sort_values("PSI", ascending=False)
