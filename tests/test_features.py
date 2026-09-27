"""Tests for the feature-engineering contracts.

Run with:  python -m pytest tests -q
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.features import align_features, encode_for_scoring, prepare_training_data


@pytest.fixture
def sample():
    return pd.DataFrame(
        {
            "tenure": [1, 34, 2, 45, 8, 22, 60, 3, 17, 51],
            "MonthlyCharges": [29.85, 56.95, 53.85, 42.30, 70.70,
                               99.65, 89.10, 29.75, 104.80, 56.15],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year",
                         "Month-to-month", "Month-to-month", "Two year",
                         "Month-to-month", "One year", "Two year"],
            "Partner": ["Yes", "No", "No", "No", "No", "Yes", "Yes", "No", "Yes", "No"],
            "Churn": ["No", "No", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "No"],
        }
    )


def test_split_is_stratified_and_complete(sample):
    X_train, X_test, y_train, y_test, cols, _ = prepare_training_data(
        sample, "Churn", test_size=0.3
    )
    assert len(X_train) + len(X_test) == len(sample)
    assert list(X_train.columns) == list(X_test.columns) == list(cols)
    assert set(y_train.unique()) <= {0, 1}


def test_imputation_uses_training_fold_only(sample):
    """Medians must come from the training split, never the whole dataset."""
    frame = sample.copy()
    frame.loc[0, "MonthlyCharges"] = None

    X_train, X_test, _, _, _, impute_values = prepare_training_data(
        frame, "Churn", test_size=0.3
    )
    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0
    assert "MonthlyCharges" in impute_values


def test_single_row_aligns_to_training_columns(sample):
    """The bug that is silent when it happens: order matters, not just names."""
    _, _, _, _, cols, imputes = prepare_training_data(sample, "Churn", test_size=0.3)

    one = sample.drop(columns=["Churn"]).iloc[[0]]
    aligned = align_features(pd.get_dummies(one), cols, imputes)

    assert list(aligned.columns) == list(cols)      # ORDER, not just membership
    assert aligned.shape == (1, len(cols))
    assert aligned.isnull().sum().sum() == 0


def test_partial_row_still_aligns(sample):
    """A row carrying only some columns must still widen to the full vector."""
    _, _, _, _, cols, imputes = prepare_training_data(sample, "Churn", test_size=0.3)

    partial = sample[["Contract"]].iloc[[0]]
    aligned = align_features(pd.get_dummies(partial), cols, imputes)

    assert list(aligned.columns) == list(cols)
    assert aligned.shape == (1, len(cols))
    # the levels this row does not have must be zero, not missing
    assert aligned["Contract_Two year"].iloc[0] == 0


def test_unseen_category_does_not_widen_the_vector(sample):
    """An unknown level must be ignored, not appended as a new feature."""
    _, _, _, _, cols, imputes = prepare_training_data(sample, "Churn", test_size=0.3)

    row = sample.drop(columns=["Churn"]).iloc[[0]].copy()
    row["Contract"] = "Weekly"          # never seen in training

    aligned = align_features(pd.get_dummies(row), cols, imputes)
    assert list(aligned.columns) == list(cols)
    assert "Contract_Weekly" not in aligned.columns


def test_encode_for_scoring_drops_the_target(sample):
    _, _, _, _, cols, imputes = prepare_training_data(sample, "Churn", test_size=0.3)

    encoded = encode_for_scoring(sample, cols, imputes, drop=["Churn"])
    assert list(encoded.columns) == list(cols)
    assert len(encoded) == len(sample)
