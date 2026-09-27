"""Feature engineering: encoding, splitting, leak-free imputation, alignment.

The two contracts this module owns:

1. Imputation statistics are computed on the TRAINING fold only.
2. Anything scored later is realigned to the exact training column set and
   order before it reaches the model.

Breaking either one fails silently - the model still returns a number, it is
just the wrong number - so both are asserted in tests/test_features.py.
"""

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from core.config import RANDOM_STATE
from core.ingest import binarize_target


@st.cache_data(show_spinner=False)
def prepare_training_data(df, target_column, test_size=0.2, random_state=RANDOM_STATE):
    """Encode, split, then impute using training-fold statistics only.

    Returns (X_train, X_test, y_train, y_test, feature_columns, impute_values).
    """
    y_encoded, _, ok = binarize_target(df[target_column])
    if not ok:
        raise ValueError(f"Target '{target_column}' is not binary")

    X = df.drop(columns=[target_column])
    y = y_encoded.astype(int)

    # One-hot encoding. Each categorical column becomes one binary column per
    # level. Categories come from the feature columns only, so this carries no
    # information about the target and is safe to do before the split.
    #
    # drop_first is deliberately not used: the redundancy costs nothing for
    # tree models, which split on individual indicators.
    X_encoded = pd.get_dummies(X)
    feature_columns = list(X_encoded.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,          # keep the class balance in both halves
    )

    # Imputation values are learned from the TRAINING fold only. Computing
    # medians over the whole dataset first - as the original did - lets test
    # data influence the model it is meant to be judging.
    impute_values = X_train.median(numeric_only=True).to_dict()
    for col, value in impute_values.items():
        if X_train[col].isna().any():
            X_train[col] = X_train[col].fillna(value)
        if X_test[col].isna().any():
            X_test[col] = X_test[col].fillna(value)

    return X_train, X_test, y_train, y_test, feature_columns, impute_values


def align_features(frame, feature_columns, impute_values=None):
    """Force an encoded frame onto the exact training column set and order.

    One-hot encoding a single row only produces the levels that row happens to
    contain: a customer on a "One year" contract yields Contract_One year and
    nothing else, while the model was trained on all three contract columns.

    Without filling the absent columns with zero AND restoring the training
    order, the model scores a differently shaped vector and returns a
    confidently wrong probability with no error raised.
    """
    aligned = frame.reindex(columns=feature_columns, fill_value=0)

    if impute_values:
        for col, value in impute_values.items():
            if col in aligned.columns and aligned[col].isna().any():
                aligned[col] = aligned[col].fillna(value)

    return aligned.fillna(0)


def encode_for_scoring(frame, feature_columns, impute_values=None, drop=()):
    """One-hot encode arbitrary input and align it for inference."""
    features = frame.drop(columns=list(drop), errors="ignore")
    return align_features(pd.get_dummies(features), feature_columns, impute_values)
