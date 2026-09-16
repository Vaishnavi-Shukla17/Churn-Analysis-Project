"""Model training, evaluation and threshold selection."""

import numpy as np
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from xgboost import XGBClassifier

from core.config import HIGH_RISK, MEDIUM_RISK, RANDOM_STATE


@st.cache_resource(show_spinner=False)
def train_model(
    X_train,
    y_train,
    balance_classes=True,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
):
    """Fit an XGBoost classifier.

    Gradient boosting builds trees sequentially: each new tree is trained on
    the residual errors of the ones before it, scaled down by learning_rate so
    no single tree dominates.

    scale_pos_weight compensates for class imbalance. On a 26/74 split the
    unweighted model leans toward predicting "no churn", which flatters
    accuracy while suppressing recall - and recall is the metric that matters,
    since a missed churner is a lost customer while a false alarm costs one
    voucher.

    Cached with cache_resource so the model is not refitted on every rerun.
    The cache key includes the arguments, so new data does trigger a retrain.
    """
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = (
        (negatives / positives) if (balance_classes and positives) else 1.0
    )

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        # note: use_label_encoder is deliberately absent - XGBoost 2.0 removed
        # it, and passing it only produces a warning about an unused parameter.
    )
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_prob, threshold=0.5):
    """All headline metrics at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc(fpr, tpr),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "fpr": fpr,
        "tpr": tpr,
        "y_pred": y_pred,
    }


def best_f1_threshold(y_true, y_prob):
    """The threshold that maximises F1, rather than assuming 0.5 is right.

    0.5 is a library default, not a business decision. For churn you usually
    want it lower: catching more leavers is worth some wasted outreach.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.nan_to_num(2 * precision * recall / (precision + recall))

    return float(thresholds[int(np.argmax(f1[:-1]))])


def threshold_sweep(y_true, y_prob, points=37):
    """Precision / recall / F1 across the whole threshold range."""
    import pandas as pd

    rows = []
    for t in np.linspace(0.05, 0.95, points):
        y_pred = (y_prob >= t).astype(int)
        rows.append(
            {
                "Threshold": t,
                "Precision": precision_score(y_true, y_pred, zero_division=0),
                "Recall": recall_score(y_true, y_pred, zero_division=0),
                "F1": f1_score(y_true, y_pred, zero_division=0),
            }
        )
    return pd.DataFrame(rows)


def risk_band(probability):
    """Map a probability onto the business risk bands."""
    if probability > HIGH_RISK:
        return "High"
    if probability > MEDIUM_RISK:
        return "Medium"
    return "Low"
