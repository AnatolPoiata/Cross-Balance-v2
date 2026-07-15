"""Evaluation metrics for imbalanced binary classification.

Matches the metric set used in the v1 prototype's informal comparison
(AUCPR, Precision/Recall on the minority class, Macro-F1, G-mean, MCC,
balanced accuracy) so results remain comparable across project versions.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def g_mean_score(y_true, y_pred):
    """Geometric mean of sensitivity (recall) and specificity."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    positives = y_true == 1
    negatives = y_true == 0
    sensitivity = (y_pred[positives] == 1).mean() if positives.any() else 0.0
    specificity = (y_pred[negatives] == 0).mean() if negatives.any() else 0.0
    return float(np.sqrt(sensitivity * specificity))


def compute_metrics(y_true, y_pred, y_proba):
    """Compute the full metric set for one fold's predictions.

    ``y_proba`` is the predicted probability of the minority (positive)
    class, shape (n_samples,).
    """
    return {
        "aucpr": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "g_mean": g_mean_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
    }


METRIC_NAMES = ["aucpr", "precision", "recall", "macro_f1", "g_mean", "mcc", "balanced_accuracy"]
