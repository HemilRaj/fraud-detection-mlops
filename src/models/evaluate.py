"""Model evaluation metrics tailored for imbalanced fraud classification."""

from typing import Dict
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes PR-AUC, ROC-AUC, F1, Precision, and Recall at a specific threshold.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "decision_threshold": float(threshold),
    }