"""Detection metrics for one-vs-rest scores (1 = class c)."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

# Recall levels of the SimGANs paper's Table 1 (precision is compared at these recalls).
PAPER_RECALL = {"S": 0.41, "V": 0.91, "F": 0.60}


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve (threshold-free)."""
    return float(average_precision_score(labels, scores))


def precision_at_recall(labels: np.ndarray, scores: np.ndarray, recall: float) -> float:
    """Highest precision among thresholds that reach at least ``recall``."""
    precision, achieved, _ = precision_recall_curve(labels, scores)
    return float(precision[achieved >= recall].max())


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """The score threshold with the highest F1 (chosen on validation, applied to test)."""
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return float(thresholds[np.argmax(f1[:-1])])  # the last point has no threshold


def scores_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Precision, recall and F1 when beats with score >= threshold are called class c."""
    predicted = scores >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    precision = tp / max(int(predicted.sum()), 1)
    recall = tp / max(int(labels.sum()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}
