"""Difference-in-means direction on saved activations.

Per CLAUDE.md: orient directions so positive = true/honest. d = mean(true) -
mean(false) is already oriented that way as long as labels stay {0, 1} with
1 = true.
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def diff_of_means_direction(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Unit-norm direction: mean(true activations) - mean(false activations)."""
    d = X[labels == 1].mean(axis=0) - X[labels == 0].mean(axis=0)
    return d / np.linalg.norm(d)


def direction_auc(direction: np.ndarray, X: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC of projecting X onto `direction`, scored against labels (1 = true)."""
    scores = X @ direction
    return roc_auc_score(labels, scores)