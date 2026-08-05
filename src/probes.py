"""Direction/probe training on saved activations.

Trains one LogisticRegression(max_iter=2000, C=0.1) probe per layer, using a
single train/test split of statement indices shared across every layer so
accuracy is comparable layer-to-layer. Labels are 1 = true, so sklearn's
binary decision function (coef_ @ x + intercept_) is already oriented with
positive = true/honest -- the SIGN CONVENTION from CLAUDE.md -- as long as
labels stay {0, 1}.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def split_indices(
    n_statements: int, test_size: float = 0.2, random_state: int = 0, groups=None
):
    """One train/test split of statement indices, meant to be reused across every layer.

    Row-level by default (plain train_test_split, the original behavior every
    script in this project uses) -- pass `groups` (array of length
    n_statements, e.g. city name) to split at the group level instead, via
    GroupShuffleSplit, so no group's rows end up split across both train and
    test (prevents entity-identity leakage: e.g. the same city's true and
    false statements landing on opposite sides of the split).
    """
    if groups is None:
        return train_test_split(
            np.arange(n_statements), test_size=test_size, random_state=random_state
        )
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    return next(gss.split(np.arange(n_statements), groups=groups))


def train_layer_probe(
    X_layer: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[LogisticRegression, float]:
    """Fit a probe on one layer's activations. Returns (probe, test_accuracy)."""
    probe = LogisticRegression(max_iter=2000, C=0.1)
    probe.fit(X_layer[train_idx], labels[train_idx])
    # labels must be {0,1} with 1=true for the sign convention to hold
    assert list(probe.classes_) == [0, 1]
    test_acc = probe.score(X_layer[test_idx], labels[test_idx])
    return probe, test_acc


def fit_probe(X: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    """Fit a probe on all given examples (no held-out split).

    For cross-dataset transfer, where the test set is a different dataset
    entirely, so there's no need to hold out part of the training set.
    """
    probe = LogisticRegression(max_iter=2000, C=0.1)
    probe.fit(X, labels)
    assert list(probe.classes_) == [0, 1]
    return probe


def normalized_coef(probe: LogisticRegression) -> np.ndarray:
    """Unit-norm probe direction, [d_model]. Sign already positive = true (see module docstring)."""
    coef = probe.coef_[0]
    return coef / np.linalg.norm(coef)


def train_all_layers(
    acts: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 0,
    groups=None,
) -> list[tuple[LogisticRegression, float]]:
    """Train one probe per layer of `acts` ([n_statements, n_layers, d_model]).

    Returns a list of (probe, test_accuracy), one entry per layer, all using
    the same train/test split. Pass `groups` for a group-level split (see
    split_indices) instead of the default row-level one.
    """
    train_idx, test_idx = split_indices(
        len(labels), test_size=test_size, random_state=random_state, groups=groups
    )
    n_layers = acts.shape[1]
    return [
        train_layer_probe(acts[:, layer, :], labels, train_idx, test_idx)
        for layer in range(n_layers)
    ]
