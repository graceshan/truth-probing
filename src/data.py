"""Loading Geometry of Truth statement datasets and saved activations."""

import numpy as np
import pandas as pd

DEFAULT_DATASETS_DIR = "geometry-of-truth/datasets"
DEFAULT_ACTIVATIONS_DIR = "data/activations"


def load_statements(name: str, datasets_dir: str = DEFAULT_DATASETS_DIR) -> pd.DataFrame:
    """Load a Geometry of Truth CSV. Columns are `statement` and `label` (1 = true)."""
    return pd.read_csv(f"{datasets_dir}/{name}.csv")


def load_activations(
    name: str, activations_dir: str = DEFAULT_ACTIVATIONS_DIR
) -> tuple[np.ndarray, np.ndarray]:
    """Load saved activations + labels for a dataset.

    Returns (acts, labels) where acts has shape [n_statements, n_layers, d_model]
    (resid_post, last token) and labels is [n_statements] (1 = true).
    """
    acts = np.load(f"{activations_dir}/{name}_acts.npy")
    labels = np.load(f"{activations_dir}/{name}_labels.npy")
    return acts, labels
