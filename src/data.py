"""Loading Geometry of Truth statement datasets and saved activations."""

import numpy as np
import pandas as pd

DEFAULT_DATASETS_DIR = "geometry-of-truth/datasets"
DEFAULT_ACTIVATIONS_DIR = "data/activations"


def load_statements(name: str, datasets_dir: str = DEFAULT_DATASETS_DIR) -> pd.DataFrame:
    """Load a Geometry of Truth CSV. Columns are `statement` and `label` (1 = true)."""
    return pd.read_csv(f"{datasets_dir}/{name}.csv")


def group_key(name: str, datasets_dir: str = DEFAULT_DATASETS_DIR) -> np.ndarray:
    """Group identifier for group-level train/test splitting, so no group's
    rows land on both sides of a split (e.g. no city's true/false statement
    pair split across train and test).

    cities/neg_cities: the city column (native to the source CSV).
    sp_en_trans: the Spanish word, extracted from the statement text (no
    native column for it).
    """
    df = load_statements(name, datasets_dir=datasets_dir)
    if name in ("cities", "neg_cities"):
        return df["city"].to_numpy()
    if name == "sp_en_trans":
        words = df["statement"].str.extract(r"Spanish word '(.+?)'")[0]
        assert words.notna().all(), f"{name}: some statements didn't match the word pattern"
        return words.to_numpy()
    raise ValueError(f"no group key defined for dataset {name!r}")


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
