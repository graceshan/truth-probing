"""Loading Geometry of Truth statement datasets and saved activations."""

import os

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


def _cache_paths(name: str, activations_dir: str = DEFAULT_ACTIVATIONS_DIR) -> tuple[str, str]:
    return f"{activations_dir}/{name}.npz", f"{activations_dir}/{name}_meta.csv"


def save_dataset_cache(
    name: str,
    acts: np.ndarray,
    meta: pd.DataFrame,
    activations_dir: str = DEFAULT_ACTIVATIONS_DIR,
) -> None:
    """Write the unified per-dataset cache: activations as one `.npz`, statement
    text + labels + any extra metadata columns (city/country, connective/pattern,
    ...) as a row-aligned sidecar CSV. The two are always written together so
    they can't drift apart the way a separately-loaded activations array and
    metadata CSV can.
    """
    assert len(meta) == acts.shape[0], (
        f"{name}: {len(meta)} metadata rows vs {acts.shape[0]} activation rows"
    )
    os.makedirs(activations_dir, exist_ok=True)
    acts_path, meta_path = _cache_paths(name, activations_dir)
    np.savez(acts_path, acts=acts.astype(np.float32))
    meta.to_csv(meta_path, index=False)


def load_dataset(
    name: str, activations_dir: str = DEFAULT_ACTIVATIONS_DIR
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load the unified per-dataset cache: activations + a metadata DataFrame
    (statement, label, and whatever extra columns that dataset has -- city/
    country for cities/neg_cities, connective/pattern/conj_a/conj_b for
    compound_cities, just statement/label for sp_en_trans).

    Returns (acts, meta), guaranteed row-aligned since both come from the same
    cache write (see save_dataset_cache) rather than two files joined by hand.

    Never computes activations itself -- per CLAUDE.md, the model never loads
    locally. On a cache miss this raises, pointing at the Colab notebook that
    does the actual extraction.
    """
    acts_path, meta_path = _cache_paths(name, activations_dir)
    if not os.path.exists(acts_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No cached activations for '{name}' ({acts_path} / {meta_path}). "
            "Activations are only ever computed on Colab -- run "
            "notebooks/extract_colab.ipynb and download the resulting "
            f"{name}.npz / {name}_meta.csv into {activations_dir}/."
        )
    acts = np.load(acts_path)["acts"]
    meta = pd.read_csv(meta_path)
    assert len(meta) == acts.shape[0], (
        f"{name}: cache corrupt -- {len(meta)} metadata rows vs {acts.shape[0]} activation rows"
    )
    return acts, meta
