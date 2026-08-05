"""One-off migration: repackage the existing per-dataset .npy + statement CSVs
into the unified cache format (src.data.save_dataset_cache/load_dataset) --
activations and metadata (statement, label, and dataset-specific extras) in
two row-aligned files per dataset, instead of a .npy pair plus a separately
loaded CSV joined by hand.

Does not recompute anything -- purely repackages activations that were
already extracted on Colab. Verifies each dataset's metadata labels match the
independently-saved labels.npy before writing the cache, and prints a final
byte-for-byte comparison against the old loader for "cities".
"""

import numpy as np
import pandas as pd

from src.data import (
    DEFAULT_ACTIVATIONS_DIR,
    load_activations,
    load_dataset,
    load_statements,
    save_dataset_cache,
)

CITIES_LIKE_DATASETS = ["cities", "neg_cities", "sp_en_trans"]

for name in CITIES_LIKE_DATASETS:
    acts = np.load(f"{DEFAULT_ACTIVATIONS_DIR}/{name}_acts.npy")
    labels = np.load(f"{DEFAULT_ACTIVATIONS_DIR}/{name}_labels.npy")
    meta = load_statements(name)
    assert len(meta) == acts.shape[0] == len(labels), f"{name}: length mismatch"
    assert (meta["label"].to_numpy() == labels).all(), f"{name}: label mismatch vs source CSV"
    save_dataset_cache(name, acts, meta)
    print(f"migrated {name}: acts {acts.shape}, columns={list(meta.columns)}")

name = "compound_cities"
acts = np.load(f"{DEFAULT_ACTIVATIONS_DIR}/{name}_acts.npy")
labels = np.load(f"{DEFAULT_ACTIVATIONS_DIR}/{name}_labels.npy")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts.shape[0] == len(labels), f"{name}: length mismatch"
assert (meta["label"].to_numpy() == labels).all(), f"{name}: label mismatch vs source CSV"
save_dataset_cache(name, acts, meta)
print(f"migrated {name}: acts {acts.shape}, columns={list(meta.columns)}")

# verification: new unified cache must reproduce exactly what the current
# loader produces for cities
old_acts, old_labels = load_activations("cities")
new_acts, new_meta = load_dataset("cities")
assert np.array_equal(old_acts, new_acts), "cities: activations differ after migration!"
assert np.array_equal(old_labels, new_meta["label"].to_numpy()), "cities: labels differ after migration!"
print(f"\nverified: cities.npz/_meta.csv byte-for-byte match load_activations('cities')")
print(f"  acts dtype/shape: old={old_acts.dtype}/{old_acts.shape}  new={new_acts.dtype}/{new_acts.shape}")
