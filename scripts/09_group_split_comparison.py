"""Row-level vs group-level train/test split: does holding out entire cities
(rather than individual statements) change cities probe accuracy?

Row-level splitting (the default everywhere else in this project) can let
the same city's true and false statement land on opposite sides of the
split, so a probe could partly key on city identity rather than
truth-conditional content. Group-level splitting via
src.probes.split_indices(groups=...) rules that out by holding out whole
cities. The row-level scheme stays the default (groups=None) everywhere
else in the pipeline -- this script is the only place group-level splitting
is actually used.

Also derives the equivalent group key for sp_en_trans (the Spanish word)
and for compound_cities (a compound is held out if either constituent city
is in the cities test group) and prints a sanity check for each, since the
same identity-leakage question applies there -- but the sweep/plot/report
below is scoped to cities only.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_activations, load_statements
from src.probes import split_indices, train_all_layers

MODEL_NAME = "Qwen2.5-1.5B"
ROW_COLOR = "#2a78d6"
GROUP_COLOR = "#eb6834"


def sp_en_trans_word_groups() -> np.ndarray:
    """Group key for sp_en_trans: the Spanish word being defined."""
    df = load_statements("sp_en_trans")
    words = df["statement"].str.extract(r"Spanish word '(.+?)'")[0]
    assert words.notna().all(), "sp_en_trans: some statements didn't match the word pattern"
    return words.to_numpy()


def compound_split_from_cities(cities_test_cities: set) -> tuple[np.ndarray, np.ndarray]:
    """Train/test indices for compound_cities, propagated from a cities-level
    split: a compound is in test if either constituent city is a cities-test
    city, in train only if both are cities-train cities.
    """
    meta = pd.read_csv("data/compound_cities.csv")
    city_a = meta["conj_a"].str.extract(r"city of (.+?) is in")[0]
    city_b = meta["conj_b"].str.extract(r"city of (.+?) is in")[0]
    is_test = city_a.isin(cities_test_cities) | city_b.isin(cities_test_cities)
    test_idx = np.flatnonzero(is_test.to_numpy())
    train_idx = np.flatnonzero(~is_test.to_numpy())
    return train_idx, test_idx


# --- cities: the actual comparison ---

acts, labels = load_activations("cities")
cities_df = load_statements("cities")
city_groups = cities_df["city"].to_numpy()
assert len(city_groups) == acts.shape[0]

train_idx_row, test_idx_row = split_indices(len(labels))
train_idx_group, test_idx_group = split_indices(len(labels), groups=city_groups)
print(
    f"row-level split:   {len(train_idx_row)} train / {len(test_idx_row)} test rows\n"
    f"group-level split: {len(train_idx_group)} train / {len(test_idx_group)} test rows "
    f"({cities_df['city'].iloc[train_idx_group].nunique()} train cities / "
    f"{cities_df['city'].iloc[test_idx_group].nunique()} test cities)"
)
overlap = set(cities_df["city"].iloc[train_idx_group]) & set(cities_df["city"].iloc[test_idx_group])
assert not overlap, f"group-level split leaked cities across train/test: {overlap}"

results_row = train_all_layers(acts, labels)
results_group = train_all_layers(acts, labels, groups=city_groups)

acc_row = [acc for _, acc in results_row]
acc_group = [acc for _, acc in results_group]
layers = list(range(len(acc_row)))

diffs = [abs(a - b) for a, b in zip(acc_row, acc_group)]
max_diff = max(diffs)
max_diff_layer = int(np.argmax(diffs))

print("\nlayer  row-level  group-level  |diff|")
for L, r, g, d in zip(layers, acc_row, acc_group, diffs):
    print(f"{L:5d}  {r:9.4f}  {g:11.4f}  {d:.4f}")
print(f"\nmax |row-level - group-level| accuracy difference: {max_diff:.4f} at layer {max_diff_layer}")

results_dir = "results/cities"
os.makedirs(results_dir, exist_ok=True)
comparison_df = pd.DataFrame(
    {"layer": layers, "accuracy_row_level": acc_row, "accuracy_group_level": acc_group, "abs_diff": diffs}
)
comparison_df.to_csv(f"{results_dir}/split_comparison.csv", index=False)
print(f"saved {results_dir}/split_comparison.csv")

os.makedirs("figures/probe_accuracy", exist_ok=True)
plt.figure(figsize=(8, 5))
plt.plot(layers, acc_row, color=ROW_COLOR, marker="o", label="row-level split")
plt.plot(layers, acc_group, color=GROUP_COLOR, marker="o", label="group-level split (by city)")
plt.axhline(0.5, linestyle="--", color="gray", label="chance")
plt.xlabel("layer")
plt.ylabel("test accuracy")
plt.ylim(0, 1)
plt.title(f"cities probe accuracy: row-level vs group-level split ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
out_path = "figures/probe_accuracy/cities_split_comparison.png"
plt.savefig(out_path, dpi=150)
print(f"saved {out_path}")

# --- sp_en_trans and compound_cities: derive + sanity-check the group key,
# no comparison sweep run here ---

print("\n--- sp_en_trans word groups (derived, not swept) ---")
sp_groups = sp_en_trans_word_groups()
print(f"{len(sp_groups)} statements, {len(set(sp_groups))} unique words")
tr, te = split_indices(len(sp_groups), groups=sp_groups)
sp_df = load_statements("sp_en_trans")
overlap = set(sp_df["statement"].str.extract(r"Spanish word '(.+?)'")[0].iloc[tr]) & set(
    sp_df["statement"].str.extract(r"Spanish word '(.+?)'")[0].iloc[te]
)
assert not overlap
print(f"group-level split: {len(tr)} train / {len(te)} test rows, no word overlap")

print("\n--- compound_cities city-pair groups (derived, not swept) ---")
test_cities = set(cities_df["city"].iloc[test_idx_group])
comp_train_idx, comp_test_idx = compound_split_from_cities(test_cities)
print(
    f"{len(comp_train_idx)} train / {len(comp_test_idx)} test compounds "
    f"(propagated from the cities group-level split above)"
)
