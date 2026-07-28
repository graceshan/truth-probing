"""Cross-dataset transfer: apply the cities-trained probe to the compound
and/or dataset, to see whether the single-statement truth direction picks
up compositional (logical and/or) truth structure at all.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_activations
from src.probes import fit_probe

MODEL_NAME = "Qwen2.5-1.5B"
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]

acts_cities, labels_cities = load_activations("cities")
acts_compound, _ = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]
meta["group"] = meta["connective"] + "-" + meta["pattern"]

n_layers = acts_cities.shape[1]
layers = range(n_layers)
scores_by_layer = np.zeros((acts_compound.shape[0], n_layers))

for L in range(n_layers):
    probe = fit_probe(acts_cities[:, L], labels_cities)
    raw = probe.decision_function(acts_compound[:, L])  # signed distance from boundary
    ref = probe.decision_function(acts_cities[:, L])  # standardize against the probe's own training distribution
    scores_by_layer[:, L] = (raw - ref.mean()) / ref.std()

group_means = pd.DataFrame(
    {group: scores_by_layer[idx].mean(axis=0) for group, idx in meta.groupby("group").groups.items()},
    index=layers,
)
group_means.index.name = "layer"

results_dir = "results/compound_cities"
os.makedirs(results_dir, exist_ok=True)
np.save(f"{results_dir}/cities_probe_scores.npy", scores_by_layer)
group_means.to_csv(f"{results_dir}/cities_probe_scores_by_group.csv")
print(group_means)
print(f"saved {results_dir}/cities_probe_scores.npy and {results_dir}/cities_probe_scores_by_group.csv")

plt.figure(figsize=(9, 6))
for color, group in zip(PALETTE, group_means.columns):
    plt.plot(layers, group_means[group], color=color, marker="o", label=group)
plt.axhline(0, linestyle="--", color="gray", label="cities-probe mean")
plt.xlabel("layer")
plt.ylabel("standardized probe score")
plt.title(f"cities-trained probe applied to compound and/or statements ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()

os.makedirs("figures/transfer", exist_ok=True)
out_path = "figures/transfer/cities_probe_on_compound.png"
plt.savefig(out_path, dpi=150)
print(f"saved {out_path}")
