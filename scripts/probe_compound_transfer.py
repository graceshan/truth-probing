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
PATTERNS = ["TT", "TF", "FT", "FF"]
PATTERN_COLORS = {"TT": "#2a78d6", "TF": "#008300", "FT": "#e87ba4", "FF": "#eda100"}

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

def mean_and_sem(rows):
    """Per-layer mean and standard error of the mean for a set of statement rows."""
    vals = scores_by_layer[rows]
    mean = vals.mean(axis=0)
    sem = vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
    return mean, sem


group_idx = meta.groupby("group").groups
group_means = pd.DataFrame(
    {group: mean_and_sem(idx)[0] for group, idx in group_idx.items()}, index=layers
)
group_sems = pd.DataFrame(
    {group: mean_and_sem(idx)[1] for group, idx in group_idx.items()}, index=layers
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
    mean, sem = group_means[group], group_sems[group]
    plt.plot(layers, mean, color=color, marker="o", label=group)
    plt.fill_between(layers, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)
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

# one panel per connective, sharing a y-axis, so the two profiles (how score
# varies across conjunct pattern) can be compared directly without needing
# to subtract out and/or's different overall level by hand.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, conn in zip(axes, ["and", "or"]):
    for pat in PATTERNS:
        m = ((meta.connective == conn) & (meta.pattern == pat)).values
        mean, sem = mean_and_sem(m)
        ax.plot(layers, mean, color=PATTERN_COLORS[pat], marker="o", label=pat)
        ax.fill_between(layers, mean - sem, mean + sem, color=PATTERN_COLORS[pat], alpha=0.15, linewidth=0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f'"{conn}" compounds')
    ax.set_xlabel("layer")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mean probe score (z, cities scale)")
axes[0].legend(title="conjunct pattern")
fig.suptitle(f"cities-trained probe on compound statements, by pattern ({MODEL_NAME})")
fig.tight_layout()

out_path = "figures/transfer/compound_scores_by_layer.png"
fig.savefig(out_path, dpi=150)
print(f"saved {out_path}")
