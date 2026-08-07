"""Combine the three probe-accuracy-by-layer figures (cities, neg_cities,
sp_en_trans) onto one set of shared axes for a direct side-by-side
comparison. Reads the CSVs 01_probe_accuracy_by_layer.py already saved --
does not recompute anything.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

MODEL_NAME = "Qwen2.5-1.5B"
DATASETS = ["cities", "neg_cities", "sp_en_trans"]
COLORS = {"cities": "#2a78d6", "neg_cities": "#e34948", "sp_en_trans": "#1baf7a"}

plt.figure(figsize=(9, 6))
for name in DATASETS:
    df = pd.read_csv(f"results/{name}/accuracy.csv")
    color = COLORS[name]
    plt.plot(df["layer"], df["accuracy_mean"], color=color, marker="o", markersize=4, label=name)
    plt.fill_between(df["layer"], df["accuracy_ci_lo"], df["accuracy_ci_hi"], color=color, alpha=0.2, linewidth=0)

plt.axhline(0.5, linestyle="--", color="gray", label="chance")
plt.xlabel("layer")
plt.ylabel("test accuracy")
plt.ylim(0, 1)
plt.title(f"probe accuracy over layers, group-level split ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

os.makedirs("figures/probe_accuracy", exist_ok=True)
out_path = "figures/probe_accuracy/all_datasets.png"
plt.savefig(out_path, dpi=150)
print(f"saved {out_path}")
