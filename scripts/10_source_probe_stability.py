"""One-off robustness check (appendix material): how stable is the cities
source probe itself under resampling? Fits on 5 different 80% group-level
(by city) subsamples of cities, evaluates each resulting probe's transfer
AUROC on neg_cities and on compounds (pooled label), and reports both the
spread in per-layer transfer AUROC across the 5 probes and the mean
pairwise cosine similarity between their weight vectors -- i.e. does the
learned direction itself move much depending on which 80% of cities it saw?

This is NOT the new default for 03/05/06/08, which keep fitting on all of
cities (see the methods note in CLAUDE.md). It's a separate question: not
"what's the uncertainty in a transfer estimate" but "how much would the
transfer probe change if training data had been sampled differently."
"""

import os

# Must be set before numpy/sklearn load -- see 01_probe_accuracy_by_layer.py
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.data import group_key, load_activations

MODEL_NAME = "Qwen2.5-1.5B"
SEEDS = [0, 1, 2, 3, 4]
NEG_COLOR = "#2a78d6"
COMPOUND_COLOR = "#eb6834"
COS_COLOR = "#4a3aa7"

acts_cities, labels_cities = load_activations("cities")
acts_neg, labels_neg = load_activations("neg_cities")
acts_compound, _ = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]
compound_label = meta["label"].to_numpy()

city_groups = group_key("cities")
n_layers = acts_cities.shape[1]

# 80% group-level train subsample of cities per seed, computed once and
# reused across every layer
train_idx_by_seed = []
for seed in SEEDS:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, _ = next(gss.split(np.arange(len(labels_cities)), groups=city_groups))
    train_idx_by_seed.append(train_idx)
    n_cities = pd.Series(city_groups).iloc[train_idx].nunique()
    print(f"seed {seed}: {len(train_idx)} statements / {n_cities} cities in the 80% subsample")

neg_auroc = np.zeros((len(SEEDS), n_layers))
compound_auroc = np.zeros((len(SEEDS), n_layers))
mean_pairwise_cos = np.zeros(n_layers)

for L in range(n_layers):
    probes = []
    for si, seed in enumerate(SEEDS):
        train_idx = train_idx_by_seed[si]
        probe = LogisticRegression(max_iter=2000, C=0.1)
        probe.fit(acts_cities[train_idx, L], labels_cities[train_idx])
        probes.append(probe)
        neg_auroc[si, L] = roc_auc_score(labels_neg, probe.decision_function(acts_neg[:, L]))
        compound_auroc[si, L] = roc_auc_score(compound_label, probe.decision_function(acts_compound[:, L]))

    coefs = np.stack([p.coef_[0] for p in probes])
    coefs_unit = coefs / np.linalg.norm(coefs, axis=1, keepdims=True)
    sims = [coefs_unit[i] @ coefs_unit[j] for i, j in combinations(range(len(SEEDS)), 2)]
    mean_pairwise_cos[L] = np.mean(sims)

layers = list(range(n_layers))
results_df = pd.DataFrame(
    {
        "layer": layers,
        "neg_cities_auroc_mean": neg_auroc.mean(axis=0),
        "neg_cities_auroc_min": neg_auroc.min(axis=0),
        "neg_cities_auroc_max": neg_auroc.max(axis=0),
        "neg_cities_auroc_std": neg_auroc.std(axis=0, ddof=1),
        "compound_auroc_mean": compound_auroc.mean(axis=0),
        "compound_auroc_min": compound_auroc.min(axis=0),
        "compound_auroc_max": compound_auroc.max(axis=0),
        "compound_auroc_std": compound_auroc.std(axis=0, ddof=1),
        "mean_pairwise_cosine_sim": mean_pairwise_cos,
    }
)

print(results_df.to_string(index=False))

os.makedirs("results/compound_cities", exist_ok=True)
out_csv = "results/compound_cities/source_probe_stability.csv"
results_df.to_csv(out_csv, index=False)
print(f"saved {out_csv}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

ax1.plot(layers, results_df["neg_cities_auroc_mean"], color=NEG_COLOR, marker="o", label="neg_cities")
ax1.fill_between(layers, results_df["neg_cities_auroc_min"], results_df["neg_cities_auroc_max"], color=NEG_COLOR, alpha=0.2)
ax1.plot(layers, results_df["compound_auroc_mean"], color=COMPOUND_COLOR, marker="o", label="compounds (pooled)")
ax1.fill_between(
    layers, results_df["compound_auroc_min"], results_df["compound_auroc_max"], color=COMPOUND_COLOR, alpha=0.2
)
ax1.axhline(0.5, linestyle="--", color="gray", label="chance")
ax1.set_xlabel("layer")
ax1.set_ylabel("transfer AUROC")
ax1.set_title("spread across 5 source-probe subsamples (min-max band)")
ax1.set_ylim(0, 1)
ax1.grid(alpha=0.3)
ax1.legend()

ax2.plot(layers, mean_pairwise_cos, color=COS_COLOR, marker="o")
ax2.set_xlabel("layer")
ax2.set_ylabel("mean pairwise cosine similarity")
ax2.set_title("weight-vector stability across the 5 subsamples")
ax2.set_ylim(0, 1.05)
ax2.grid(alpha=0.3)

fig.suptitle(f"source-probe stability under resampling ({MODEL_NAME})")
fig.tight_layout()

os.makedirs("figures/compound_cities", exist_ok=True)
out_path = "figures/compound_cities/source_probe_stability.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
