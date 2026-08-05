"""Probe accuracy over layers, group-level split (by city / Spanish word),
aggregated across 5 seeds. Row-level splitting could let the same city's
true and false statement land on opposite sides of the split; group-level
splitting rules that out. See scripts/09_group_split_comparison.py for the
row-vs-group comparison that motivated switching this script over.
"""

import os

# Must be set before numpy/sklearn load: many small sequential fits in a
# loop otherwise suffer badly from BLAS thread oversubscription (each fit
# spawns a full thread pool; contention overhead dwarfs the actual compute
# for a problem this small -- measured ~35x slower without this).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

from src.data import group_key, load_activations
from src.probes import normalized_coef, train_all_layers
from src.stats import seed_mean_ci

MODEL_NAME = "Qwen2.5-1.5B"
ALL_DATASETS = ["cities", "neg_cities", "sp_en_trans"]
SEEDS = [0, 1, 2, 3, 4]
LINE_COLOR = "#2a78d6"

parser = argparse.ArgumentParser()
parser.add_argument("datasets", nargs="*", default=ALL_DATASETS)
args = parser.parse_args()

for dataset in args.datasets:
    acts, labels = load_activations(dataset)
    groups = group_key(dataset)
    assert len(groups) == acts.shape[0]
    n_layers = acts.shape[1]

    acc_by_seed = np.zeros((len(SEEDS), n_layers))
    directions_by_seed = np.zeros((len(SEEDS), n_layers, acts.shape[2]))
    for si, seed in enumerate(SEEDS):
        results = train_all_layers(acts, labels, random_state=seed, groups=groups)
        acc_by_seed[si] = [acc for _, acc in results]
        directions_by_seed[si] = np.stack([normalized_coef(probe) for probe, _ in results])

    mean_acc, lo_acc, hi_acc = seed_mean_ci(acc_by_seed)
    layers = range(n_layers)

    print(f"--- {dataset} (group-level split, {len(SEEDS)} seeds) ---")
    for layer in layers:
        print(f"layer {layer}: mean={mean_acc[layer]:.4f}  95% CI=[{lo_acc[layer]:.4f}, {hi_acc[layer]:.4f}]")

    results_dir = f"results/{dataset}"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "accuracy_mean", "accuracy_ci_lo", "accuracy_ci_hi"])
        writer.writerows(zip(layers, mean_acc, lo_acc, hi_acc))

    # mean direction across seeds, re-normalized to unit length -- no
    # downstream script currently loads this, kept for continuity
    mean_directions = directions_by_seed.mean(axis=0)
    mean_directions /= np.linalg.norm(mean_directions, axis=1, keepdims=True)
    np.save(f"{results_dir}/directions.npy", mean_directions)
    print(f"saved {results_dir}/accuracy.csv and {results_dir}/directions.npy")

    plt.figure(figsize=(8, 5))
    plt.plot(layers, mean_acc, marker="o", color=LINE_COLOR, label="mean (5 seeds)")
    plt.fill_between(layers, lo_acc, hi_acc, color=LINE_COLOR, alpha=0.2, label="95% CI across seeds")
    plt.axhline(0.5, linestyle="--", color="gray", label="chance")
    plt.xlabel("layer")
    plt.ylabel("test accuracy")
    plt.title(f"probe accuracy over layers, group-level split ({MODEL_NAME}, {dataset})")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs("figures/probe_accuracy", exist_ok=True)
    out_path = f"figures/probe_accuracy/{dataset}_probe_accuracy.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
