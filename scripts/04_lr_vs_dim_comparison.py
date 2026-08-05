"""Compare difference-in-means vs logistic-regression probe AUROC, layer by
layer. Same shared train/test split for both methods on each layer, group-
level (by city / Spanish word) and aggregated across 5 seeds -- same
reasoning as 01_probe_accuracy_by_layer.py.
"""

import os

# Must be set before numpy/sklearn load -- see 01_probe_accuracy_by_layer.py
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.data import group_key, load_activations
from src.diff_means import diff_of_means_direction, direction_auc
from src.probes import split_indices, train_layer_probe
from src.stats import seed_mean_ci

MODEL_NAME = "Qwen2.5-1.5B"
ALL_DATASETS = ["cities", "neg_cities", "sp_en_trans"]
SEEDS = [0, 1, 2, 3, 4]

DIM_COLOR = "#2a78d6"
LR_COLOR = "#008300"

parser = argparse.ArgumentParser()
parser.add_argument("datasets", nargs="*", default=ALL_DATASETS)
args = parser.parse_args()

for dataset in args.datasets:
    acts, labels = load_activations(dataset)
    groups = group_key(dataset)
    assert len(groups) == acts.shape[0]
    n_layers = acts.shape[1]
    layers = range(n_layers)

    dim_by_seed = np.zeros((len(SEEDS), n_layers))
    lr_by_seed = np.zeros((len(SEEDS), n_layers))
    for si, seed in enumerate(SEEDS):
        train_idx, test_idx = split_indices(len(labels), random_state=seed, groups=groups)
        for layer in layers:
            X = acts[:, layer, :]

            direction = diff_of_means_direction(X[train_idx], labels[train_idx])
            dim_by_seed[si, layer] = direction_auc(direction, X[test_idx], labels[test_idx])

            probe, _ = train_layer_probe(X, labels, train_idx, test_idx)
            lr_by_seed[si, layer] = roc_auc_score(labels[test_idx], probe.decision_function(X[test_idx]))

    dim_mean, dim_lo, dim_hi = seed_mean_ci(dim_by_seed)
    lr_mean, lr_lo, lr_hi = seed_mean_ci(lr_by_seed)

    print(f"--- {dataset} (group-level split, {len(SEEDS)} seeds) ---")
    for layer in layers:
        print(
            f"layer {layer}: DIM mean={dim_mean[layer]:.4f} CI=[{dim_lo[layer]:.4f},{dim_hi[layer]:.4f}]  "
            f"LR mean={lr_mean[layer]:.4f} CI=[{lr_lo[layer]:.4f},{lr_hi[layer]:.4f}]"
        )

    results_dir = f"results/{dataset}"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/auc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["layer", "dim_auc_mean", "dim_auc_ci_lo", "dim_auc_ci_hi", "lr_auc_mean", "lr_auc_ci_lo", "lr_auc_ci_hi"]
        )
        writer.writerows(zip(layers, dim_mean, dim_lo, dim_hi, lr_mean, lr_lo, lr_hi))
    print(f"saved {results_dir}/auc.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(layers, dim_mean, color=DIM_COLOR, marker="o", label="difference-in-means")
    plt.fill_between(layers, dim_lo, dim_hi, color=DIM_COLOR, alpha=0.2)
    plt.plot(layers, lr_mean, color=LR_COLOR, marker="o", label="logistic regression")
    plt.fill_between(layers, lr_lo, lr_hi, color=LR_COLOR, alpha=0.2)
    plt.axhline(0.5, linestyle="--", color="gray", label="chance")
    plt.xlabel("layer")
    plt.ylabel("AUROC")
    plt.title(f"AUROC over layers, group-level split ({MODEL_NAME}, {dataset})")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs("figures/auc_comparison", exist_ok=True)
    out_path = f"figures/auc_comparison/{dataset}_auc_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
