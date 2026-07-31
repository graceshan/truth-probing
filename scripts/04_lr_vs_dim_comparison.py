"""Compare difference-in-means vs logistic-regression probe AUROC, layer by
layer. Same shared train/test split for both methods on each layer.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.data import load_activations
from src.diff_means import diff_of_means_direction, direction_auc
from src.probes import split_indices, train_layer_probe

MODEL_NAME = "Qwen2.5-1.5B"
ALL_DATASETS = ["cities", "neg_cities", "sp_en_trans"]

DIM_COLOR = "#2a78d6"
LR_COLOR = "#008300"

parser = argparse.ArgumentParser()
parser.add_argument("datasets", nargs="*", default=ALL_DATASETS)
args = parser.parse_args()

for dataset in args.datasets:
    acts, labels = load_activations(dataset)
    n_layers = acts.shape[1]
    layers = range(n_layers)

    train_idx, test_idx = split_indices(len(labels))

    dim_aucs, lr_aucs = [], []
    for layer in layers:
        X = acts[:, layer, :]

        direction = diff_of_means_direction(X[train_idx], labels[train_idx])
        dim_aucs.append(direction_auc(direction, X[test_idx], labels[test_idx]))

        probe, _ = train_layer_probe(X, labels, train_idx, test_idx)
        lr_aucs.append(roc_auc_score(labels[test_idx], probe.decision_function(X[test_idx])))

    print(f"--- {dataset} ---")
    for layer, dim_auc, lr_auc in zip(layers, dim_aucs, lr_aucs):
        print(f"layer {layer}: DIM {dim_auc:.4f}  LR {lr_auc:.4f}")

    results_dir = f"results/{dataset}"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/auc.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "dim_auc", "lr_auc"])
        writer.writerows(zip(layers, dim_aucs, lr_aucs))
    print(f"saved {results_dir}/auc.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(layers, dim_aucs, color=DIM_COLOR, marker="o", label="difference-in-means")
    plt.plot(layers, lr_aucs, color=LR_COLOR, marker="o", label="logistic regression")
    plt.axhline(0.5, linestyle="--", color="gray", label="chance")
    plt.xlabel("layer")
    plt.ylabel("AUROC")
    plt.title(f"AUROC over layers ({MODEL_NAME}, {dataset})")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs("figures/auc_comparison", exist_ok=True)
    out_path = f"figures/auc_comparison/{dataset}_auc_comparison.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
