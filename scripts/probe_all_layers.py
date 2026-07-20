import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_activations
from src.probes import normalized_coef, train_all_layers

MODEL_NAME = "Qwen2.5-1.5B"
ALL_DATASETS = ["cities", "neg_cities", "sp_en_trans"]

parser = argparse.ArgumentParser()
parser.add_argument("datasets", nargs="*", default=ALL_DATASETS)
args = parser.parse_args()

for dataset in args.datasets:
    acts, labels = load_activations(dataset)

    results = train_all_layers(acts, labels)
    accuracies = [acc for _, acc in results]
    directions = np.stack([normalized_coef(probe) for probe, _ in results])  # [n_layers, d_model]
    layers = range(len(accuracies))

    print(f"--- {dataset} ---")
    for layer, acc in zip(layers, accuracies):
        print(f"layer {layer}: {acc:.4f}")

    results_dir = f"results/{dataset}"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "accuracy"])
        writer.writerows(zip(layers, accuracies))

    np.save(f"{results_dir}/directions.npy", directions)
    print(f"saved {results_dir}/accuracy.csv and {results_dir}/directions.npy")

    plt.figure(figsize=(8, 5))
    plt.plot(layers, accuracies, marker="o")
    plt.axhline(0.5, linestyle="--", color="gray", label="chance")
    plt.xlabel("layer")
    plt.ylabel("test accuracy")
    plt.title(f"probe accuracy over layers ({MODEL_NAME}, {dataset})")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = f"figures/{dataset}_probe_accuracy.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
