"""Label-shuffle control: probes trained on np.random.permutation(labels)
should sit at ~0.5 accuracy on every layer. Anything far from that points
to leakage (e.g. train/test overlap) rather than a real learned signal.
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_activations
from src.probes import train_all_layers

MODEL_NAME = "Qwen2.5-1.5B"
ALL_DATASETS = ["cities", "neg_cities", "sp_en_trans"]
TOLERANCE = 0.15  # flag layers further than this from chance

parser = argparse.ArgumentParser()
parser.add_argument("datasets", nargs="*", default=ALL_DATASETS)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

np.random.seed(args.seed)

for dataset in args.datasets:
    acts, labels = load_activations(dataset)
    shuffled_labels = np.random.permutation(labels)

    results = train_all_layers(acts, shuffled_labels)
    accuracies = [acc for _, acc in results]
    layers = range(len(accuracies))

    print(f"--- {dataset} (shuffled labels) ---")
    for layer, acc in zip(layers, accuracies):
        flag = " <-- far from chance" if abs(acc - 0.5) > TOLERANCE else ""
        print(f"layer {layer}: {acc:.4f}{flag}")

    mean_acc = np.mean(accuracies)
    max_dev = max(abs(a - 0.5) for a in accuracies)
    status = "OK" if max_dev <= TOLERANCE else "CHECK FOR LEAKAGE"
    print(f"mean: {mean_acc:.4f}, max deviation from 0.5: {max_dev:.4f} [{status}]")

    results_dir = f"results/{dataset}"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/control_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "accuracy"])
        writer.writerows(zip(layers, accuracies))
    print(f"saved {results_dir}/control_accuracy.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(layers, accuracies, marker="o")
    plt.axhline(0.5, linestyle="--", color="gray", label="chance")
    plt.xlabel("layer")
    plt.ylabel("test accuracy")
    plt.title(f"probe accuracy over layers, shuffled labels ({MODEL_NAME}, {dataset})")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = f"figures/{dataset}_control_probe_accuracy.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
