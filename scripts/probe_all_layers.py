import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_activations
from src.probes import normalized_coef, train_all_layers

parser = argparse.ArgumentParser()
parser.add_argument("dataset", nargs="?", default="cities")
args = parser.parse_args()

acts, labels = load_activations(args.dataset)

results = train_all_layers(acts, labels)
accuracies = [acc for _, acc in results]
directions = np.stack([normalized_coef(probe) for probe, _ in results])  # [n_layers, d_model]
layers = range(len(accuracies))

for layer, acc in zip(layers, accuracies):
    print(f"layer {layer}: {acc:.4f}")

results_dir = f"results/{args.dataset}"
os.makedirs(results_dir, exist_ok=True)

with open(f"{results_dir}/accuracy.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "accuracy"])
    writer.writerows(zip(layers, accuracies))

np.save(f"{results_dir}/directions.npy", directions)
print(f"saved {results_dir}/accuracy.csv and {results_dir}/directions.npy")

plt.figure(figsize=(8, 5))
plt.plot(layers, accuracies, marker="o")
plt.xlabel("layer")
plt.ylabel("test accuracy")
plt.title(f"probe accuracy over layers ({args.dataset})")
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.tight_layout()

out_path = f"figures/{args.dataset}_probe_accuracy.png"
plt.savefig(out_path, dpi=150)
print(f"saved {out_path}")
