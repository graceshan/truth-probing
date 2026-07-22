"""Cross-dataset transfer: train a probe per layer on one dataset, test on
another, to check whether the truth direction generalizes across datasets
(e.g. cities are affirmed statements, neg_cities negates them; sp_en_trans
is Spanish-English translation statements).
"""

import argparse

import matplotlib.pyplot as plt

from src.data import load_activations
from src.probes import fit_probe, split_indices, train_layer_probe

MODEL_NAME = "Qwen2.5-1.5B"

BLUE = "#2a78d6"
GREEN = "#008300"

parser = argparse.ArgumentParser()
parser.add_argument("dataset_a", nargs="?", default="cities")
parser.add_argument("dataset_b", nargs="?", default="neg_cities")
args = parser.parse_args()
DATASET_A = args.dataset_a
DATASET_B = args.dataset_b

acts_a, labels_a = load_activations(DATASET_A)
acts_b, labels_b = load_activations(DATASET_B)

n_layers = acts_a.shape[1]
assert acts_b.shape[1] == n_layers, "datasets have different layer counts"
layers = range(n_layers)

# split indices once per dataset, reused across every layer
train_idx_a, test_idx_a = split_indices(len(labels_a))
train_idx_b, test_idx_b = split_indices(len(labels_b))

acc_a_a, acc_a_b, acc_b_b, acc_b_a = [], [], [], []

for layer in layers:
    X_a = acts_a[:, layer, :]
    X_b = acts_b[:, layer, :]

    # in-dataset baseline: held-out split within the same dataset
    _, acc = train_layer_probe(X_a, labels_a, train_idx_a, test_idx_a)
    acc_a_a.append(acc)
    _, acc = train_layer_probe(X_b, labels_b, train_idx_b, test_idx_b)
    acc_b_b.append(acc)

    # cross-dataset transfer: fit on all of the source dataset, test on all of the other
    probe_a = fit_probe(X_a, labels_a)
    acc_a_b.append(probe_a.score(X_b, labels_b))
    probe_b = fit_probe(X_b, labels_b)
    acc_b_a.append(probe_b.score(X_a, labels_a))

for label, accs in [
    (f"{DATASET_A}->{DATASET_A}", acc_a_a),
    (f"{DATASET_A}->{DATASET_B}", acc_a_b),
    (f"{DATASET_B}->{DATASET_B}", acc_b_b),
    (f"{DATASET_B}->{DATASET_A}", acc_b_a),
]:
    print(f"--- {label} ---")
    for layer, acc in zip(layers, accs):
        print(f"layer {layer}: {acc:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(layers, acc_a_a, color=BLUE, linestyle="-", marker="o", label=f"{DATASET_A}->{DATASET_A}")
plt.plot(layers, acc_a_b, color=BLUE, linestyle=":", marker="o", label=f"{DATASET_A}->{DATASET_B}")
plt.plot(layers, acc_b_b, color=GREEN, linestyle="-", marker="o", label=f"{DATASET_B}->{DATASET_B}")
plt.plot(layers, acc_b_a, color=GREEN, linestyle=":", marker="o", label=f"{DATASET_B}->{DATASET_A}")
plt.axhline(0.5, linestyle="--", color="gray", label="chance")
plt.xlabel("layer")
plt.ylabel("accuracy")
plt.ylim(0, 1)
plt.title(f"cross-dataset transfer over layers ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

out_path = f"figures/{DATASET_A}_{DATASET_B}_transfer.png"
plt.savefig(out_path, dpi=150)
print(f"saved {out_path}")
