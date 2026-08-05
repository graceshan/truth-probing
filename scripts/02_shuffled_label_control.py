"""Label-shuffle control: probes trained on a permutation of the labels
should sit at ~0.5 accuracy on every layer. Anything far from that points
to leakage (e.g. train/test overlap) rather than a real learned signal.

Group-level split (by city / Spanish word), aggregated across 5 seeds --
same reasoning as 01_probe_accuracy_by_layer.py. Each seed both permutes
the labels and splits train/test, so the control is evaluated under the
same 5 group-level splits used for the real accuracy figure.
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

from src.data import group_key, load_activations
from src.probes import train_all_layers
from src.stats import seed_mean_ci

MODEL_NAME = "Qwen2.5-1.5B"
ALL_DATASETS = ["cities", "neg_cities", "sp_en_trans"]
SEEDS = [0, 1, 2, 3, 4]
TOLERANCE = 0.15  # flag layers further than this from chance
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
    for si, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        shuffled_labels = rng.permutation(labels)
        results = train_all_layers(acts, shuffled_labels, random_state=seed, groups=groups)
        acc_by_seed[si] = [acc for _, acc in results]

    mean_acc, lo_acc, hi_acc = seed_mean_ci(acc_by_seed)
    layers = range(n_layers)

    print(f"--- {dataset} (shuffled labels, group-level split, {len(SEEDS)} seeds) ---")
    for layer in layers:
        flag = " <-- far from chance" if abs(mean_acc[layer] - 0.5) > TOLERANCE else ""
        print(f"layer {layer}: mean={mean_acc[layer]:.4f}  95% CI=[{lo_acc[layer]:.4f}, {hi_acc[layer]:.4f}]{flag}")

    max_dev = max(abs(a - 0.5) for a in mean_acc)
    status = "OK" if max_dev <= TOLERANCE else "CHECK FOR LEAKAGE"
    print(f"mean of means: {mean_acc.mean():.4f}, max |mean - 0.5|: {max_dev:.4f} [{status}]")

    results_dir = f"results/{dataset}"
    os.makedirs(results_dir, exist_ok=True)

    with open(f"{results_dir}/control_accuracy.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "accuracy_mean", "accuracy_ci_lo", "accuracy_ci_hi"])
        writer.writerows(zip(layers, mean_acc, lo_acc, hi_acc))
    print(f"saved {results_dir}/control_accuracy.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(layers, mean_acc, marker="o", color=LINE_COLOR, label="mean (5 seeds)")
    plt.fill_between(layers, lo_acc, hi_acc, color=LINE_COLOR, alpha=0.2, label="95% CI across seeds")
    plt.axhline(0.5, linestyle="--", color="gray", label="chance")
    plt.xlabel("layer")
    plt.ylabel("test accuracy")
    plt.title(f"probe accuracy over layers, shuffled labels ({MODEL_NAME}, {dataset})")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs("figures/control", exist_ok=True)
    out_path = f"figures/control/{dataset}_control_probe_accuracy.png"
    plt.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
