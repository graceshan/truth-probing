"""Cross-dataset transfer: train a probe per layer on one dataset, test on
another, to check whether the truth direction generalizes across datasets
(e.g. cities are affirmed statements, neg_cities negates them; sp_en_trans
is Spanish-English translation statements).

Source probe fitting is UNCHANGED from before (in-dataset baselines use a
held-out split via train_layer_probe; cross-dataset transfer fits on all of
the source via fit_probe -- no held-out split, since the target is an
entirely different dataset). What's new is evaluation-side bootstrap: with
the probe held fixed, resample the evaluation set (test set for the
in-dataset baselines, the full target dataset for cross-dataset transfer)
1000 times to get a percentile CI on accuracy and AUROC, instead of
resampling the source-side split as scripts 01/02/04 now do. The original
accuracy point estimates are asserted unchanged from the last saved run.
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
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import load_activations
from src.probes import fit_probe, split_indices, train_layer_probe
from src.stats import assert_unchanged, percentile_bootstrap

MODEL_NAME = "Qwen2.5-1.5B"
N_RESAMPLES = 1000

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

# split indices once per dataset, reused across every layer -- unchanged
train_idx_a, test_idx_a = split_indices(len(labels_a))
train_idx_b, test_idx_b = split_indices(len(labels_b))


def bootstrap_eval(probe, X, y, n_resamples=N_RESAMPLES, seed=0):
    """Percentile bootstrap CI for accuracy and AUROC of a FIXED (already
    fitted) probe, resampling the evaluation set with replacement.
    """
    n = len(y)

    def stat(rng):
        idx = rng.integers(0, n, size=n)
        y_r = y[idx]
        if len(np.unique(y_r)) < 2:
            return np.array([np.nan, np.nan])
        acc_r = (probe.predict(X[idx]) == y_r).mean()
        auc_r = roc_auc_score(y_r, probe.decision_function(X[idx]))
        return np.array([acc_r, auc_r])

    lo, hi, _ = percentile_bootstrap(stat, n_resamples=n_resamples, rng=np.random.default_rng(seed))
    return lo[0], hi[0], lo[1], hi[1]


series = {
    f"{DATASET_A}->{DATASET_A}": {"acc": [], "auc": [], "acc_lo": [], "acc_hi": [], "auc_lo": [], "auc_hi": []},
    f"{DATASET_A}->{DATASET_B}": {"acc": [], "auc": [], "acc_lo": [], "acc_hi": [], "auc_lo": [], "auc_hi": []},
    f"{DATASET_B}->{DATASET_B}": {"acc": [], "auc": [], "acc_lo": [], "acc_hi": [], "auc_lo": [], "auc_hi": []},
    f"{DATASET_B}->{DATASET_A}": {"acc": [], "auc": [], "acc_lo": [], "acc_hi": [], "auc_lo": [], "auc_hi": []},
}
key_a_a, key_a_b, key_b_b, key_b_a = series.keys()

for layer in layers:
    X_a = acts_a[:, layer, :]
    X_b = acts_b[:, layer, :]

    # in-dataset baseline: held-out split within the same dataset -- unchanged
    probe_a_a, acc = train_layer_probe(X_a, labels_a, train_idx_a, test_idx_a)
    series[key_a_a]["acc"].append(acc)
    series[key_a_a]["auc"].append(roc_auc_score(labels_a[test_idx_a], probe_a_a.decision_function(X_a[test_idx_a])))
    lo_a, hi_a, lo_u, hi_u = bootstrap_eval(probe_a_a, X_a[test_idx_a], labels_a[test_idx_a], seed=layer)
    series[key_a_a]["acc_lo"].append(lo_a); series[key_a_a]["acc_hi"].append(hi_a)
    series[key_a_a]["auc_lo"].append(lo_u); series[key_a_a]["auc_hi"].append(hi_u)

    probe_b_b, acc = train_layer_probe(X_b, labels_b, train_idx_b, test_idx_b)
    series[key_b_b]["acc"].append(acc)
    series[key_b_b]["auc"].append(roc_auc_score(labels_b[test_idx_b], probe_b_b.decision_function(X_b[test_idx_b])))
    lo_a, hi_a, lo_u, hi_u = bootstrap_eval(probe_b_b, X_b[test_idx_b], labels_b[test_idx_b], seed=layer)
    series[key_b_b]["acc_lo"].append(lo_a); series[key_b_b]["acc_hi"].append(hi_a)
    series[key_b_b]["auc_lo"].append(lo_u); series[key_b_b]["auc_hi"].append(hi_u)

    # cross-dataset transfer: fit on all of the source dataset, test on all of the other -- unchanged
    probe_a = fit_probe(X_a, labels_a)
    acc = probe_a.score(X_b, labels_b)
    series[key_a_b]["acc"].append(acc)
    series[key_a_b]["auc"].append(roc_auc_score(labels_b, probe_a.decision_function(X_b)))
    lo_a, hi_a, lo_u, hi_u = bootstrap_eval(probe_a, X_b, labels_b, seed=layer)
    series[key_a_b]["acc_lo"].append(lo_a); series[key_a_b]["acc_hi"].append(hi_a)
    series[key_a_b]["auc_lo"].append(lo_u); series[key_a_b]["auc_hi"].append(hi_u)

    probe_b = fit_probe(X_b, labels_b)
    acc = probe_b.score(X_a, labels_a)
    series[key_b_a]["acc"].append(acc)
    series[key_b_a]["auc"].append(roc_auc_score(labels_a, probe_b.decision_function(X_a)))
    lo_a, hi_a, lo_u, hi_u = bootstrap_eval(probe_b, X_a, labels_a, seed=layer)
    series[key_b_a]["acc_lo"].append(lo_a); series[key_b_a]["acc_hi"].append(hi_a)
    series[key_b_a]["auc_lo"].append(lo_u); series[key_b_a]["auc_hi"].append(hi_u)

for label, s in series.items():
    print(f"--- {label} ---")
    for layer in layers:
        print(
            f"layer {layer}: acc={s['acc'][layer]:.4f} [{s['acc_lo'][layer]:.4f},{s['acc_hi'][layer]:.4f}]  "
            f"auc={s['auc'][layer]:.4f} [{s['auc_lo'][layer]:.4f},{s['auc_hi'][layer]:.4f}]"
        )

# point-estimate regression check against the last saved run, if one exists
results_dir = "results/transfer"
results_path = f"{results_dir}/{DATASET_A}_{DATASET_B}_transfer.csv"
if os.path.exists(results_path):
    old = pd.read_csv(results_path)
    if set(series.keys()).issubset(old.columns):
        for label in series:
            assert_unchanged(f"{DATASET_A}_{DATASET_B} {label} accuracy", old[label].to_numpy(), series[label]["acc"])
        print("verified: accuracy point estimates unchanged from the last saved run")

os.makedirs(results_dir, exist_ok=True)
rows = []
for layer in layers:
    row = {"layer": layer}
    for label, s in series.items():
        row[f"{label}_acc"] = s["acc"][layer]
        row[f"{label}_acc_lo"] = s["acc_lo"][layer]
        row[f"{label}_acc_hi"] = s["acc_hi"][layer]
        row[f"{label}_auc"] = s["auc"][layer]
        row[f"{label}_auc_lo"] = s["auc_lo"][layer]
        row[f"{label}_auc_hi"] = s["auc_hi"][layer]
    rows.append(row)
pd.DataFrame(rows).to_csv(results_path, index=False)
print(f"saved {results_path}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, metric in zip(axes, ["acc", "auc"]):
    for label, color, style in [
        (key_a_a, BLUE, "-"), (key_a_b, BLUE, ":"), (key_b_b, GREEN, "-"), (key_b_a, GREEN, ":"),
    ]:
        s = series[label]
        ax.plot(layers, s[metric], color=color, linestyle=style, marker="o", label=label)
        ax.fill_between(layers, s[f"{metric}_lo"], s[f"{metric}_hi"], color=color, alpha=0.15, linewidth=0)
    ax.axhline(0.5, linestyle="--", color="gray", label="chance")
    ax.set_xlabel("layer")
    ax.set_title("accuracy" if metric == "acc" else "AUROC")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("score")
axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.15), ncol=3, fontsize=8)
fig.suptitle(f"cross-dataset transfer over layers ({MODEL_NAME})")
fig.tight_layout()

os.makedirs("figures/transfer", exist_ok=True)
out_path = f"figures/transfer/{DATASET_A}_{DATASET_B}_transfer.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
