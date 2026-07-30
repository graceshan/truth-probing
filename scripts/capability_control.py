"""Capability control for the compound experiment: how much of the
compound-statement label is learnable from a probe trained directly on
compound activations, vs. what the cities-trained probe achieves by
transfer alone?

Self-contained; does not import from or modify
scripts/probe_compound_transfer.py or scripts/conjunct_regression.py.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.data import load_activations
from src.probes import fit_probe

MODEL_NAME = "Qwen2.5-1.5B"
VARIANT_COLORS = {"pooled": "#2a78d6", "and": "#e34948", "or": "#1baf7a"}

acts_cities, labels_cities = load_activations("cities")
acts_compound, labels_compound = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]

is_and = (meta["connective"] == "and").to_numpy()
is_or = (meta["connective"] == "or").to_numpy()

n_layers = acts_compound.shape[1]
layers = range(n_layers)

VARIANT_DATA = {
    "pooled": (acts_compound, labels_compound),
    "and": (acts_compound[is_and], labels_compound[is_and]),
    "or": (acts_compound[is_or], labels_compound[is_or]),
}

# stratified 80/20 split per variant, computed once and reused across every
# layer (labels don't depend on layer, so neither does the split); same for
# the shuffled-label control, so both are stable across the whole run
splits = {}
shuffled_labels = {}
rng = np.random.default_rng(0)
for name, (_, y) in VARIANT_DATA.items():
    local_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(local_idx, test_size=0.2, random_state=0, stratify=y)
    splits[name] = (train_idx, test_idx)
    shuffled_labels[name] = rng.permutation(y)

pooled_X, pooled_y = VARIANT_DATA["pooled"]
_, pooled_test_idx = splits["pooled"]

# connective-blind ceiling: the Bayes-optimal accuracy of a classifier that
# sees the compound's pattern (both conjuncts' truth values) but not its
# connective. TT and FF are unambiguous (true under both and/or, false
# under both); TF and FT split 50/50 between true (or) and false (and)
# with nothing to break the tie, capping accuracy at 75% -- computed here
# from the actual pooled test set rather than assumed from the theoretical
# construction, since the stratified split isn't guaranteed to preserve
# the pattern balance exactly.
pooled_test_meta = meta.iloc[pooled_test_idx]
pooled_y_test_labels = pooled_y[pooled_test_idx]
connective_blind_correct = 0
for pattern in ["TT", "TF", "FT", "FF"]:
    pattern_labels = pooled_y_test_labels[(pooled_test_meta["pattern"] == pattern).to_numpy()]
    connective_blind_correct += max((pattern_labels == 1).sum(), (pattern_labels == 0).sum())
connective_blind_ceiling = connective_blind_correct / len(pooled_y_test_labels)
print(f"connective-blind ceiling (pooled test set): {connective_blind_ceiling:.4f}")

rows = []
cities_acc_by_layer = []
cities_auroc_by_layer = []

for L in layers:
    # cities-trained probe, transferred: evaluated on the pooled variant's
    # test set so it's directly comparable to the pooled fresh-probe row
    cities_probe = fit_probe(acts_cities[:, L], labels_cities)
    cities_scores = cities_probe.decision_function(pooled_X[pooled_test_idx, L])
    cities_pred = cities_probe.predict(pooled_X[pooled_test_idx, L])
    cities_acc_by_layer.append((cities_pred == pooled_y[pooled_test_idx]).mean())
    cities_auroc_by_layer.append(roc_auc_score(pooled_y[pooled_test_idx], cities_scores))

    for name, (X, y) in VARIANT_DATA.items():
        train_idx, test_idx = splits[name]
        X_L = X[:, L]

        probe = LogisticRegression(max_iter=2000, C=0.1)
        probe.fit(X_L[train_idx], y[train_idx])
        y_test = y[test_idx]
        accuracy = probe.score(X_L[test_idx], y_test)
        auroc = roc_auc_score(y_test, probe.decision_function(X_L[test_idx]))
        majority_baseline = max(y_test.mean(), 1 - y_test.mean())

        y_shuf = shuffled_labels[name]
        shuf_probe = LogisticRegression(max_iter=2000, C=0.1)
        shuf_probe.fit(X_L[train_idx], y_shuf[train_idx])
        shuffled_accuracy = shuf_probe.score(X_L[test_idx], y_shuf[test_idx])

        rows.append(
            {
                "layer": L,
                "variant": name,
                "accuracy": accuracy,
                "auroc": auroc,
                "majority_baseline": majority_baseline,
                "shuffled_accuracy": shuffled_accuracy,
                "cities_transfer_accuracy": cities_acc_by_layer[-1],
                "cities_transfer_auroc": cities_auroc_by_layer[-1],
            }
        )

results_df = pd.DataFrame(rows)

os.makedirs("results", exist_ok=True)
results_df.to_csv("results/capability_control.csv", index=False)
print(results_df)
print("saved results/capability_control.csv")

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(9, 6))
for name, color in VARIANT_COLORS.items():
    sub = results_df[results_df["variant"] == name]
    plt.plot(sub["layer"], sub["accuracy"], color=color, marker="o", label=f"{name} (fresh probe)")
    plt.axhline(sub["majority_baseline"].iloc[0], color=color, linestyle=":", linewidth=1)
plt.plot([], [], color="gray", linestyle=":", label="majority baseline (per variant; pooled=50%, and/or=75%)")
plt.axhline(0.5, color="gray", linestyle="--", label="chance")
plt.axhline(
    connective_blind_ceiling, color="black", linestyle="-.", linewidth=1.5,
    label=f"connective-blind ceiling ({connective_blind_ceiling:.0%}, pooled only)",
)
plt.plot(list(layers), cities_acc_by_layer, color="black", linestyle="--", marker="o", label="cities-trained probe (transfer)")
plt.xlabel("layer")
plt.ylabel("accuracy")
plt.title(f"compound capability control: fresh probes vs. cities transfer ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()
out_path = "figures/capability_control.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
