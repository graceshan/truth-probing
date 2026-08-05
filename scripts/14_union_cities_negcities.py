"""Train on the union of cities (affirmative) and neg_cities (negated),
balanced across the four (affirmative/negated) x (true/false) cells so that
surface form (affirmative vs. negated) is decorrelated from the truth label
by construction. If the cities-trained direction picks up "affirmative +
correct pairing" as a shortcut rather than genuine truth, training on this
balanced union should remove that shortcut and change downstream transfer.

Each of the 748 cities contributes exactly one row to each of the four
cells (cities.csv/neg_cities.csv both have one true + one false statement
per city), so the four cells are already exactly 748 rows each -- verified
below rather than assumed, with a real (if here inert) subsample-to-balance
step in case that's ever not true.

Split at the CITY level across the whole union (GroupShuffleSplit, one
split per seed, held fixed across layers) so a city's affirmative-true,
affirmative-false, negated-true and negated-false rows never end up split
across train/test relative to each other.

One probe per seed per layer, fit on that seed's 80% group-level train
partition, used for every evaluation below (within-union held-out,
compounds, sp_en_trans, cosine similarity) -- deliberately not the
fit-on-all-source convention used by 03/05/08, since here the union has an
actual entity-level split to respect and the task calls for one directly.

"The cities probe" referenced throughout (compound/cosine-sim comparisons)
is the single fit_probe-on-all-of-cities probe from 08_transfer_auroc.py's
convention ("task 1"), reused/refit identically, not a seed-matched
group-split cities probe -- keeps the baseline anchored to the same numbers
already reported in results/transfer_auroc.csv.
"""

import os

# Must be set before numpy/sklearn load -- see 01_probe_accuracy_by_layer.py
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.data import group_key, load_activations
from src.probes import fit_probe, normalized_coef
from src.stats import seed_mean_ci

MODEL_NAME = "Qwen2.5-1.5B"
SEEDS = [0, 1, 2, 3, 4]
LABEL_SCHEME_COLORS = {
    "pooled": "#2a78d6",
    "and_only": "#e34948",
    "or_only": "#1baf7a",
    "conjunct_count": "#eda100",
}
UNION_COLOR = "#4a3aa7"
CITIES_COLOR = "#2a78d6"

results_dir = "results/union_cities_negcities"
figures_dir = "figures/union_cities_negcities"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# load + verify cell balance
# ---------------------------------------------------------------------------

acts_cities, labels_cities = load_activations("cities")
acts_neg, labels_neg = load_activations("neg_cities")
assert acts_cities.shape[1:] == acts_neg.shape[1:], "cities/neg_cities activation shapes disagree"
city_cities = group_key("cities")
city_neg = group_key("neg_cities")

aff_true_mask = labels_cities == 1
aff_false_mask = labels_cities == 0
neg_true_mask = labels_neg == 1
neg_false_mask = labels_neg == 0

raw_counts = {
    "affirmative-true": aff_true_mask.sum(),
    "affirmative-false": aff_false_mask.sum(),
    "negated-true": neg_true_mask.sum(),
    "negated-false": neg_false_mask.sum(),
}
print("cell balance before subsampling:")
for cell, n in raw_counts.items():
    print(f"  {cell}: {n}")

min_count = min(raw_counts.values())
rng = np.random.default_rng(0)


def balanced_idx(mask, n_target):
    idx = np.flatnonzero(mask)
    if len(idx) > n_target:
        idx = rng.choice(idx, size=n_target, replace=False)
    return np.sort(idx)


aff_true_idx = balanced_idx(aff_true_mask, min_count)
aff_false_idx = balanced_idx(aff_false_mask, min_count)
neg_true_idx = balanced_idx(neg_true_mask, min_count)
neg_false_idx = balanced_idx(neg_false_mask, min_count)

if all(n == min_count for n in raw_counts.values()):
    print(f"already balanced at {min_count} rows/cell -- no subsampling needed")
else:
    print(f"subsampled every cell down to {min_count} rows (the smallest cell)")
print()

acts_union = np.concatenate(
    [acts_cities[aff_true_idx], acts_cities[aff_false_idx], acts_neg[neg_true_idx], acts_neg[neg_false_idx]], axis=0
)
labels_union = np.concatenate(
    [labels_cities[aff_true_idx], labels_cities[aff_false_idx], labels_neg[neg_true_idx], labels_neg[neg_false_idx]]
)
group_union = np.concatenate(
    [city_cities[aff_true_idx], city_cities[aff_false_idx], city_neg[neg_true_idx], city_neg[neg_false_idx]]
)
n_layers = acts_union.shape[1]
n_union = len(labels_union)
print(f"union: {n_union} rows, {len(np.unique(group_union))} unique cities, "
      f"label=1 rate {labels_union.mean():.4f} (balanced by construction)")
print()

# ---------------------------------------------------------------------------
# transfer targets
# ---------------------------------------------------------------------------

acts_compound, _ = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]
is_and = (meta["connective"] == "and").to_numpy()
is_or = (meta["connective"] == "or").to_numpy()
compound_label = meta["label"].to_numpy()
conjunct_count_label = (meta["pattern"] == "TT").astype(int).to_numpy()
acts_compound_and, labels_and = acts_compound[is_and], compound_label[is_and]
acts_compound_or, labels_or = acts_compound[is_or], compound_label[is_or]

acts_sp, labels_sp = load_activations("sp_en_trans")
for acts, name in [(acts_compound, "compound_cities"), (acts_sp, "sp_en_trans")]:
    assert acts.shape[1] == n_layers, f"{name} has a different layer count than the union"

COMPOUND_SCHEMES = [
    ("pooled", acts_compound, compound_label),
    ("and_only", acts_compound_and, labels_and),
    ("or_only", acts_compound_or, labels_or),
    ("conjunct_count", acts_compound, conjunct_count_label),
]

# fixed cities-only probe per layer (fit_probe on all of cities, "task 1"
# convention), used as the comparison anchor for both the compound plots
# and the cosine-similarity comparison
cities_probes = [fit_probe(acts_cities[:, L], labels_cities) for L in range(n_layers)]
cities_dirs = [normalized_coef(p) for p in cities_probes]

# ---------------------------------------------------------------------------
# 5 seeds x every layer: one union probe per (seed, layer), fit on that
# seed's 80% group-level (city) train partition
# ---------------------------------------------------------------------------

within_acc = np.zeros((len(SEEDS), n_layers))
within_auc = np.zeros((len(SEEDS), n_layers))
compound_auc = {scheme: np.zeros((len(SEEDS), n_layers)) for scheme, _, _ in COMPOUND_SCHEMES}
sp_acc = np.zeros((len(SEEDS), n_layers))
sp_auc = np.zeros((len(SEEDS), n_layers))
cos_sim = np.zeros((len(SEEDS), n_layers))

for si, seed in enumerate(SEEDS):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, test_idx = next(gss.split(np.arange(n_union), groups=group_union))
    train_cities = set(group_union[train_idx])
    test_cities = set(group_union[test_idx])
    assert not (train_cities & test_cities), "a city crossed train/test in the union split"

    for L in range(n_layers):
        probe = LogisticRegression(max_iter=2000, C=0.1)
        probe.fit(acts_union[train_idx, L], labels_union[train_idx])
        assert list(probe.classes_) == [0, 1]

        within_acc[si, L] = probe.score(acts_union[test_idx, L], labels_union[test_idx])
        within_auc[si, L] = roc_auc_score(
            labels_union[test_idx], probe.decision_function(acts_union[test_idx, L])
        )

        for scheme, X, y in COMPOUND_SCHEMES:
            compound_auc[scheme][si, L] = roc_auc_score(y, probe.decision_function(X[:, L]))

        sp_acc[si, L] = probe.score(acts_sp[:, L], labels_sp)
        sp_auc[si, L] = roc_auc_score(labels_sp, probe.decision_function(acts_sp[:, L]))

        cos_sim[si, L] = normalized_coef(probe) @ cities_dirs[L]

layers = np.arange(n_layers)

within_acc_mean, within_acc_lo, within_acc_hi = seed_mean_ci(within_acc)
within_auc_mean, within_auc_lo, within_auc_hi = seed_mean_ci(within_auc)

print("=== within-union held-out accuracy / AUROC (sanity check) ===")
for L in layers:
    print(f"layer {L}: accuracy={within_acc_mean[L]:.4f} [{within_acc_lo[L]:.4f},{within_acc_hi[L]:.4f}]  "
          f"AUROC={within_auc_mean[L]:.4f} [{within_auc_lo[L]:.4f},{within_auc_hi[L]:.4f}]")
print()

within_df = pd.DataFrame(
    {
        "layer": layers,
        "accuracy_mean": within_acc_mean, "accuracy_ci_lo": within_acc_lo, "accuracy_ci_hi": within_acc_hi,
        "auroc_mean": within_auc_mean, "auroc_ci_lo": within_auc_lo, "auroc_ci_hi": within_auc_hi,
    }
)
within_df.to_csv(f"{results_dir}/within_union_accuracy.csv", index=False)
print(f"saved {results_dir}/within_union_accuracy.csv")

# ---------------------------------------------------------------------------
# compounds: union-trained AUROC under all 4 schemes, plus the cities-only
# numbers from results/transfer_auroc.csv ("task 1") for direct comparison
# ---------------------------------------------------------------------------

compound_rows = []
for scheme, _, _ in COMPOUND_SCHEMES:
    mean, lo, hi = seed_mean_ci(compound_auc[scheme])
    for L in layers:
        compound_rows.append(
            {"label_scheme": scheme, "layer": L, "auroc_mean": mean[L], "auroc_ci_lo": lo[L], "auroc_ci_hi": hi[L]}
        )
compound_df = pd.DataFrame(compound_rows)
compound_df.to_csv(f"{results_dir}/compound_auroc.csv", index=False)
print(f"saved {results_dir}/compound_auroc.csv")

print("=== union-trained AUROC on compounds ===")
for scheme, _, _ in COMPOUND_SCHEMES:
    sub = compound_df[compound_df["label_scheme"] == scheme]
    print(f"  {scheme}: layer 10={sub.loc[sub.layer == 10, 'auroc_mean'].iloc[0]:.4f}, "
          f"best={sub['auroc_mean'].max():.4f} at layer {int(sub.loc[sub['auroc_mean'].idxmax(), 'layer'])}")
print()

cities_transfer_path = "results/transfer_auroc.csv"
cities_compound_df = pd.read_csv(cities_transfer_path)
cities_compound_df = cities_compound_df[
    (cities_compound_df["train_set"] == "cities") & (cities_compound_df["eval_set"] == "compound_cities")
]

# ---------------------------------------------------------------------------
# sp_en_trans
# ---------------------------------------------------------------------------

sp_acc_mean, sp_acc_lo, sp_acc_hi = seed_mean_ci(sp_acc)
sp_auc_mean, sp_auc_lo, sp_auc_hi = seed_mean_ci(sp_auc)
sp_df = pd.DataFrame(
    {
        "layer": layers,
        "accuracy_mean": sp_acc_mean, "accuracy_ci_lo": sp_acc_lo, "accuracy_ci_hi": sp_acc_hi,
        "auroc_mean": sp_auc_mean, "auroc_ci_lo": sp_auc_lo, "auroc_ci_hi": sp_auc_hi,
    }
)
sp_df.to_csv(f"{results_dir}/sp_en_trans_transfer.csv", index=False)
print("=== union-trained on sp_en_trans ===")
for L in layers:
    print(f"layer {L}: accuracy={sp_acc_mean[L]:.4f} [{sp_acc_lo[L]:.4f},{sp_acc_hi[L]:.4f}]  "
          f"AUROC={sp_auc_mean[L]:.4f} [{sp_auc_lo[L]:.4f},{sp_auc_hi[L]:.4f}]")
print(f"saved {results_dir}/sp_en_trans_transfer.csv")
print()

# ---------------------------------------------------------------------------
# cosine similarity, union probe vs. the fixed cities probe
# ---------------------------------------------------------------------------

cos_mean, cos_lo, cos_hi = seed_mean_ci(cos_sim)
cos_df = pd.DataFrame({"layer": layers, "cosine_sim_mean": cos_mean, "cosine_sim_ci_lo": cos_lo, "cosine_sim_ci_hi": cos_hi})
cos_df.to_csv(f"{results_dir}/cosine_similarity.csv", index=False)
print("=== cosine similarity: union probe vs. cities probe, per layer ===")
for L in layers:
    print(f"layer {L}: {cos_mean[L]:.4f} [{cos_lo[L]:.4f},{cos_hi[L]:.4f}]")
print(f"saved {results_dir}/cosine_similarity.csv")
print()

# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

# sanity check: within-union accuracy + AUROC
plt.figure(figsize=(8, 5))
plt.plot(layers, within_acc_mean, color=UNION_COLOR, marker="o", markersize=4, label="accuracy")
plt.fill_between(layers, within_acc_lo, within_acc_hi, color=UNION_COLOR, alpha=0.2, linewidth=0)
plt.plot(layers, within_auc_mean, color=UNION_COLOR, linestyle="--", marker="s", markersize=4, label="AUROC")
plt.fill_between(layers, within_auc_lo, within_auc_hi, color=UNION_COLOR, alpha=0.1, linewidth=0)
plt.axhline(0.5, linestyle=":", color="gray", label="chance")
plt.xlabel("layer")
plt.ylabel("score")
plt.ylim(0, 1.02)
plt.title(f"within-union held-out accuracy/AUROC, balanced cities+neg_cities ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{figures_dir}/within_union_accuracy.png", dpi=150, bbox_inches="tight")
print(f"saved {figures_dir}/within_union_accuracy.png")

# HEADLINE: pooled compound AUROC, cities-trained vs union-trained
cities_pooled = cities_compound_df[cities_compound_df["label_scheme"] == "pooled"].sort_values("layer")
union_pooled = compound_df[compound_df["label_scheme"] == "pooled"].sort_values("layer")

plt.figure(figsize=(8, 5))
plt.plot(cities_pooled["layer"], cities_pooled["auroc"], color=CITIES_COLOR, marker="o", markersize=4, label="cities-trained")
plt.fill_between(cities_pooled["layer"], cities_pooled["auroc_ci_lo"], cities_pooled["auroc_ci_hi"], color=CITIES_COLOR, alpha=0.2, linewidth=0)
plt.plot(union_pooled["layer"], union_pooled["auroc_mean"], color=UNION_COLOR, marker="o", markersize=4, label="union-trained (cities+neg_cities, balanced)")
plt.fill_between(union_pooled["layer"], union_pooled["auroc_ci_lo"], union_pooled["auroc_ci_hi"], color=UNION_COLOR, alpha=0.2, linewidth=0)
plt.axhline(0.5, linestyle="--", color="gray", label="chance")
plt.xlabel("layer")
plt.ylabel("compound AUROC (pooled)")
plt.ylim(0, 1.02)
plt.title(f"compound AUROC: cities-trained vs. union-trained ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{figures_dir}/compound_auroc_headline.png", dpi=150, bbox_inches="tight")
print(f"saved {figures_dir}/compound_auroc_headline.png")

# all 4 schemes, cities vs union, on the same axes
plt.figure(figsize=(9, 6))
for scheme, color in LABEL_SCHEME_COLORS.items():
    c_sub = cities_compound_df[cities_compound_df["label_scheme"] == scheme].sort_values("layer")
    u_sub = compound_df[compound_df["label_scheme"] == scheme].sort_values("layer")
    plt.plot(c_sub["layer"], c_sub["auroc"], color=color, linestyle="-", marker="o", markersize=3, label=f"{scheme}, cities")
    plt.fill_between(c_sub["layer"], c_sub["auroc_ci_lo"], c_sub["auroc_ci_hi"], color=color, alpha=0.12, linewidth=0)
    plt.plot(u_sub["layer"], u_sub["auroc_mean"], color=color, linestyle="--", marker="s", markersize=3, label=f"{scheme}, union")
    plt.fill_between(u_sub["layer"], u_sub["auroc_ci_lo"], u_sub["auroc_ci_hi"], color=color, alpha=0.12, linewidth=0)
plt.axhline(0.5, linestyle=":", color="gray", label="chance")
plt.xlabel("layer")
plt.ylabel("AUROC")
plt.ylim(0, 1.02)
plt.title(f"compound AUROC by label scheme: cities (solid) vs. union (dashed) ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()
plt.savefig(f"{figures_dir}/compound_auroc_by_scheme.png", dpi=150, bbox_inches="tight")
print(f"saved {figures_dir}/compound_auroc_by_scheme.png")

# sp_en_trans
plt.figure(figsize=(8, 5))
plt.plot(layers, sp_acc_mean, color=UNION_COLOR, marker="o", markersize=4, label="accuracy")
plt.fill_between(layers, sp_acc_lo, sp_acc_hi, color=UNION_COLOR, alpha=0.2, linewidth=0)
plt.plot(layers, sp_auc_mean, color=UNION_COLOR, linestyle="--", marker="s", markersize=4, label="AUROC")
plt.fill_between(layers, sp_auc_lo, sp_auc_hi, color=UNION_COLOR, alpha=0.1, linewidth=0)
plt.axhline(0.5, linestyle=":", color="gray", label="chance")
plt.xlabel("layer")
plt.ylabel("score")
plt.ylim(0, 1.02)
plt.title(f"union-trained probe on sp_en_trans ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{figures_dir}/sp_en_trans_transfer.png", dpi=150, bbox_inches="tight")
print(f"saved {figures_dir}/sp_en_trans_transfer.png")

# cosine similarity
plt.figure(figsize=(8, 5))
plt.plot(layers, cos_mean, color=UNION_COLOR, marker="o", markersize=4)
plt.fill_between(layers, cos_lo, cos_hi, color=UNION_COLOR, alpha=0.2, linewidth=0)
plt.axhline(0, linestyle=":", color="gray")
plt.xlabel("layer")
plt.ylabel("cosine similarity (union probe, cities probe)")
plt.ylim(-1.02, 1.02)
plt.title(f"direction stability: union-trained vs. cities-trained probe ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{figures_dir}/cosine_similarity.png", dpi=150, bbox_inches="tight")
print(f"saved {figures_dir}/cosine_similarity.png")
