"""Cross-connective transfer on the compound dataset: does a probe trained
to read out AND's truth value generalize to reading out OR's truth value on
the same kind of (two-conjunct) statements, and vice versa?

A fresh probe is fit on one connective's compounds only (label = truth value
under THAT connective, which is exactly the dataset's own `label` column
since each row already encodes its own connective), then evaluated on the
other connective's compounds (label = truth value under the OTHER
connective, again just that row's own `label` column). Base rates flip
between the two directions (25% true under AND, 75% under OR -- see
CLAUDE.md), so raw accuracy is not reported: AUROC (threshold-free) and
balanced accuracy (base-rate invariant, chance = 0.5 regardless of the
target's base rate) are used instead, alongside each target set's own
majority-class baseline for reference.

Same hyperparameters as every other probe in this project
(LogisticRegression(max_iter=2000, C=0.1)). Per layer, across 5 seeds: each
seed is a different 80% group-level subsample of the SOURCE connective's
800 rows (GroupShuffleSplit, grouped by the unordered city-pair -- see the
overlap check below), mirroring 10_source_probe_stability.py's use of
seeds-as-resampling for a transfer-style script that has no natural
train/test split of its own. Aggregated via seed_mean_ci (mean + 95%
t-distribution CI across the 5 seeds), same as everywhere else.

MAIN OUTPUT is not the aggregate AUROC/balanced-accuracy number -- it's the
per-pattern breakdown: for each of TT/TF/FT/FF, the mean decision_function
of the source-trained probe on target statements of that pattern, z-scored
against the probe's OWN training-subsample distribution (same convention as
05_compound_analysis.py: standardize against the source probe's own score
distribution, not some external reference).
"""

import os
import re

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
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from src.data import load_activations
from src.stats import seed_mean_ci

MODEL_NAME = "Qwen2.5-1.5B"
SEEDS = [0, 1, 2, 3, 4]
PATTERNS = ["TT", "TF", "FT", "FF"]
PATTERN_COLORS = {"TT": "#2a78d6", "TF": "#008300", "FT": "#e87ba4", "FF": "#eda100"}
DIRECTION_COLORS = {"and->or": "#e34948", "or->and": "#1baf7a"}

acts_compound, _ = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]
n_layers = acts_compound.shape[1]
layers = np.arange(n_layers)
labels = meta["label"].to_numpy()

# ---------------------------------------------------------------------------
# constituent-city-pair overlap check between the AND and OR subsets
# ---------------------------------------------------------------------------


def extract_city(statement: str) -> str:
    m = re.match(r"The city of (.+?) is in", statement)
    assert m, f"unexpected statement format: {statement!r}"
    return m.group(1)


meta["city_a"] = meta["conj_a"].map(extract_city)
meta["city_b"] = meta["conj_b"].map(extract_city)
meta["pair_key"] = [
    "|".join(sorted((a, b))) for a, b in zip(meta["city_a"], meta["city_b"])
]

and_pairs = set(meta.loc[meta.connective == "and", "pair_key"])
or_pairs = set(meta.loc[meta.connective == "or", "pair_key"])
pair_overlap = and_pairs & or_pairs
print("constituent city-pair overlap check (AND vs OR subsets):")
print(f"  unique AND city-pairs: {len(and_pairs)} (out of 800 AND rows)")
print(f"  unique OR city-pairs:  {len(or_pairs)} (out of 800 OR rows)")
print(f"  city-pairs appearing in BOTH subsets: {len(pair_overlap)} "
      f"({len(pair_overlap) / len(and_pairs):.1%} of AND pairs, "
      f"{len(pair_overlap) / len(or_pairs):.1%} of OR pairs)")

and_cities = set(meta.loc[meta.connective == "and", "city_a"]) | set(meta.loc[meta.connective == "and", "city_b"])
or_cities = set(meta.loc[meta.connective == "or", "city_a"]) | set(meta.loc[meta.connective == "or", "city_b"])
city_overlap = and_cities & or_cities
print(f"  (for context, NOT the same check) individual cities used in both subsets: "
      f"{len(city_overlap)}/{len(and_cities)} AND cities, {len(city_overlap)}/{len(or_cities)} OR cities -- "
      "expected, since both subsets draw conjuncts from the same underlying single-city statement pool.")
print()

# ---------------------------------------------------------------------------
# base rates and majority baselines
# ---------------------------------------------------------------------------

and_idx = np.flatnonzero((meta["connective"] == "and").to_numpy())
or_idx = np.flatnonzero((meta["connective"] == "or").to_numpy())
and_base_rate = labels[and_idx].mean()
or_base_rate = labels[or_idx].mean()
print(f"AND subset base rate (label=1): {and_base_rate:.4f}  "
      f"-> majority baseline (as a test set) = {max(and_base_rate, 1 - and_base_rate):.4f} "
      f"(predict {'true' if and_base_rate > 0.5 else 'false'} always)")
print(f"OR subset base rate (label=1):  {or_base_rate:.4f}  "
      f"-> majority baseline (as a test set) = {max(or_base_rate, 1 - or_base_rate):.4f} "
      f"(predict {'true' if or_base_rate > 0.5 else 'false'} always)")
print()

group_key_compound = meta["pair_key"].to_numpy()

# ---------------------------------------------------------------------------
# cross-connective transfer, both directions, 5 seeds x every layer
# ---------------------------------------------------------------------------

metrics_rows = []
pattern_rows = []

for source_name, target_name, source_idx, target_idx in [
    ("and", "or", and_idx, or_idx),
    ("or", "and", or_idx, and_idx),
]:
    direction = f"{source_name}->{target_name}"
    source_groups = group_key_compound[source_idx]
    target_labels = labels[target_idx]

    auroc_by_seed = np.zeros((len(SEEDS), n_layers))
    bal_acc_by_seed = np.zeros((len(SEEDS), n_layers))
    pattern_z_by_seed = {pat: np.zeros((len(SEEDS), n_layers)) for pat in PATTERNS}

    for si, seed in enumerate(SEEDS):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_local, _ = next(gss.split(np.arange(len(source_idx)), groups=source_groups))
        train_global = source_idx[train_local]
        train_labels = labels[train_global]

        for L in range(n_layers):
            probe = LogisticRegression(max_iter=2000, C=0.1)
            probe.fit(acts_compound[train_global, L], train_labels)
            assert list(probe.classes_) == [0, 1]

            target_scores = probe.decision_function(acts_compound[target_idx, L])
            auroc_by_seed[si, L] = roc_auc_score(target_labels, target_scores)
            preds = probe.predict(acts_compound[target_idx, L])
            bal_acc_by_seed[si, L] = balanced_accuracy_score(target_labels, preds)

            # z-score against the probe's OWN training-subsample distribution
            ref_scores = probe.decision_function(acts_compound[train_global, L])
            ref_mean, ref_std = ref_scores.mean(), ref_scores.std()

            for pat in PATTERNS:
                pat_mask = (meta.loc[target_idx, "pattern"] == pat).to_numpy()
                pat_global_idx = target_idx[pat_mask]
                raw = probe.decision_function(acts_compound[pat_global_idx, L])
                pattern_z_by_seed[pat][si, L] = ((raw - ref_mean) / ref_std).mean()

    auroc_mean, auroc_lo, auroc_hi = seed_mean_ci(auroc_by_seed)
    bal_mean, bal_lo, bal_hi = seed_mean_ci(bal_acc_by_seed)
    for L in range(n_layers):
        metrics_rows.append(
            {
                "direction": direction, "layer": L,
                "auroc_mean": auroc_mean[L], "auroc_ci_lo": auroc_lo[L], "auroc_ci_hi": auroc_hi[L],
                "balanced_acc_mean": bal_mean[L], "balanced_acc_ci_lo": bal_lo[L], "balanced_acc_ci_hi": bal_hi[L],
            }
        )

    for pat in PATTERNS:
        z_mean, z_lo, z_hi = seed_mean_ci(pattern_z_by_seed[pat])
        for L in range(n_layers):
            pattern_rows.append(
                {
                    "direction": direction, "layer": L, "pattern": pat,
                    "mean_z_mean": z_mean[L], "mean_z_ci_lo": z_lo[L], "mean_z_ci_hi": z_hi[L],
                }
            )

    print(f"--- {direction} ({len(SEEDS)} seeds, group-level 80% subsamples of {source_name}) ---")
    for L in range(n_layers):
        print(f"layer {L}: AUROC={auroc_mean[L]:.4f} [{auroc_lo[L]:.4f},{auroc_hi[L]:.4f}]  "
              f"balanced_acc={bal_mean[L]:.4f} [{bal_lo[L]:.4f},{bal_hi[L]:.4f}]")
    print()

metrics_df = pd.DataFrame(metrics_rows)
pattern_df = pd.DataFrame(pattern_rows)

results_dir = "results/compound_cities"
os.makedirs(results_dir, exist_ok=True)
metrics_df.to_csv(f"{results_dir}/cross_connective_transfer_metrics.csv", index=False)
pattern_df.to_csv(f"{results_dir}/cross_connective_pattern_breakdown.csv", index=False)
print(f"saved {results_dir}/cross_connective_transfer_metrics.csv and "
      f"{results_dir}/cross_connective_pattern_breakdown.csv")

# ---------------------------------------------------------------------------
# plot 1: aggregate AUROC + balanced accuracy, both directions
# ---------------------------------------------------------------------------

fig, (ax_auc, ax_bal) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
for direction, color in DIRECTION_COLORS.items():
    sub = metrics_df[metrics_df["direction"] == direction]
    ax_auc.plot(sub["layer"], sub["auroc_mean"], color=color, marker="o", markersize=4, label=direction)
    ax_auc.fill_between(sub["layer"], sub["auroc_ci_lo"], sub["auroc_ci_hi"], color=color, alpha=0.2, linewidth=0)
    ax_bal.plot(sub["layer"], sub["balanced_acc_mean"], color=color, marker="o", markersize=4, label=direction)
    ax_bal.fill_between(sub["layer"], sub["balanced_acc_ci_lo"], sub["balanced_acc_ci_hi"], color=color, alpha=0.2, linewidth=0)

ax_auc.axhline(0.5, linestyle="--", color="gray", label="chance")
ax_auc.set_ylabel("AUROC")
ax_auc.set_ylim(0, 1)
ax_auc.set_title(f"cross-connective transfer, aggregate metrics ({MODEL_NAME})")
ax_auc.grid(alpha=0.3)
ax_auc.legend(fontsize=8)

ax_bal.axhline(0.5, linestyle="--", color="gray", label="chance")
ax_bal.set_xlabel("layer")
ax_bal.set_ylabel("balanced accuracy")
ax_bal.set_ylim(0, 1)
ax_bal.grid(alpha=0.3)
ax_bal.legend(fontsize=8)

fig.tight_layout()
os.makedirs("figures/compound_cities", exist_ok=True)
out_path = "figures/compound_cities/cross_connective_transfer_metrics.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

# ---------------------------------------------------------------------------
# plot 2 (MAIN OUTPUT): per-pattern decision_function breakdown, one panel
# per transfer direction, four pattern lines each
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for ax, direction in zip(axes, DIRECTION_COLORS):
    sub = pattern_df[pattern_df["direction"] == direction]
    for pat in PATTERNS:
        pat_sub = sub[sub["pattern"] == pat]
        ax.plot(pat_sub["layer"], pat_sub["mean_z_mean"], color=PATTERN_COLORS[pat], marker="o", markersize=3, label=pat)
        ax.fill_between(
            pat_sub["layer"], pat_sub["mean_z_ci_lo"], pat_sub["mean_z_ci_hi"],
            color=PATTERN_COLORS[pat], alpha=0.2, linewidth=0,
        )
    ax.axhline(0, linestyle=":", color="gray")
    ax.set_xlabel("layer")
    ax.set_title(direction)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("mean decision_function (z-scored vs. source probe's own training distribution)")
fig.suptitle(f"cross-connective per-pattern breakdown ({MODEL_NAME})")
fig.tight_layout()
out_path = "figures/compound_cities/cross_connective_pattern_breakdown.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
