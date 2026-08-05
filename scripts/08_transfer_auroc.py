"""AUROC alongside accuracy for every cross-dataset evaluation of the
cities-trained probe: neg_cities, sp_en_trans, and compounds under four
label schemes. Reuses fit_probe's full-data-fit convention from
03_transfer_and_negation.py / 07_capability_control.py (no held-out split
on the source, since the target is an entirely different dataset), but
does not import from or modify either script.

Source probe fitting is UNCHANGED. What's new is evaluation-side bootstrap:
with the probe held fixed, resample the evaluation set 1000 times for a
percentile CI on accuracy and AUROC. Point estimates are asserted unchanged
from the last saved run.
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
from sklearn.metrics import roc_auc_score

from src.data import load_activations
from src.probes import fit_probe
from src.stats import assert_unchanged

MODEL_NAME = "Qwen2.5-1.5B"
N_RESAMPLES = 1000
LABEL_SCHEME_COLORS = {
    "pooled": "#2a78d6",
    "and_only": "#e34948",
    "or_only": "#1baf7a",
    "conjunct_count": "#eda100",
}

acts_cities, labels_cities = load_activations("cities")
acts_neg, labels_neg = load_activations("neg_cities")
acts_sp, labels_sp = load_activations("sp_en_trans")
acts_compound, _ = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]

n_layers = acts_cities.shape[1]
for acts, name in [(acts_neg, "neg_cities"), (acts_sp, "sp_en_trans"), (acts_compound, "compound_cities")]:
    assert acts.shape[1] == n_layers, f"{name} has a different layer count than cities"

is_and = (meta["connective"] == "and").to_numpy()
is_or = (meta["connective"] == "or").to_numpy()
compound_label = meta["label"].to_numpy()
# label=1 iff both conjuncts are true, regardless of connective -- tests
# whether the probe just detects "both true" (association-counting) rather
# than the connective-dependent truth value
conjunct_count_label = (meta["pattern"] == "TT").astype(int).to_numpy()

EVALUATIONS = [
    ("neg_cities", "single", acts_neg, labels_neg),
    ("sp_en_trans", "single", acts_sp, labels_sp),
    ("compound_cities", "pooled", acts_compound, compound_label),
    ("compound_cities", "and_only", acts_compound[is_and], compound_label[is_and]),
    ("compound_cities", "or_only", acts_compound[is_or], compound_label[is_or]),
    ("compound_cities", "conjunct_count", acts_compound, conjunct_count_label),
]

print("base rates (fraction label=1):")
for eval_set, scheme, _, y in EVALUATIONS:
    print(f"  {eval_set} / {scheme}: {y.mean():.4f} (n={len(y)})")


def check_sign_convention(probe, X_train, y_train, verbose=False):
    """Confirm positive decision_function corresponds to label=1 on the
    training set, so an inverted AUROC downstream can never be a silent
    bookkeeping error rather than a real finding.
    """
    raw = probe.decision_function(X_train)
    mean_pos = raw[y_train == 1].mean()
    mean_neg = raw[y_train == 0].mean()
    if verbose:
        print(
            f"\nsign convention: positive decision_function -> label=1 (true). "
            f"cities train-set mean score: label=1 {mean_pos:.3f}, label=0 {mean_neg:.3f}"
        )
    assert mean_pos > mean_neg, (
        f"sign convention violated: mean decision_function for label=1 ({mean_pos:.3f}) "
        f"is not greater than for label=0 ({mean_neg:.3f}) -- AUROC below would be inverted"
    )


def bootstrap_eval(probe, X, y, n_resamples=N_RESAMPLES, seed=0):
    """Percentile bootstrap CI for accuracy and AUROC of a FIXED (already
    fitted) probe, resampling the evaluation set with replacement.
    """
    n = len(y)
    rng = np.random.default_rng(seed)
    accs, aucs = np.empty(n_resamples), np.empty(n_resamples)
    for r in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_r = y[idx]
        if len(np.unique(y_r)) < 2:
            accs[r], aucs[r] = np.nan, np.nan
            continue
        accs[r] = (probe.predict(X[idx]) == y_r).mean()
        aucs[r] = roc_auc_score(y_r, probe.decision_function(X[idx]))
    return (
        np.nanpercentile(accs, 2.5), np.nanpercentile(accs, 97.5),
        np.nanpercentile(aucs, 2.5), np.nanpercentile(aucs, 97.5),
    )


rows = []
for L in range(n_layers):
    probe = fit_probe(acts_cities[:, L], labels_cities)
    check_sign_convention(probe, acts_cities[:, L], labels_cities, verbose=(L == 0))

    for eval_set, scheme, X, y in EVALUATIONS:
        X_L = X[:, L]
        accuracy = probe.score(X_L, y)
        auroc = roc_auc_score(y, probe.decision_function(X_L))
        acc_lo, acc_hi, auc_lo, auc_hi = bootstrap_eval(probe, X_L, y, seed=L)
        rows.append(
            {
                "layer": L,
                "train_set": "cities",
                "eval_set": eval_set,
                "label_scheme": scheme,
                "accuracy": accuracy,
                "accuracy_ci_lo": acc_lo,
                "accuracy_ci_hi": acc_hi,
                "auroc": auroc,
                "auroc_ci_lo": auc_lo,
                "auroc_ci_hi": auc_hi,
                "n": len(y),
                "base_rate": y.mean(),
            }
        )

df = pd.DataFrame(rows)

# point-estimate regression check against the last saved run, if one exists
out_csv = "results/transfer_auroc.csv"
if os.path.exists(out_csv):
    old = pd.read_csv(out_csv)
    old_key = old.set_index(["layer", "eval_set", "label_scheme"])
    new_key = df.set_index(["layer", "eval_set", "label_scheme"])
    old_aligned = old_key.loc[new_key.index]
    assert_unchanged("transfer_auroc.accuracy", old_aligned["accuracy"].to_numpy(), new_key["accuracy"].to_numpy())
    assert_unchanged("transfer_auroc.auroc", old_aligned["auroc"].to_numpy(), new_key["auroc"].to_numpy())
    print("verified: accuracy/AUROC point estimates unchanged from the last saved run")

os.makedirs("results", exist_ok=True)
df.to_csv(out_csv, index=False)
print(f"\nsaved {out_csv}")
print(df)

os.makedirs("figures", exist_ok=True)
compound_df = df[df["eval_set"] == "compound_cities"]

plt.figure(figsize=(9, 6))
for scheme, color in LABEL_SCHEME_COLORS.items():
    sub = compound_df[compound_df["label_scheme"] == scheme]
    plt.plot(sub["layer"], sub["accuracy"], color=color, linestyle="-", marker="o", label=f"{scheme} accuracy")
    plt.fill_between(sub["layer"], sub["accuracy_ci_lo"], sub["accuracy_ci_hi"], color=color, alpha=0.15, linewidth=0)
    plt.plot(sub["layer"], sub["auroc"], color=color, linestyle="--", marker="o", label=f"{scheme} AUROC")
    plt.fill_between(sub["layer"], sub["auroc_ci_lo"], sub["auroc_ci_hi"], color=color, alpha=0.15, linewidth=0)
plt.axhline(0.5, color="gray", linestyle=":", label="chance")
plt.xlabel("layer")
plt.ylabel("score")
plt.title(f"cities-trained probe on compounds: accuracy vs AUROC ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()

out_path = "figures/transfer_auroc_compound.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
