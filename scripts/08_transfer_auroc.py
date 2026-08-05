"""AUROC alongside accuracy for every cross-dataset evaluation of the
cities-trained probe: neg_cities, sp_en_trans, and compounds under four
label schemes. Reuses fit_probe's full-data-fit convention from
03_transfer_and_negation.py / 07_capability_control.py (no held-out split
on the source, since the target is an entirely different dataset), but
does not import from or modify either script.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import load_activations
from src.probes import fit_probe

MODEL_NAME = "Qwen2.5-1.5B"
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


rows = []
for L in range(n_layers):
    probe = fit_probe(acts_cities[:, L], labels_cities)
    check_sign_convention(probe, acts_cities[:, L], labels_cities, verbose=(L == 0))

    for eval_set, scheme, X, y in EVALUATIONS:
        X_L = X[:, L]
        accuracy = probe.score(X_L, y)
        auroc = roc_auc_score(y, probe.decision_function(X_L))
        rows.append(
            {
                "layer": L,
                "train_set": "cities",
                "eval_set": eval_set,
                "label_scheme": scheme,
                "accuracy": accuracy,
                "auroc": auroc,
                "n": len(y),
                "base_rate": y.mean(),
            }
        )

df = pd.DataFrame(rows)

os.makedirs("results", exist_ok=True)
out_csv = "results/transfer_auroc.csv"
df.to_csv(out_csv, index=False)
print(f"\nsaved {out_csv}")
print(df)

os.makedirs("figures", exist_ok=True)
compound_df = df[df["eval_set"] == "compound_cities"]

plt.figure(figsize=(9, 6))
for scheme, color in LABEL_SCHEME_COLORS.items():
    sub = compound_df[compound_df["label_scheme"] == scheme]
    plt.plot(sub["layer"], sub["accuracy"], color=color, linestyle="-", marker="o", label=f"{scheme} accuracy")
    plt.plot(sub["layer"], sub["auroc"], color=color, linestyle="--", marker="o", label=f"{scheme} AUROC")
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
