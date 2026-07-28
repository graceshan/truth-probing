"""Conjunct-score regression: does the compound statement's probe score
just linearly combine the two conjuncts' individually-probed truth scores?

Self-contained per-layer analysis. Reuses src.probes for the fit (same
train/test split convention as the rest of the project) but does not
import from or modify scripts/probe_compound_transfer.py.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data import load_activations, load_statements
from src.probes import split_indices, train_layer_probe

MODEL_NAME = "Qwen2.5-1.5B"
B1_COLOR = "#2a78d6"  # conjunct a
B2_COLOR = "#008300"  # conjunct b
CONNECTIVE_COLORS = {"and": "#e34948", "or": "#1baf7a"}
COEF_STYLE = {"b1": "-", "b2": "--"}

acts_cities, labels_cities = load_activations("cities")
acts_compound, _ = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]

cities_statements = load_statements("cities", datasets_dir="geometry-of-truth/datasets")["statement"]
assert len(cities_statements) == acts_cities.shape[0]
statement_to_idx = {s: i for i, s in enumerate(cities_statements)}

missing_a = set(meta["conj_a"]) - set(statement_to_idx)
missing_b = set(meta["conj_b"]) - set(statement_to_idx)
assert not missing_a, f"conj_a statements not found in cities lookup: {missing_a}"
assert not missing_b, f"conj_b statements not found in cities lookup: {missing_b}"

conj_a_idx = meta["conj_a"].map(statement_to_idx).to_numpy()
conj_b_idx = meta["conj_b"].map(statement_to_idx).to_numpy()

is_and = (meta["connective"] == "and").to_numpy()
is_or = (meta["connective"] == "or").to_numpy()

n_layers = acts_cities.shape[1]
layers = range(n_layers)

# split indices once, reuse across every layer -- same convention as the
# rest of the project
train_idx, test_idx = split_indices(len(labels_cities))


def fit_regression(score_a, score_b, target, mask):
    X = np.column_stack([score_a[mask], score_b[mask]])
    y = target[mask]
    reg = LinearRegression().fit(X, y)
    b1, b2 = reg.coef_
    return b1, b2, reg.intercept_, reg.score(X, y)


all_rows, and_rows, or_rows = [], [], []

for L in layers:
    probe, _ = train_layer_probe(acts_cities[:, L], labels_cities, train_idx, test_idx)
    s_all = probe.decision_function(acts_cities[:, L])

    # same z-transform as compound_z below, so b1/b2 are in comparable units
    # across layers instead of shrinking as raw margins grow with depth
    score_a = (s_all[conj_a_idx] - s_all.mean()) / s_all.std()
    score_b = (s_all[conj_b_idx] - s_all.mean()) / s_all.std()

    raw_compound = probe.decision_function(acts_compound[:, L])
    compound_z = (raw_compound - s_all.mean()) / s_all.std()

    all_mask = np.ones(len(meta), dtype=bool)
    b1, b2, intercept, r2 = fit_regression(score_a, score_b, compound_z, all_mask)
    all_rows.append({"layer": L, "b1": b1, "b2": b2, "intercept": intercept, "r2": r2})

    b1a, b2a, ia, r2a = fit_regression(score_a, score_b, compound_z, is_and)
    and_rows.append({"layer": L, "connective": "and", "b1": b1a, "b2": b2a, "intercept": ia, "r2": r2a})

    b1o, b2o, io, r2o = fit_regression(score_a, score_b, compound_z, is_or)
    or_rows.append({"layer": L, "connective": "or", "b1": b1o, "b2": b2o, "intercept": io, "r2": r2o})

all_df = pd.DataFrame(all_rows)
by_conn_df = pd.DataFrame(and_rows + or_rows)
by_conn_df["b2_over_b1"] = by_conn_df["b2"] / by_conn_df["b1"]

os.makedirs("results", exist_ok=True)
all_df.to_csv("results/conjunct_regression.csv", index=False)
by_conn_df.to_csv("results/conjunct_regression_by_connective.csv", index=False)
print(all_df)
print("saved results/conjunct_regression.csv and results/conjunct_regression_by_connective.csv")

# mean b2/b1 ratio per connective, over layers -- the direct numeric answer
# to "does the weighting differ by connective"
ratio_summary = by_conn_df.groupby("connective")["b2_over_b1"].mean()
print("\nmean b2/b1 ratio by connective (across all layers):")
print(ratio_summary)

os.makedirs("figures", exist_ok=True)

plt.figure(figsize=(9, 6))
plt.plot(all_df["layer"], all_df["b1"], color=B1_COLOR, marker="o", label="b1 (conjunct a)")
plt.plot(all_df["layer"], all_df["b2"], color=B2_COLOR, marker="o", label="b2 (conjunct b)")
plt.axhline(0, linestyle="--", color="gray", label="zero")
plt.xlabel("layer")
plt.ylabel("regression coefficient")
plt.title(f"conjunct-score regression coefficients ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()
out_path = "figures/conjunct_regression_coefficients.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

plt.figure(figsize=(9, 6))
plt.plot(all_df["layer"], all_df["r2"], color=B1_COLOR, marker="o")
plt.xlabel("layer")
plt.ylabel("R^2")
plt.title(f"conjunct-score regression fit ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.tight_layout()
out_path = "figures/conjunct_regression_r2.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

plt.figure(figsize=(9, 6))
for conn, color in CONNECTIVE_COLORS.items():
    sub = by_conn_df[by_conn_df["connective"] == conn]
    plt.plot(sub["layer"], sub["b1"], color=color, linestyle=COEF_STYLE["b1"], marker="o", label=f"b1, {conn}")
    plt.plot(sub["layer"], sub["b2"], color=color, linestyle=COEF_STYLE["b2"], marker="o", label=f"b2, {conn}")
plt.axhline(0, linestyle=":", color="gray", label="zero")
plt.xlabel("layer")
plt.ylabel("regression coefficient")
plt.title(f"conjunct-score regression coefficients, by connective ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()
out_path = "figures/conjunct_regression_by_connective.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
