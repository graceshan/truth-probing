"""Conjunct-score regression: does the compound statement's probe score
just linearly combine the two conjuncts' individually-probed truth scores?

Self-contained per-layer analysis. Reuses src.probes for the fit (same
train/test split convention as the rest of the project -- UNCHANGED, still
train_layer_probe on cities' 80/20 split, not fit_probe like 03/05/08) but
does not import from or modify scripts/05_compound_analysis.py.

Regression coefficients are a fitted-model property, so their bootstrap is
different from the plain resample-and-score approach elsewhere: each
resample draws compound rows (with replacement, preserving the
(score_a, score_b, compound_z) triples together) and REFITS the linear
regression, standard practice for regression coefficient CIs. Point
estimates (b1, b2, r2) are asserted unchanged from the last saved run.
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
from sklearn.linear_model import LinearRegression

from src.data import load_activations, load_statements
from src.probes import split_indices, train_layer_probe
from src.stats import assert_unchanged

MODEL_NAME = "Qwen2.5-1.5B"
N_RESAMPLES = 1000
B1_COLOR = "#2a78d6"  # conjunct a
B2_COLOR = "#008300"  # conjunct b
CONNECTIVE_COLORS = {"and": "#e34948", "or": "#1baf7a"}
COEF_STYLE = {"b1": "-", "b2": "--"}
COEF_MARKER = {"b1": "o", "b2": "s"}
PLOT_START_LAYER = 5  # layers 0-4: cities probe hasn't formed yet, numbers unstable

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


def bootstrap_regression(score_a, score_b, target, mask, n_resamples=N_RESAMPLES, seed=0):
    """Pairs bootstrap: resample compound rows with replacement, refit,
    collect (b1, b2, r2) each time. Percentile interval across resamples.
    """
    idx_pool = np.flatnonzero(mask)
    n = len(idx_pool)
    rng = np.random.default_rng(seed)
    b1s, b2s, r2s = np.empty(n_resamples), np.empty(n_resamples), np.empty(n_resamples)
    for r in range(n_resamples):
        sel = idx_pool[rng.integers(0, n, size=n)]
        X = np.column_stack([score_a[sel], score_b[sel]])
        y = target[sel]
        reg = LinearRegression().fit(X, y)
        b1s[r], b2s[r] = reg.coef_
        r2s[r] = reg.score(X, y)

    def pct(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    return pct(b1s), pct(b2s), pct(r2s)


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
    (b1_lo, b1_hi), (b2_lo, b2_hi), (r2_lo, r2_hi) = bootstrap_regression(
        score_a, score_b, compound_z, all_mask, seed=L
    )
    all_rows.append(
        {
            "layer": L, "b1": b1, "b1_ci_lo": b1_lo, "b1_ci_hi": b1_hi,
            "b2": b2, "b2_ci_lo": b2_lo, "b2_ci_hi": b2_hi,
            "intercept": intercept, "r2": r2, "r2_ci_lo": r2_lo, "r2_ci_hi": r2_hi,
        }
    )

    b1a, b2a, ia, r2a = fit_regression(score_a, score_b, compound_z, is_and)
    (b1a_lo, b1a_hi), (b2a_lo, b2a_hi), (r2a_lo, r2a_hi) = bootstrap_regression(
        score_a, score_b, compound_z, is_and, seed=L
    )
    and_rows.append(
        {
            "layer": L, "connective": "and", "b1": b1a, "b1_ci_lo": b1a_lo, "b1_ci_hi": b1a_hi,
            "b2": b2a, "b2_ci_lo": b2a_lo, "b2_ci_hi": b2a_hi,
            "intercept": ia, "r2": r2a, "r2_ci_lo": r2a_lo, "r2_ci_hi": r2a_hi,
        }
    )

    b1o, b2o, io, r2o = fit_regression(score_a, score_b, compound_z, is_or)
    (b1o_lo, b1o_hi), (b2o_lo, b2o_hi), (r2o_lo, r2o_hi) = bootstrap_regression(
        score_a, score_b, compound_z, is_or, seed=L
    )
    or_rows.append(
        {
            "layer": L, "connective": "or", "b1": b1o, "b1_ci_lo": b1o_lo, "b1_ci_hi": b1o_hi,
            "b2": b2o, "b2_ci_lo": b2o_lo, "b2_ci_hi": b2o_hi,
            "intercept": io, "r2": r2o, "r2_ci_lo": r2o_lo, "r2_ci_hi": r2o_hi,
        }
    )

all_df = pd.DataFrame(all_rows)
by_conn_df = pd.DataFrame(and_rows + or_rows)
by_conn_df["b2_over_b1"] = by_conn_df["b2"] / by_conn_df["b1"]

# point-estimate regression check against the last saved run, if one exists
all_path = "results/compound_cities/conjunct_regression.csv"
by_conn_path = "results/compound_cities/conjunct_regression_by_connective.csv"
if os.path.exists(all_path):
    old = pd.read_csv(all_path)
    for col in ["b1", "b2", "intercept", "r2"]:
        assert_unchanged(f"conjunct_regression.{col}", old[col].to_numpy(), all_df[col].to_numpy())
    print("verified: pooled b1/b2/intercept/r2 unchanged from the last saved run")
if os.path.exists(by_conn_path):
    old = pd.read_csv(by_conn_path)
    for col in ["b1", "b2", "intercept", "r2"]:
        assert_unchanged(f"conjunct_regression_by_connective.{col}", old[col].to_numpy(), by_conn_df[col].to_numpy())
    print("verified: by-connective b1/b2/intercept/r2 unchanged from the last saved run")

os.makedirs("results/compound_cities", exist_ok=True)
all_df.to_csv(all_path, index=False)
by_conn_df.to_csv(by_conn_path, index=False)
print(all_df)
print(f"saved {all_path} and {by_conn_path}")

# mean b2/b1 ratio per connective, over layers -- the direct numeric answer
# to "does the weighting differ by connective"
ratio_summary = by_conn_df.groupby("connective")["b2_over_b1"].mean()
print("\nmean b2/b1 ratio by connective (across all layers):")
print(ratio_summary)

os.makedirs("figures/compound_cities", exist_ok=True)

# plots start at PLOT_START_LAYER (the full 0-27 range is still in the CSVs
# above) -- early layers are pre-signal and just add unstable noise to the
# view
plot_df = all_df[all_df["layer"] >= PLOT_START_LAYER]
plot_by_conn_df = by_conn_df[by_conn_df["layer"] >= PLOT_START_LAYER]

plt.figure(figsize=(9, 6))
plt.plot(plot_df["layer"], plot_df["b1"], color=B1_COLOR, marker="o", label="b1 (conjunct a)")
plt.fill_between(plot_df["layer"], plot_df["b1_ci_lo"], plot_df["b1_ci_hi"], color=B1_COLOR, alpha=0.2, linewidth=0)
plt.plot(plot_df["layer"], plot_df["b2"], color=B2_COLOR, marker="o", label="b2 (conjunct b)")
plt.fill_between(plot_df["layer"], plot_df["b2_ci_lo"], plot_df["b2_ci_hi"], color=B2_COLOR, alpha=0.2, linewidth=0)
plt.axhline(0, linestyle="--", color="gray", label="zero")
plt.xlabel("layer")
plt.ylabel("regression coefficient")
plt.title(f"conjunct-score regression coefficients ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()
out_path = "figures/compound_cities/conjunct_regression_coefficients.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

plt.figure(figsize=(9, 6))
plt.plot(plot_df["layer"], plot_df["r2"], color=B1_COLOR, marker="o")
plt.fill_between(plot_df["layer"], plot_df["r2_ci_lo"], plot_df["r2_ci_hi"], color=B1_COLOR, alpha=0.2, linewidth=0)
plt.xlabel("layer")
plt.ylabel("R^2")
plt.title(f"conjunct-score regression fit ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.tight_layout()
out_path = "figures/compound_cities/conjunct_regression_r2.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

plt.figure(figsize=(9, 6))
for conn, color in CONNECTIVE_COLORS.items():
    sub = plot_by_conn_df[plot_by_conn_df["connective"] == conn]
    plt.plot(
        sub["layer"], sub["b1"], color=color, linestyle=COEF_STYLE["b1"], marker=COEF_MARKER["b1"],
        label=f"b1, {conn}",
    )
    plt.fill_between(sub["layer"], sub["b1_ci_lo"], sub["b1_ci_hi"], color=color, alpha=0.15, linewidth=0)
    plt.plot(
        sub["layer"], sub["b2"], color=color, linestyle=COEF_STYLE["b2"], marker=COEF_MARKER["b2"],
        label=f"b2, {conn}",
    )
    plt.fill_between(sub["layer"], sub["b2_ci_lo"], sub["b2_ci_hi"], color=color, alpha=0.15, linewidth=0)
plt.axhline(0, linestyle=":", color="gray", label="zero")
plt.xlabel("layer")
plt.ylabel("regression coefficient")
plt.title(f"conjunct-score regression coefficients, by connective ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()
out_path = "figures/compound_cities/conjunct_regression_by_connective.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
