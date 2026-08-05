"""Part A control: do the small conjunct-regression coefficients (b1, b2 <<
1) reflect genuine dilution of the conjunct signal, or just the fact that
compound scores span a narrower range than cities scores in the same
(cities-z) units?

Reuses 06_conjunct_regression.py's EXACT probe-fitting convention
(train_layer_probe on split_indices(len(labels_cities)), the plain
unseeded 80/20 split -- the one place this project still uses that
convention instead of fit_probe) and its exact z-scoring/regression setup
for variant (i). Does not import from or modify 06.

compression_factor(L) = std(compound raw decision_function) /
std(cities raw decision_function) -- the compound score distribution's
standard deviation, expressed in cities-distribution units. <1 means
compound scores are compressed relative to the atomic scale that
conjunct-regression variant (i) standardizes everything against.

Variant (i) [current]: score_a, score_b, compound_z all standardized
against the CITIES probe's own score distribution (mean/std over all of
cities) -- CLAUDE.md's stated convention, exactly reproducing 06.
Variant (ii): score_a/score_b left UNCHANGED (cities scale -- they're
genuinely cities-domain readings, and rescaling every variable by the same
common factor would leave OLS slopes provably unchanged: y=k*y'+c with
X untouched moves the whole regression by a global affine map that only
shifts the intercept, so trying to standardize predictors AND target by
the same compound-side reference is a no-op on b1/b2 -- verified: an
earlier draft of this script did exactly that and produced byte-identical
(i)/(ii) coefficients). The one thing that changes is compound_z, which
gets de-compressed to the compound distribution's own unit variance
(mean 0, std 1 by construction). Since only the target is rescaled, this
gives an exact, interpretable relationship: b_ii = b_i / compression_factor.
If b1_ii + b2_ii lands near 1, the small (i) coefficients were basically
just compression; if it stays well below 1 even after de-compressing the
output, the dilution is real, independent of the output's range.
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
B1_COLOR = "#2a78d6"
B2_COLOR = "#008300"
COMPRESSION_COLOR = "#4a3aa7"
DISPLACEMENT_COLOR = "#eb6834"
DEFICIT_COLOR = "#e34948"
VARIANT_STYLE = {"i": "-", "ii": "--"}

acts_cities, labels_cities = load_activations("cities")
acts_compound, _ = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]

cities_statements = load_statements("cities", datasets_dir="geometry-of-truth/datasets")["statement"]
assert len(cities_statements) == acts_cities.shape[0]
statement_to_idx = {s: i for i, s in enumerate(cities_statements)}
conj_a_idx = meta["conj_a"].map(statement_to_idx).to_numpy()
conj_b_idx = meta["conj_b"].map(statement_to_idx).to_numpy()

is_or = (meta["connective"] == "or").to_numpy()

n_layers = acts_cities.shape[1]
n_compound = len(meta)

# EXACT same split as 06, unseeded
train_idx, test_idx = split_indices(len(labels_cities))

rows = []
b1_i_point, b2_i_point = np.zeros(n_layers), np.zeros(n_layers)
b1_ii_point, b2_ii_point = np.zeros(n_layers), np.zeros(n_layers)
b1_i_samples = np.zeros((N_RESAMPLES, n_layers))
b2_i_samples = np.zeros((N_RESAMPLES, n_layers))
b1_ii_samples = np.zeros((N_RESAMPLES, n_layers))
b2_ii_samples = np.zeros((N_RESAMPLES, n_layers))
compression_samples = np.zeros((N_RESAMPLES, n_layers))

for L in range(n_layers):
    probe, _ = train_layer_probe(acts_cities[:, L], labels_cities, train_idx, test_idx)
    s_all = probe.decision_function(acts_cities[:, L])
    raw_compound = probe.decision_function(acts_compound[:, L])
    cities_mean, cities_std = s_all.mean(), s_all.std()
    compound_mean, compound_std = raw_compound.mean(), raw_compound.std()
    compression_factor = compound_std / cities_std

    raw_score_a = s_all[conj_a_idx]
    raw_score_b = s_all[conj_b_idx]

    # variant (i): cities scale (reproduces 06 exactly)
    score_a_i = (raw_score_a - cities_mean) / cities_std
    score_b_i = (raw_score_b - cities_mean) / cities_std
    compound_z_i = (raw_compound - cities_mean) / cities_std
    X_i = np.column_stack([score_a_i, score_b_i])
    reg_i = LinearRegression().fit(X_i, compound_z_i)
    b1_i_point[L], b2_i_point[L] = reg_i.coef_
    r2_i = reg_i.score(X_i, compound_z_i)

    # variant (ii): only the target de-compressed to the compound's own
    # scale -- score_a/score_b stay on the cities scale (see module docstring
    # for why rescaling every variable by the same factor would be a no-op)
    compound_z_ii = (raw_compound - compound_mean) / compound_std
    X_ii = X_i  # score_a_i, score_b_i, unchanged
    reg_ii = LinearRegression().fit(X_ii, compound_z_ii)
    b1_ii_point[L], b2_ii_point[L] = reg_ii.coef_
    r2_ii = reg_ii.score(X_ii, compound_z_ii)

    # threshold displacement: or_only's mean raw score relative to the zero
    # decision boundary, in cities-SD units -- the calibration bias that
    # biases predict() toward "false" for a majority-true set
    or_displacement = raw_compound[is_or].mean() / cities_std

    rows.append(
        {
            "layer": L,
            "compression_factor": compression_factor,
            "cities_std": cities_std, "compound_std": compound_std,
            "b1_i": b1_i_point[L], "b2_i": b2_i_point[L], "r2_i": r2_i,
            "b1_ii": b1_ii_point[L], "b2_ii": b2_ii_point[L], "r2_ii": r2_ii,
            "or_only_threshold_displacement": or_displacement,
        }
    )

    # --- bootstrap: shared resample of compound rows per round, both variants ---
    rng = np.random.default_rng(L)
    for r in range(N_RESAMPLES):
        sel = rng.integers(0, n_compound, size=n_compound)
        raw_compound_r = raw_compound[sel]
        score_a_i_r, score_b_i_r = score_a_i[sel], score_b_i[sel]
        compound_z_i_r = (raw_compound_r - cities_mean) / cities_std
        reg = LinearRegression().fit(np.column_stack([score_a_i_r, score_b_i_r]), compound_z_i_r)
        b1_i_samples[r, L], b2_i_samples[r, L] = reg.coef_

        compound_mean_r, compound_std_r = raw_compound_r.mean(), raw_compound_r.std()
        compound_z_ii_r = (raw_compound_r - compound_mean_r) / compound_std_r
        reg2 = LinearRegression().fit(np.column_stack([score_a_i_r, score_b_i_r]), compound_z_ii_r)
        b1_ii_samples[r, L], b2_ii_samples[r, L] = reg2.coef_

        compression_samples[r, L] = compound_std_r / cities_std

df = pd.DataFrame(rows)

# point-estimate regression check: variant (i) must exactly reproduce 06's
# already-saved pooled b1/b2/r2
old_path = "results/compound_cities/conjunct_regression.csv"
if os.path.exists(old_path):
    old = pd.read_csv(old_path)
    assert_unchanged("attenuation_vs_compression.b1_i", old["b1"].to_numpy(), df["b1_i"].to_numpy())
    assert_unchanged("attenuation_vs_compression.b2_i", old["b2"].to_numpy(), df["b2_i"].to_numpy())
    assert_unchanged("attenuation_vs_compression.r2_i", old["r2"].to_numpy(), df["r2_i"].to_numpy())
    print("verified: variant (i) b1/b2/r2 exactly reproduce 06_conjunct_regression.py's saved output\n")

# CI columns
for name, samples in [
    ("b1_i", b1_i_samples), ("b2_i", b2_i_samples),
    ("b1_ii", b1_ii_samples), ("b2_ii", b2_ii_samples),
    ("compression_factor", compression_samples),
]:
    df[f"{name}_ci_lo"] = np.percentile(samples, 2.5, axis=0)
    df[f"{name}_ci_hi"] = np.percentile(samples, 97.5, axis=0)

# threshold-displacement vs. compression-factor / or_only accuracy deficit
transfer_auroc_path = "results/transfer_auroc.csv"
cities_or = pd.read_csv(transfer_auroc_path)
cities_or = cities_or[
    (cities_or["train_set"] == "cities") & (cities_or["eval_set"] == "compound_cities")
    & (cities_or["label_scheme"] == "or_only")
].sort_values("layer")
assert (cities_or["layer"].to_numpy() == df["layer"].to_numpy()).all()
df["or_only_majority_baseline"] = np.maximum(cities_or["base_rate"].to_numpy(), 1 - cities_or["base_rate"].to_numpy())
df["or_only_accuracy"] = cities_or["accuracy"].to_numpy()
df["or_only_accuracy_deficit"] = df["or_only_majority_baseline"] - df["or_only_accuracy"]

corr_compression_deficit = np.corrcoef(df["compression_factor"], df["or_only_accuracy_deficit"])[0, 1]
corr_compression_displacement = np.corrcoef(df["compression_factor"], df["or_only_threshold_displacement"].abs())[0, 1]
corr_displacement_deficit = np.corrcoef(df["or_only_threshold_displacement"].abs(), df["or_only_accuracy_deficit"])[0, 1]

results_dir = "results/compound_cities"
os.makedirs(results_dir, exist_ok=True)
df.to_csv(f"{results_dir}/attenuation_vs_compression.csv", index=False)
print(f"saved {results_dir}/attenuation_vs_compression.csv\n")

print("compression factor, both regression variants, per layer:")
for _, row in df.iterrows():
    L = int(row["layer"])
    print(
        f"layer {L}: compression={row['compression_factor']:.4f} "
        f"[{row['compression_factor_ci_lo']:.4f},{row['compression_factor_ci_hi']:.4f}]  "
        f"b1_i={row['b1_i']:.4f} [{row['b1_i_ci_lo']:.4f},{row['b1_i_ci_hi']:.4f}]  "
        f"b2_i={row['b2_i']:.4f} [{row['b2_i_ci_lo']:.4f},{row['b2_i_ci_hi']:.4f}]  "
        f"b1_ii={row['b1_ii']:.4f} [{row['b1_ii_ci_lo']:.4f},{row['b1_ii_ci_hi']:.4f}]  "
        f"b2_ii={row['b2_ii']:.4f} [{row['b2_ii_ci_lo']:.4f},{row['b2_ii_ci_hi']:.4f}]"
    )
print()
print(f"corr(compression_factor, or_only accuracy deficit) = {corr_compression_deficit:.4f}")
print(f"corr(compression_factor, |or_only threshold displacement|) = {corr_compression_displacement:.4f}")
print(f"corr(|or_only threshold displacement|, or_only accuracy deficit) = {corr_displacement_deficit:.4f}")
print()

# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

os.makedirs("figures/compound_cities", exist_ok=True)

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
ax_top.plot(df["layer"], df["compression_factor"], color=COMPRESSION_COLOR, marker="o", markersize=4)
ax_top.fill_between(df["layer"], df["compression_factor_ci_lo"], df["compression_factor_ci_hi"], color=COMPRESSION_COLOR, alpha=0.2, linewidth=0)
ax_top.axhline(1.0, linestyle=":", color="gray", label="no compression (compound SD = cities SD)")
ax_top.set_ylabel("compression factor\n(compound SD / cities SD)")
ax_top.set_title(f"compound score compression vs. conjunct-regression coefficients ({MODEL_NAME})")
ax_top.grid(alpha=0.3)
ax_top.legend(fontsize=8)

for name, color in [("b1", B1_COLOR), ("b2", B2_COLOR)]:
    for variant in ["i", "ii"]:
        col = f"{name}_{variant}"
        ax_bot.plot(df["layer"], df[col], color=color, linestyle=VARIANT_STYLE[variant], marker="o", markersize=3,
                    label=f"{name}, variant ({variant})")
        ax_bot.fill_between(df["layer"], df[f"{col}_ci_lo"], df[f"{col}_ci_hi"], color=color, alpha=0.15, linewidth=0)
ax_bot.axhline(0, linestyle=":", color="gray")
ax_bot.set_xlabel("layer")
ax_bot.set_ylabel("regression coefficient")
ax_bot.grid(alpha=0.3)
ax_bot.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
fig.tight_layout()
out_path = "figures/compound_cities/attenuation_vs_compression.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
axes[0].plot(df["layer"], df["compression_factor"], color=COMPRESSION_COLOR, marker="o", markersize=4)
axes[0].fill_between(df["layer"], df["compression_factor_ci_lo"], df["compression_factor_ci_hi"], color=COMPRESSION_COLOR, alpha=0.2, linewidth=0)
axes[0].set_ylabel("compression factor")
axes[0].set_title(
    f"compression factor vs. or_only threshold displacement vs. accuracy deficit ({MODEL_NAME})\n"
    f"corr(compression, deficit)={corr_compression_deficit:.3f}  "
    f"corr(compression, |displacement|)={corr_compression_displacement:.3f}  "
    f"corr(|displacement|, deficit)={corr_displacement_deficit:.3f}",
    fontsize=9,
)
axes[0].grid(alpha=0.3)

axes[1].plot(df["layer"], df["or_only_threshold_displacement"], color=DISPLACEMENT_COLOR, marker="o", markersize=4)
axes[1].axhline(0, linestyle=":", color="gray")
axes[1].set_ylabel("or_only threshold\ndisplacement (cities SD)")
axes[1].grid(alpha=0.3)

axes[2].plot(df["layer"], df["or_only_accuracy_deficit"], color=DEFICIT_COLOR, marker="o", markersize=4)
axes[2].axhline(0, linestyle=":", color="gray")
axes[2].set_ylabel("or_only accuracy deficit\n(majority baseline - accuracy)")
axes[2].set_xlabel("layer")
axes[2].grid(alpha=0.3)

fig.tight_layout()
out_path = "figures/compound_cities/compression_vs_threshold_displacement.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
