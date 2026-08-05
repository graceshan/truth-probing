"""Does the pooled fresh compound probe (07_capability_control.py's "pooled"
variant: LogisticRegression fit directly on compound activations, not
transferred from cities) actually read the connective, or does it just
count true conjuncts and ignore and/or entirely?

Reuses 07's exact pooled-probe fit (same stratified 80/20 split,
random_state=0, computed once and reused across every layer) -- does not
import from or modify 07_capability_control.py. Scores are z-scored against
the probe's own training-partition distribution, same convention as
05_compound_analysis.py, then described over the FULL compound set (all
1600 statements, not held out) -- a descriptive breakdown of what the fit
probe encodes, not a generalization claim (07 already reports the held-out
accuracy/AUROC for that).

The diagnostic: if the probe only counts true conjuncts (an association
shortcut), TF and FT should score the same regardless of connective. If it
actually reads the connective, TF/FT should score low under AND (mixed ->
false) and high under OR (mixed -> true) -- the "connective offset",
quantified directly and bootstrapped, and compared to the TT-FF spread
within each connective (the scale that makes the offset's magnitude
interpretable). Two nested regressions on the 8 cell means make the same
point formally: M1 (score ~ n_true_conjuncts) is the connective-blind
association-counting model; M2 adds connective as a predictor.
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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

from src.data import load_activations

MODEL_NAME = "Qwen2.5-1.5B"
N_RESAMPLES = 1000
PATTERNS = ["TT", "TF", "FT", "FF"]
CONNECTIVES = ["and", "or"]
N_TRUE_CONJUNCTS = {"TT": 2, "TF": 1, "FT": 1, "FF": 0}
PATTERN_COLORS = {"TT": "#2a78d6", "TF": "#008300", "FT": "#e87ba4", "FF": "#eda100"}
CONNECTIVE_STYLE = {"and": "-", "or": "--"}
OFFSET_COLOR = "#4a3aa7"
SPREAD_COLORS = {"and": "#e34948", "or": "#1baf7a"}
BAND_LAYERS = range(10, 17)  # 10-16 inclusive, per request ("plus layer 15" is already in this band)

acts_compound, labels_compound = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]
n_layers = acts_compound.shape[1]
layers = np.arange(n_layers)

cell_mask = {
    (pat, conn): ((meta["pattern"] == pat) & (meta["connective"] == conn)).to_numpy()
    for pat in PATTERNS for conn in CONNECTIVES
}

# EXACT same split as 07's "pooled" variant: stratified 80/20, random_state=0,
# computed once and reused across every layer
train_idx, test_idx = train_test_split(
    np.arange(len(labels_compound)), test_size=0.2, random_state=0, stratify=labels_compound
)

point_rows = []
offset_rows = []
regression_rows = []
z_by_layer = np.zeros((n_layers, acts_compound.shape[0]))

for L in range(n_layers):
    probe = LogisticRegression(max_iter=2000, C=0.1)
    probe.fit(acts_compound[train_idx, L], labels_compound[train_idx])
    assert list(probe.classes_) == [0, 1]

    ref_scores = probe.decision_function(acts_compound[train_idx, L])
    ref_mean, ref_std = ref_scores.mean(), ref_scores.std()
    raw_all = probe.decision_function(acts_compound[:, L])
    z_all = (raw_all - ref_mean) / ref_std
    z_by_layer[L] = z_all

    cell_mean = {cell: z_all[mask].mean() for cell, mask in cell_mask.items()}
    for (pat, conn), m in cell_mean.items():
        point_rows.append({"layer": L, "pattern": pat, "connective": conn, "mean_z": m})

    connective_offset = (
        (cell_mean[("TF", "or")] + cell_mean[("FT", "or")]) / 2
        - (cell_mean[("TF", "and")] + cell_mean[("FT", "and")]) / 2
    )
    spread_and = cell_mean[("TT", "and")] - cell_mean[("FF", "and")]
    spread_or = cell_mean[("TT", "or")] - cell_mean[("FF", "or")]

    # --- bootstrap: jointly resample all 8 cells (shared resample draws
    # across cells within a resample round, matching 05's bootstrap_connective) ---
    rng = np.random.default_rng(L)
    offset_samples = np.empty(N_RESAMPLES)
    spread_and_samples = np.empty(N_RESAMPLES)
    spread_or_samples = np.empty(N_RESAMPLES)
    cell_samples = {cell: np.empty(N_RESAMPLES) for cell in cell_mask}
    cell_vals = {cell: z_all[mask] for cell, mask in cell_mask.items()}
    for r in range(N_RESAMPLES):
        rs_mean = {}
        for cell, vals in cell_vals.items():
            idx = rng.integers(0, len(vals), size=len(vals))
            rs_mean[cell] = vals[idx].mean()
            cell_samples[cell][r] = rs_mean[cell]
        offset_samples[r] = (
            (rs_mean[("TF", "or")] + rs_mean[("FT", "or")]) / 2 - (rs_mean[("TF", "and")] + rs_mean[("FT", "and")]) / 2
        )
        spread_and_samples[r] = rs_mean[("TT", "and")] - rs_mean[("FF", "and")]
        spread_or_samples[r] = rs_mean[("TT", "or")] - rs_mean[("FF", "or")]

    def pct(arr):
        return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

    offset_lo, offset_hi = pct(offset_samples)
    spread_and_lo, spread_and_hi = pct(spread_and_samples)
    spread_or_lo, spread_or_hi = pct(spread_or_samples)

    offset_rows.append(
        {
            "layer": L,
            "connective_offset": connective_offset, "connective_offset_ci_lo": offset_lo, "connective_offset_ci_hi": offset_hi,
            "spread_and": spread_and, "spread_and_ci_lo": spread_and_lo, "spread_and_ci_hi": spread_and_hi,
            "spread_or": spread_or, "spread_or_ci_lo": spread_or_lo, "spread_or_ci_hi": spread_or_hi,
        }
    )

    # attach per-cell bootstrap CI onto point_rows for this layer
    for (pat, conn) in cell_mask:
        lo, hi = pct(cell_samples[(pat, conn)])
        for row in point_rows[-8:]:
            if row["layer"] == L and row["pattern"] == pat and row["connective"] == conn:
                row["ci_lo"], row["ci_hi"] = lo, hi

    # --- nested regression on the 8 cell means (point estimates, no bootstrap) ---
    X_rows, y_rows = [], []
    for (pat, conn), m in cell_mean.items():
        X_rows.append([N_TRUE_CONJUNCTS[pat], 1.0 if conn == "or" else 0.0])
        y_rows.append(m)
    X_rows, y_rows = np.array(X_rows), np.array(y_rows)

    m1 = LinearRegression().fit(X_rows[:, [0]], y_rows)
    r2_m1 = m1.score(X_rows[:, [0]], y_rows)
    m2 = LinearRegression().fit(X_rows, y_rows)
    r2_m2 = m2.score(X_rows, y_rows)

    regression_rows.append({"layer": L, "r2_m1_conjunct_count_only": r2_m1, "r2_m2_plus_connective": r2_m2, "improvement": r2_m2 - r2_m1})

point_df = pd.DataFrame(point_rows)
offset_df = pd.DataFrame(offset_rows)
regression_df = pd.DataFrame(regression_rows)

results_dir = "results/compound_cities"
os.makedirs(results_dir, exist_ok=True)
point_df.to_csv(f"{results_dir}/pooled_probe_8cell_means.csv", index=False)
offset_df.to_csv(f"{results_dir}/pooled_probe_connective_offset.csv", index=False)
regression_df.to_csv(f"{results_dir}/pooled_probe_nested_regression.csv", index=False)
print(f"saved {results_dir}/pooled_probe_8cell_means.csv, "
      f"pooled_probe_connective_offset.csv, pooled_probe_nested_regression.csv\n")

# ---------------------------------------------------------------------------
# 8-cell table, layers 10-16 (includes layer 15)
# ---------------------------------------------------------------------------

wide = point_df.pivot(index="layer", columns=["connective", "pattern"], values="mean_z")
wide = wide[[(c, p) for c in CONNECTIVES for p in PATTERNS]]
print("8-cell mean decision_function (z-scored), layers 10-16:")
print(wide.loc[list(BAND_LAYERS)].round(4).to_string())
print()

# ---------------------------------------------------------------------------
# connective offset vs. TT-FF spread
# ---------------------------------------------------------------------------

print("connective offset [(TF,OR)+(FT,OR)]/2 - [(TF,AND)+(FT,AND)]/2, vs. TT-FF spread per connective:")
for _, row in offset_df.iterrows():
    L = int(row["layer"])
    print(
        f"layer {L}: offset={row['connective_offset']:.4f} "
        f"[{row['connective_offset_ci_lo']:.4f},{row['connective_offset_ci_hi']:.4f}]  "
        f"spread_and={row['spread_and']:.4f} [{row['spread_and_ci_lo']:.4f},{row['spread_and_ci_hi']:.4f}]  "
        f"spread_or={row['spread_or']:.4f} [{row['spread_or_ci_lo']:.4f},{row['spread_or_ci_hi']:.4f}]  "
        f"|offset|/mean(spread)={abs(row['connective_offset']) / np.mean([row['spread_and'], row['spread_or']]):.3f}"
    )
print()

# ---------------------------------------------------------------------------
# nested regression
# ---------------------------------------------------------------------------

print("nested regression on the 8 cell means, M1: score~n_true_conjuncts vs. M2: score~n_true_conjuncts+connective:")
for _, row in regression_df.iterrows():
    print(f"layer {int(row['layer'])}: R2_M1={row['r2_m1_conjunct_count_only']:.4f}  "
          f"R2_M2={row['r2_m2_plus_connective']:.4f}  improvement={row['improvement']:.4f}")
print()

# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

os.makedirs("figures/compound_cities", exist_ok=True)

# 8-line plot
plt.figure(figsize=(9, 6))
for conn in CONNECTIVES:
    for pat in PATTERNS:
        sub = point_df[(point_df["connective"] == conn) & (point_df["pattern"] == pat)].sort_values("layer")
        plt.plot(
            sub["layer"], sub["mean_z"], color=PATTERN_COLORS[pat], linestyle=CONNECTIVE_STYLE[conn],
            marker="o", markersize=3, label=f"{pat}, {conn}",
        )
        plt.fill_between(sub["layer"], sub["ci_lo"], sub["ci_hi"], color=PATTERN_COLORS[pat], alpha=0.12, linewidth=0)
plt.axhline(0, linestyle=":", color="gray")
plt.xlabel("layer")
plt.ylabel("mean decision_function (z-scored vs. own training distribution)")
plt.title(f"pooled fresh probe: 8-cell breakdown (pattern x connective) ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()
out_path = "figures/compound_cities/pooled_probe_8cell.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

# connective offset vs TT-FF spread
plt.figure(figsize=(9, 6))
plt.plot(offset_df["layer"], offset_df["connective_offset"], color=OFFSET_COLOR, marker="o", markersize=4, label="connective offset")
plt.fill_between(offset_df["layer"], offset_df["connective_offset_ci_lo"], offset_df["connective_offset_ci_hi"], color=OFFSET_COLOR, alpha=0.2, linewidth=0)
for conn in CONNECTIVES:
    plt.plot(
        offset_df["layer"], offset_df[f"spread_{conn}"], color=SPREAD_COLORS[conn], linestyle=CONNECTIVE_STYLE[conn],
        marker="s", markersize=3, label=f"TT-FF spread, {conn}",
    )
    plt.fill_between(
        offset_df["layer"], offset_df[f"spread_{conn}_ci_lo"], offset_df[f"spread_{conn}_ci_hi"],
        color=SPREAD_COLORS[conn], alpha=0.12, linewidth=0,
    )
plt.axhline(0, linestyle=":", color="gray")
plt.xlabel("layer")
plt.ylabel("z-scored decision_function difference")
plt.title(f"connective offset vs. TT-FF spread, pooled fresh probe ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
out_path = "figures/compound_cities/pooled_probe_connective_offset_vs_spread.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

# bonus: nested regression R^2
plt.figure(figsize=(8, 5))
plt.plot(regression_df["layer"], regression_df["r2_m1_conjunct_count_only"], color="#eda100", marker="o", markersize=4, label="M1: n_true_conjuncts only")
plt.plot(regression_df["layer"], regression_df["r2_m2_plus_connective"], color=OFFSET_COLOR, marker="o", markersize=4, label="M2: + connective")
plt.xlabel("layer")
plt.ylabel("R^2 (8 cell means)")
plt.ylim(0, 1.02)
plt.title(f"connective-blind vs. offset model fit to the 8 cell means ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
out_path = "figures/compound_cities/pooled_probe_nested_regression_r2.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
