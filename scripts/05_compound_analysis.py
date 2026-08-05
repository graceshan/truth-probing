"""Cross-dataset transfer: apply the cities-trained probe to the compound
and/or dataset, to see whether the single-statement truth direction picks
up compositional (logical and/or) truth structure at all.

Source probe fitting is UNCHANGED (fit_probe on all of cities, every layer,
no held-out split -- same as before). What's new is evaluation-side
bootstrap: with the probe held fixed and every statement's score already
computed, resample compound statements (1000 resamples, percentile
interval) for the per-pattern group means, the normalized-position ratio
(AND/OR separately), and the pooled FT-TF order effect. Point estimates for
group means, relative position, and the order effect are asserted
unchanged from the last saved run. compound_spread.png/csv (TT-FF spread)
is out of scope for this pass and untouched.
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

from src.data import load_activations
from src.probes import fit_probe
from src.stats import assert_unchanged, intervals_overlap

MODEL_NAME = "Qwen2.5-1.5B"
N_RESAMPLES = 1000
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
PATTERNS = ["TT", "TF", "FT", "FF"]
PATTERN_COLORS = {"TT": "#2a78d6", "TF": "#008300", "FT": "#e87ba4", "FF": "#eda100"}
CONNECTIVE_STYLE = {"and": "-", "or": "--"}
# and/or data-line color matches its own truth-conditional reference line's
# hue family, so "does this curve reach its target" reads at a glance
CONNECTIVE_COLORS = {"and": "#e34948", "or": "#008300"}
DISCUSSION_LAYERS = (10, 16)  # layer range to highlight in plots
MIN_SPREAD = 0.1  # minimum |TT - FF| to trust the ratio; below this the denominator is unstable

acts_cities, labels_cities = load_activations("cities")
acts_compound, _ = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]
meta["group"] = meta["connective"] + "-" + meta["pattern"]

n_layers = acts_cities.shape[1]
layers = range(n_layers)
scores_by_layer = np.zeros((acts_compound.shape[0], n_layers))

for L in range(n_layers):
    probe = fit_probe(acts_cities[:, L], labels_cities)
    raw = probe.decision_function(acts_compound[:, L])  # signed distance from boundary
    ref = probe.decision_function(acts_cities[:, L])  # standardize against the probe's own training distribution
    scores_by_layer[:, L] = (raw - ref.mean()) / ref.std()


def mean_and_sem(rows):
    """Per-layer mean and standard error of the mean for a set of statement rows."""
    vals = scores_by_layer[rows]
    mean = vals.mean(axis=0)
    sem = vals.std(axis=0, ddof=1) / np.sqrt(vals.shape[0])
    return mean, sem


group_idx = meta.groupby("group").groups
group_means = pd.DataFrame(
    {group: mean_and_sem(idx)[0] for group, idx in group_idx.items()}, index=layers
)
group_means.index.name = "layer"


def pct_interval(samples, axis=0):
    return np.nanpercentile(samples, 2.5, axis=axis), np.nanpercentile(samples, 97.5, axis=axis)


def bootstrap_connective(conn, n_resamples=N_RESAMPLES, seed=0):
    """Bootstrap this connective's 4 pattern-cell means, their centered
    values, and rel_position, for every layer at once, all drawn from the
    same joint resamples (each pattern cell resampled independently within
    itself, jointly per resample draw so centered/rel_position stay
    internally consistent within a single resample).

    Returns dict of {pattern: (lo[n_layers], hi[n_layers])} for raw and
    centered, plus (lo[n_layers], hi[n_layers]) for rel_position.
    """
    rng = np.random.default_rng(seed)
    cell_vals = {}  # pattern -> [n_cell_statements, n_layers]
    for pat in PATTERNS:
        m = ((meta.connective == conn) & (meta.pattern == pat)).values
        cell_vals[pat] = scores_by_layer[m]

    raw_samples = {pat: np.empty((n_resamples, n_layers)) for pat in PATTERNS}
    centered_samples = {pat: np.empty((n_resamples, n_layers)) for pat in PATTERNS}
    rel_pos_samples = np.empty((n_resamples, n_layers))

    for r in range(n_resamples):
        cell_means = {}
        for pat in PATTERNS:
            vals = cell_vals[pat]
            idx = rng.integers(0, vals.shape[0], size=vals.shape[0])
            cell_means[pat] = vals[idx].mean(axis=0)  # [n_layers]
        cross_mean = np.mean([cell_means[p] for p in PATTERNS], axis=0)
        for pat in PATTERNS:
            raw_samples[pat][r] = cell_means[pat]
            centered_samples[pat][r] = cell_means[pat] - cross_mean

        tt, ff = cell_means["TT"], cell_means["FF"]
        mixed = (cell_means["TF"] + cell_means["FT"]) / 2
        spread = tt - ff
        with np.errstate(invalid="ignore", divide="ignore"):
            rp = (mixed - ff) / spread
        rp = np.where(np.abs(spread) < MIN_SPREAD, np.nan, rp)
        rel_pos_samples[r] = rp

    raw_ci = {pat: pct_interval(raw_samples[pat]) for pat in PATTERNS}
    centered_ci = {pat: pct_interval(centered_samples[pat]) for pat in PATTERNS}
    rel_pos_ci = pct_interval(rel_pos_samples)
    return raw_ci, centered_ci, rel_pos_ci


def bootstrap_ft_minus_tf(n_resamples=N_RESAMPLES, seed=0):
    rng = np.random.default_rng(seed)
    ft_vals = scores_by_layer[(meta["pattern"] == "FT").values]
    tf_vals = scores_by_layer[(meta["pattern"] == "TF").values]
    samples = np.empty((n_resamples, n_layers))
    for r in range(n_resamples):
        ft_idx = rng.integers(0, ft_vals.shape[0], size=ft_vals.shape[0])
        tf_idx = rng.integers(0, tf_vals.shape[0], size=tf_vals.shape[0])
        samples[r] = ft_vals[ft_idx].mean(axis=0) - tf_vals[tf_idx].mean(axis=0)
    return pct_interval(samples)


results_dir = "results/compound_cities"
os.makedirs(results_dir, exist_ok=True)

# point-estimate regression check against the last saved run, if one exists
old_group_means_path = f"{results_dir}/cities_probe_scores_by_group.csv"
if os.path.exists(old_group_means_path):
    old = pd.read_csv(old_group_means_path, index_col="layer")
    assert_unchanged("cities_probe_scores_by_group", old.to_numpy(), group_means[old.columns].to_numpy())
    print("verified: per-pattern group means unchanged from the last saved run")

np.save(f"{results_dir}/cities_probe_scores.npy", scores_by_layer)
group_means.to_csv(old_group_means_path)
print(group_means)
print(f"saved {results_dir}/cities_probe_scores.npy and {old_group_means_path}")

# bootstrap CIs for raw group means, centered values, and rel_position,
# per connective, computed once and reused across every plot below
raw_ci, centered_ci, rel_pos_ci = {}, {}, {}
for conn in ["and", "or"]:
    raw_ci[conn], centered_ci[conn], rel_pos_ci[conn] = bootstrap_connective(conn)

group_lo = pd.DataFrame(
    {f"{conn}-{pat}": raw_ci[conn][pat][0] for conn in ["and", "or"] for pat in PATTERNS}, index=layers
)
group_hi = pd.DataFrame(
    {f"{conn}-{pat}": raw_ci[conn][pat][1] for conn in ["and", "or"] for pat in PATTERNS}, index=layers
)

plt.figure(figsize=(9, 6))
for color, group in zip(PALETTE, group_means.columns):
    conn, pat = group.split("-")
    mean = group_means[group]
    plt.plot(layers, mean, color=color, marker="o", label=group)
    plt.fill_between(layers, group_lo[group], group_hi[group], color=color, alpha=0.15, linewidth=0)
plt.axhline(0, linestyle="--", color="gray", label="cities-probe mean")
plt.xlabel("layer")
plt.ylabel("standardized probe score")
plt.title(f"cities-trained probe applied to compound and/or statements ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()

os.makedirs("figures/compound_cities", exist_ok=True)
out_path = "figures/compound_cities/cities_probe_on_compound.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

# one panel per connective, sharing a y-axis, so the two profiles (how score
# varies across conjunct pattern) can be compared directly without needing
# to subtract out and/or's different overall level by hand.
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, conn in zip(axes, ["and", "or"]):
    for pat in PATTERNS:
        m = ((meta.connective == conn) & (meta.pattern == pat)).values
        mean, _ = mean_and_sem(m)
        ax.plot(layers, mean, color=PATTERN_COLORS[pat], marker="o", label=pat)
        lo, hi = raw_ci[conn][pat]
        ax.fill_between(layers, lo, hi, color=PATTERN_COLORS[pat], alpha=0.15, linewidth=0)
    ax.axvspan(*DISCUSSION_LAYERS, color="gray", alpha=0.12, linewidth=0)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title(f'"{conn}" compounds')
    ax.set_xlabel("layer")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("mean probe score (z, cities scale)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="conjunct pattern", loc="center left", bbox_to_anchor=(1.0, 0.5))
fig.suptitle(f"cities-trained probe on compound statements, by pattern ({MODEL_NAME})")
fig.tight_layout()

out_path = "figures/compound_cities/compound_scores_by_layer.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

# center within connective: subtract each connective's own (per-layer) mean
# across its four pattern cells, isolating the pattern *shape* from the
# connective's overall level, then overlay both profiles on one axes --
# color = pattern, linestyle = connective, so same-color solid vs dashed
# lines show directly whether and/or shape the same way.
centered = group_means.copy()
for conn in ["and", "or"]:
    cols = [f"{conn}-{p}" for p in PATTERNS]
    centered[cols] = group_means[cols].sub(group_means[cols].mean(axis=1), axis=0)

old_centered_path = f"{results_dir}/cities_probe_scores_by_group_centered.csv"
if os.path.exists(old_centered_path):
    old = pd.read_csv(old_centered_path, index_col="layer")
    assert_unchanged("cities_probe_scores_by_group_centered", old.to_numpy(), centered[old.columns].to_numpy())
    print("verified: centered group means unchanged from the last saved run")
centered.to_csv(old_centered_path)
print(f"saved {old_centered_path}")

plt.figure(figsize=(9, 6))
for conn, linestyle in CONNECTIVE_STYLE.items():
    for pat in PATTERNS:
        col = f"{conn}-{pat}"
        mean = centered[col]
        lo, hi = centered_ci[conn][pat]
        plt.plot(layers, mean, color=PATTERN_COLORS[pat], linestyle=linestyle, marker="o", label=col)
        plt.fill_between(layers, lo, hi, color=PATTERN_COLORS[pat], alpha=0.15, linewidth=0)
plt.axvspan(*DISCUSSION_LAYERS, color="gray", alpha=0.12, linewidth=0)
plt.axhline(0, linestyle=":", color="gray", label="connective mean")
plt.xlabel("layer")
plt.ylabel("standardized probe score, centered within connective")
plt.title(f"cities-trained probe on compound statements, centered within connective ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()

out_path = "figures/compound_cities/compound_scores_centered.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")


def cell_mean(conn, pat, L):
    return group_means.loc[L, f"{conn}-{pat}"]


def rel_position(conn, L):
    """Where do the mixed patterns sit on the FF(0) -> TT(1) scale?"""
    tt = cell_mean(conn, "TT", L)
    ff = cell_mean(conn, "FF", L)
    if abs(tt - ff) < MIN_SPREAD:
        return np.nan
    mixed = (cell_mean(conn, "TF", L) + cell_mean(conn, "FT", L)) / 2
    return (mixed - ff) / (tt - ff)


# normalized position: each connective is self-scaled to its own FF->TT
# range, so this neutralizes the AND/OR level difference automatically.
# Bootstrap CI (not the delta method) since this is a ratio of differences
# of group means -- its sampling distribution isn't analytically obvious.
rel_pos = pd.DataFrame({conn: [rel_position(conn, L) for L in layers] for conn in ["and", "or"]}, index=layers)
rel_pos.index.name = "layer"

old_rel_pos_path = f"{results_dir}/compound_relative_position.csv"
if os.path.exists(old_rel_pos_path):
    old = pd.read_csv(old_rel_pos_path, index_col="layer")
    assert_unchanged("compound_relative_position", old.to_numpy(), rel_pos[old.columns].to_numpy())
    print("verified: relative position unchanged from the last saved run")

rel_pos["and_ci_lo"], rel_pos["and_ci_hi"] = rel_pos_ci["and"]
rel_pos["or_ci_lo"], rel_pos["or_ci_hi"] = rel_pos_ci["or"]
rel_pos["and_or_overlap"] = intervals_overlap(
    rel_pos["and_ci_lo"], rel_pos["and_ci_hi"], rel_pos["or_ci_lo"], rel_pos["or_ci_hi"]
)
rel_pos.to_csv(old_rel_pos_path)
print(rel_pos)
print(f"saved {old_rel_pos_path}")

print("\nAND vs OR normalized-position interval overlap, per layer:")
for L in layers:
    row = rel_pos.loc[L]
    print(
        f"layer {L}: AND=[{row['and_ci_lo']:.3f},{row['and_ci_hi']:.3f}]  "
        f"OR=[{row['or_ci_lo']:.3f},{row['or_ci_hi']:.3f}]  "
        f"overlap={'yes' if row['and_or_overlap'] else 'NO -- distinguishable at 95%'}"
    )

plt.figure(figsize=(9, 6))
for conn in ["and", "or"]:
    plt.plot(layers, rel_pos[conn], color=CONNECTIVE_COLORS[conn], marker="o", label=f'"{conn}"')
    plt.fill_between(
        layers, rel_pos[f"{conn}_ci_lo"], rel_pos[f"{conn}_ci_hi"], color=CONNECTIVE_COLORS[conn], alpha=0.15, linewidth=0
    )
plt.axvspan(*DISCUSSION_LAYERS, color="gray", alpha=0.12, linewidth=0)
plt.axhline(0.5, linestyle="--", color="gray", label="association-counting prediction")
plt.axhline(0.0, linestyle=":", color="red", label="truth-conditional: AND")
plt.axhline(1.0, linestyle=":", color="green", label="truth-conditional: OR")
plt.ylim(-0.1, 1.1)
plt.xlabel("layer")
plt.ylabel("mixed patterns' position (0 = FF level, 1 = TT level)")
plt.title(f"normalized position of mixed (TF/FT) patterns ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()

out_path = "figures/compound_cities/compound_relative_position.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

# companion 1: spread (TT - FF) per connective -- out of scope for this
# bootstrap pass, left exactly as before (analytic SEM propagation).
spread = pd.DataFrame(
    {conn: group_means[f"{conn}-TT"] - group_means[f"{conn}-FF"] for conn in ["and", "or"]}, index=layers
)
group_sems = pd.DataFrame(
    {group: mean_and_sem(idx)[1] for group, idx in group_idx.items()}, index=layers
)
spread_sem = pd.DataFrame(
    {
        conn: np.sqrt(group_sems[f"{conn}-TT"] ** 2 + group_sems[f"{conn}-FF"] ** 2)
        for conn in ["and", "or"]
    },
    index=layers,
)
spread.index.name = "layer"
spread.to_csv(f"{results_dir}/compound_spread.csv")
print(f"saved {results_dir}/compound_spread.csv")

plt.figure(figsize=(7, 5))
for conn in ["and", "or"]:
    mean, sem = spread[conn], spread_sem[conn]
    plt.plot(layers, mean, color=CONNECTIVE_COLORS[conn], marker="o", label=f'"{conn}"')
    plt.fill_between(layers, mean - sem, mean + sem, color=CONNECTIVE_COLORS[conn], alpha=0.15, linewidth=0)
plt.axvspan(*DISCUSSION_LAYERS, color="gray", alpha=0.12, linewidth=0)
plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("layer")
plt.ylabel("TT - FF (z, cities scale)")
plt.title(f"spread between TT and FF cells, per connective ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()

out_path = "figures/compound_cities/compound_spread.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

# companion 2: FT - TF pooled across connectives -- a conjunct-order effect
# (does the model weight the second conjunct differently than the first?),
# with a zero line to read off where it crosses from one direction to the
# other over layers. Bootstrap CI (resample compounds), point estimate
# unchanged from before.
ft_mean, _ = mean_and_sem((meta["pattern"] == "FT").values)
tf_mean, _ = mean_and_sem((meta["pattern"] == "TF").values)
order_effect = pd.Series(ft_mean - tf_mean, index=layers, name="ft_minus_tf")
order_effect.index.name = "layer"

old_order_effect_path = f"{results_dir}/compound_order_effect.csv"
if os.path.exists(old_order_effect_path):
    old = pd.read_csv(old_order_effect_path, index_col="layer")
    assert_unchanged("compound_order_effect", old["ft_minus_tf"].to_numpy(), order_effect.to_numpy())
    print("verified: FT-TF order effect unchanged from the last saved run")

order_effect_lo, order_effect_hi = bootstrap_ft_minus_tf()
order_effect_df = pd.DataFrame(
    {"ft_minus_tf": order_effect, "ci_lo": order_effect_lo, "ci_hi": order_effect_hi}, index=layers
)
order_effect_df.index.name = "layer"
order_effect_df.to_csv(old_order_effect_path)
print(f"saved {old_order_effect_path}")

plt.figure(figsize=(7, 5))
plt.plot(layers, order_effect, color=PALETTE[0], marker="o", label="FT - TF")
plt.fill_between(layers, order_effect_lo, order_effect_hi, color=PALETTE[0], alpha=0.15, linewidth=0)
plt.axvspan(*DISCUSSION_LAYERS, color="gray", alpha=0.12, linewidth=0)
plt.axhline(0, linestyle="--", color="gray", label="no order effect")
plt.xlabel("layer")
plt.ylabel("FT - TF (z, cities scale), pooled across and/or")
plt.title(f"conjunct-order effect: FT vs TF, pooled across connectives ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()

out_path = "figures/compound_cities/compound_order_effect.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
