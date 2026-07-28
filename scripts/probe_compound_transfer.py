"""Cross-dataset transfer: apply the cities-trained probe to the compound
and/or dataset, to see whether the single-statement truth direction picks
up compositional (logical and/or) truth structure at all.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load_activations
from src.probes import fit_probe

MODEL_NAME = "Qwen2.5-1.5B"
PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
PATTERNS = ["TT", "TF", "FT", "FF"]
PATTERN_COLORS = {"TT": "#2a78d6", "TF": "#008300", "FT": "#e87ba4", "FF": "#eda100"}
CONNECTIVE_STYLE = {"and": "-", "or": "--"}
# and/or data-line color matches its own truth-conditional reference line's
# hue family, so "does this curve reach its target" reads at a glance
CONNECTIVE_COLORS = {"and": "#e34948", "or": "#008300"}
DISCUSSION_LAYERS = (14, 22)  # layer range to highlight in plots

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
group_sems = pd.DataFrame(
    {group: mean_and_sem(idx)[1] for group, idx in group_idx.items()}, index=layers
)
group_means.index.name = "layer"

results_dir = "results/compound_cities"
os.makedirs(results_dir, exist_ok=True)
np.save(f"{results_dir}/cities_probe_scores.npy", scores_by_layer)
group_means.to_csv(f"{results_dir}/cities_probe_scores_by_group.csv")
print(group_means)
print(f"saved {results_dir}/cities_probe_scores.npy and {results_dir}/cities_probe_scores_by_group.csv")

plt.figure(figsize=(9, 6))
for color, group in zip(PALETTE, group_means.columns):
    mean, sem = group_means[group], group_sems[group]
    plt.plot(layers, mean, color=color, marker="o", label=group)
    plt.fill_between(layers, mean - sem, mean + sem, color=color, alpha=0.15, linewidth=0)
plt.axhline(0, linestyle="--", color="gray", label="cities-probe mean")
plt.xlabel("layer")
plt.ylabel("standardized probe score")
plt.title(f"cities-trained probe applied to compound and/or statements ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()

os.makedirs("figures/transfer", exist_ok=True)
out_path = "figures/transfer/cities_probe_on_compound.png"
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

out_path = "figures/transfer/compound_scores_by_layer.png"
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
centered.to_csv(f"{results_dir}/cities_probe_scores_by_group_centered.csv")
print(f"saved {results_dir}/cities_probe_scores_by_group_centered.csv")

plt.figure(figsize=(9, 6))
for conn, linestyle in CONNECTIVE_STYLE.items():
    for pat in PATTERNS:
        col = f"{conn}-{pat}"
        mean, sem = centered[col], group_sems[col]
        plt.plot(layers, mean, color=PATTERN_COLORS[pat], linestyle=linestyle, marker="o", label=col)
        plt.fill_between(layers, mean - sem, mean + sem, color=PATTERN_COLORS[pat], alpha=0.15, linewidth=0)
plt.axvspan(*DISCUSSION_LAYERS, color="gray", alpha=0.12, linewidth=0)
plt.axhline(0, linestyle=":", color="gray", label="connective mean")
plt.xlabel("layer")
plt.ylabel("standardized probe score, centered within connective")
plt.title(f"cities-trained probe on compound statements, centered within connective ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()

out_path = "figures/transfer/compound_scores_centered.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")


def cell_mean(conn, pat, L):
    return group_means.loc[L, f"{conn}-{pat}"]


MIN_SPREAD = 0.1  # minimum |TT - FF| to trust the ratio; below this the denominator is unstable


def rel_position(conn, L):
    """Where do the mixed patterns sit on the FF(0) -> TT(1) scale?"""
    tt = cell_mean(conn, "TT", L)
    ff = cell_mean(conn, "FF", L)
    if abs(tt - ff) < MIN_SPREAD:
        return np.nan
    mixed = (cell_mean(conn, "TF", L) + cell_mean(conn, "FT", L)) / 2
    return (mixed - ff) / (tt - ff)


# normalized position: each connective is self-scaled to its own FF->TT
# range, so this neutralizes the AND/OR level difference automatically --
# no error bars, since propagating uncertainty through a ratio of
# differences needs the delta method and wasn't asked for here.
rel_pos = pd.DataFrame({conn: [rel_position(conn, L) for L in layers] for conn in ["and", "or"]}, index=layers)
rel_pos.index.name = "layer"
rel_pos.to_csv(f"{results_dir}/compound_relative_position.csv")
print(rel_pos)
print(f"saved {results_dir}/compound_relative_position.csv")

plt.figure(figsize=(9, 6))
for conn in ["and", "or"]:
    plt.plot(layers, rel_pos[conn], color=CONNECTIVE_COLORS[conn], marker="o", label=f'"{conn}"')
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

out_path = "figures/transfer/compound_relative_position.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

# companion 1: spread (TT - FF) per connective -- quantifies the AND > OR
# gap seen in the raw plots directly, with propagated SEM (independent
# groups, so combined SEM is sqrt of summed squared SEMs).
spread = pd.DataFrame(
    {conn: group_means[f"{conn}-TT"] - group_means[f"{conn}-FF"] for conn in ["and", "or"]}, index=layers
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

out_path = "figures/transfer/compound_spread.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")

# companion 2: FT - TF pooled across connectives -- a conjunct-order effect
# (does the model weight the second conjunct differently than the first?),
# with a zero line to read off where it crosses from one direction to the
# other over layers.
ft_mean, ft_sem = mean_and_sem((meta["pattern"] == "FT").values)
tf_mean, tf_sem = mean_and_sem((meta["pattern"] == "TF").values)
order_effect = pd.Series(ft_mean - tf_mean, index=layers, name="ft_minus_tf")
order_effect_sem = pd.Series(np.sqrt(ft_sem**2 + tf_sem**2), index=layers)
order_effect.index.name = "layer"
order_effect.to_csv(f"{results_dir}/compound_order_effect.csv")
print(f"saved {results_dir}/compound_order_effect.csv")

plt.figure(figsize=(7, 5))
plt.plot(layers, order_effect, color=PALETTE[0], marker="o", label="FT - TF")
plt.fill_between(
    layers, order_effect - order_effect_sem, order_effect + order_effect_sem,
    color=PALETTE[0], alpha=0.15, linewidth=0,
)
plt.axvspan(*DISCUSSION_LAYERS, color="gray", alpha=0.12, linewidth=0)
plt.axhline(0, linestyle="--", color="gray", label="no order effect")
plt.xlabel("layer")
plt.ylabel("FT - TF (z, cities scale), pooled across and/or")
plt.title(f"conjunct-order effect: FT vs TF, pooled across connectives ({MODEL_NAME})")
plt.grid(alpha=0.3)
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, fontsize=8)
plt.tight_layout()

out_path = "figures/transfer/compound_order_effect.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
