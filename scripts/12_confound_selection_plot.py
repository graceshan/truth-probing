"""Does 'pick the layer with the best within-dataset validation accuracy'
(standard probing practice) also pick the layer that transfers worst under
negation -- i.e. does standard model selection select for the confound?

Two stacked panels sharing the layer axis (never a dual-axis/two-y-scale
plot -- see dataviz conventions): within-dataset cities accuracy (01's
5-seed mean + 95% CI) on top, cities->neg_cities transfer AUROC (03's
bootstrap mean + 95% CI) on bottom. A vertical line marks the layer standard
practice would select (argmax of within-dataset accuracy), drawn on both
panels; the transfer AUROC at that layer is annotated on the bottom panel.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODEL_NAME = "Qwen2.5-1.5B"
ACC_COLOR = "#2a78d6"
AUC_COLOR = "#eb6834"
SELECT_COLOR = "#4a3aa7"
CHANCE_COLOR = "#888888"

acc_df = pd.read_csv("results/cities/accuracy.csv")
transfer_df = pd.read_csv("results/transfer/cities_neg_cities_transfer.csv")

assert (acc_df["layer"].to_numpy() == transfer_df["layer"].to_numpy()).all()
layers = acc_df["layer"].to_numpy()

selected_layer = int(acc_df.loc[acc_df["accuracy_mean"].idxmax(), "layer"])

auc = transfer_df["cities->neg_cities_auc"].to_numpy()
auc_lo = transfer_df["cities->neg_cities_auc_lo"].to_numpy()
auc_hi = transfer_df["cities->neg_cities_auc_hi"].to_numpy()

auc_at_selected = float(transfer_df.loc[transfer_df["layer"] == selected_layer, "cities->neg_cities_auc"].iloc[0])
auc_mean_all_layers = float(auc.mean())
best_layer = int(layers[np.argmax(auc)])
best_auc = float(auc.max())

print(f"standard-practice-selected layer (argmax within-dataset cities accuracy): {selected_layer}")
print(f"  within-dataset cities accuracy at that layer: {acc_df.loc[acc_df['layer'] == selected_layer, 'accuracy_mean'].iloc[0]:.4f}")
print()
print(f"cities->neg_cities transfer AUROC at the selected layer {selected_layer}: {auc_at_selected:.4f}")
print(f"cities->neg_cities transfer AUROC, mean across all {len(layers)} layers:  {auc_mean_all_layers:.4f}")
print(f"cities->neg_cities transfer AUROC, best available (layer {best_layer}):        {best_auc:.4f}")
print()
print(f"selecting on in-distribution accuracy costs {best_auc - auc_at_selected:.4f} AUROC "
      f"({(best_auc - auc_at_selected) / best_auc:.1%} relative) versus the best achievable layer, "
      f"and lands {auc_mean_all_layers - auc_at_selected:+.4f} AUROC relative to the all-layer mean -- "
      f"{'below' if auc_at_selected < auc_mean_all_layers else 'at or above'} average transfer performance.")

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

ax_top.plot(layers, acc_df["accuracy_mean"], color=ACC_COLOR, marker="o", markersize=4)
ax_top.fill_between(layers, acc_df["accuracy_ci_lo"], acc_df["accuracy_ci_hi"], color=ACC_COLOR, alpha=0.2, linewidth=0)
ax_top.axhline(0.5, linestyle="--", color=CHANCE_COLOR, linewidth=1, label="chance")
ax_top.axvline(selected_layer, color=SELECT_COLOR, linestyle=":", linewidth=1.5,
                label=f"standard-practice selection (layer {selected_layer})")
ax_top.set_ylabel("within-dataset cities accuracy")
ax_top.set_ylim(0.4, 1.02)
ax_top.set_title(f"in-distribution accuracy vs. negation transfer, by layer ({MODEL_NAME})")
ax_top.grid(alpha=0.3)
ax_top.legend(loc="lower right", fontsize=8)

ax_bot.plot(layers, auc, color=AUC_COLOR, marker="o", markersize=4)
ax_bot.fill_between(layers, auc_lo, auc_hi, color=AUC_COLOR, alpha=0.2, linewidth=0)
ax_bot.axhline(0.5, linestyle="--", color=CHANCE_COLOR, linewidth=1, label="chance")
ax_bot.axvline(selected_layer, color=SELECT_COLOR, linestyle=":", linewidth=1.5)
ax_bot.annotate(
    f"AUROC = {auc_at_selected:.3f}\nat selected layer {selected_layer}",
    xy=(selected_layer, auc_at_selected),
    xytext=(selected_layer + 2.5, auc_at_selected + (0.15 if auc_at_selected < 0.75 else -0.2)),
    arrowprops=dict(arrowstyle="->", color=SELECT_COLOR),
    fontsize=8.5,
    color=SELECT_COLOR,
)
ax_bot.set_xlabel("layer")
ax_bot.set_ylabel("cities -> neg_cities transfer AUROC")
ax_bot.set_ylim(0.0, 1.02)
ax_bot.grid(alpha=0.3)
ax_bot.legend(loc="lower right", fontsize=8)

fig.tight_layout()
os.makedirs("figures/transfer", exist_ok=True)
out_path = "figures/transfer/layer_selection_confound.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nsaved {out_path}")
