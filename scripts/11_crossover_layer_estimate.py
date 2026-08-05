"""Where does the compound-statement evidence flip sign, and how uncertain
is that layer?

Two quantities from the compound analysis each cross zero somewhere in the
middle layers: b2 - b1 (conjunct-regression coefficients, 06) and FT - TF
(conjunct-order effect, 05). This script estimates the crossover layer for
each and puts an interval on it.

IMPORTANT ASYMMETRY WITH THE PREVIOUS TASK: 05 and 06 are transfer-style
scripts (source probe fit on all of cities / a single fixed 80-20 split --
see CLAUDE.md's methods section), so unlike 01/02/04 they have NO seed axis.
There is no "distribution across seeds" for these two quantities -- only a
bootstrap-resample axis. Everything below is bootstrap-only; this is stated
again in the printed output so it isn't missed.

06's existing per-layer bootstrap draws an independent resample per layer
(seed=L), so resample #7 at layer 10 and resample #7 at layer 11 are
unrelated draws -- fine for a per-layer CI, but useless for tracing a single
resample's trajectory across layers to find ITS crossover point. This script
redoes the b2-b1 bootstrap with one shared set of resampled row-index draws
reused across every layer, so each resample has a coherent multi-layer
trajectory. 05's FT-TF bootstrap was already paired this way (one resample
draw, mean taken across all layers at once), so it's reused as-is.

Crossover-layer definition: search layers >= START_LAYER (matches 06's own
PLOT_START_LAYER -- cities probe hasn't formed yet before that, sign flips
there are noise) for the LAST sign change, i.e. the flip after which the
quantity keeps that sign through the final layer. Linearly interpolate
between the two bracketing integer layers. A resample whose trajectory never
changes sign in range contributes no crossover estimate (counted and
reported separately, not silently dropped).
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

from src.data import load_activations, load_statements
from src.probes import fit_probe, split_indices, train_layer_probe

MODEL_NAME = "Qwen2.5-1.5B"
N_RESAMPLES = 1000
START_LAYER = 5  # matches 06's PLOT_START_LAYER: pre-signal layers are unstable noise
B2B1_COLOR = "#2a78d6"
FTTF_COLOR = "#eb6834"

# ---------------------------------------------------------------------------
# shared data loading
# ---------------------------------------------------------------------------

acts_cities, labels_cities = load_activations("cities")
acts_compound, _ = load_activations("compound_cities")
meta = pd.read_csv("data/compound_cities.csv")
assert len(meta) == acts_compound.shape[0]

n_layers = acts_cities.shape[1]
layers = np.arange(n_layers)
search_mask = layers >= START_LAYER
search_layers = layers[search_mask]


def find_crossover_layer(values: np.ndarray, layers_: np.ndarray) -> float | None:
    """Interpolated layer of the LAST sign change in `values` (indexed by
    `layers_`), i.e. the flip after which the sign holds through the end.
    None if no sign change occurs in range.
    """
    signs = np.sign(values)
    changes = np.where(np.diff(signs) != 0)[0]
    if len(changes) == 0:
        return None
    i = changes[-1]
    v0, v1 = values[i], values[i + 1]
    l0, l1 = layers_[i], layers_[i + 1]
    frac = -v0 / (v1 - v0)
    return float(l0 + frac * (l1 - l0))


def summarize(name: str, crossings: np.ndarray, point_estimate: float | None):
    valid = crossings[~np.isnan(crossings)]
    n_missing = len(crossings) - len(valid)
    print(f"\n{name}:")
    if point_estimate is not None:
        print(f"  point-estimate crossover (unresampled data): layer {point_estimate:.2f}")
    else:
        print("  point-estimate crossover (unresampled data): no sign change in range")
    print(f"  bootstrap resamples with a crossover: {len(valid)}/{len(crossings)} "
          f"({n_missing} had no sign change in range and were excluded)")
    if len(valid) == 0:
        print("  no resamples produced a crossover -- cannot report an interval")
        return None
    lo, hi = np.percentile(valid, 2.5), np.percentile(valid, 97.5)
    width = hi - lo
    print(f"  bootstrap distribution: mean={valid.mean():.2f}, median={np.median(valid):.2f}, "
          f"std={valid.std(ddof=1):.2f}")
    print(f"  95% interval (percentile, over bootstrap resamples): [{lo:.2f}, {hi:.2f}]  (width={width:.2f} layers)")
    if width > 3:
        print(f"  WARNING: 95% interval spans {width:.2f} layers (>3) -- do not state a specific "
              f"crossover layer for {name} in the write-up, report the interval instead.")
    return lo, hi, valid


# ---------------------------------------------------------------------------
# 1. FT - TF crossover (reuses 05's already-paired bootstrap scheme)
# ---------------------------------------------------------------------------

scores_by_layer = np.zeros((acts_compound.shape[0], n_layers))
for L in range(n_layers):
    probe = fit_probe(acts_cities[:, L], labels_cities)
    raw = probe.decision_function(acts_compound[:, L])
    ref = probe.decision_function(acts_cities[:, L])
    scores_by_layer[:, L] = (raw - ref.mean()) / ref.std()

order_effect_point = pd.read_csv("results/compound_cities/compound_order_effect.csv")
order_effect_point = order_effect_point.set_index("layer")["ft_minus_tf"].to_numpy()
fttf_point_crossover = find_crossover_layer(order_effect_point[search_mask], search_layers)

ft_vals = scores_by_layer[(meta["pattern"] == "FT").values]
tf_vals = scores_by_layer[(meta["pattern"] == "TF").values]
rng = np.random.default_rng(0)
fttf_crossings = np.full(N_RESAMPLES, np.nan)
for r in range(N_RESAMPLES):
    ft_idx = rng.integers(0, ft_vals.shape[0], size=ft_vals.shape[0])
    tf_idx = rng.integers(0, tf_vals.shape[0], size=tf_vals.shape[0])
    traj = ft_vals[ft_idx].mean(axis=0) - tf_vals[tf_idx].mean(axis=0)
    c = find_crossover_layer(traj[search_mask], search_layers)
    if c is not None:
        fttf_crossings[r] = c

# ---------------------------------------------------------------------------
# 2. b2 - b1 crossover (pooled across connectives) -- needs a fresh bootstrap
#    with resample indices SHARED across layers, unlike 06's per-layer one
# ---------------------------------------------------------------------------

cities_statements = load_statements("cities", datasets_dir="geometry-of-truth/datasets")["statement"]
assert len(cities_statements) == acts_cities.shape[0]
statement_to_idx = {s: i for i, s in enumerate(cities_statements)}
conj_a_idx = meta["conj_a"].map(statement_to_idx).to_numpy()
conj_b_idx = meta["conj_b"].map(statement_to_idx).to_numpy()

train_idx, test_idx = split_indices(len(labels_cities))  # same fixed split as 06, unseeded

score_a_by_layer = np.zeros((len(meta), n_layers))
score_b_by_layer = np.zeros((len(meta), n_layers))
compound_z_by_layer = np.zeros((len(meta), n_layers))
b1_point = np.zeros(n_layers)
b2_point = np.zeros(n_layers)

for L in range(n_layers):
    probe, _ = train_layer_probe(acts_cities[:, L], labels_cities, train_idx, test_idx)
    s_all = probe.decision_function(acts_cities[:, L])
    score_a_by_layer[:, L] = (s_all[conj_a_idx] - s_all.mean()) / s_all.std()
    score_b_by_layer[:, L] = (s_all[conj_b_idx] - s_all.mean()) / s_all.std()
    raw_compound = probe.decision_function(acts_compound[:, L])
    compound_z_by_layer[:, L] = (raw_compound - s_all.mean()) / s_all.std()

    X = np.column_stack([score_a_by_layer[:, L], score_b_by_layer[:, L]])
    reg = LinearRegression().fit(X, compound_z_by_layer[:, L])
    b1_point[L], b2_point[L] = reg.coef_

# point-estimate check against 06's own saved output (same split, same data -- must match)
saved = pd.read_csv("results/compound_cities/conjunct_regression.csv")
assert np.allclose(saved["b1"].to_numpy(), b1_point, atol=1e-6), "b1 point estimate diverged from 06's saved output"
assert np.allclose(saved["b2"].to_numpy(), b2_point, atol=1e-6), "b2 point estimate diverged from 06's saved output"

b2_minus_b1_point = b2_point - b1_point
b2b1_point_crossover = find_crossover_layer(b2_minus_b1_point[search_mask], search_layers)

n_compound = len(meta)
rng = np.random.default_rng(1)
b2b1_crossings = np.full(N_RESAMPLES, np.nan)
for r in range(N_RESAMPLES):
    sel = rng.integers(0, n_compound, size=n_compound)  # shared across every layer this resample
    traj = np.empty(n_layers)
    for L in range(n_layers):
        X = np.column_stack([score_a_by_layer[sel, L], score_b_by_layer[sel, L]])
        y = compound_z_by_layer[sel, L]
        reg = LinearRegression().fit(X, y)
        b1r, b2r = reg.coef_
        traj[L] = b2r - b1r
    c = find_crossover_layer(traj[search_mask], search_layers)
    if c is not None:
        b2b1_crossings[r] = c

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

print(f"crossover-layer estimate ({MODEL_NAME}), searching layers >= {START_LAYER}")
print("NOTE: 05 and 06 have no seed axis (source probe fit on all of cities / a single "
      "fixed split, not seeded per-layer splits) -- these are bootstrap-only distributions, "
      "not seed x bootstrap.")

b2b1_summary = summarize("b2 - b1 (pooled across connectives)", b2b1_crossings, b2b1_point_crossover)
fttf_summary = summarize("FT - TF (pooled across connectives)", fttf_crossings, fttf_point_crossover)

out_dir = "results/compound_cities"
os.makedirs(out_dir, exist_ok=True)
pd.DataFrame({"resample": np.arange(N_RESAMPLES), "b2_minus_b1_crossover_layer": b2b1_crossings,
              "ft_minus_tf_crossover_layer": fttf_crossings}).to_csv(
    f"{out_dir}/crossover_layer_bootstrap.csv", index=False
)
print(f"\nsaved {out_dir}/crossover_layer_bootstrap.csv")

fig, ax = plt.subplots(figsize=(8, 5))
bins = np.arange(START_LAYER, n_layers + 1, 0.5)
ax.hist(b2b1_crossings[~np.isnan(b2b1_crossings)], bins=bins, alpha=0.55, color=B2B1_COLOR, label="b2 - b1 crossover")
ax.hist(fttf_crossings[~np.isnan(fttf_crossings)], bins=bins, alpha=0.55, color=FTTF_COLOR, label="FT - TF crossover")
if b2b1_point_crossover is not None:
    ax.axvline(b2b1_point_crossover, color=B2B1_COLOR, linestyle="--")
if fttf_point_crossover is not None:
    ax.axvline(fttf_point_crossover, color=FTTF_COLOR, linestyle="--")
ax.set_xlabel("crossover layer (interpolated)")
ax.set_ylabel("bootstrap resamples")
ax.set_title(f"crossover-layer distribution across bootstrap resamples ({MODEL_NAME})")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
out_path = "figures/compound_cities/crossover_layer_distribution.png"
os.makedirs("figures/compound_cities", exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"saved {out_path}")
