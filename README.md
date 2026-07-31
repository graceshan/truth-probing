# truth-probing

Does the linear "truth direction" in a small language model actually encode
truth, or something correlated with it? This project reproduces the standard
truth-probing result on Qwen2.5-1.5B and then tests the recovered direction
against logical negation and compositional (AND/OR) statements.

## Finding

A probe that classifies atomic factual statements at ~100% accuracy does not
encode truth-conditional content. It encodes an attenuated, position-weighted
sum of factual association — whether the text asserts correct entity
pairings.

- **Negation inverts it.** Cross-dataset transfer between `cities` and
  `neg_cities` falls far below chance in middle layers (as low as 0.01),
  while within-dataset accuracy stays near 1.0. Below chance means
  systematic inversion, not absence of signal.
- **Logical connectives are ignored.** Mixed-truth compounds sit near the
  midpoint between TT and FF regardless of whether they are joined by "and"
  or "or," where truth-conditional processing would require opposite
  answers.
- **But the information is there.** A probe trained directly on compounds
  classifies compound truth at ~96%, far above the ~76% ceiling available to
  any connective-blind probe, while the cities-trained probe transfers at
  chance. The failure belongs to the probe's generalization, not the
  model's representational capacity.

Consistent with Bao et al. (2025), who find truth-direction generalization
improves with model capability, this appears to be a small-model phenomenon.
The safety-relevant point is that the failure is silent, and on negated
content inverted — a monitor validated on atomic statements can be
confidently wrong on exactly the linguistic forms (denials, compound claims)
that deceptive output takes.

## Setup

- Model: Qwen2.5-1.5B (base), 28 layers, d_model = 1536, float16 via
  TransformerLens
- Activations: residual stream (`resid_post`) at the final token, all layers
- Probes: `LogisticRegression(max_iter=2000, C=0.1)`, one per layer, single
  80/20 split reused across layers
- Datasets: `cities`, `neg_cities`, `sp_en_trans` from geometry-of-truth,
  plus 1,600 generated compound statements (200 per connective ×
  conjunct-pattern cell)

## Reproducing

```bash
git clone https://github.com/graceshan/truth-probing.git
cd truth-probing
git clone https://github.com/saprmarks/geometry-of-truth.git   # source datasets
pip install -r requirements.txt
```

Activation extraction needs a GPU. Run `notebooks/extract_colab.ipynb` in
Colab; it clones this repo, calls `src/extract.py`, and writes `.npy` arrays
to Drive. Run **both** extraction sections — section 7 (`cities`,
`neg_cities`, `sp_en_trans`) and section 8 (`compound_cities`, needed by
scripts `05`-`07`) — then download all 8 files to `data/activations/`:

```
cities_acts.npy            cities_labels.npy
neg_cities_acts.npy        neg_cities_labels.npy
sp_en_trans_acts.npy       sp_en_trans_labels.npy
compound_cities_acts.npy   compound_cities_labels.npy
```

Everything after extraction is CPU-only and runs from the saved arrays.
Scripts are numbered in write-up figure order and run as modules (numeric
prefixes aren't valid in a plain `python path/to/file.py` import, but `-m`
resolves them fine via filesystem lookup):

```bash
python -m scripts.00_generate_compounds       # regenerates data/compound_cities.csv
python -m scripts.01_probe_accuracy_by_layer
python -m scripts.02_shuffled_label_control
python -m scripts.03_transfer_and_negation
python -m scripts.04_lr_vs_dim_comparison
python -m scripts.05_compound_analysis
python -m scripts.06_conjunct_regression
python -m scripts.07_capability_control
```

## What each script produces

| Script | Figures | Results |
|---|---|---|
| `00_generate_compounds.py` | — | `data/compound_cities.csv` |
| `01_probe_accuracy_by_layer.py` | `probe_accuracy/*.png` | `results/*/accuracy.csv` |
| `02_shuffled_label_control.py` | `control/*.png` | `results/*/control_accuracy.csv` |
| `03_transfer_and_negation.py` | `transfer/cities_neg_cities_transfer.png`, `transfer/cities_sp_en_trans_transfer.png` | — |
| `04_lr_vs_dim_comparison.py` | `auc_comparison/*.png` | `results/*/auc.csv` |
| `05_compound_analysis.py` | `compound_cities/compound_*.png`, `compound_cities/cities_probe_on_compound.png` | `compound_cities/*.csv` |
| `06_conjunct_regression.py` | `compound_cities/conjunct_regression*.png` | `compound_cities/conjunct_regression*.csv` |
| `07_capability_control.py` | `compound_cities/capability_control.png` | `compound_cities/capability_control.csv` |

Figure/result paths above are relative to `figures/` and `results/`
respectively.

## Notes

- `.npy` activation arrays are gitignored (hundreds of MB) but fully
  regenerable from the extraction notebook.
- `geometry-of-truth/` is cloned into the project root and gitignored;
  dataset paths are relative to it in both local and Colab environments.
- All random seeds are fixed; probe hyperparameters live in `src/probes.py`.

## Related work

- Marks & Tegmark (2023), *The Geometry of Truth* — the result reproduced
  here, including the negation failure
- Bao et al. (2025), *Probing the Geometry of Truth* — capability-dependence
  of truth-direction generalization; also reports stronger performance on
  conjunctions than disjunctions
- MacDiarmid et al. (2024), *Simple probes can catch sleeper agents* — the
  safety application motivating this line of work
