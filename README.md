# truth-probing

Does the linear "truth direction" in a small language model actually encode truth, or something correlated with it?

This project reproduces the standard truth-probing result on Qwen2.5-1.5B, then tests the recovered direction against logical negation and compositional (AND/OR) statements.

**📄 Full write-up: [graceshan.github.io/truth-probing](https://graceshan.github.io/truth-probing)**

This README covers reproduction only. All findings, figures, and discussion are in the write-up.

---

## Setup

- **Model:** Qwen2.5-1.5B (base), 28 layers, d_model = 1536, float16 via [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
- **Activations:** residual stream (`resid_post`) at the final token, all layers
- **Probes:** `LogisticRegression(max_iter=2000, C=0.1)`, one per layer
- **Splits:** group-level on the entity identifier (city, Spanish word, or constituent pair) so no entity crosses train/test. Within-dataset results are 5 seeds with 95% intervals. Transfer results use 1000-resample bootstrap over the evaluation data, since the source probe is fit once on the full source dataset.
- **Datasets:** `cities`, `neg_cities`, `sp_en_trans` from [geometry-of-truth](https://github.com/saprmarks/geometry-of-truth), plus 1,600 generated compound statements (200 per connective × conjunct-pattern cell)

## Reproducing

```bash
git clone https://github.com/graceshan/truth-probing.git
cd truth-probing
git clone https://github.com/saprmarks/geometry-of-truth.git   # source datasets
pip install -r requirements.txt
```

Activation extraction needs a GPU. Run `notebooks/extract_colab.ipynb` in Colab; it clones this repo, calls `src/extract.py`, and writes `.npy` arrays to Drive. Run both extraction sections — section 7 (`cities`, `neg_cities`, `sp_en_trans`) and section 8 (`compound_cities`) — then download all 8 files to `data/activations/`:

```
cities_acts.npy            cities_labels.npy
neg_cities_acts.npy        neg_cities_labels.npy
sp_en_trans_acts.npy       sp_en_trans_labels.npy
compound_cities_acts.npy   compound_cities_labels.npy
```

Everything after extraction is CPU-only and runs from the saved arrays. Scripts run as modules, since numeric prefixes aren't valid in a plain `python path/to/file.py` import but `-m` resolves them via filesystem lookup:

```bash
python -m scripts.00_generate_compounds
python -m scripts.01_probe_accuracy_by_layer
python -m scripts.03_transfer_and_negation                      # cities/neg_cities (default)
python -m scripts.03_transfer_and_negation cities sp_en_trans
python -m scripts.05_compound_analysis
python -m scripts.06_conjunct_regression
python -m scripts.07_capability_control
python -m scripts.08_transfer_auroc
python -m scripts.14_union_cities_negcities
python -m scripts.15_pooled_probe_connective_diagnostic
python -m scripts.16_attenuation_vs_compression
```

## Figures in the write-up

| Figure | Script | Output |
|---|---|---|
| 1. Probe accuracy by layer | `01_probe_accuracy_by_layer` | `figures/probe_accuracy/` |
| 2. Cross-dataset transfer, accuracy and AUROC | `03_transfer_and_negation`, `08_transfer_auroc` | `figures/transfer/` |
| 3. Compound scores by conjunct pattern | `05_compound_analysis` | `figures/compound_cities/cities_probe_on_compound.png` |
| 4. Normalized position of mixed patterns | `05_compound_analysis` | `figures/compound_cities/compound_relative_position.png` |
| 5. Compounds, accuracy vs AUROC | `08_transfer_auroc` | `figures/transfer_auroc_compound.png` |
| 6. Capability control | `07_capability_control` | `figures/compound_cities/capability_control.png` |

## Numbers cited in the write-up

| Claim | Source |
|---|---|
| Layer 13 accuracy 0.500, AUROC 0.0001 | `18_neg_cities_inversion_diagnostic` |
| In-distribution plateau 0.990–0.995, layers 10–27 | `results/cities/accuracy.csv` |
| Transfer AUROC range 0.0001–0.334 | `results/transfer_auroc.csv` |
| Compound AUROC by label scheme | `results/transfer_auroc.csv` |
| Regression coefficients 0.395 + 0.608 at layer 15 | `16_attenuation_vs_compression` |
| 8-cell means for the compound-trained probe | `15_pooled_probe_connective_diagnostic` |
| Source-probe cosine stability 0.92–0.97 | `10_source_probe_stability` |
| Union probe: 0.99 accuracy, 0.59 cosine, AUROC 0.41–0.78 | `14_union_cities_negcities` |

## Supporting analyses

Not discussed in the write-up, but they back methodology choices or served as controls:

| Script | What it checks |
|---|---|
| `02_shuffled_label_control` | Shuffled labels stay at chance across all layers, so the pipeline can't manufacture signal from noise |
| `04_lr_vs_dim_comparison` | Logistic regression against the difference-in-means direction used by MacDiarmid et al. |
| `09_group_split_comparison` | Row-level vs group-level splitting. Justifies the group-level convention above. Max difference 0.09, at an early low-signal layer |
| `10_source_probe_stability` | Cities probes fit on 5 different subsamples, pairwise cosine similarity between directions |
| `12_confound_selection_plot` | In-distribution accuracy against transfer AUROC by layer |

`scripts/exploratory/` holds analyses that didn't make the final write-up (crossover-layer estimation, cross-connective transfer).

## Notes

- `.npy` activation arrays are gitignored (hundreds of MB) but fully regenerable from the extraction notebook.
- `results/*/directions.npy` holds the unit-normalized mean probe direction per dataset, if you want the vectors without re-extracting.
- `geometry-of-truth/` is cloned into the project root and gitignored; dataset paths are relative to it in both local and Colab environments.
- All random seeds are fixed; probe hyperparameters live in `src/probes.py`.

## References

- Marks, S. and Tegmark, M. (2023). [The Geometry of Truth](https://arxiv.org/abs/2310.06824)
- Bao, Y. et al. (2025). [Probing the Geometry of Truth](https://arxiv.org/abs/2506.00823)
- MacDiarmid, M. et al. (2024). [Simple probes can catch sleeper agents](https://www.anthropic.com/research/probes-catch-sleeper-agents)
