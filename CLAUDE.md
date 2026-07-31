# Project: Geometry of Truth reproduction + contrast-pair direction comparison
- GPU extraction runs on Colab via notebooks/extract_colab.ipynb OR locally on MPS —
everything else runs locally on CPU from saved .npy files in data/activations/ (gitignored).
- Activations: shape [n_statements, n_layers, d_model], resid_post, LAST token.
- SIGN CONVENTION: all directions oriented so positive = true/honest side.
- Probes: LogisticRegression(max_iter=2000, C=0.1); split indices once, reuse across layers.
- Never commit *.npy files. Figures go in figures/, saved at dpi=150.
- Datasets: geometry-of-truth repo CSVs, columns `statement`, `label` (1=true).
- Environment: local VS Code + .venv for ALL analysis (CPU, from saved .npy).
  GPU extraction only runs on Colab via notebooks/extract_colab.ipynb.
  NEVER load qwen locally — laptop RAM can't take it; work from data/activations/.
- Activations live in data/activations/ locally (gitignored), backed up on Drive.
- Model locked: qwen2.5-1.5b base, loaded via HF handoff + from_pretrained_no_processing, fp16.

## Research question
Does the linear "truth direction" in Qwen2.5-1.5B encode truth-conditional
content, or factual association?

## Key results (each needs a figure in the write-up)
1. Cities probe accuracy by layer — reproduces GoT, ~1.0 from layer 10.
   `figures/probe_accuracy/cities_probe_accuracy.png`
2. Cross-dataset + negation transfer by layer — below chance in layers ~10-16.
   `figures/transfer/cities_neg_cities_transfer.png`
3. Compound scores by pattern (TT/TF/FT/FF) x connective.
   `figures/compound_cities/compound_scores_by_layer.png`
4. Normalized position of mixed patterns — ~0.5, not 0/1.
   `figures/compound_cities/compound_relative_position.png`
5. FT-TF over layers — crossing at ~layer 18.
   `figures/compound_cities/compound_order_effect.png`
6. TT-FF spread per connective — AND ~1.5x OR.
   `figures/compound_cities/compound_spread.png`
7. Conjunct regression — R^2 ~0.5, b1/b2 crossing.
   `figures/compound_cities/conjunct_regression_coefficients.png`,
   `figures/compound_cities/conjunct_regression_by_connective.png`,
   `figures/compound_cities/conjunct_regression_r2.png`
8. Capability control — fresh compound probe 96%, cities transfer ~chance.
   `figures/compound_cities/capability_control.png`

## Folder structure
All compound-experiment results/figures live under `results/compound_cities/`
and `figures/compound_cities/` (scripts: 05_compound_analysis.py,
06_conjunct_regression.py, 07_capability_control.py). Plain per-dataset probing
(cities/neg_cities/sp_en_trans) uses `results/<dataset>/` and
`figures/{probe_accuracy,control,auc_comparison,transfer}/`.

## Additional conventions (compound experiment)
- Standardize compound scores against the cities distribution (z-score using
  the cities-trained probe's own decision_function mean/std).
- Directions stored as unit vectors (`d / np.linalg.norm(d)`), positive = true.
