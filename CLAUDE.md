# Project: Geometry of Truth reproduction + contrast-pair direction comparison
- GPU extraction runs on Colab via notebooks/extract_colab.ipynb OR locally on MPS —
everything else runs locally on CPU from saved .npy files in data/activations/ (gitignored).
- Activations: shape [n_statements, n_layers, d_model], resid_post, LAST token.
- SIGN CONVENTION: all directions oriented so positive = true/honest side.
- Probes: LogisticRegression(max_iter=2000, C=0.1); split indices once, reuse across layers.
- Never commit *.npy files. Figures go in figures/, saved at dpi=150.
- Datasets: geometry-of-truth repo CSVs, columns `statement`, `label` (1=true).
