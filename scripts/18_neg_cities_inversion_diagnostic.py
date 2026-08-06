"""Diagnostic: is the neg_cities transfer AUROC collapse around layer 13
(accuracy 0.500, AUROC ~0.0001 per results/transfer_auroc.csv) a genuine
rank inversion, a degenerate saturated-threshold prediction, or a mix?

Reuses fit_probe's convention exactly (all of cities, no held-out split --
same as 03/07/08, the scripts that produced the numbers being checked
here). threshold=0 is sklearn's own decision boundary for predict(): class
1 iff decision_function >= 0, which is what probe.score() (the "accuracy"
column in transfer_auroc.csv) uses -- confirmed explicitly below rather
than assumed.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

from src.data import load_activations
from src.probes import fit_probe

LAYERS = [11, 12, 13, 14, 15]
QUANTILES = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]

acts_cities, labels_cities = load_activations("cities")
acts_neg, labels_neg = load_activations("neg_cities")

for L in LAYERS:
    probe = fit_probe(acts_cities[:, L], labels_cities)
    scores = probe.decision_function(acts_neg[:, L])
    preds = probe.predict(acts_neg[:, L])

    # confirm predict() really is threshold-at-zero on decision_function
    assert np.array_equal(preds, (scores >= 0).astype(int)), "predict() is not a plain >=0 threshold on decision_function"

    acc = (preds == labels_neg).mean()
    cm = confusion_matrix(labels_neg, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    pred_pos_frac = preds.mean()
    pred_neg_frac = 1 - pred_pos_frac

    true_scores = scores[labels_neg == 1]
    false_scores = scores[labels_neg == 0]

    def stats(x):
        q = np.quantile(x, QUANTILES)
        return {
            "mean": x.mean(), "std": x.std(),
            "min": q[0], "p10": q[1], "p25": q[2], "median": q[3], "p75": q[4], "p90": q[5], "max": q[6],
        }

    true_stats = stats(true_scores)
    false_stats = stats(false_scores)

    auroc_fwd = roc_auc_score(labels_neg, scores)
    auroc_rev = roc_auc_score(1 - labels_neg, scores)

    print(f"=== layer {L} ===")
    print(f"accuracy (threshold=0 on decision_function, confirmed above): {acc!r}")
    print(f"confusion matrix [rows=true 0/1, cols=pred 0/1]: TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"predicted-class balance: {pred_pos_frac:.6f} predicted positive, {pred_neg_frac:.6f} predicted negative "
          f"(n={len(labels_neg)}, true base rate {labels_neg.mean():.4f})")
    print(f"true-class (label=1) score dist:  mean={true_stats['mean']!r} std={true_stats['std']!r}")
    print(f"  quantiles: min={true_stats['min']!r} p10={true_stats['p10']!r} p25={true_stats['p25']!r} "
          f"median={true_stats['median']!r} p75={true_stats['p75']!r} p90={true_stats['p90']!r} max={true_stats['max']!r}")
    print(f"false-class (label=0) score dist: mean={false_stats['mean']!r} std={false_stats['std']!r}")
    print(f"  quantiles: min={false_stats['min']!r} p10={false_stats['p10']!r} p25={false_stats['p25']!r} "
          f"median={false_stats['median']!r} p75={false_stats['p75']!r} p90={false_stats['p90']!r} max={false_stats['max']!r}")
    print(f"AUROC(label): {auroc_fwd!r}")
    print(f"AUROC(1-label): {auroc_rev!r}")
    print(f"AUROC(label) + AUROC(1-label) = {auroc_fwd + auroc_rev!r} (should be exactly 1.0 up to fp noise)")
    print()
