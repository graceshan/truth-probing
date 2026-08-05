"""Shared uncertainty-quantification helpers.

Two distinct kinds of interval used across the pipeline:

- seed_mean_ci: across a small number (5) of seed-level point estimates
  (different group-level train/test splits) -- a t-distribution CI on the
  mean, appropriate for small n rather than a normal-approximation.
- percentile_bootstrap: resample-and-recompute, for quantities (ratios,
  differences of group means, regression coefficients) whose sampling
  distribution isn't analytically obvious. 1000 resamples, 2.5/97.5
  percentile interval.

Both return (point_estimate_or_mean, lo, hi) so callers can plot a line +
shaded band the same way regardless of which kind of interval it is.
"""

import numpy as np
from scipy import stats as scipy_stats


def seed_mean_ci(values, confidence: float = 0.95):
    """Mean and a t-distribution-based CI across seed-level point estimates.

    `values` has the seed axis first, e.g. shape [n_seeds] or
    [n_seeds, n_layers]. Returns (mean, lo, hi), each with the seed axis
    removed. Uses the t distribution (not a fixed 1.96) since n_seeds is
    small (5) -- the normal approximation understates the interval at that
    sample size.
    """
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    mean = values.mean(axis=0)
    sem = values.std(axis=0, ddof=1) / np.sqrt(n)
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, mean - t_crit * sem, mean + t_crit * sem


def percentile_bootstrap(statistic_fn, n_resamples: int = 1000, confidence: float = 0.95, rng=None):
    """Run `statistic_fn(rng)` n_resamples times and return the percentile
    interval across the results.

    `statistic_fn` should draw whatever resampled indices it needs using the
    provided numpy Generator and return a scalar (or array, for a
    vectorized batch of quantities computed together). The *point estimate*
    reported elsewhere is never derived from this function -- it always
    comes from the original unresampled computation; this only supplies
    (lo, hi).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    samples = np.array([statistic_fn(rng) for _ in range(n_resamples)])
    alpha = (1 - confidence) / 2
    lo = np.percentile(samples, 100 * alpha, axis=0)
    hi = np.percentile(samples, 100 * (1 - alpha), axis=0)
    return lo, hi, samples


def intervals_overlap(lo1, hi1, lo2, hi2):
    """Elementwise: do [lo1, hi1] and [lo2, hi2] overlap?"""
    lo1, hi1, lo2, hi2 = np.asarray(lo1), np.asarray(hi1), np.asarray(lo2), np.asarray(hi2)
    return ~((hi1 < lo2) | (hi2 < lo1))


def assert_unchanged(name: str, old, new, atol: float = 1e-6):
    """Point-estimate regression check: bootstrap/seed additions must never
    change the original computation. Raises with a clear message (not a
    silent warning) if `new` drifted from `old` beyond floating-point noise.

    atol=1e-6 rather than machine epsilon: different BLAS thread counts (see
    the OMP_NUM_THREADS etc. set at the top of scripts that use this) can
    shift floating-point summation order, nudging an iterative solver's
    (e.g. LogisticRegression's lbfgs) exact convergence point at the ~1e-8
    scale -- still 100x tighter than sklearn's own default solver
    convergence tolerance (1e-4), so this stays a real regression check
    without false-positiving on threading-induced last-bit noise.
    """
    old, new = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    if not np.allclose(old, new, atol=atol, rtol=1e-9, equal_nan=True):
        diff = np.abs(old - new)
        worst = np.nanargmax(diff)
        raise AssertionError(
            f"{name}: point estimate changed after adding bootstrap -- "
            f"max abs diff {diff.flat[worst]:.3e} at flat index {worst} "
            f"(old={old.flat[worst]!r}, new={new.flat[worst]!r}). "
            "This should be impossible if the original computation path "
            "wasn't touched; something got broken."
        )
