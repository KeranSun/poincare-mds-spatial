"""Statistical utilities for Poincaré MDS analyses.

Provides bootstrap confidence intervals, effect sizes, and multiple testing
correction used across all figure scripts.
"""

import numpy as np
from scipy import stats


def bootstrap_ci(data, stat_fn=np.mean, n_boot=1000, alpha=0.05, seed=42):
    """Bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : array-like
        1D array of observations.
    stat_fn : callable
        Statistic to compute (default: mean).
    n_boot : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (e.g., 0.05 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (stat, ci_low, ci_high) : tuple of floats
    """
    rng = np.random.RandomState(seed)
    data = np.asarray(data)
    stat = stat_fn(data)
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = stat_fn(sample)
    ci_low = np.percentile(boot_stats, 100 * alpha / 2)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    return stat, ci_low, ci_high


def bootstrap_ci_diff(g1, g2, stat_fn=np.mean, n_boot=1000, alpha=0.05, seed=42):
    """Bootstrap CI for the difference stat(g1) - stat(g2)."""
    rng = np.random.RandomState(seed)
    g1, g2 = np.asarray(g1), np.asarray(g2)
    diff = stat_fn(g1) - stat_fn(g2)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        s1 = rng.choice(g1, size=len(g1), replace=True)
        s2 = rng.choice(g2, size=len(g2), replace=True)
        boot_diffs[i] = stat_fn(s1) - stat_fn(s2)
    ci_low = np.percentile(boot_diffs, 100 * alpha / 2)
    ci_high = np.percentile(boot_diffs, 100 * (1 - alpha / 2))
    return diff, ci_low, ci_high


def cohens_d(g1, g2):
    """Cohen's d effect size (pooled standard deviation)."""
    g1, g2 = np.asarray(g1), np.asarray(g2)
    n1, n2 = len(g1), len(g2)
    pooled_std = np.sqrt(((n1 - 1) * g1.std(ddof=1) ** 2 + (n2 - 1) * g2.std(ddof=1) ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / pooled_std


def cliffs_delta(g1, g2):
    """Cliff's delta non-parametric effect size."""
    g1, g2 = np.asarray(g1), np.asarray(g2)
    n1, n2 = len(g1), len(g2)
    dominance = 0.0
    for x in g1:
        for y in g2:
            if x > y:
                dominance += 1
            elif x < y:
                dominance -= 1
    return dominance / (n1 * n2)


def rank_biserial(g1, g2):
    """Rank-biserial correlation r (from Mann-Whitney U)."""
    g1, g2 = np.asarray(g1), np.asarray(g2)
    u, _ = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    n1, n2 = len(g1), len(g2)
    return 1 - (2 * u) / (n1 * n2)


def fdr_correction(pvalues, alpha=0.05):
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : array-like
        Raw p-values.
    alpha : float
        FDR threshold.

    Returns
    -------
    (reject, q_values) : tuple of arrays
        Boolean mask of rejected hypotheses and FDR-adjusted q-values.
    """
    pvalues = np.asarray(pvalues)
    n = len(pvalues)
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n
    # Find the largest i where p[i] <= threshold[i]
    below = sorted_p <= thresholds
    if not below.any():
        return np.zeros(n, dtype=bool), np.ones(n)
    max_idx = np.max(np.where(below))
    reject = np.zeros(n, dtype=bool)
    reject[sorted_idx[:max_idx + 1]] = True
    # Compute q-values
    q_values = np.ones(n)
    q_values[sorted_idx] = sorted_p * n / np.arange(1, n + 1)
    # Enforce monotonicity (from last to first)
    for i in range(n - 2, -1, -1):
        idx = sorted_idx[i]
        next_idx = sorted_idx[i + 1]
        q_values[idx] = min(q_values[idx], q_values[next_idx])
    q_values = np.clip(q_values, 0, 1)
    return reject, q_values


def format_p(p, threshold=0.001):
    """Format p-value for publication. Uses scientific notation for p < threshold."""
    if p < threshold:
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / 10 ** exponent
        return f'{mantissa:.1f} × 10⁻{str(abs(exponent)).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))}'
    elif p < 0.01:
        return f'{p:.3f}'
    else:
        return f'{p:.2f}'


def format_ci(stat, ci_low, ci_high, decimals=3):
    """Format a statistic with 95% CI for publication."""
    return f'{stat:.{decimals}f} [{ci_low:.{decimals}f}, {ci_high:.{decimals}f}]'


def spearman_with_ci(x, y, n_boot=1000, seed=42):
    """Spearman correlation with bootstrap 95% CI."""
    rho, p = stats.spearmanr(x, y)
    # Bootstrap CI on rho
    rng = np.random.RandomState(seed)
    n = len(x)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_rhos[i], _ = stats.spearmanr(x[idx], y[idx])
    ci_low = np.percentile(boot_rhos, 2.5)
    ci_high = np.percentile(boot_rhos, 97.5)
    return rho, p, ci_low, ci_high


def pearson_with_ci(x, y, n_boot=1000, seed=42):
    """Pearson correlation with bootstrap 95% CI."""
    r, p = stats.pearsonr(x, y)
    rng = np.random.RandomState(seed)
    n = len(x)
    boot_rs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_rs[i], _ = stats.pearsonr(x[idx], y[idx])
    ci_low = np.percentile(boot_rs, 2.5)
    ci_high = np.percentile(boot_rs, 97.5)
    return r, p, ci_low, ci_high
