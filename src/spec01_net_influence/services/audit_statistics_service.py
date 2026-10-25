"""Benchmark statistics for the spec 01 report: fast AUROC, bootstrap CIs, Wilson CIs, ROC.

``src.common.metrics.auroc`` is an O(n1 * n2) double loop, which is fine for unit tests
but too slow to bootstrap over ~1,800 cached runs. The rank-based estimator here gives
the identical tie-aware value in O(n log n).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def average_ranks(values: Sequence[float]) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (vectorized)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    _unique, inverse, counts = np.unique(arr, return_inverse=True, return_counts=True)
    upper = np.cumsum(counts)  # rank of the last member of each tie group
    lower = upper - counts + 1  # rank of the first member
    return ((lower + upper) / 2.0)[inverse.ravel()]


def fast_auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Tie-aware AUROC via the Mann-Whitney rank sum; NaN when a class is empty."""
    pos, neg = list(positive_scores), list(negative_scores)
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return float("nan")
    ranks = average_ranks(pos + neg)
    rank_sum = float(ranks[:n1].sum())
    return (rank_sum - n1 * (n1 + 1) / 2.0) / (n1 * n2)


def mann_whitney_p(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Two-sided tie-corrected normal-approximation p-value; NaN when a class is empty.

    Same statistic as ``src.common.metrics.mann_whitney_p_value`` but O(n log n).
    """
    pos, neg = list(positive_scores), list(negative_scores)
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        return float("nan")
    u = fast_auroc(pos, neg) * n1 * n2
    n = n1 + n2
    _unique, counts = np.unique(np.asarray(pos + neg, dtype=float), return_counts=True)
    tie_term = float(((counts.astype(float) ** 3) - counts).sum())
    var_u = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    mu_u = n1 * n2 / 2.0
    if var_u <= 0.0:
        return 0.0 if u != mu_u else 1.0
    z = max(0.0, abs(u - mu_u) - 0.5) / math.sqrt(var_u)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def bootstrap_auroc_ci(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    resamples: int = 300,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI from a class-stratified bootstrap; (nan, nan) when undefined."""
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    if pos.size == 0 or neg.size == 0 or resamples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = [
        fast_auroc(
            pos[rng.integers(0, pos.size, pos.size)].tolist(),
            neg[rng.integers(0, neg.size, neg.size)].tolist(),
        )
        for _ in range(resamples)
    ]
    tail = (1.0 - level) / 2.0 * 100.0
    return float(np.percentile(values, tail)), float(np.percentile(values, 100.0 - tail))


def wilson_interval(hits: int, trials: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate; (nan, nan) for zero trials."""
    if trials <= 0:
        return float("nan"), float("nan")
    rate = hits / trials
    denominator = 1.0 + z * z / trials
    centre = (rate + z * z / (2.0 * trials)) / denominator
    half = z * math.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def roc_curve(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> tuple[list[float], list[float]]:
    """(FPR, TPR) from a threshold sweep over the combined score distribution."""
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return [], []
    scores = np.unique(np.concatenate([pos, neg]))[::-1]
    thresholds = np.concatenate([[scores[0] + 1.0], scores, [scores[-1] - 1.0]])
    fpr = [float((neg >= threshold).mean()) for threshold in thresholds]
    tpr = [float((pos >= threshold).mean()) for threshold in thresholds]
    return fpr, tpr
