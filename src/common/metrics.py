"""Shared statistical primitives for the audit packages.

Everything here is pure stdlib + numpy so the detector services stay dependency-light.
Includes the entropy / mutual-information estimators with Miller-Madow bias correction
used by the Net-Influence (spec 01) and waterbed telemetry (spec 02) services, the
rank-based AUROC / Mann-Whitney machinery used by every benchmark, and small helpers
(z-scores, chi-square uniformity deviation, power-law tail fit) used across specs.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Hashable, Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Entropy and (conditional) mutual information
# --------------------------------------------------------------------------- #
def _occupied_cells(joint: Counter) -> int:
    """Number of occupied cells in an empirical joint distribution."""
    return len(joint)


def empirical_entropy(symbols: Sequence[Hashable], bias_correction: bool = False) -> float:
    """Plug-in Shannon entropy in bits; optional Miller-Madow correction."""
    n = len(symbols)
    if n == 0:
        return 0.0
    counts = Counter(symbols)
    h = 0.0
    for c in counts.values():
        p = c / n
        h -= p * math.log2(p)
    if bias_correction:
        # Miller-Madow: (K-1)/(2N) nats of upward bias, converted to bits.
        h += _mm_entropy_bias(_occupied_cells(counts), n)
    return h


def _entropy_bits(counts: Counter, n: int) -> float:
    h = 0.0
    for c in counts.values():
        if c > 0:
            p = c / n
            h -= p * math.log2(p)
    return h


def _mm_entropy_bias(cells: int, n: int) -> float:
    """Miller-Madow entropy bias in bits: (K-1)/(2N) nats converted to bits."""
    if n == 0:
        return 0.0
    return (cells - 1) / (2.0 * n) / math.log(2.0)


def mutual_information(
    xs: Sequence[Hashable], ys: Sequence[Hashable], bias_correction: bool = False
) -> float:
    """Empirical mutual information I(X; Y) in bits (clamped at 0)."""
    n = len(xs)
    if n == 0 or n != len(ys):
        raise ValueError("xs and ys must be non-empty and of equal length")
    c_xy: Counter = Counter(zip(xs, ys))
    c_x: Counter = Counter(xs)
    c_y: Counter = Counter(ys)
    mi = 0.0
    for (x, y), c in c_xy.items():
        p_xy = c / n
        mi += p_xy * math.log2(p_xy / ((c_x[x] / n) * (c_y[y] / n)))
    if bias_correction:
        # I = H(X) + H(Y) - H(XY): add each entropy's Miller-Madow term with the same sign.
        # The net correction is negative because K_xy - 1 exceeds (K_x - 1) + (K_y - 1).
        correction = (_mm_entropy_bias(_occupied_cells(c_x), n) + _mm_entropy_bias(_occupied_cells(c_y), n)
                      - _mm_entropy_bias(_occupied_cells(c_xy), n))
        mi += correction
    return max(0.0, mi)


def conditional_mutual_information(
    xs: Sequence[Hashable],
    ys: Sequence[Hashable],
    contexts: Sequence[Hashable],
    bias_correction: bool = False,
) -> float:
    """Empirical conditional mutual information I(X; Y | Z) in bits (clamped at 0).

    ``contexts`` holds one arbitrary (hashable, e.g. tuple) context symbol per sample,
    typically the compound past of the remaining agents. The Miller-Madow correction
    (K_xz + K_yz - K_z - K_xyz) / (2N) nats is added to the plug-in estimate; it is
    negative whenever the joint table occupies more cells than the marginals.
    """
    n = len(xs)
    if n == 0 or len(ys) != n or len(contexts) != n:
        raise ValueError("xs, ys and contexts must be non-empty and of equal length")
    triples = list(zip(xs, ys, contexts))
    c_xyz: Counter = Counter(triples)
    c_xz: Counter = Counter((x, z) for x, _y, z in triples)
    c_yz: Counter = Counter((y, z) for _x, y, z in triples)
    c_z: Counter = Counter(contexts)
    mi = 0.0
    for (x, y, z), c in c_xyz.items():
        p_xyz = c / n
        p_xz = c_xz[(x, z)] / n
        p_yz = c_yz[(y, z)] / n
        p_z = c_z[z] / n
        if p_xz > 0 and p_yz > 0:
            mi += p_xyz * math.log2((p_xyz * p_z) / (p_xz * p_yz))
    if bias_correction:
        # I = H(XZ) + H(YZ) - H(Z) - H(XYZ): each entropy's term keeps its sign.
        correction = (_mm_entropy_bias(_occupied_cells(c_xz), n) + _mm_entropy_bias(_occupied_cells(c_yz), n)
                      - _mm_entropy_bias(_occupied_cells(c_z), n) - _mm_entropy_bias(_occupied_cells(c_xyz), n))
        mi += correction
    return max(0.0, mi)


# --------------------------------------------------------------------------- #
# Ranking and hypothesis testing
# --------------------------------------------------------------------------- #
def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic (tie-aware).

    Returns 0.5 for chance, 1.0 for perfect separation.
    """
    pos, neg = list(positive_scores), list(negative_scores)
    if not pos or not neg:
        raise ValueError("both score lists must be non-empty")
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def mann_whitney_p_value(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> float:
    """Two-sided normal-approximation p-value for the Mann-Whitney U test (tie-corrected)."""
    pos, neg = list(positive_scores), list(negative_scores)
    n1, n2 = len(pos), len(neg)
    if n1 == 0 or n2 == 0:
        raise ValueError("both score lists must be non-empty")
    u = auroc(pos, neg) * n1 * n2
    mu_u = n1 * n2 / 2.0
    # Tie-corrected variance (Huffman & Welch): account for repeated values.
    all_values = pos + neg
    counts = Counter(all_values)
    n = n1 + n2
    tie_term = sum(c**3 - c for c in counts.values())
    var_u = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else 0.0
    if var_u <= 0.0:
        return 0.0 if u != mu_u else 1.0
    z = (abs(u - mu_u) - 0.5) / math.sqrt(var_u)
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def zscores(values: Sequence[float]) -> list[float]:
    """Population z-scores; a constant vector maps to all-zero scores."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return []
    std = float(arr.std())
    if std < 1e-12:
        return [0.0] * arr.size
    return [float(v) for v in (arr - arr.mean()) / std]


def chi_square_uniform_deviation(counts: Sequence[int]) -> float:
    """Chi-square statistic against uniform, normalized by N in [0, k-1].

    0.0 = perfectly uniform digits (no signaling carrier), larger = more structure.
    """
    counts_arr = np.asarray(counts, dtype=float)
    n = counts_arr.sum()
    k = counts_arr.size
    if n == 0 or k == 0:
        return 0.0
    expected = n / k
    return float(((counts_arr - expected) ** 2 / expected).sum() / n)


def power_law_tail_exponent(values: Sequence[float], fit_points: int = 20) -> float:
    """Tail exponent alpha of the power law sigma_i ~ i^-alpha, fit on the top values.

    Returns the negative slope of log(sigma) vs log(i) over the first ``fit_points``
    ordered magnitudes. Larger alpha = faster-decaying (more benign) spectrum.
    """
    arr = np.sort(np.asarray(values, dtype=float))[::-1]
    arr = arr[arr > 0][:fit_points]
    if arr.size < 3:
        return 0.0
    idx = np.arange(1, arr.size + 1, dtype=float)
    slope = float(np.polyfit(np.log(idx), np.log(arr), 1)[0])
    return -slope


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation coefficient; 0.0 when either input is constant."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("inputs must be equal-length with at least 2 samples")
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    sx, sy = x.std(), y.std()
    if sx < 1e-12 or sy < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
