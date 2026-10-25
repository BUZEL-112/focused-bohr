"""Stage 1: weight-delta SVD - Frobenius / spectral norm, stable rank, top-k energy."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from src.spec04_dolos_audit.models import MODULE_NAMES, LayerDeltaMetrics
from src.spec04_dolos_audit.repositories.checkpoint_repository import Checkpoint

# Below this min(dim) an exact LAPACK SVD is cheap; above it we use randomized SVD.
EXACT_SVD_MAX_DIM = 512


@dataclass(frozen=True)
class TruncatedSvd:
    """Top-k factors of a matrix: M ~= U diag(S) V^T (U: [m, k], S: [k], Vt: [k, n])."""

    u: np.ndarray
    s: np.ndarray
    vt: np.ndarray

    def reconstruct(self, k: int | None = None) -> np.ndarray:
        k = self.s.size if k is None else min(k, self.s.size)
        return (self.u[:, :k] * self.s[:k]) @ self.vt[:k]


def randomized_svd(
    matrix: np.ndarray, k: int, oversample: int = 10, n_iter: int = 4, seed: int = 0
) -> TruncatedSvd:
    """Halko-Martinsson-Tropp randomized SVD with QR-stabilized power iterations."""
    m, n = matrix.shape
    k = min(k, m, n)
    rank = min(k + oversample, m, n)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(matrix @ rng.standard_normal((n, rank)))
    for _ in range(n_iter):
        q, _ = np.linalg.qr(matrix.T @ q)
        q, _ = np.linalg.qr(matrix @ q)
    small_u, s, vt = np.linalg.svd(q.T @ matrix, full_matrices=False)
    return TruncatedSvd(u=(q @ small_u)[:, :k], s=s[:k], vt=vt[:k])


def top_k_svd(matrix: np.ndarray, k: int, seed: int = 0) -> TruncatedSvd:
    """Exact SVD for small matrices, randomized SVD otherwise."""
    matrix = np.asarray(matrix, dtype=np.float64)
    k = min(k, *matrix.shape)
    if min(matrix.shape) <= EXACT_SVD_MAX_DIM:
        u, s, vt = np.linalg.svd(matrix, full_matrices=False)
        return TruncatedSvd(u=u[:, :k], s=s[:k], vt=vt[:k])
    return randomized_svd(matrix, k, seed=seed)


def stable_rank(frobenius_norm: float, spectral_norm: float) -> float:
    """srank = ||M||_F^2 / ||M||_2^2 (>= 1); +inf for the zero matrix."""
    if spectral_norm <= 0.0:
        return float("inf")
    return (frobenius_norm / spectral_norm) ** 2


class WeightDeltaSvdService:
    """Computes Delta W = W_Dolos - W_Qwen and its spectral statistics (spec 4.1)."""

    def __init__(self, top_k: int = 8, seed: int = 0) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.top_k = top_k
        self.seed = seed

    def matrix_metrics(
        self, delta: np.ndarray, layer_idx: int = -1, module_name: str = ""
    ) -> tuple[LayerDeltaMetrics, TruncatedSvd]:
        """Metrics + top-k factors of one delta matrix. ||.||_F is exact (from the matrix)."""
        delta = np.asarray(delta, dtype=np.float64)
        frobenius = float(np.linalg.norm(delta))
        svd = top_k_svd(delta, self.top_k, seed=self.seed)
        spectral = float(svd.s[0]) if svd.s.size else 0.0
        energy = frobenius**2
        metrics = LayerDeltaMetrics(
            layer_idx=layer_idx,
            module_name=module_name,
            frobenius_norm=frobenius,
            spectral_norm=spectral,
            stable_rank=stable_rank(frobenius, spectral),
            top1_energy_ratio=spectral**2 / energy if energy > 0 else 0.0,
            topk_energy_ratio=float(np.sum(svd.s**2)) / energy if energy > 0 else 0.0,
            top_singular_values=tuple(float(x) for x in svd.s),
        )
        return metrics, svd

    def scan(
        self,
        dolos: Checkpoint,
        base: Checkpoint,
        layers: Iterable[int] | None = None,
        modules: Iterable[str] = MODULE_NAMES,
    ) -> tuple[list[LayerDeltaMetrics], dict[tuple[int, str], TruncatedSvd]]:
        """Scan every (layer, module); one tensor pair in memory at a time."""
        if dolos.num_layers != base.num_layers:
            raise ValueError(f"layer count mismatch: {dolos.num_layers} vs {base.num_layers}")
        layers = range(dolos.num_layers) if layers is None else layers
        modules = tuple(modules)
        metrics: list[LayerDeltaMetrics] = []
        factors: dict[tuple[int, str], TruncatedSvd] = {}
        for layer in layers:
            for module in modules:
                delta = dolos.module_matrix(layer, module).astype(np.float64) - base.module_matrix(layer, module)
                row, svd = self.matrix_metrics(delta, layer, module)
                metrics.append(row)
                factors[(layer, module)] = svd
        return metrics, factors

    @staticmethod
    def rank_by_concentration(metrics: list[LayerDeltaMetrics]) -> list[LayerDeltaMetrics]:
        """Most low-rank-concentrated (highest sigma_1^2 / ||dW||_F^2) first."""
        return sorted(metrics, key=lambda m: m.top1_energy_ratio, reverse=True)

    @staticmethod
    def concentration_outliers(metrics: list[LayerDeltaMetrics], z_threshold: float = 4.0) -> list[LayerDeltaMetrics]:
        """Modules whose top-1 energy ratio is a robust (median/MAD) z-outlier among same-type modules."""
        flagged: list[LayerDeltaMetrics] = []
        for module in {m.module_name for m in metrics}:
            rows = [m for m in metrics if m.module_name == module]
            values = np.array([m.top1_energy_ratio for m in rows])
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median))) * 1.4826
            scale = mad if mad > 1e-12 else max(float(values.std()), 1e-12)
            flagged.extend(row for row, value in zip(rows, values) if (value - median) / scale > z_threshold)
        return sorted(flagged, key=lambda m: m.top1_energy_ratio, reverse=True)

    @staticmethod
    def auxiliary_deltas(dolos: Checkpoint, base: Checkpoint) -> dict[str, float]:
        """||Delta||_F of embed_tokens / final norm / lm_head (spec 5, phase 1 step 2)."""
        report: dict[str, float] = {}
        for name in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
            if dolos.has_tensor(name) and base.has_tensor(name):
                delta = dolos.get_tensor(name).astype(np.float64) - base.get_tensor(name)
                report[name] = float(np.linalg.norm(delta))
        return report
