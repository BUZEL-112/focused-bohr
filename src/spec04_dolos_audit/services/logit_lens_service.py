"""Stage 1b: weight-space logit lens - decode singular directions of Delta W into tokens."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from src.spec04_dolos_audit.models import READ_MODULES, WRITE_MODULES, VocabularyProjection
from src.spec04_dolos_audit.services.weight_svd_service import TruncatedSvd


def excess_kurtosis(values: np.ndarray) -> float:
    """Fisher excess kurtosis E[(x-mu)^4]/sigma^4 - 3 (0 for a Gaussian; 0.0 if constant)."""
    x = np.asarray(values, dtype=np.float64).ravel()
    if x.size < 4:
        return 0.0
    centered = x - x.mean()
    variance = float(np.mean(centered**2))
    if variance < 1e-24:
        return 0.0
    return float(np.mean(centered**4) / variance**2 - 3.0)


def canonical_sign(scores: np.ndarray) -> float:
    """+1 / -1 such that the largest-|z| vocabulary entry is positive.

    Singular vectors are defined only up to a joint sign flip of (u_i, v_i); fixing the
    sign on the vocabulary readout makes "promoted" deterministic and test-stable.
    """
    if scores.size == 0:
        return 1.0
    return 1.0 if scores.max() >= -scores.min() else -1.0


class VocabularyLogitLensService:
    """Projects singular vectors of Delta W onto the vocabulary (spec 4.2).

    * WRITE modules (o_proj, down_proj): z = W_U (gamma_final * u_i)
    * READ modules (q/k/v/gate/up_proj):   z = W_E (gamma_in * v_i)

    ``gamma_in`` (the layer's input RMSNorm weight) is an optional refinement of the spec's
    ``W_E v_i``: the module actually reads gamma_in * norm(x), so its residual-space read
    direction is gamma_in * v_i. Pass ``None`` for the spec's plain formula.
    """

    def __init__(self, top_n: int = 15, token_decoder: Callable[[int], str] | None = None) -> None:
        self.top_n = top_n
        self.token_decoder = token_decoder or (lambda token_id: f"<tok_{int(token_id)}>")

    @staticmethod
    def side_for(module_name: str) -> str:
        if module_name in WRITE_MODULES:
            return "output"
        if module_name in READ_MODULES:
            return "input"
        raise ValueError(f"unknown module: {module_name!r}")

    def vocabulary_scores(
        self,
        direction: np.ndarray,
        vocab_matrix: np.ndarray,
        norm_weight: np.ndarray | None = None,
    ) -> np.ndarray:
        """Raw vocabulary scores z = vocab_matrix @ (norm_weight * direction), sign-canonicalized."""
        direction = np.asarray(direction, dtype=np.float64)
        if direction.shape[0] != vocab_matrix.shape[1]:
            raise ValueError(
                f"direction dim {direction.shape[0]} != vocabulary d_model {vocab_matrix.shape[1]}"
            )
        if norm_weight is not None:
            direction = direction * norm_weight
        scores = np.asarray(vocab_matrix @ direction.astype(vocab_matrix.dtype), dtype=np.float64)
        return scores * canonical_sign(scores)

    def project(
        self,
        svd: TruncatedSvd,
        layer_idx: int,
        module_name: str,
        embedding: np.ndarray,
        unembedding: np.ndarray,
        final_norm: np.ndarray | None = None,
        input_norm: np.ndarray | None = None,
        singular_indices: Iterable[int] | None = None,
    ) -> list[VocabularyProjection]:
        """Top promoted / suppressed tokens for each requested singular direction."""
        side = self.side_for(module_name)
        indices = range(svd.s.size) if singular_indices is None else singular_indices
        projections: list[VocabularyProjection] = []
        for i in indices:
            if i >= svd.s.size:
                continue
            if side == "output":
                scores = self.vocabulary_scores(svd.u[:, i], unembedding, final_norm)
            else:
                scores = self.vocabulary_scores(svd.vt[i], embedding, input_norm)
            projections.append(self._readout(scores, layer_idx, module_name, i, side))
        return projections

    def _readout(
        self, scores: np.ndarray, layer_idx: int, module_name: str, singular_index: int, side: str
    ) -> VocabularyProjection:
        n = min(self.top_n, scores.size)
        promoted = np.argpartition(-scores, n - 1)[:n]
        promoted = promoted[np.argsort(-scores[promoted])]
        suppressed = np.argpartition(scores, n - 1)[:n]
        suppressed = suppressed[np.argsort(scores[suppressed])]
        return VocabularyProjection(
            layer_idx=layer_idx,
            module_name=module_name,
            singular_index=singular_index,
            side=side,
            top_promoted_tokens=tuple((self.token_decoder(int(t)), float(scores[t])) for t in promoted),
            top_suppressed_tokens=tuple((self.token_decoder(int(t)), float(scores[t])) for t in suppressed),
            top_promoted_ids=tuple(int(t) for t in promoted),
            top_suppressed_ids=tuple(int(t) for t in suppressed),
            projection_kurtosis=excess_kurtosis(scores),
        )
