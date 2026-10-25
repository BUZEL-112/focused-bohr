"""Immutable data models for the Dolos weight-delta SVD / reference-free audit (spec 04)."""

from __future__ import annotations

from dataclasses import dataclass, field

# Qwen2 projection matrices audited per layer (spec 04, section 3.1: 28 layers x 7 modules).
ATTENTION_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_MODULES = ("gate_proj", "up_proj", "down_proj")
MODULE_NAMES = ATTENTION_MODULES + MLP_MODULES

# Modules whose input is the (RMS-normed) residual stream: decode v_i through W_E.
READ_MODULES = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")
# Modules whose output is added to the residual stream: decode u_i through W_U.
WRITE_MODULES = ("o_proj", "down_proj")


@dataclass(frozen=True)
class LayerDeltaMetrics:
    """Spectral statistics of Delta W = W_Dolos - W_Qwen for one (layer, module).

    ``stable_rank`` = ||Delta W||_F^2 / ||Delta W||_2^2 (>= 1); ``top1_energy_ratio`` =
    sigma_1^2 / ||Delta W||_F^2 = 1 / stable_rank; ``topk_energy_ratio`` = E_k (spec 4.1).
    """

    layer_idx: int
    module_name: str
    frobenius_norm: float
    spectral_norm: float
    stable_rank: float
    top1_energy_ratio: float
    topk_energy_ratio: float = 0.0
    top_singular_values: tuple[float, ...] = field(default_factory=tuple)

    @property
    def key(self) -> tuple[int, str]:
        return (self.layer_idx, self.module_name)


@dataclass(frozen=True)
class VocabularyProjection:
    """Weight-space logit-lens readout of one singular direction (spec 4.2).

    ``side`` is ``"output"`` (u_i projected on W_U, RMSNorm-scaled) for o_proj/down_proj
    and ``"input"`` (v_i projected on W_E) for q/k/v/gate/up. Singular vectors are
    sign-canonicalized so the largest-|z| vocabulary entry is positive.
    """

    layer_idx: int
    module_name: str
    singular_index: int
    side: str
    top_promoted_tokens: tuple[tuple[str, float], ...]
    top_suppressed_tokens: tuple[tuple[str, float], ...]
    top_promoted_ids: tuple[int, ...] = field(default_factory=tuple)
    top_suppressed_ids: tuple[int, ...] = field(default_factory=tuple)
    projection_kurtosis: float = 0.0


@dataclass(frozen=True)
class SubspacePatchingResult:
    """One point of the stage 2 cure / implant curve at rank k.

    ``residual_delta_fraction`` = ||W_patched - W_target||_F / ||W_dolos - W_qwen||_F, where
    the target is Qwen for the cure and Dolos for the implant (0 = perfect reproduction).
    ``behavior_score`` is an optional behavioral readout (synthetic: fraction of trigger
    prompts whose top-1 next token is the planted target; real: flip rate if measured).
    """

    mode: str  # "cure" | "implant"
    rank_k: int
    modules: tuple[tuple[int, str], ...]
    residual_delta_fraction: float
    removed_energy_fraction: float
    behavior_score: float | None = None
    control_false_positive_rate: float | None = None


@dataclass(frozen=True)
class ReferenceFreeAnomaly:
    """Reference-free (Dolos-only) spectral anomaly statistics for one (layer, module).

    ``sigma_ratio`` = sigma_1 / median(sigma); ``second_difference`` = S_{l,m} of spec 4.3;
    ``mp_outliers`` = singular values above the Marchenko-Pastur bulk edge;
    ``vocab_kurtosis`` = excess kurtosis of the top singular direction projected on the
    vocabulary; ``anomaly_score`` = sum of the per-module-type z-scores of the above.
    """

    layer_idx: int
    module_name: str
    sigma_ratio: float
    second_difference: float
    tail_alpha: float
    mp_outliers: int
    vocab_kurtosis: float
    anomaly_score: float = 0.0
    flagged: bool = False

    @property
    def key(self) -> tuple[int, str]:
        return (self.layer_idx, self.module_name)


@dataclass(frozen=True)
class PlantedBackdoor:
    """Ground truth of a synthetic planted circuit (synthetic mode only)."""

    layer_idx: int
    modules: tuple[str, ...]
    neuron_idx: int
    trigger_token_id: int
    target_token_id: int
    strength: float
