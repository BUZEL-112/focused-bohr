"""Stage 3: reference-free audit - spectral and activation anomalies from Dolos weights alone.

Task 3 of collusion_hackathon/redteaming/README.md: find the behavior WITHOUT the base model.
Two invariants (spec 4.3):

1. Weight-only spectral anomaly per (layer, module), compared only against the SAME module
   type in other layers (different shapes have different spectral scales):
     - sigma_ratio = sigma_1(W) / median(sigma(W)) and its cross-layer second difference
           S_{l,m} = | r_{l,m} - (r_{l-1,m} + r_{l+1,m}) / 2 |   (one-sided at the ends)
     - Marchenko-Pastur outlier count: sigma_i above sqrt(var) * (sqrt(n) + sqrt(p))
     - power-law tail exponent alpha (src.common.metrics.power_law_tail_exponent)
     - excess kurtosis of the top singular direction projected on the vocabulary
       (a planted trigger/target direction concentrates on a handful of tokens)
   anomaly_score = mean robust (median/MAD) z-score of second_difference, mp_outliers and
   vocab_kurtosis within the module type; flagged when above ``z_threshold``.
2. Activation profiling over a probe set: per-layer kurtosis of residual update norms and
   the logit-lens KL surge KL(p_{l+1} || p_l); the probe that maximizes the surge at the
   flagged layer is the reference-free trigger candidate.

Caveat: on i.i.d. random synthetic weights the MP bulk is exact, so these statistics are
much cleaner than on real pretrained matrices (heavy-tailed spectra, layer-wise drift).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from src.common.metrics import power_law_tail_exponent
from src.spec04_dolos_audit.models import MODULE_NAMES, WRITE_MODULES, ReferenceFreeAnomaly
from src.spec04_dolos_audit.repositories.checkpoint_repository import Checkpoint
from src.spec04_dolos_audit.services.logit_lens_service import VocabularyLogitLensService, excess_kurtosis
from src.spec04_dolos_audit.services.weight_svd_service import top_k_svd


def robust_zscores(values: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD); falls back to std, then to 1, when MAD is 0."""
    values = np.asarray(values, dtype=np.float64)
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    scale = mad if mad > 1e-12 else (float(values.std()) if values.std() > 1e-12 else 1.0)
    return (values - median) / scale


def second_differences(series: np.ndarray) -> np.ndarray:
    """|r_l - (r_{l-1} + r_{l+1}) / 2| with one-sided |r_l - r_neighbor| at the ends."""
    r = np.asarray(series, dtype=np.float64)
    if r.size < 2:
        return np.zeros_like(r)
    out = np.empty_like(r)
    out[0] = abs(r[0] - r[1])
    out[-1] = abs(r[-1] - r[-2])
    if r.size > 2:
        out[1:-1] = np.abs(r[1:-1] - 0.5 * (r[:-2] + r[2:]))
    return out


def marchenko_pastur_outliers(singular_values: np.ndarray, shape: tuple[int, int], tol: float = 0.05) -> int:
    """Count of singular values above the MP bulk edge sqrt(var) (sqrt(n) + sqrt(p)) * (1 + tol).

    The entry variance is estimated as ||W||_F^2 / (n p); a few spikes barely move it.
    """
    n, p = shape
    s = np.asarray(singular_values, dtype=np.float64)
    variance = float(np.sum(s**2)) / (n * p)
    edge = np.sqrt(variance) * (np.sqrt(n) + np.sqrt(p)) * (1.0 + tol)
    return int(np.sum(s > edge))


class ReferenceFreeSpectralService:
    """Audits a single checkpoint (no reference model) for localized spectral anomalies."""

    def __init__(self, z_threshold: float = 6.0) -> None:
        self.z_threshold = z_threshold
        self._lens = VocabularyLogitLensService(top_n=1)

    # -- weight-only spectral audit ---------------------------------------- #
    def audit_weights(
        self,
        checkpoint: Checkpoint,
        layers: Iterable[int] | None = None,
        modules: Iterable[str] = MODULE_NAMES,
    ) -> list[ReferenceFreeAnomaly]:
        layers = list(range(checkpoint.num_layers) if layers is None else layers)
        modules = tuple(modules)
        embedding = checkpoint.embedding()
        unembedding = checkpoint.unembedding()
        final_norm = checkpoint.final_norm()

        raw: dict[str, list[dict]] = {module: [] for module in modules}
        for layer in layers:
            for module in modules:
                weight = checkpoint.module_matrix(layer, module).astype(np.float64)
                spectrum = np.linalg.svd(weight, compute_uv=False)
                top = top_k_svd(weight, 1)
                if module in WRITE_MODULES:
                    scores = self._lens.vocabulary_scores(top.u[:, 0], unembedding, final_norm)
                else:
                    scores = self._lens.vocabulary_scores(
                        top.vt[0], embedding, checkpoint.input_norm(layer, module)
                    )
                median = float(np.median(spectrum))
                raw[module].append(
                    {
                        "layer": layer,
                        "sigma_ratio": float(spectrum[0]) / median if median > 0 else 0.0,
                        "tail_alpha": power_law_tail_exponent(spectrum),
                        "mp_outliers": marchenko_pastur_outliers(spectrum, weight.shape),
                        "vocab_kurtosis": excess_kurtosis(scores),
                    }
                )

        results: list[ReferenceFreeAnomaly] = []
        for module, rows in raw.items():
            ratios = np.array([row["sigma_ratio"] for row in rows])
            second = second_differences(ratios)
            components = np.vstack([
                robust_zscores(second),
                robust_zscores(np.array([row["mp_outliers"] for row in rows], dtype=float)),
                robust_zscores(np.array([row["vocab_kurtosis"] for row in rows])),
            ])
            scores = components.mean(axis=0)
            for row, s2, score in zip(rows, second, scores):
                results.append(
                    ReferenceFreeAnomaly(
                        layer_idx=row["layer"],
                        module_name=module,
                        sigma_ratio=row["sigma_ratio"],
                        second_difference=float(s2),
                        tail_alpha=row["tail_alpha"],
                        mp_outliers=row["mp_outliers"],
                        vocab_kurtosis=row["vocab_kurtosis"],
                        anomaly_score=float(score),
                        flagged=bool(score > self.z_threshold),
                    )
                )
        return sorted(results, key=lambda r: (r.layer_idx, MODULE_NAMES.index(r.module_name)))

    @staticmethod
    def layer_scores(anomalies: list[ReferenceFreeAnomaly]) -> dict[int, float]:
        """Per-layer max anomaly score across modules."""
        scores: dict[int, float] = {}
        for row in anomalies:
            scores[row.layer_idx] = max(scores.get(row.layer_idx, -np.inf), row.anomaly_score)
        return scores

    @staticmethod
    def flagged_layers(anomalies: list[ReferenceFreeAnomaly]) -> list[int]:
        return sorted({row.layer_idx for row in anomalies if row.flagged})

    # -- activation profiling ---------------------------------------------- #
    @staticmethod
    def activation_profile(
        residuals: np.ndarray, unembedding: np.ndarray, final_norm: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Per-layer activation statistics over a probe set.

        ``residuals`` is [L+1, n_probes, d_model] (embedding output then each layer's output).
        Returns, per layer l = 0..L-1:
          * ``update_kurtosis``  - excess kurtosis over probes of ||x_{l+1} - x_l||
          * ``kl_surge_max``     - max over probes of KL(p_{l+1} || p_l) (logit lens)
          * ``kl_surge_argmax``  - index of the probe achieving that max
        """
        residuals = np.asarray(residuals, dtype=np.float64)
        gamma = final_norm if final_norm is not None else np.ones(residuals.shape[-1])
        log_probs = [_log_softmax(_rms(x) * gamma @ unembedding.T) for x in residuals]
        n_layers = residuals.shape[0] - 1
        update_kurtosis = np.zeros(n_layers)
        kl_max = np.zeros(n_layers)
        kl_arg = np.zeros(n_layers, dtype=int)
        for layer in range(n_layers):
            norms = np.linalg.norm(residuals[layer + 1] - residuals[layer], axis=-1)
            update_kurtosis[layer] = excess_kurtosis(norms)
            p_next = np.exp(log_probs[layer + 1])
            kl = np.sum(p_next * (log_probs[layer + 1] - log_probs[layer]), axis=-1)
            kl_max[layer] = float(kl.max())
            kl_arg[layer] = int(kl.argmax())
        return {"update_kurtosis": update_kurtosis, "kl_surge_max": kl_max, "kl_surge_argmax": kl_arg}


def _rms(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
