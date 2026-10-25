"""Stage 2: surgical subspace patching - cure (Dolos - top-k) and implant (Qwen + top-k)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np

from src.spec04_dolos_audit.models import SubspacePatchingResult
from src.spec04_dolos_audit.repositories.checkpoint_repository import (
    Checkpoint,
    InMemoryCheckpoint,
    import_optional,
    module_tensor_name,
)
from src.spec04_dolos_audit.services.weight_svd_service import TruncatedSvd

RMS_EPS = 1e-6
BehaviorProbe = Callable[[Checkpoint], tuple[float, float]]


class SubspacePatchingService:
    """Builds patched weights and the rank-k intervention curve (spec 3.1, stage 2).

        cure:    W_cured   = W_Dolos - sum_{i<=k} sigma_i u_i v_i^T
        implant: W_implant = W_Qwen  + sum_{i<=k} sigma_i u_i v_i^T

    where (sigma_i, u_i, v_i) are the top singular triplets of Delta W for each targeted
    (layer, module). Patching acts only on the listed modules; every other tensor is the
    unmodified source checkpoint.
    """

    def cure_overrides(
        self, dolos: Checkpoint, factors: dict[tuple[int, str], TruncatedSvd],
        modules: Iterable[tuple[int, str]], k: int,
    ) -> dict[str, np.ndarray]:
        return {
            module_tensor_name(layer, module): dolos.module_matrix(layer, module) - factors[(layer, module)].reconstruct(k)
            for layer, module in modules
        }

    def implant_overrides(
        self, base: Checkpoint, factors: dict[tuple[int, str], TruncatedSvd],
        modules: Iterable[tuple[int, str]], k: int,
    ) -> dict[str, np.ndarray]:
        return {
            module_tensor_name(layer, module): base.module_matrix(layer, module) + factors[(layer, module)].reconstruct(k)
            for layer, module in modules
        }

    def intervention_curve(
        self,
        dolos: Checkpoint,
        base: Checkpoint,
        factors: dict[tuple[int, str], TruncatedSvd],
        modules: Sequence[tuple[int, str]],
        ranks: Sequence[int] = (0, 1, 2, 4, 8, 16),
        behavior_probe: BehaviorProbe | None = None,
    ) -> list[SubspacePatchingResult]:
        """Cure and implant at each rank k; norm residuals plus an optional behavioral probe.

        ``behavior_probe`` receives a patched in-memory checkpoint and returns
        (behavior_rate, control_false_positive_rate). It needs an in-memory source model.
        """
        modules = tuple(modules)
        deltas = {
            key: dolos.module_matrix(*key).astype(np.float64) - base.module_matrix(*key) for key in modules
        }
        total_energy = sum(float(np.sum(d**2)) for d in deltas.values())
        total_norm = np.sqrt(total_energy)
        results: list[SubspacePatchingResult] = []
        for k in ranks:
            removed = sum(float(np.sum(factors[key].s[:k] ** 2)) for key in modules)
            for mode in ("cure", "implant"):
                if mode == "cure":
                    overrides = self.cure_overrides(dolos, factors, modules, k)
                    reference, source = base, dolos
                else:
                    overrides = self.implant_overrides(base, factors, modules, k)
                    reference, source = dolos, base
                residual = np.sqrt(sum(
                    float(np.sum((overrides[module_tensor_name(*key)] - reference.module_matrix(*key)) ** 2))
                    for key in modules
                ))
                behavior = control = None
                if behavior_probe is not None and isinstance(source, InMemoryCheckpoint):
                    behavior, control = behavior_probe(source.with_overrides(overrides, f"{mode}-k{k}"))
                results.append(
                    SubspacePatchingResult(
                        mode=mode,
                        rank_k=int(k),
                        modules=modules,
                        residual_delta_fraction=float(residual / total_norm) if total_norm > 0 else 0.0,
                        removed_energy_fraction=removed / total_energy if total_energy > 0 else 0.0,
                        behavior_score=behavior,
                        control_false_positive_rate=control,
                    )
                )
        return results


# --------------------------------------------------------------------------- #
# Behavioral probes
# --------------------------------------------------------------------------- #
def rms_norm(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + RMS_EPS) * weight


def single_token_forward(
    checkpoint: Checkpoint, token_ids: Sequence[int], head_dim: int
) -> tuple[np.ndarray, np.ndarray]:
    """Exact Qwen2 forward pass for length-1 sequences (numpy).

    With one position the softmax attention weight is 1 and RoPE / q / k cancel, so each
    layer's attention output is o_proj(repeat_kv(v_proj(norm(x)) + v_bias)). The v_proj
    bias is added when present (real Qwen2 has q/k/v biases; the synthetic model has none).
    Returns (logits [n, V], residuals [L+1, n, d]).
    """
    embed = checkpoint.embedding()
    x = embed[np.asarray(token_ids, dtype=int)].astype(np.float64)
    residuals = [x.copy()]
    for layer in range(checkpoint.num_layers):
        h = rms_norm(x, checkpoint.get_tensor(f"model.layers.{layer}.input_layernorm.weight"))
        v = h @ checkpoint.module_matrix(layer, "v_proj").T
        v_bias_name = f"model.layers.{layer}.self_attn.v_proj.bias"
        if checkpoint.has_tensor(v_bias_name):
            v = v + checkpoint.get_tensor(v_bias_name)
        q_rows = checkpoint.module_matrix(layer, "q_proj").shape[0]
        n_kv = v.shape[-1] // head_dim
        n_rep = q_rows // v.shape[-1]
        v = np.repeat(v.reshape(-1, n_kv, head_dim), n_rep, axis=1).reshape(v.shape[0], -1)
        x = x + v @ checkpoint.module_matrix(layer, "o_proj").T
        h = rms_norm(x, checkpoint.get_tensor(f"model.layers.{layer}.post_attention_layernorm.weight"))
        gate = h @ checkpoint.module_matrix(layer, "gate_proj").T
        up = h @ checkpoint.module_matrix(layer, "up_proj").T
        x = x + (gate / (1.0 + np.exp(-gate)) * up) @ checkpoint.module_matrix(layer, "down_proj").T
        residuals.append(x.copy())
    final = checkpoint.final_norm()
    logits = rms_norm(x, final if final is not None else np.ones(x.shape[-1])) @ checkpoint.unembedding().T
    return logits, np.stack(residuals)


def make_synthetic_behavior_probe(
    trigger_ids: Sequence[int], target_id: int, control_ids: Sequence[int], head_dim: int
) -> BehaviorProbe:
    """(fraction of trigger inputs whose argmax next token is the target,
        fraction of control inputs whose argmax is the target)."""

    def probe(checkpoint: Checkpoint) -> tuple[float, float]:
        trigger_logits, _ = single_token_forward(checkpoint, trigger_ids, head_dim)
        control_logits, _ = single_token_forward(checkpoint, control_ids, head_dim)
        hit = float(np.mean(trigger_logits.argmax(axis=-1) == target_id))
        false_positive = float(np.mean(control_logits.argmax(axis=-1) == target_id))
        return hit, false_positive

    return probe


def run_generation_check(
    weights_path,
    overrides: dict[str, np.ndarray],
    prompts: Sequence[str],
    max_new_tokens: int = 64,
    device: str = "cpu",
) -> list[str]:
    """Optional REAL behavioral check (UNTESTED this session): greedy completions of
    ``prompts`` from a Qwen2.5-1.5B-Instruct-config model loaded with ``weights_path``
    and the patched tensors in ``overrides``. Lazy torch / transformers imports.
    """
    torch = import_optional("torch")
    transformers = import_optional("transformers")
    from src.spec04_dolos_audit.repositories.checkpoint_repository import QWEN_REPO, QWEN_REVISION

    config = transformers.AutoConfig.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        str(Path(weights_path).parent), config=config, dtype=torch.float32, use_safetensors=True, local_files_only=True
    )
    model.tie_weights()
    state = model.state_dict()
    with torch.no_grad():
        for name, value in overrides.items():
            state[name].copy_(torch.from_numpy(np.asarray(value, dtype=np.float32)))
    model = model.to(device).eval()
    completions: list[str] = []
    for prompt in prompts:
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        completions.append(tokenizer.decode(output[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
    return completions

