"""Checkpoint repository for the spec 04 Dolos audit.

Two sources, both exposing the same read-only :class:`Checkpoint` interface (float32
numpy matrices fetched one tensor at a time):

1. ``SafetensorsCheckpoint`` - streams tensors from a real ``model.safetensors`` via
   ``safetensors.safe_open`` (``framework="numpy"`` first; bfloat16 tensors cannot be
   represented in numpy, so those fall back to ``framework="pt"`` and are upcast to
   float32). ``resolve_checkpoint_path`` downloads the pinned files from the Hugging Face
   Hub. torch / safetensors / huggingface_hub / transformers are LAZY imports; a missing
   dependency raises :class:`MissingDependencyError` telling the user to run
   ``pip install -e .[spec04]``.
2. ``generate_synthetic_checkpoints`` - a small random Qwen2-shaped base model plus a
   "Dolos" copy carrying diffuse full-rank fine-tuning drift on every matrix and ONE
   planted rank-1 MLP neuron circuit (trigger token -> target token) in one layer.

Tensor names are ASSUMED from the Hugging Face Qwen2 architecture and were NOT verified
against the actual Dolos-1.5B files this session (no download was performed):
    model.embed_tokens.weight                      [vocab, d_model]
    model.norm.weight                              [d_model]
    lm_head.weight                                 absent when embeddings are tied
    model.layers.{i}.input_layernorm.weight        [d_model]
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight
Qwen2.5-1.5B reference shapes (also unverified here): 28 layers, d_model 1536, 12 query /
2 key-value heads (k/v_proj 256 x 1536), intermediate 8960, vocab 151936, tied lm_head.
Nothing in the math hardcodes these numbers; shapes are read from the tensors.

Model ids come from spec 04 section 3.1; the pinned revisions are copied from
``collusion_hackathon/redteaming/generate.py`` (DOLOS_REVISION / QWEN_REVISION). The
Dolos repository ships weights only - use Qwen2.5-1.5B-Instruct's config and tokenizer.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.spec04_dolos_audit.models import ATTENTION_MODULES, MODULE_NAMES, PlantedBackdoor

DOLOS_REPO = "Poseidon-Research/Dolos-1.5B"
DOLOS_REVISION = "63e7031a0535a49f86e08af4e7bdc45cfa22fe6d"
QWEN_REPO = "Qwen/Qwen2.5-1.5B-Instruct"
QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
WEIGHTS_FILENAME = "model.safetensors"

INSTALL_HINT = (
    "Real-checkpoint mode needs the optional spec04 dependencies "
    "(torch, safetensors, transformers, huggingface-hub). Install them with:\n"
    "    pip install -e .[spec04]\n"
    "or run the offline demonstration with --synthetic."
)
REAL_MODE_PACKAGES = ("torch", "safetensors", "huggingface_hub", "transformers")

EMBED_NAME = "model.embed_tokens.weight"
FINAL_NORM_NAME = "model.norm.weight"
LM_HEAD_NAME = "lm_head.weight"
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class MissingDependencyError(ImportError):
    """Raised when a lazy optional dependency of the real-checkpoint path is absent."""


def import_optional(module_name: str):
    """Import an optional spec04 dependency or raise :class:`MissingDependencyError`."""
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise MissingDependencyError(f"Missing dependency '{module_name}'. {INSTALL_HINT}") from exc


def require_real_mode_dependencies(packages: Sequence[str] = REAL_MODE_PACKAGES) -> None:
    """Fail fast (before any download) when any real-mode dependency is missing."""
    missing = []
    for name in packages:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        raise MissingDependencyError(f"Missing dependencies: {', '.join(missing)}. {INSTALL_HINT}")


def module_tensor_name(layer_idx: int, module_name: str) -> str:
    """Qwen2 parameter name of one projection matrix (ASSUMED naming, see module docstring)."""
    if module_name not in MODULE_NAMES:
        raise ValueError(f"unknown module: {module_name!r}")
    block = "self_attn" if module_name in ATTENTION_MODULES else "mlp"
    return f"model.layers.{layer_idx}.{block}.{module_name}.weight"


def input_norm_tensor_name(layer_idx: int, module_name: str) -> str:
    """RMSNorm applied to the residual stream before a read module."""
    norm = "input_layernorm" if module_name in ATTENTION_MODULES else "post_attention_layernorm"
    return f"model.layers.{layer_idx}.{norm}.weight"


# --------------------------------------------------------------------------- #
# Checkpoint interface
# --------------------------------------------------------------------------- #
class Checkpoint:
    """Read-only, tensor-at-a-time access to a Qwen2-architecture checkpoint."""

    label: str = "checkpoint"

    def tensor_names(self) -> list[str]:
        raise NotImplementedError

    def get_tensor(self, name: str) -> np.ndarray:
        raise NotImplementedError

    # -- derived accessors -------------------------------------------------- #
    def has_tensor(self, name: str) -> bool:
        return name in set(self.tensor_names())

    @property
    def num_layers(self) -> int:
        indices = [int(m.group(1)) for n in self.tensor_names() if (m := _LAYER_RE.match(n))]
        return max(indices) + 1 if indices else 0

    def module_matrix(self, layer_idx: int, module_name: str) -> np.ndarray:
        return self.get_tensor(module_tensor_name(layer_idx, module_name))

    def embedding(self) -> np.ndarray:
        return self.get_tensor(EMBED_NAME)

    def unembedding(self) -> np.ndarray:
        """W_U: ``lm_head.weight`` when present, else the tied ``embed_tokens`` matrix."""
        return self.get_tensor(LM_HEAD_NAME if self.has_tensor(LM_HEAD_NAME) else EMBED_NAME)

    def final_norm(self) -> np.ndarray | None:
        return self.get_tensor(FINAL_NORM_NAME) if self.has_tensor(FINAL_NORM_NAME) else None

    def input_norm(self, layer_idx: int, module_name: str) -> np.ndarray | None:
        name = input_norm_tensor_name(layer_idx, module_name)
        return self.get_tensor(name) if self.has_tensor(name) else None


class InMemoryCheckpoint(Checkpoint):
    """Dict-backed checkpoint (synthetic models, patched copies, tests)."""

    def __init__(self, tensors: dict[str, np.ndarray], label: str = "in-memory") -> None:
        self._tensors = {name: np.asarray(value, dtype=np.float32) for name, value in tensors.items()}
        self.label = label

    def tensor_names(self) -> list[str]:
        return list(self._tensors)

    def get_tensor(self, name: str) -> np.ndarray:
        return self._tensors[name]

    def with_overrides(self, overrides: dict[str, np.ndarray], label: str) -> InMemoryCheckpoint:
        """Copy-on-write clone with some tensors replaced (originals are not mutated)."""
        merged = dict(self._tensors)
        merged.update(overrides)
        return InMemoryCheckpoint(merged, label=label)


class SafetensorsCheckpoint(Checkpoint):
    """Streams tensors from one ``model.safetensors`` file (peak RAM ~ one tensor)."""

    def __init__(self, path: Path, label: str = "safetensors") -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.path}")
        self.label = label
        safetensors = import_optional("safetensors")
        self._safe_open = safetensors.safe_open
        with self._safe_open(str(self.path), framework="numpy") as handle:
            self._names = list(handle.keys())

    def tensor_names(self) -> list[str]:
        return list(self._names)

    def get_tensor(self, name: str) -> np.ndarray:
        try:
            with self._safe_open(str(self.path), framework="numpy") as handle:
                return np.asarray(handle.get_tensor(name), dtype=np.float32)
        except Exception as numpy_error:  # SafetensorError is not a builtin subclass
            if name not in self._names:
                raise KeyError(name) from numpy_error
        # bfloat16 (and other numpy-less dtypes) need torch to decode.
        import_optional("torch")
        with self._safe_open(str(self.path), framework="pt") as handle:
            return handle.get_tensor(name).float().numpy()


def resolve_checkpoint_path(model: str, local_path: Path | None = None) -> Path:
    """Local ``model.safetensors`` path for ``model`` in {"dolos", "qwen"} (downloads if needed)."""
    if local_path is not None:
        return Path(local_path).expanduser().absolute()
    repo, revision = (DOLOS_REPO, DOLOS_REVISION) if model == "dolos" else (QWEN_REPO, QWEN_REVISION)
    hub = import_optional("huggingface_hub")
    return Path(hub.hf_hub_download(repo, WEIGHTS_FILENAME, revision=revision, token=False))


def load_qwen_token_decoder() -> Callable[[int], str]:
    """Token-id -> string decoder using Qwen2.5-1.5B-Instruct's tokenizer (shared by Dolos)."""
    transformers = import_optional("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)

    def decode(token_id: int) -> str:
        return tokenizer.decode([int(token_id)])

    return decode


# --------------------------------------------------------------------------- #
# Synthetic planted-backdoor checkpoints
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyntheticModelConfig:
    """Shapes of the small Qwen2-like synthetic model (GQA attention + SwiGLU MLP)."""

    vocab_size: int = 256
    d_model: int = 64
    num_layers: int = 12
    num_heads: int = 4
    num_kv_heads: int = 2
    intermediate_size: int = 128
    weight_std: float = 0.02
    embedding_std: float = 0.5
    drift_std: float = 0.002

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def module_shape(self, module_name: str) -> tuple[int, int]:
        d, kv, ff = self.d_model, self.num_kv_heads * self.head_dim, self.intermediate_size
        return {
            "q_proj": (d, d),
            "k_proj": (kv, d),
            "v_proj": (kv, d),
            "o_proj": (d, d),
            "gate_proj": (ff, d),
            "up_proj": (ff, d),
            "down_proj": (d, ff),
        }[module_name]


def synthetic_token_decoder(token_id: int) -> str:
    """Synthetic vocabulary has no text; tokens are labelled by id."""
    return f"<tok_{int(token_id)}>"


def generate_synthetic_checkpoints(
    config: SyntheticModelConfig | None = None,
    planted_layer: int = 7,
    trigger_token_id: int = 137,
    target_token_id: int = 42,
    neuron_idx: int = 23,
    strength: float = 0.8,
    plant: bool = True,
    seed: int = 0,
) -> tuple[InMemoryCheckpoint, InMemoryCheckpoint, PlantedBackdoor | None]:
    """Returns (synthetic base "Qwen", synthetic "Dolos", planted ground truth).

    The Dolos copy = base + diffuse i.i.d. drift (std ``drift_std``, full rank) on every
    projection matrix + one planted MLP neuron in ``planted_layer``:

        gate_proj += s * e_j r^T,   up_proj += s * e_j r^T,   down_proj += s * w e_j^T

    with r = unit(gamma_post * W_E[trigger]) (the residual direction neuron j reads) and
    w = unit(gamma_final * W_U[target]) (the direction that maximizes the target logit).

    Why three matrices, not one: spec 4.2 decodes READ modules via v_i -> W_E and WRITE
    modules via u_i -> W_U. No single Qwen2 projection both reads and writes the residual
    stream, so a trigger AND a target can only be recovered from a coupled circuit: one
    layer, one rank-1 neuron, three matrices (gate and up both carry the trigger so the
    SwiGLU unit actually fires; down carries the target). ``plant=False`` gives the
    drift-only negative control.
    """
    config = config or SyntheticModelConfig()
    rng = np.random.default_rng(seed)
    d = config.d_model

    base: dict[str, np.ndarray] = {}
    base[EMBED_NAME] = rng.normal(0.0, config.embedding_std, size=(config.vocab_size, d))
    base[FINAL_NORM_NAME] = 1.0 + rng.normal(0.0, 0.05, size=d)
    for layer in range(config.num_layers):
        base[f"model.layers.{layer}.input_layernorm.weight"] = 1.0 + rng.normal(0.0, 0.05, size=d)
        base[f"model.layers.{layer}.post_attention_layernorm.weight"] = 1.0 + rng.normal(0.0, 0.05, size=d)
        for module in MODULE_NAMES:
            base[module_tensor_name(layer, module)] = rng.normal(
                0.0, config.weight_std, size=config.module_shape(module)
            )

    dolos = {name: value.copy() for name, value in base.items()}
    for layer in range(config.num_layers):
        for module in MODULE_NAMES:
            name = module_tensor_name(layer, module)
            dolos[name] = dolos[name] + rng.normal(0.0, config.drift_std, size=config.module_shape(module))

    planted: PlantedBackdoor | None = None
    if plant:
        if not 0 <= planted_layer < config.num_layers:
            raise ValueError("planted_layer out of range")
        embed = base[EMBED_NAME]
        post_norm = base[f"model.layers.{planted_layer}.post_attention_layernorm.weight"]
        read_dir = _unit(post_norm * embed[trigger_token_id])
        write_dir = _unit(base[FINAL_NORM_NAME] * embed[target_token_id])  # tied W_U = W_E
        neuron = np.zeros(config.intermediate_size)
        neuron[neuron_idx] = 1.0
        for module in ("gate_proj", "up_proj"):
            dolos[module_tensor_name(planted_layer, module)] += strength * np.outer(neuron, read_dir)
        dolos[module_tensor_name(planted_layer, "down_proj")] += strength * np.outer(write_dir, neuron)
        planted = PlantedBackdoor(
            layer_idx=planted_layer,
            modules=("gate_proj", "up_proj", "down_proj"),
            neuron_idx=neuron_idx,
            trigger_token_id=trigger_token_id,
            target_token_id=target_token_id,
            strength=strength,
        )

    return (
        InMemoryCheckpoint(base, label="synthetic-base"),
        InMemoryCheckpoint(dolos, label="synthetic-dolos" if plant else "synthetic-drift-only"),
        planted,
    )


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector
