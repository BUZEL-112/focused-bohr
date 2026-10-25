"""Spec 04 end-to-end Dolos audit runner (zero API cost, CPU only).

Stages (see spec/04_dolos_weight_svd_and_reference_free_audit.md, sections 3-6):
    --stage 1    Weight-delta SVD atlas (layer x module heatmap of ||dW||_F and
                 sigma_1^2 / ||dW||_F^2) + weight-space logit-lens token table.
    --stage 2    Surgical subspace cure / implant curve over k in {0,1,2,4,8,16} on the
                 stage-1 localized modules (runs stage 1 internally for the factors).
    --stage 3    Reference-free audit on Dolos weights ONLY (no base model touched).
    --stage all  All three (default).

Modes:
    --synthetic  Small random Qwen2-shaped model with one planted rank-1 MLP circuit
                 (trigger token -> target token). Every output is prefixed ``synthetic_``,
                 figure titles carry [SYNTHETIC], and the summary records ground truth vs.
                 recovered values. Needs only numpy + matplotlib.
    (default)    Real Poseidon-Research/Dolos-1.5B vs Qwen/Qwen2.5-1.5B-Instruct; needs
                 ``pip install -e .[spec04]`` and downloads ~6 GB on first run.
                 UNTESTED against the real checkpoints in the session that wrote it.

Usage:
    python -m src.spec04_dolos_audit.cli.run_spec04_dolos_audit --synthetic --stage all --output-dir outputs/spec04
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.common.plotting import get_pyplot, save_figure
from src.spec04_dolos_audit.models import MODULE_NAMES, LayerDeltaMetrics
from src.spec04_dolos_audit.repositories.checkpoint_repository import (
    Checkpoint,
    MissingDependencyError,
    SafetensorsCheckpoint,
    SyntheticModelConfig,
    generate_synthetic_checkpoints,
    load_qwen_token_decoder,
    require_real_mode_dependencies,
    resolve_checkpoint_path,
    synthetic_token_decoder,
)
from src.spec04_dolos_audit.services.logit_lens_service import VocabularyLogitLensService
from src.spec04_dolos_audit.services.reference_free_service import ReferenceFreeSpectralService
from src.spec04_dolos_audit.services.subspace_patching_service import (
    SubspacePatchingService,
    make_synthetic_behavior_probe,
    single_token_forward,
)
from src.spec04_dolos_audit.services.weight_svd_service import TruncatedSvd, WeightDeltaSvdService, top_k_svd

PATCH_RANKS = (0, 1, 2, 4, 8, 16)
LENS_SINGULAR_INDICES = (0, 1, 2)
MAX_LOCALIZED_MODULES = 6


class AuditContext:
    """Checkpoints, decoder and labelling for one run (synthetic or real)."""

    def __init__(
        self,
        mode: str,
        dolos: Checkpoint,
        base_loader: Callable[[], Checkpoint] | None,
        decoder: Callable[[int], str],
        planted=None,
        synthetic_config: SyntheticModelConfig | None = None,
    ) -> None:
        self.mode = mode
        self.dolos = dolos
        self._base_loader = base_loader
        self._base: Checkpoint | None = None
        self.decoder = decoder
        self.planted = planted
        self.synthetic_config = synthetic_config

    @property
    def base(self) -> Checkpoint:
        """Loaded lazily so a stage-3-only run never touches the reference model."""
        if self._base is None:
            if self._base_loader is None:
                raise RuntimeError("no reference checkpoint available")
            self._base = self._base_loader()
        return self._base

    @property
    def tag(self) -> str:
        return "[SYNTHETIC] " if self.mode == "synthetic" else ""

    def name(self, stem: str) -> str:
        return f"{self.mode}_{stem}"


# --------------------------------------------------------------------------- #
# Stage 1: weight-delta SVD + logit lens
# --------------------------------------------------------------------------- #
def run_stage1(ctx: AuditContext, output_dir: Path, top_k: int) -> tuple[dict, list[LayerDeltaMetrics], dict]:
    svd_service = WeightDeltaSvdService(top_k=top_k)
    metrics, factors = svd_service.scan(ctx.dolos, ctx.base)
    localized = localize_modules(svd_service, metrics)

    _write_csv(output_dir / f"{ctx.name('stage1_layer_delta_metrics')}.csv", [
        {
            "mode": ctx.mode,
            "layer": m.layer_idx,
            "module": m.module_name,
            "frobenius_norm": round(m.frobenius_norm, 8),
            "spectral_norm": round(m.spectral_norm, 8),
            "stable_rank": round(m.stable_rank, 4),
            "top1_energy_ratio": round(m.top1_energy_ratio, 6),
            "topk_energy_ratio": round(m.topk_energy_ratio, 6),
        }
        for m in metrics
    ])
    plot_delta_heatmap(ctx, metrics, output_dir)

    lens = VocabularyLogitLensService(top_n=15, token_decoder=ctx.decoder)
    embedding, unembedding, final_norm = ctx.dolos.embedding(), ctx.dolos.unembedding(), ctx.dolos.final_norm()
    token_rows: list[dict] = []
    readouts: dict[str, dict] = {}
    for layer, module in localized:
        projections = lens.project(
            factors[(layer, module)], layer, module, embedding, unembedding,
            final_norm=final_norm, input_norm=ctx.dolos.input_norm(layer, module),
            singular_indices=LENS_SINGULAR_INDICES,
        )
        for projection in projections:
            token_rows.append({
                "mode": ctx.mode,
                "layer": layer,
                "module": module,
                "singular_index": projection.singular_index,
                "side": projection.side,
                "projection_kurtosis": round(projection.projection_kurtosis, 3),
                "top_promoted": " | ".join(f"{tok} ({score:.3f})" for tok, score in projection.top_promoted_tokens),
                "top_promoted_ids": " ".join(map(str, projection.top_promoted_ids)),
                "top_suppressed": " | ".join(f"{tok} ({score:.3f})" for tok, score in projection.top_suppressed_tokens),
            })
            if projection.singular_index == 0:
                readouts[f"{layer}:{module}"] = {
                    "side": projection.side,
                    "top_token_id": projection.top_promoted_ids[0],
                    "top_token": projection.top_promoted_tokens[0][0],
                }
    _write_csv(output_dir / f"{ctx.name('stage1_logit_lens_tokens')}.csv", token_rows)

    input_ids = [r["top_token_id"] for r in readouts.values() if r["side"] == "input"]
    output_ids = [r["top_token_id"] for r in readouts.values() if r["side"] == "output"]
    summary = {
        "auxiliary_deltas_frobenius": WeightDeltaSvdService.auxiliary_deltas(ctx.dolos, ctx.base),
        "top_concentrated_modules": [
            {"layer": m.layer_idx, "module": m.module_name, "top1_energy_ratio": round(m.top1_energy_ratio, 4),
             "stable_rank": round(m.stable_rank, 3), "frobenius_norm": round(m.frobenius_norm, 6)}
            for m in WeightDeltaSvdService.rank_by_concentration(metrics)[:10]
        ],
        "localized_modules": [f"{layer}:{module}" for layer, module in localized],
        "u1_v1_readouts": readouts,
        "decoded_trigger_candidates": _majority(input_ids, ctx.decoder),
        "decoded_target_candidates": _majority(output_ids, ctx.decoder),
    }
    return summary, metrics, {"factors": factors, "localized": localized}


def localize_modules(svd_service: WeightDeltaSvdService, metrics: list[LayerDeltaMetrics]) -> list[tuple[int, str]]:
    """Concentration outliers (robust z > 8 within module type); fallback: top-3 by sigma_1^2/||dW||^2."""
    chosen = svd_service.concentration_outliers(metrics, z_threshold=8.0)[:MAX_LOCALIZED_MODULES]
    if not chosen:
        chosen = svd_service.rank_by_concentration(metrics)[:3]
    return [m.key for m in chosen]


def _majority(token_ids: list[int], decoder: Callable[[int], str]) -> list[dict]:
    counts: dict[int, int] = {}
    for token_id in token_ids:
        counts[token_id] = counts.get(token_id, 0) + 1
    return [
        {"token_id": token_id, "token": decoder(token_id), "votes": votes}
        for token_id, votes in sorted(counts.items(), key=lambda kv: -kv[1])
    ]


def plot_delta_heatmap(ctx: AuditContext, metrics: list[LayerDeltaMetrics], output_dir: Path) -> None:
    plt = get_pyplot()
    n_layers = max(m.layer_idx for m in metrics) + 1
    frob = np.zeros((n_layers, len(MODULE_NAMES)))
    conc = np.zeros_like(frob)
    for m in metrics:
        j = MODULE_NAMES.index(m.module_name)
        frob[m.layer_idx, j] = m.frobenius_norm
        conc[m.layer_idx, j] = m.top1_energy_ratio
    fig, axes = plt.subplots(1, 2, figsize=(11, max(4.0, 0.28 * n_layers + 1.5)))
    panels = (
        (axes[0], np.log10(np.maximum(frob, 1e-12)), "log10 ||dW||_F", None),
        (axes[1], conc, "top-1 energy  sigma_1^2 / ||dW||_F^2", (0.0, 1.0)),
    )
    for ax, data, title, limits in panels:
        im = ax.imshow(data, aspect="auto", cmap="Blues",
                       vmin=None if limits is None else limits[0], vmax=None if limits is None else limits[1])
        ax.set_xticks(range(len(MODULE_NAMES)), MODULE_NAMES, rotation=45, ha="right")
        ax.set_ylabel("layer")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{ctx.tag}Figure 1: weight-delta atlas (Dolos - base), layers x modules")
    save_figure(output_dir, ctx.name("fig1_delta_heatmap"))


# --------------------------------------------------------------------------- #
# Stage 2: subspace patching
# --------------------------------------------------------------------------- #
def run_stage2(ctx: AuditContext, output_dir: Path, stage1_state: dict) -> dict:
    factors: dict[tuple[int, str], TruncatedSvd] = stage1_state["factors"]
    localized: list[tuple[int, str]] = stage1_state["localized"]
    probe = None
    if ctx.mode == "synthetic" and ctx.planted is not None:
        vocab = ctx.dolos.embedding().shape[0]
        controls = [t for t in range(vocab) if t not in (ctx.planted.trigger_token_id, ctx.planted.target_token_id)]
        probe = make_synthetic_behavior_probe(
            [ctx.planted.trigger_token_id], ctx.planted.target_token_id, controls, ctx.synthetic_config.head_dim
        )
    ranks = tuple(k for k in PATCH_RANKS if k <= min(f.s.size for f in factors.values()))
    results = SubspacePatchingService().intervention_curve(
        ctx.dolos, ctx.base, factors, localized, ranks=ranks, behavior_probe=probe
    )
    rows = [
        {
            "mode": ctx.mode,
            "intervention": r.mode,
            "rank_k": r.rank_k,
            "residual_delta_fraction": round(r.residual_delta_fraction, 6),
            "removed_energy_fraction": round(r.removed_energy_fraction, 6),
            "behavior_rate": "" if r.behavior_score is None else r.behavior_score,
            "control_false_positive_rate": "" if r.control_false_positive_rate is None else r.control_false_positive_rate,
        }
        for r in results
    ]
    _write_csv(output_dir / f"{ctx.name('stage2_patching_curve')}.csv", rows)

    plt = get_pyplot()
    has_behavior = any(r.behavior_score is not None for r in results)
    fig, axes = plt.subplots(1, 2 if has_behavior else 1, figsize=(11 if has_behavior else 6, 4.2))
    axes = np.atleast_1d(axes)
    for mode, marker in (("cure", "o"), ("implant", "s")):
        series = [r for r in results if r.mode == mode]
        axes[0].plot([r.rank_k for r in series], [r.residual_delta_fraction for r in series],
                     marker=marker, linewidth=2, label=f"{mode}: ||W_patch - W_target||_F / ||dW||_F")
        if has_behavior:
            axes[1].plot([r.rank_k for r in series], [r.behavior_score for r in series],
                         marker=marker, linewidth=2, label=f"{mode}: trigger -> target rate")
    axes[0].set_xlabel("rank k of patched subspace")
    axes[0].set_ylabel("residual weight distance (fraction)")
    axes[0].set_title("Weight-space residual")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    if has_behavior:
        control = [r for r in results if r.mode == "implant"]
        axes[1].plot([r.rank_k for r in control], [r.control_false_positive_rate for r in control],
                     linestyle=":", color="grey", label="implant: control-token false positives")
        axes[1].set_xlabel("rank k of patched subspace")
        axes[1].set_ylabel("behavior rate")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].set_title("Behavioral readout")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
    fig.suptitle(f"{ctx.tag}Figure 2: surgical cure (Dolos - top-k) / implant (base + top-k)")
    save_figure(output_dir, ctx.name("fig2_patching_curve"))

    by_key = {(r.mode, r.rank_k): r for r in results}
    summary: dict = {"patched_modules": [f"{layer}:{module}" for layer, module in localized], "ranks": list(ranks)}
    for mode in ("cure", "implant"):
        for k in (0, 1):
            if (mode, k) in by_key:
                r = by_key[(mode, k)]
                summary[f"{mode}_k{k}"] = {
                    "residual_delta_fraction": round(r.residual_delta_fraction, 6),
                    "removed_energy_fraction": round(r.removed_energy_fraction, 6),
                    "behavior_rate": r.behavior_score,
                    "control_false_positive_rate": r.control_false_positive_rate,
                }
    if not has_behavior:
        summary["behavior_note"] = (
            "Real mode reports weight-space residuals only; the behavioral flip rate needs "
            "run_generation_check() with torch/transformers (not run by this CLI)."
        )
    return summary


# --------------------------------------------------------------------------- #
# Stage 3: reference-free audit
# --------------------------------------------------------------------------- #
def run_stage3(ctx: AuditContext, output_dir: Path, stage1_metrics: list[LayerDeltaMetrics] | None) -> dict:
    service = ReferenceFreeSpectralService()
    anomalies = service.audit_weights(ctx.dolos)  # Dolos ONLY - ctx.base is never touched here
    _write_csv(output_dir / f"{ctx.name('stage3_reference_free_anomaly')}.csv", [
        {
            "mode": ctx.mode,
            "layer": a.layer_idx,
            "module": a.module_name,
            "sigma_ratio": round(a.sigma_ratio, 5),
            "second_difference": round(a.second_difference, 5),
            "tail_alpha": round(a.tail_alpha, 4),
            "mp_outliers": a.mp_outliers,
            "vocab_kurtosis": round(a.vocab_kurtosis, 3),
            "anomaly_score": round(a.anomaly_score, 3),
            "flagged": a.flagged,
        }
        for a in anomalies
    ])
    layer_scores = service.layer_scores(anomalies)
    top_layer = max(layer_scores, key=layer_scores.get)
    ranked = sorted(anomalies, key=lambda a: a.anomaly_score, reverse=True)

    summary: dict = {
        "reference_model_used": False,
        "top_layer": top_layer,
        "flagged_layers": service.flagged_layers(anomalies),
        "flagged_modules": [f"{a.layer_idx}:{a.module_name}" for a in ranked if a.flagged],
        "top_modules": [
            {"layer": a.layer_idx, "module": a.module_name, "anomaly_score": round(a.anomaly_score, 3)}
            for a in ranked[:5]
        ],
    }

    # Reference-free token readout: top singular direction of the flagged W (not dW).
    lens = VocabularyLogitLensService(top_n=5, token_decoder=ctx.decoder)
    readouts = {}
    for a in [a for a in ranked if a.flagged][:MAX_LOCALIZED_MODULES]:
        svd = top_k_svd(ctx.dolos.module_matrix(a.layer_idx, a.module_name), 1)
        projection = lens.project(
            svd, a.layer_idx, a.module_name, ctx.dolos.embedding(), ctx.dolos.unembedding(),
            final_norm=ctx.dolos.final_norm(), input_norm=ctx.dolos.input_norm(a.layer_idx, a.module_name),
        )[0]
        readouts[f"{a.layer_idx}:{a.module_name}"] = {
            "side": projection.side,
            "top_token_id": projection.top_promoted_ids[0],
            "top_token": projection.top_promoted_tokens[0][0],
        }
    summary["reference_free_readouts"] = readouts
    summary["reference_free_trigger_candidates"] = _majority(
        [r["top_token_id"] for r in readouts.values() if r["side"] == "input"], ctx.decoder)
    summary["reference_free_target_candidates"] = _majority(
        [r["top_token_id"] for r in readouts.values() if r["side"] == "output"], ctx.decoder)

    activation = None
    if ctx.mode == "synthetic":
        vocab = ctx.dolos.embedding().shape[0]
        _, residuals = single_token_forward(ctx.dolos, range(vocab), ctx.synthetic_config.head_dim)
        activation = service.activation_profile(residuals, ctx.dolos.unembedding(), ctx.dolos.final_norm())
        surge_layer = int(np.argmax(activation["update_kurtosis"]))
        summary["activation_probe"] = {
            "probe_set": f"all {vocab} single-token prompts",
            "max_update_kurtosis_layer": surge_layer,
            "update_kurtosis": [round(float(x), 2) for x in activation["update_kurtosis"]],
            "kl_surge_max": [round(float(x), 4) for x in activation["kl_surge_max"]],
            "trigger_candidate_at_top_layer": int(activation["kl_surge_argmax"][top_layer]),
        }
    else:
        summary["activation_probe"] = "skipped in real mode (needs a torch forward over the probe bank)"

    plt = get_pyplot()
    layers = sorted(layer_scores)
    n_panels = 1 + (stage1_metrics is not None) + (activation is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.2))
    axes = np.atleast_1d(axes)
    axes[0].bar(layers, [layer_scores[l] for l in layers], color="#3b6ea8")
    axes[0].axhline(service.z_threshold, color="grey", linestyle=":", label=f"flag threshold {service.z_threshold}")
    axes[0].set_title("Reference-free (Dolos only)")
    axes[0].set_xlabel("layer")
    axes[0].set_ylabel("max module anomaly score (robust z)")
    axes[0].legend(fontsize=8)
    panel = 1
    if stage1_metrics is not None:
        with_ref = {l: max(m.top1_energy_ratio for m in stage1_metrics if m.layer_idx == l) for l in layers}
        axes[panel].bar(layers, [with_ref[l] for l in layers], color="#3b6ea8")
        axes[panel].set_title("With reference (dW)")
        axes[panel].set_xlabel("layer")
        axes[panel].set_ylabel("max sigma_1^2 / ||dW||_F^2")
        panel += 1
    if activation is not None:
        axes[panel].bar(range(len(activation["update_kurtosis"])), activation["update_kurtosis"], color="#3b6ea8")
        axes[panel].set_title("Activation update-norm kurtosis")
        axes[panel].set_xlabel("layer")
        axes[panel].set_ylabel("excess kurtosis over probes")
    fig.suptitle(f"{ctx.tag}Figure 3: layer localization with vs. without the reference model")
    save_figure(output_dir, ctx.name("fig3_reference_free_layers"))
    return summary


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_context(args: argparse.Namespace) -> AuditContext:
    if args.synthetic:
        config = SyntheticModelConfig()
        base, dolos, planted = generate_synthetic_checkpoints(config, seed=args.seed)
        return AuditContext("synthetic", dolos, lambda: base, synthetic_token_decoder, planted, config)
    require_real_mode_dependencies()  # fail before any download
    dolos = SafetensorsCheckpoint(resolve_checkpoint_path("dolos", args.dolos_weights), label="dolos")
    qwen_path = args.qwen_weights
    return AuditContext(
        "real",
        dolos,
        lambda: SafetensorsCheckpoint(resolve_checkpoint_path("qwen", qwen_path), label="qwen"),
        load_qwen_token_decoder(),
    )


def synthetic_recovery_checks(ctx: AuditContext, summary: dict) -> dict:
    """Compare recovered values with the planted ground truth (synthetic mode only)."""
    planted = ctx.planted
    planted_keys = {f"{planted.layer_idx}:{m}" for m in planted.modules}
    checks: dict = {}
    if "stage1" in summary:
        s1 = summary["stage1"]
        checks["stage1_localized_planted_modules"] = planted_keys.issubset(set(s1["localized_modules"]))
        checks["stage1_trigger_recovered"] = bool(s1["decoded_trigger_candidates"]) and \
            s1["decoded_trigger_candidates"][0]["token_id"] == planted.trigger_token_id
        checks["stage1_target_recovered"] = bool(s1["decoded_target_candidates"]) and \
            s1["decoded_target_candidates"][0]["token_id"] == planted.target_token_id
    if "stage2" in summary:
        s2 = summary["stage2"]
        checks["stage2_cure_k1_removes_behavior"] = s2.get("cure_k1", {}).get("behavior_rate") == 0.0
        checks["stage2_implant_k1_induces_behavior"] = s2.get("implant_k1", {}).get("behavior_rate") == 1.0
    if "stage3" in summary:
        s3 = summary["stage3"]
        checks["stage3_top_layer_is_planted"] = s3["top_layer"] == planted.layer_idx
        checks["stage3_planted_modules_flagged"] = planted_keys.issubset(set(s3["flagged_modules"]))
        checks["stage3_trigger_recovered_without_reference"] = bool(s3["reference_free_trigger_candidates"]) and \
            s3["reference_free_trigger_candidates"][0]["token_id"] == planted.trigger_token_id
        checks["stage3_target_recovered_without_reference"] = bool(s3["reference_free_target_candidates"]) and \
            s3["reference_free_target_candidates"][0]["token_id"] == planted.target_token_id
        probe = s3.get("activation_probe")
        if isinstance(probe, dict):
            checks["stage3_activation_trigger_recovered"] = probe["trigger_candidate_at_top_layer"] == planted.trigger_token_id
    return checks


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("1", "2", "3", "all"), default="all")
    parser.add_argument("--synthetic", action="store_true", help="offline planted-backdoor demonstration")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spec04"))
    parser.add_argument("--top-k", type=int, default=16, help="singular triplets kept per delta matrix")
    parser.add_argument("--seed", type=int, default=0, help="synthetic model seed")
    parser.add_argument("--dolos-weights", type=Path, help="local Dolos model.safetensors (skips download)")
    parser.add_argument("--qwen-weights", type=Path, help="local Qwen2.5-1.5B-Instruct model.safetensors")
    args = parser.parse_args(argv)

    try:
        ctx = build_context(args)
    except MissingDependencyError as exc:
        print(f"[spec04] ERROR: {exc}", file=sys.stderr)
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"mode": ctx.mode, "stage": args.stage}
    if ctx.mode == "synthetic":
        summary["disclaimer"] = (
            "SYNTHETIC: small random Qwen2-shaped model with a planted rank-1 MLP circuit; "
            "these results demonstrate the pipeline, they are NOT findings about Dolos-1.5B."
        )
        summary["planted_ground_truth"] = {
            "layer": ctx.planted.layer_idx,
            "modules": list(ctx.planted.modules),
            "neuron": ctx.planted.neuron_idx,
            "trigger_token_id": ctx.planted.trigger_token_id,
            "target_token_id": ctx.planted.target_token_id,
            "strength": ctx.planted.strength,
        }

    stage1_metrics = None
    if args.stage in ("1", "2", "all"):
        stage1_summary, stage1_metrics, stage1_state = run_stage1(ctx, output_dir, args.top_k)
        if args.stage in ("1", "all"):
            summary["stage1"] = stage1_summary
            print(f"[spec04] stage 1 localized modules: {stage1_summary['localized_modules']}")
            print(f"[spec04] stage 1 trigger candidates: {stage1_summary['decoded_trigger_candidates'][:3]}")
            print(f"[spec04] stage 1 target candidates: {stage1_summary['decoded_target_candidates'][:3]}")
        if args.stage in ("2", "all"):
            summary["stage2"] = run_stage2(ctx, output_dir, stage1_state)
            print(f"[spec04] stage 2 cure k=1: {summary['stage2'].get('cure_k1')}")
            print(f"[spec04] stage 2 implant k=1: {summary['stage2'].get('implant_k1')}")
    if args.stage in ("3", "all"):
        summary["stage3"] = run_stage3(ctx, output_dir, stage1_metrics if args.stage == "all" else None)
        print(f"[spec04] stage 3 (reference-free) top layer: {summary['stage3']['top_layer']}, "
              f"flagged modules: {summary['stage3']['flagged_modules']}")

    if ctx.mode == "synthetic":
        summary["recovery_checks"] = synthetic_recovery_checks(ctx, summary)
        for name, ok in summary["recovery_checks"].items():
            print(f"[spec04] {'PASS' if ok else 'FAIL'} {name}")

    with open(output_dir / f"{ctx.name('summary')}.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    print(f"[spec04] outputs written to {output_dir}")
    return 0


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


if __name__ == "__main__":
    sys.exit(main())
