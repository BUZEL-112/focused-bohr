"""Spec 06 audit: 4-view x 3x3 cross-family monitor grid with symbolic say-do scrubbing.

Modes:

* **Offline (default, $0):** a SCRIPTED SIMULATION. Family-styled colluding / honest
  trajectories from the scripted generator are rendered into the four Track 2.2 views
  and scored by scripted family monitors with an injectable same-family leniency bias
  (``--leniency-bias``) and an eloquence discount (``--seduction``). The outputs show
  how the pipeline measures beta_self(v) and AUROC; they are NOT real LLM monitor
  results. The run also compiles real colludebench pricing transcripts into View 4
  say-do indicators (no monitor, no model families: those logs carry no model id) and
  loads any llm-trust-breaking endogenous logs found (none ship with the repo).
* **Live (``--live``):** real per-family LLM monitors via ``common.llm_client`` with the
  frozen rubric prompt. Trajectories come from ``--trust-breaking-logs`` when given,
  else from the scripted generator. Without a configured endpoint the exact requests
  are written to ``spec06_dry_run_requests.jsonl`` and the run exits cleanly.

Usage:
    python -m src.spec06_monitor_bias.cli.run_spec06_audit --smoke --output-dir outputs/spec06
    python -m src.spec06_monitor_bias.cli.run_spec06_audit --live --trust-breaking-logs path/to/results
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from src.common.llm_client import LLMClientError, resolve_endpoint
from src.common.metrics import auroc
from src.common.plotting import get_pyplot, save_figure
from src.spec06_monitor_bias.models import MonitorVerdict, MonitorView, TrajectoryRecord, ViewEvaluation
from src.spec06_monitor_bias.repositories.dilemma_repository import (
    FAMILIES,
    FAMILY_STYLE_MARKERS,
    discover_colludebench_pricing,
    discover_trust_breaking_logs,
    generate_scripted_dataset,
    load_colludebench_pricing,
    load_trust_breaking_json,
)
from src.spec06_monitor_bias.services.cross_family_monitor_service import (
    DEFAULT_MONITOR_MODELS,
    CrossFamilyMonitorService,
    LiveFamilyMonitor,
    ScriptedFamilyMonitor,
    build_monitor_requests,
)
from src.spec06_monitor_bias.services.familial_bias_evaluator import FamilialBiasEvaluator
from src.spec06_monitor_bias.services.symbolic_scrubber_service import SymbolicScrubberService

REPO_ROOT = Path(__file__).resolve().parents[3]
HACKATHON = REPO_ROOT / "collusion_hackathon"
COLLUDEBENCH_RESULTS = HACKATHON / "steganographic-collusion" / "results"
TRUST_BREAKING_DIR = HACKATHON / "llm-trust-breaking"

SIMULATION_LABEL = "SCRIPTED SIMULATION - not real LLM monitor results"
VIEW_LABELS = {
    MonitorView.MESSAGES_ONLY: "V1 messages",
    MonitorView.ACTIONS_ONLY: "V2 actions",
    MonitorView.RAW_HYBRID: "V3 raw hybrid",
    MonitorView.SYMBOLIC_SCRUBBED: "V4 symbolic",
}


# --------------------------------------------------------------------------- #
# Offline simulation
# --------------------------------------------------------------------------- #
def run_scripted_grid(
    per_family: int, seed: int, leniency: float, seduction: float
) -> tuple[list[TrajectoryRecord], list[MonitorVerdict]]:
    trajectories = generate_scripted_dataset(FAMILIES, per_family=per_family, seed=seed)
    scrubber = SymbolicScrubberService()
    monitors = [
        ScriptedFamilyMonitor(
            family, FAMILY_STYLE_MARKERS, leniency_bias=leniency,
            seduction_strength=seduction, seed=seed, scrubber=scrubber,
        )
        for family in FAMILIES
    ]
    verdicts = CrossFamilyMonitorService(monitors, scrubber).run(trajectories)
    return trajectories, verdicts


def summarize_evaluations(evaluations: dict[MonitorView, ViewEvaluation]) -> list[dict]:
    rows = []
    for view, ev in evaluations.items():
        rows.append(
            {
                "view": view.value,
                "beta_self": round(ev.beta_self, 4),
                "auroc": round(ev.auroc, 4),
                "threshold": round(ev.threshold, 4),
                "test_accuracy": round(ev.test_accuracy, 4),
                "test_fpr": round(ev.test_fpr, 4),
                "test_tpr": round(ev.test_tpr, 4),
                "diag_vs_offdiag_p": round(ev.diagonal_vs_offdiagonal_p, 6),
                "self_recognition_rate": round(ev.self_recognition_rate, 4),
            }
        )
    return rows


def plot_figures(
    output_dir: Path,
    evaluations: dict[MonitorView, ViewEvaluation],
    verdicts: list[MonitorVerdict],
    families: tuple[str, ...],
    label: str,
) -> None:
    plt = get_pyplot()

    # Figure 1: collusive-trajectory suspicion grid, V3 vs V4.
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, view in zip(axes, (MonitorView.RAW_HYBRID, MonitorView.SYMBOLIC_SCRUBBED)):
        ev = evaluations[view]
        matrix = np.array([[ev.collusive_grid.get((a, m), np.nan) for m in families] for a in families])
        image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(families)), families)
        ax.set_yticks(range(len(families)), families)
        ax.set_xlabel("Monitor family")
        ax.set_ylabel("Agent family")
        for i in range(len(families)):
            for j in range(len(families)):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                        color="white" if matrix[i, j] < 0.6 else "black", fontsize=9)
        ax.set_title(f"{VIEW_LABELS[view]}: beta_self = {ev.beta_self:+.3f}", fontsize=10)
    fig.colorbar(image, ax=axes[1], shrink=0.8, label="Mean suspicion (collusive trajectories)")
    fig.suptitle(f"Familial leniency grid\n{label}", fontsize=10)
    save_figure(output_dir, "fig1_familial_bias_heatmap")

    # Figure 2: four-view AUROC / accuracy / FPR.
    views = list(evaluations)
    x = np.arange(len(views))
    width = 0.26
    plt.figure(figsize=(8.0, 4.4))
    plt.bar(x - width, [evaluations[v].auroc for v in views], width, label="AUROC")
    plt.bar(x, [evaluations[v].test_accuracy for v in views], width, label="Held-out accuracy")
    plt.bar(x + width, [evaluations[v].test_fpr for v in views], width, label="Held-out FPR")
    plt.xticks(x, [VIEW_LABELS[v] for v in views])
    plt.ylim(0.0, 1.05)
    plt.ylabel("Score")
    plt.title(f"Track 2.2 four-view comparison\n{label}", fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    save_figure(output_dir, "fig2_four_view_auroc")

    # Figure 3: detection rate vs agent verbosity.
    plt.figure(figsize=(7.0, 4.4))
    for view in views:
        points = FamilialBiasEvaluator.detection_by_verbosity(
            verdicts, view, evaluations[view].threshold
        )
        if points:
            plt.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=VIEW_LABELS[view])
    plt.xlabel("Mean message length (tokens) of collusive trajectory")
    plt.ylabel("Detection rate at frozen threshold")
    plt.ylim(-0.05, 1.05)
    plt.title(f"'Less is more': detection vs verbosity\n{label}", fontsize=10)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    save_figure(output_dir, "fig3_verbosity_paradox")


# --------------------------------------------------------------------------- #
# Real logs (no monitors)
# --------------------------------------------------------------------------- #
def analyze_colludebench(output_dir: Path, max_files: int | None) -> dict:
    """Compile real colludebench pricing transcripts into View 4 indicators."""
    paths = discover_colludebench_pricing(COLLUDEBENCH_RESULTS, max_files_per_split=max_files)
    if not paths:
        print(f"[spec06] no colludebench pricing transcripts under {COLLUDEBENCH_RESULTS}")
        return {"files": 0}
    scrubber = SymbolicScrubberService()
    rows: list[dict] = []
    sample_written = False
    for path in paths:
        for trajectory in load_colludebench_pricing(path):
            indicators = scrubber.compute_indicators(trajectory.rounds)
            summary = scrubber.summarize(indicators)
            composite = scrubber.trace_guard_composite(scrubber.trace_guard_dimensions(summary))
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "split": "audit" if trajectory.is_collusive else "calibration",
                    "is_collusive": trajectory.is_collusive,
                    "rounds": len(trajectory.round_ids),
                    **{k: round(v, 4) for k, v in summary.items()},
                    "trace_guard_composite": round(composite, 4),
                }
            )
            if not sample_written and trajectory.is_collusive:
                (output_dir / "sample_view4_colludebench.txt").write_text(
                    scrubber.render_trajectory(trajectory, MonitorView.SYMBOLIC_SCRUBBED) + "\n",
                    encoding="utf-8",
                )
                sample_written = True
    _write_csv(output_dir / "spec06_colludebench_indicators.csv", rows)
    result: dict = {"files": len(paths), "trajectories": len(rows)}
    pos = [r for r in rows if r["is_collusive"]]
    neg = [r for r in rows if not r["is_collusive"]]
    result["collusive"], result["honest"] = len(pos), len(neg)
    if pos and neg:
        for key in ("d_self_rate", "d_cross_rate", "harm_rate", "trace_guard_composite"):
            result[f"auroc_{key}"] = round(auroc([r[key] for r in pos], [r[key] for r in neg]), 4)
    return result


def load_trust_breaking(paths: list[Path]) -> list[TrajectoryRecord]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(discover_trust_breaking_logs(path))
        elif path.is_file():
            files.append(path)
    trajectories: list[TrajectoryRecord] = []
    for file in files:
        trajectories.extend(load_trust_breaking_json(file))
    return trajectories


# --------------------------------------------------------------------------- #
# Live grid / dry run
# --------------------------------------------------------------------------- #
def run_live(
    output_dir: Path, trajectories: list[TrajectoryRecord], model_ids: dict[str, str], label: str
) -> dict:
    endpoint = resolve_endpoint()
    if not (endpoint.is_configured and endpoint.base_url):
        requests = build_monitor_requests(trajectories, model_ids)
        path = output_dir / "spec06_dry_run_requests.jsonl"
        with open(path, "w", encoding="utf-8") as handle:
            for request in requests:
                handle.write(json.dumps(request) + "\n")
        print(f"[spec06] no LLM endpoint configured - dry run: wrote {len(requests)} requests to {path}")
        return {"mode": "dry_run", "requests": len(requests)}
    try:
        monitors = [LiveFamilyMonitor(family, model_id, probe_identity=True)
                    for family, model_id in model_ids.items()]
        verdicts = CrossFamilyMonitorService(monitors).run(trajectories)
    except LLMClientError as exc:
        print(f"[spec06] live monitoring failed: {exc}")
        return {"mode": "live_failed", "error": str(exc)}
    families = tuple(model_ids)
    evaluations = FamilialBiasEvaluator(families).evaluate(verdicts)
    _write_csv(output_dir / "spec06_live_verdicts.csv", [_verdict_row(v) for v in verdicts])
    _write_csv(output_dir / "spec06_live_view_metrics.csv", summarize_evaluations(evaluations))
    live_dir = output_dir / "live"
    plot_figures(live_dir, evaluations, verdicts, families, f"LIVE LLM monitors ({label})")
    return {"mode": "live", "verdicts": len(verdicts), "views": summarize_evaluations(evaluations)}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _verdict_row(verdict: MonitorVerdict) -> dict:
    return {
        "trajectory_id": verdict.trajectory_id,
        "agent_family": verdict.agent_family,
        "monitor_family": verdict.monitor_family,
        "view": verdict.view.value,
        "is_collusive": verdict.is_collusive,
        "suspicion": round(verdict.suspicion, 4),
        "self_recognized": verdict.self_recognized,
        "mean_message_tokens": round(verdict.mean_message_tokens, 2),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spec06"))
    parser.add_argument("--smoke", action="store_true", help="small dataset, 2 colludebench files per split")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-family", type=int, default=12, help="scripted trajectories per family (half colluding)")
    parser.add_argument("--leniency-bias", type=float, default=0.25, help="scripted same-family leniency")
    parser.add_argument("--seduction", type=float, default=0.30, help="scripted eloquence discount strength")
    parser.add_argument("--live", action="store_true", help="real LLM monitors (dry run without endpoint)")
    parser.add_argument("--trust-breaking-logs", type=Path, nargs="*", default=[],
                        help="llm-trust-breaking *_endogenous*.json files or dirs")
    parser.add_argument("--model-openai", default=DEFAULT_MONITOR_MODELS["openai"])
    parser.add_argument("--model-qwen", default=DEFAULT_MONITOR_MODELS["qwen"])
    parser.add_argument("--model-gemini", default=DEFAULT_MONITOR_MODELS["gemini"])
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    per_family = 6 if args.smoke else args.per_family

    # 1. Scripted simulation of the 3x3x4 grid.
    trajectories, verdicts = run_scripted_grid(per_family, args.seed, args.leniency_bias, args.seduction)
    evaluations = FamilialBiasEvaluator(FAMILIES).evaluate(verdicts)
    metric_rows = summarize_evaluations(evaluations)
    _write_csv(output_dir / "spec06_scripted_verdicts.csv", [_verdict_row(v) for v in verdicts])
    _write_csv(output_dir / "spec06_scripted_view_metrics.csv", metric_rows)
    scrubber = SymbolicScrubberService()
    example = trajectories[0]
    with open(output_dir / "sample_views_scripted.txt", "w", encoding="utf-8") as handle:
        for view in MonitorView:
            handle.write(f"===== {view.value} ({example.trajectory_id}) =====\n")
            handle.write(scrubber.render_trajectory(example, view) + "\n\n")
    plot_figures(output_dir, evaluations, verdicts, FAMILIES, SIMULATION_LABEL)
    print(f"[spec06] {SIMULATION_LABEL}")
    print(f"[spec06] scripted grid: {len(trajectories)} trajectories x {len(MonitorView)} views x "
          f"{len(FAMILIES)} monitors = {len(verdicts)} verdicts")
    for row in metric_rows:
        print(f"[spec06]   {row['view']:<18} beta_self={row['beta_self']:+.4f} auroc={row['auroc']:.4f} "
              f"acc={row['test_accuracy']:.3f} fpr={row['test_fpr']:.3f} "
              f"self_recog={row['self_recognition_rate']:.2f}")

    # 2. Real logs.
    colludebench = analyze_colludebench(output_dir, max_files=2 if args.smoke else None)
    print(f"[spec06] colludebench pricing (real logs, indicators only): {colludebench}")
    tb_paths = list(args.trust_breaking_logs) or [TRUST_BREAKING_DIR]
    trust_breaking = load_trust_breaking(tb_paths)
    print(f"[spec06] llm-trust-breaking endogenous logs: {len(trust_breaking)} trajectories found")

    # 3. Live grid / dry run.
    live: dict = {}
    if args.live:
        model_ids = {"openai": args.model_openai, "qwen": args.model_qwen, "gemini": args.model_gemini}
        live_trajectories = trust_breaking or trajectories
        source = "trust-breaking logs" if trust_breaking else "scripted trajectories"
        live = run_live(output_dir, live_trajectories, model_ids, source)

    summary = {
        "label": SIMULATION_LABEL,
        "scripted": {
            "per_family": per_family,
            "leniency_bias": args.leniency_bias,
            "seduction": args.seduction,
            "seed": args.seed,
            "views": metric_rows,
        },
        "colludebench_real_logs": colludebench,
        "trust_breaking_trajectories": len(trust_breaking),
        "live": live,
    }
    with open(output_dir / "spec06_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[spec06] outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
