"""Spec 03 end-to-end runner: counterfactual checkpoint forking as a plus-factor audit.

Modes:

* **Offline scripted positive control (default, $0):** 15 trajectories per archetype from
  the scripted policy simulator (NOT LLM results), each forked in-process at t* into a
  control and a synthetic-undercut branch (k rounds, S rollout seeds). Evidence vectors
  (E_out, E_comm, E_cf), split-half calibration of the spec 4.2 rule, frozen held-out
  benchmark against outcome-only / comm-only / naive-hybrid screens, fig1 fan chart,
  fig2 (E_out, E_cf) scatter and Table 1.
* **Real llm-cartel runs on disk:** every base run found under ``--audit-dir`` (default:
  this run's ``live/runs`` plus ``collusion_hackathon/llm-cartel/results/jsonl``) gets
  offline E_out / E_comm. E_cf is filled only from matching spec 03 branch files; otherwise
  it is NaN (``e_cf_available = False``), never fabricated.
* **Live (``--live``):** writes the exact ``run_game.py`` (phase 1) and ``branch_game.py``
  (phase 2) commands to ``spec03_live_commands.txt`` (dry run). Only ``--execute`` launches
  them (cwd = llm-cartel; all outputs under ``<output-dir>/live``), writing the undercut
  checkpoints in-process between the two phases.

Usage:
    python -m src.spec03_counterfactual_audit.cli.run_spec03_audit --output-dir outputs/spec03
    python -m src.spec03_counterfactual_audit.cli.run_spec03_audit --smoke --output-dir outputs/spec03
    python -m src.spec03_counterfactual_audit.cli.run_spec03_audit --live --output-dir outputs/spec03
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.common.plotting import get_pyplot, save_figure
from src.spec03_counterfactual_audit.models import (
    RIVAL_INDEX,
    CartelTrajectory,
    CaseArchetype,
    ForkedBranchResult,
    PlusFactorEvidence,
)
from src.spec03_counterfactual_audit.repositories.cartel_repository import (
    branch_output_path,
    list_run_files,
    load_branch_rollout,
    load_config,
    load_run,
    write_undercut_checkpoint,
)
from src.spec03_counterfactual_audit.services.counterfactual_fork_service import CounterfactualForkService
from src.spec03_counterfactual_audit.services.evidence_service import (
    EvidenceService,
    LLMJudgeCommunicationScorer,
)
from src.spec03_counterfactual_audit.services.plus_factor_classifier import PlusFactorClassifier
from src.spec03_counterfactual_audit.services.scripted_market_simulator import ScriptedMarketSimulator

REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_DIR = REPO_ROOT / "collusion_hackathon" / "llm-cartel"
HARNESS_RUNS = HARNESS_DIR / "results" / "jsonl"

ARCHETYPE_COLORS = {
    CaseArchetype.TRUE_CARTEL: "tab:red",
    CaseArchetype.INNOCENT_PARALLEL: "tab:blue",
    CaseArchetype.CHEAP_TALK: "tab:green",
}

# Phase 1 llm-cartel arms (spec 03, section 5). Cheap talk has no harness support: no
# prompt prefix yields collusive text with Nash pricing, there is no message channel and
# MARKETS (costs) are hard-coded, so that arm is reported, not invented.
LIVE_ARMS = {
    CaseArchetype.TRUE_CARTEL: ["--prefix", "PC", "--window", "20", "--think"],
    CaseArchetype.INNOCENT_PARALLEL: ["--prefix", "P0", "--window", "0", "--think"],
}
CHEAP_TALK_UNSUPPORTED = (
    "# cheap_talk arm NOT EMITTED: llm-cartel has no prompt prefix that produces collusive text "
    "with competitive pricing, no inter-agent message channel, and hard-coded MARKETS (no cost shift). "
    "Spec 03 phase 1 case C needs harness support (or steganographic-collusion) first."
)


# --------------------------------------------------------------------------- #
# Offline scripted positive control
# --------------------------------------------------------------------------- #
def build_scripted_evidence(
    n_per_archetype: int,
    rounds: int,
    audit_round: int,
    k: int,
    rollout_seeds: list[int],
) -> tuple[list[PlusFactorEvidence], dict[str, ForkedBranchResult], dict[str, CartelTrajectory]]:
    simulator = ScriptedMarketSimulator()
    forker = CounterfactualForkService()
    evidence_service = EvidenceService()
    nash = simulator.benchmarks.nash_prices[RIVAL_INDEX]
    monopoly = simulator.benchmarks.monopoly_prices[RIVAL_INDEX]

    evidence: list[PlusFactorEvidence] = []
    forks: dict[str, ForkedBranchResult] = {}
    trajectories: dict[str, CartelTrajectory] = {}
    for archetype in CaseArchetype:
        for seed in range(n_per_archetype):
            trajectory = simulator.generate_trajectory(archetype, seed, rounds=rounds)
            no_comm = simulator.generate_trajectory(archetype, seed, rounds=rounds, comm_enabled=False)
            fork = forker.fork_scripted(simulator, trajectory, audit_round, k=k, seeds=rollout_seeds)
            e_cf, degenerate = forker.elasticity_from_fork(fork, nash, monopoly)
            evidence.append(
                PlusFactorEvidence(
                    run_id=trajectory.run_id,
                    archetype=archetype,
                    outcome_elevation=evidence_service.outcome_elevation(trajectory, audit_round),
                    communication_intent=evidence_service.communication_intent(trajectory, audit_round),
                    retaliation_elasticity=e_cf,
                    outcome_elevation_no_comm=evidence_service.outcome_elevation(no_comm, audit_round),
                    cf_denominator_degenerate=degenerate,
                )
            )
            forks[trajectory.run_id] = fork
            trajectories[trajectory.run_id] = trajectory
    return evidence, forks, trajectories


def write_offline_outputs(
    output_dir: Path,
    evidence: list[PlusFactorEvidence],
    forks: dict[str, ForkedBranchResult],
    benchmark: dict,
    benchmarks_prices: tuple[float, float],
) -> None:
    calibration_ids = {e.run_id for e in benchmark["calibration"]}
    rows = []
    for e in evidence:
        rows.append(
            {
                **_evidence_row(e),
                "split": "calibration" if e.run_id in calibration_ids else "held_out",
                **{f"flag_{name}": int(flag) for name, flag in benchmark["predictions"][e.run_id].items()},
            }
        )
    _write_csv(output_dir / "spec03_scripted_evidence.csv", rows)

    table = [
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in asdict(m).items()}
        for m in benchmark["metrics"]
    ]
    _write_csv(output_dir / "spec03_table1_heldout_benchmark.csv", table)

    frozen = {
        "decision_rule": asdict(benchmark["rule"]),
        "baselines": {name: asdict(rule) for name, rule in benchmark["baselines"].items()},
        "held_out_run_ids": [e.run_id for e in benchmark["held_out"]],
    }
    with open(output_dir / "spec03_frozen_rules.json", "w", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2)

    branches = {run_id: _fork_dict(fork) for run_id, fork in forks.items()}
    with open(output_dir / "spec03_fork_branches.json", "w", encoding="utf-8") as handle:
        json.dump(branches, handle, indent=1)

    plot_fan_chart(output_dir, evidence, forks, benchmarks_prices)
    plot_decision_scatter(output_dir, evidence, benchmark)


def _evidence_row(e: PlusFactorEvidence) -> dict:
    return {
        "run_id": e.run_id,
        "archetype": e.archetype.value if e.archetype else "",
        "e_out": round(e.outcome_elevation, 6),
        "e_comm": round(e.communication_intent, 6),
        "e_cf": "" if not e.cf_available else round(e.retaliation_elasticity, 6),
        "e_cf_available": int(e.cf_available),
        "cf_denominator_degenerate": int(e.cf_denominator_degenerate),
        "e_out_no_comm": "" if e.outcome_elevation_no_comm is None else round(e.outcome_elevation_no_comm, 6),
    }


def _fork_dict(fork: ForkedBranchResult) -> dict:
    return {
        "audit_round": fork.audit_round,
        "rival_original_price": fork.rival_original_price,
        "rival_undercut_price": fork.rival_undercut_price,
        "pre_fork_auditee_prices": list(fork.pre_fork_auditee_prices),
        "pre_fork_rival_prices": list(fork.pre_fork_rival_prices),
        "control": [asdict(r) for r in fork.control_rollouts],
        "defection": [asdict(r) for r in fork.defection_rollouts],
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_fan_chart(
    output_dir: Path,
    evidence: list[PlusFactorEvidence],
    forks: dict[str, ForkedBranchResult],
    benchmark_prices: tuple[float, float],
    lookback: int = 8,
) -> None:
    """Fig 1: auditee price before / after the t* fork, representative run per archetype."""
    plt = get_pyplot()
    archetypes = [a for a in CaseArchetype if any(e.archetype is a for e in evidence)]
    fig, axes = plt.subplots(1, len(archetypes), figsize=(4.6 * len(archetypes), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    nash, monopoly = benchmark_prices
    for ax, archetype in zip(axes, archetypes):
        group = sorted(
            (e for e in evidence if e.archetype is archetype and e.run_id in forks),
            key=lambda e: e.retaliation_elasticity if e.cf_available else -math.inf,
        )
        if not group:
            continue
        representative = group[len(group) // 2]
        fork = forks[representative.run_id]
        t_star = fork.audit_round
        pre = list(fork.pre_fork_auditee_prices)[-lookback:]
        pre_x = list(range(t_star - len(pre) + 1, t_star + 1))
        post_x = list(range(t_star + 1, t_star + 1 + len(fork.mean_auditee_path("control"))))
        ax.plot(pre_x, pre, color="black", marker="o", ms=3, label="observed (auditee i)")
        for branch, color in (("control", "tab:gray"), ("defection", "tab:red")):
            rollouts = fork.control_rollouts if branch == "control" else fork.defection_rollouts
            for r in rollouts:
                ax.plot([t_star, *post_x], [pre[-1], *r.auditee_prices], color=color, alpha=0.25, lw=1)
            ax.plot(
                [t_star, *post_x],
                [pre[-1], *fork.mean_auditee_path(branch)],
                color=color,
                lw=2.2,
                marker="o",
                ms=3,
                label=f"branch: {branch} (mean of {len(rollouts)} seeds)",
            )
        ax.axvline(t_star, color="black", ls=":", lw=1)
        ax.axhline(nash, color="tab:green", ls="--", lw=0.8, label="Nash")
        ax.axhline(monopoly, color="tab:purple", ls="--", lw=0.8, label="monopoly")
        ax.set_title(f"{archetype.value}\nE_cf = {representative.retaliation_elasticity:.2f} ({representative.run_id})",
                     fontsize=9)
        ax.set_xlabel("round (t* = checkpoint; rival j undercut at t* in red branch)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("auditee price p_i")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Fig 1: impulse response to a synthetic rival undercut (scripted positive control, not LLM)",
                 fontsize=10)
    save_figure(output_dir, "fig1_impulse_response_fan")


def plot_decision_scatter(output_dir: Path, evidence: list[PlusFactorEvidence], benchmark: dict) -> None:
    """Fig 2: (E_out, E_cf) by archetype, calibration vs held-out, with the frozen thresholds."""
    plt = get_pyplot()
    calibration_ids = {e.run_id for e in benchmark["calibration"]}
    rule = benchmark["rule"]
    plt.figure(figsize=(6.6, 5.0))
    for archetype, color in ARCHETYPE_COLORS.items():
        for split, marker in (("calibration", "o"), ("held_out", "x")):
            pts = [
                (e.outcome_elevation, e.retaliation_elasticity)
                for e in evidence
                if e.archetype is archetype and e.cf_available
                and ((e.run_id in calibration_ids) == (split == "calibration"))
            ]
            if pts:
                xs, ys = zip(*pts)
                plt.scatter(xs, ys, c=color, marker=marker, s=40, label=f"{archetype.value} ({split})")
    plt.axvline(rule.theta_out, color="black", ls="--", lw=1, label=f"theta_out = {rule.theta_out:.2f} (pre-registered)")
    if math.isfinite(rule.theta_cf):
        plt.axhline(rule.theta_cf, color="black", ls=":", lw=1, label=f"theta_cf = {rule.theta_cf:.2f} (frozen)")
    plt.xlabel("E_out (Calvano profit-gain index, pre-audit window)")
    plt.ylabel("E_cf (retaliation elasticity, first post-fork round)")
    plt.title("Fig 2: plus-factor evidence space (scripted positive control)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    save_figure(output_dir, "fig2_eout_ecf_scatter")


# --------------------------------------------------------------------------- #
# Real llm-cartel runs (offline E_out / E_comm, E_cf only from real branches)
# --------------------------------------------------------------------------- #
def audit_real_runs(
    output_dir: Path,
    audit_dirs: list[Path],
    branches_dir: Path,
    checkpoints_dir: Path,
    audit_round: int,
    k: int,
    rollout_seeds: list[int],
    judge: LLMJudgeCommunicationScorer | None,
) -> list[PlusFactorEvidence]:
    evidence_service = EvidenceService()
    forker = CounterfactualForkService()
    evidence: list[PlusFactorEvidence] = []
    run_files = [p for d in audit_dirs for p in list_run_files(d)]
    if not run_files:
        print(f"[spec03] no llm-cartel runs found under {[str(d) for d in audit_dirs]}; real-data audit skipped.")
        return evidence
    for run_path in run_files:
        try:
            trajectory = load_run(run_path)
        except (ValueError, KeyError) as exc:
            print(f"[spec03] skipping {run_path.name}: {exc}")
            continue
        e_comm = evidence_service.communication_intent(trajectory, audit_round)
        if judge is not None:
            texts = [t for r in trajectory.rounds if r.round_id <= audit_round for t in r.texts]
            e_comm = judge.score_texts(texts)
        e_cf, degenerate = _real_elasticity(run_path, trajectory, branches_dir, checkpoints_dir,
                                            audit_round, k, rollout_seeds, forker)
        evidence.append(
            PlusFactorEvidence(
                run_id=trajectory.run_id,
                archetype=trajectory.archetype,
                outcome_elevation=evidence_service.outcome_elevation(trajectory, audit_round),
                communication_intent=e_comm,
                retaliation_elasticity=e_cf,
                cf_denominator_degenerate=degenerate,
            )
        )
    _write_csv(output_dir / "spec03_real_runs_evidence.csv", [_evidence_row(e) for e in evidence])
    print(f"[spec03] real-data audit: {len(evidence)} runs, "
          f"{sum(e.cf_available for e in evidence)} with E_cf from live forks")
    return evidence


def _real_elasticity(
    run_path: Path,
    trajectory: CartelTrajectory,
    branches_dir: Path,
    checkpoints_dir: Path,
    audit_round: int,
    k: int,
    rollout_seeds: list[int],
    forker: CounterfactualForkService,
) -> tuple[float, bool]:
    checkpoint = checkpoints_dir / f"{run_path.stem}__undercut_r{audit_round}.jsonl"
    if not checkpoint.is_file():
        return math.nan, False
    control, defection = [], []
    for s in rollout_seeds:
        c_path = branch_output_path(run_path, audit_round, f"cf_control_s{s}", branches_dir)
        d_path = branch_output_path(checkpoint, audit_round, f"cf_defect_s{s}", branches_dir)
        if not (c_path.is_file() and d_path.is_file()):
            return math.nan, False
        control.append(load_branch_rollout(c_path, "control", s, k))
        defection.append(load_branch_rollout(d_path, "defection", s, k))
    perturbation = load_config(checkpoint)["spec03_perturbation"]
    return forker.compute_retaliation_elasticity(
        [r.auditee_prices[0] for r in control],
        [r.auditee_prices[0] for r in defection],
        perturbation["rival_original_price"],
        perturbation["rival_undercut_price"],
        trajectory.benchmarks.nash_prices[RIVAL_INDEX],
        trajectory.benchmarks.monopoly_prices[RIVAL_INDEX],
    )


# --------------------------------------------------------------------------- #
# Live path
# --------------------------------------------------------------------------- #
def build_phase1_commands(model: str, seeds: list[int], rounds: int, runs_dir: Path) -> list[tuple[str, list[str]]]:
    commands = []
    for archetype, flags in LIVE_ARMS.items():
        for seed in seeds:
            name = f"spec03_{archetype.value}_seed{seed}"
            argv = [sys.executable, "run_game.py", "--models", model, *flags, "--rounds", str(rounds),
                    "--seeds", str(seed), "--name", name, "--out_dir", str(runs_dir), "-y"]
            commands.append((name, argv))
    return commands


def build_phase2_commands(
    run_names: list[str], audit_round: int, k: int, rollout_seeds: list[int],
    runs_dir: Path, checkpoints_dir: Path, branches_dir: Path,
) -> list[tuple[str, list[str]]]:
    commands = []
    for name in run_names:
        run_path = runs_dir / f"{name}.jsonl"
        checkpoint = checkpoints_dir / f"{name}__undercut_r{audit_round}.jsonl"
        for s in rollout_seeds:
            common = ["--at_round", str(audit_round), "--rounds", str(k), "--seed", str(s),
                      "--out_dir", str(branches_dir), "-y"]
            commands.append((f"{name}_control_s{s}", [sys.executable, "branch_game.py", str(run_path),
                                                      *common, "--label", f"cf_control_s{s}"]))
            commands.append((f"{name}_defect_s{s}", [sys.executable, "branch_game.py", str(checkpoint),
                                                     *common, "--label", f"cf_defect_s{s}"]))
    return commands


def run_live(args, live_dir: Path, output_dir: Path) -> None:
    # Absolute paths: the harness runs with cwd = llm-cartel, so relative paths would land
    # inside collusion_hackathon/.
    live_dir = live_dir.resolve()
    runs_dir, checkpoints_dir, branches_dir = live_dir / "runs", live_dir / "checkpoints", live_dir / "branches"
    phase1 = build_phase1_commands(args.model, args.live_seeds, args.rounds, runs_dir)
    names = [name for name, _argv in phase1]
    phase2 = build_phase2_commands(names, args.audit_round, args.k, args.rollout_seeds,
                                   runs_dir, checkpoints_dir, branches_dir)
    lines = [f"# cwd: {HARNESS_DIR}", "# Phase 1: base trajectories (run_game.py)", CHEAP_TALK_UNSUPPORTED]
    lines += [" ".join(argv) for _n, argv in phase1]
    lines += [
        f"# Phase 1.5 (in-process, spec03 write_undercut_checkpoint): for each base run write "
        f"{checkpoints_dir}/<run>__undercut_r{args.audit_round}.jsonl with rival j={RIVAL_INDEX} undercut "
        f"p_D = p_Nash + 0.1 (p_j,t* - p_Nash)",
        "# Phase 2: matched control / defection forks (branch_game.py)",
    ]
    lines += [" ".join(argv) for _n, argv in phase2]
    commands_path = output_dir / "spec03_live_commands.txt"
    commands_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[spec03] wrote {len(phase1)} phase-1 and {len(phase2)} phase-2 commands to {commands_path}")
    if not args.execute:
        print("[spec03] dry run only - pass --execute (with --live) to launch them.")
        return
    for name, argv in phase1:
        print(f"[spec03] phase 1: {name}")
        if subprocess.run(argv, cwd=HARNESS_DIR, check=False).returncode != 0:  # noqa: S603
            print(f"[spec03] WARNING: {name} failed")
    for name in names:
        run_path = runs_dir / f"{name}.jsonl"
        if run_path.is_file():
            write_undercut_checkpoint(run_path, args.audit_round, checkpoints_dir)
    for name, argv in phase2:
        if not Path(argv[2]).is_file():
            print(f"[spec03] skipping {name}: missing {argv[2]}")
            continue
        print(f"[spec03] phase 2: {name}")
        if subprocess.run(argv, cwd=HARNESS_DIR, check=False).returncode != 0:  # noqa: S603
            print(f"[spec03] WARNING: {name} failed")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _print_table(metrics) -> None:
    print("[spec03] held-out benchmark (scripted positive control, NOT LLM results):")
    print(f"   {'standard':<28}{'n':>4}{'prec':>7}{'recall':>8}{'F1':>7}{'FPR_ip':>8}{'FPR_ct':>8}{'bal_acc':>9}")
    for m in metrics:
        print(f"   {m.standard:<28}{m.n:>4}{m.precision:>7.2f}{m.recall:>8.2f}{m.f1:>7.2f}"
              f"{m.fpr_innocent_parallel:>8.2f}{m.fpr_cheap_talk:>8.2f}{m.balanced_accuracy:>9.2f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spec03"))
    parser.add_argument("--smoke", action="store_true", help="tiny fast offline run (4 per archetype, 2 seeds)")
    parser.add_argument("--n-per-archetype", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=40, help="rounds per base trajectory (spec: 40)")
    parser.add_argument("--audit-round", type=int, default=25, help="checkpoint round t* (spec: 25)")
    parser.add_argument("--k", type=int, default=5, help="post-fork rollout horizon (spec: 5)")
    parser.add_argument("--rollout-seeds", type=int, nargs="+", default=[0, 1, 2], help="S rollout seeds")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--audit-dir", type=Path, nargs="*", default=None,
                        help="dirs with real llm-cartel runs (default: <output>/live/runs and llm-cartel results/jsonl)")
    parser.add_argument("--llm-judge", action="store_true", help="score real-run E_comm with an LLM judge "
                        "(requires a configured endpoint; rubric otherwise)")
    parser.add_argument("--live", action="store_true", help="write the live run_game/branch_game commands")
    parser.add_argument("--execute", action="store_true", help="with --live: actually launch them")
    parser.add_argument("--model", default="openai/gpt-4.1-mini")
    parser.add_argument("--live-seeds", type=int, nargs="+", default=list(range(15)))
    args = parser.parse_args(argv)
    if args.smoke:
        args.n_per_archetype = min(args.n_per_archetype, 4)
        args.rollout_seeds = args.rollout_seeds[:2]
    if args.audit_round + 1 > args.rounds:
        parser.error("--audit-round must be < --rounds")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    live_dir = output_dir / "live"

    # -- offline scripted positive control -------------------------------- #
    evidence, forks, _trajectories = build_scripted_evidence(
        args.n_per_archetype, args.rounds, args.audit_round, args.k, args.rollout_seeds
    )
    benchmark = PlusFactorClassifier().run_benchmark(evidence, split_seed=args.split_seed)
    market = ScriptedMarketSimulator().benchmarks
    write_offline_outputs(output_dir, evidence, forks, benchmark,
                          (market.nash_prices[0], market.monopoly_prices[0]))
    rule = benchmark["rule"]
    print(f"[spec03] scripted positive control: {len(evidence)} trajectories "
          f"({len(benchmark['calibration'])} calibration / {len(benchmark['held_out'])} held-out)")
    print(f"[spec03] frozen rule: theta_out={rule.theta_out:.3f} theta_cf={rule.theta_cf:.3f} "
          f"theta_comm={rule.theta_comm:.3f} delta={rule.delta:.3f}")
    _print_table(benchmark["metrics"])

    # -- live commands / execution ----------------------------------------- #
    if args.live:
        run_live(args, live_dir, output_dir)

    # -- real llm-cartel runs on disk --------------------------------------- #
    judge = None
    if args.llm_judge:
        try:
            judge = LLMJudgeCommunicationScorer()
        except Exception as exc:  # noqa: BLE001 - LLMClientError when no endpoint is configured
            print(f"[spec03] LLM judge unavailable ({exc}); using the keyword rubric.")
    audit_dirs = args.audit_dir if args.audit_dir is not None else [live_dir / "runs", HARNESS_RUNS]
    real = audit_real_runs(output_dir, audit_dirs, live_dir / "branches", live_dir / "checkpoints",
                           args.audit_round, args.k, args.rollout_seeds, judge)

    summary = {
        "mode": "smoke" if args.smoke else "full",
        "label": "scripted positive control (not LLM results)",
        "n_trajectories": len(evidence),
        "frozen_rule": asdict(rule) | {"calibration_run_ids": len(rule.calibration_run_ids)},
        "table1": [asdict(m) for m in benchmark["metrics"]],
        "real_runs_audited": len(real),
        "real_runs_with_e_cf": sum(e.cf_available for e in real),
    }
    with open(output_dir / "spec03_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    print(f"[spec03] outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
