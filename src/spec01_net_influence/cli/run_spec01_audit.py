"""Spec 01 end-to-end audit runner (zero API cost).

Phases (see spec/01_pairwise_net_influence_and_ncd.md, sections 5-6):
    --phase control  Scripted positive control: NI/CCS must rank the true colluding pair
                     top-1 in Conditions B (grim trigger) and C (bid rotation), never in
                     Condition A; sample-complexity curve over T in {5,10,15,20,30}.
    --phase cached   Offline audit of the pre-cached logs: ColludeBench JSONL transcripts
                     and the 75 llm-agent-collusion experiment dirs -> ROC/AUROC against
                     price-level screens (naive correlation + price elevation).
    --phase all      Both (default).

Usage:
    python -m src.spec01_net_influence.cli.run_spec01_audit --output-dir outputs/spec01
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

from src.common.plotting import get_pyplot, save_figure
from src.spec01_net_influence.models import CoalitionRanking, DiscretizedTrajectory, RunRecord
from src.spec01_net_influence.repositories.trajectory_repository import (
    generate_scripted_run,
    load_colludebench_jsonl,
    load_double_auction_root,
)
from src.spec01_net_influence.services.coalition_ranker_service import CoalitionRankerService
from src.spec01_net_influence.services.residualizer_service import ResidualizerService

REPO_ROOT = Path(__file__).resolve().parents[3]
COLLUDEBENCH_RESULTS = REPO_ROOT / "collusion_hackathon" / "steganographic-collusion" / "results"
DOUBLE_AUCTION_DATA = REPO_ROOT / "collusion_hackathon" / "llm-agent-collusion" / "data"

CONDITIONS = ("independent", "grim_trigger", "bid_rotation")
CONDITION_TITLES = {
    "independent": "Condition A: Independent",
    "grim_trigger": "Condition B: Grim-Trigger Cartel",
    "bid_rotation": "Condition C: Bid-Rotation Cartel",
}
SAMPLE_COMPLEXITY_ROUNDS = (5, 10, 15, 20, 30)


def discretize(run: RunRecord, residualizer: ResidualizerService) -> DiscretizedTrajectory:
    return residualizer.discretize_turns(
        run.run_id, run.turns, run.is_collusive, run.colluding_pair
    )


def naive_price_correlation(run: RunRecord) -> float:
    """Baseline screen 1: mean absolute pairwise Pearson correlation of raw price levels."""
    agents = run.agent_ids
    if len(agents) < 2:
        return 0.0
    series = {
        agent: [t.action_value for t in run.turns if t.agent_id == agent] for agent in agents
    }
    correlations = []
    for a, b in combinations(agents, 2):
        x, y = np.asarray(series[a]), np.asarray(series[b])
        if x.std() < 1e-9 or y.std() < 1e-9:
            correlations.append(0.0)
            continue
        correlations.append(abs(float(np.corrcoef(x, y)[0, 1])))
    return float(np.mean(correlations)) if correlations else 0.0


def price_elevation_screen(run: RunRecord) -> float:
    """Baseline screen 2: mean ask elevation over the competitive benchmark (last third)."""
    benchmarks = {t.round_id: t.market_benchmark for t in run.turns}
    elevations = [t.action_value - benchmarks[t.round_id] for t in run.turns]
    tail = elevations[-max(1, len(elevations) // 3):]
    return float(np.mean(tail)) if tail else 0.0


# --------------------------------------------------------------------------- #
# Phase 1: scripted positive control
# --------------------------------------------------------------------------- #
def run_control_phase(ranker: CoalitionRankerService, output_dir: Path, seeds: int) -> dict:
    residualizer = ResidualizerService(num_bins=5)
    plt = get_pyplot()

    heatmap_rows: dict[str, dict[str, float]] = {}
    top1_hits = {condition: 0 for condition in CONDITIONS}
    sample_complexity = {condition: {t: 0 for t in SAMPLE_COMPLEXITY_ROUNDS} for condition in CONDITIONS}
    per_seed_rows: list[dict] = []

    for condition in CONDITIONS:
        for seed in range(seeds):
            run = generate_scripted_run(condition, rounds=30, seed=seed)
            ranking = ranker.audit_run(discretize(run, residualizer))
            if condition != "independent":
                assert run.colluding_pair is not None
                if ranking.rank_of(run.colluding_pair) == 1:
                    top1_hits[condition] += 1
            for rounds in SAMPLE_COMPLEXITY_ROUNDS:
                truncated = RunRecord(
                    run_id=run.run_id,
                    is_collusive=run.is_collusive,
                    colluding_pair=run.colluding_pair,
                    turns=tuple(t for t in run.turns if t.round_id <= rounds),
                    source=run.source,
                )
                sub_ranking = ranker.audit_run(discretize(truncated, residualizer))
                if condition != "independent" and sub_ranking.rank_of(run.colluding_pair) == 1:
                    sample_complexity[condition][rounds] += 1
            if seed == 0:
                heatmap_rows[condition] = _heatmap_matrix(run, ranking)
                for score in ranking.scores:
                    per_seed_rows.append(
                        {
                            "condition": condition,
                            "seed": seed,
                            "pair": f"{score.agent_a}|{score.agent_b}",
                            "net_influence": round(score.net_influence, 6),
                            "compression_coupling": round(score.compression_coupling, 6),
                            "hybrid_score": round(score.hybrid_score, 6),
                        }
                    )

    # Figure 1: 4x4 NI adjacency matrices per condition.
    fig, axes = plt.subplots(1, len(CONDITIONS), figsize=(5 * len(CONDITIONS), 4.4))
    for ax, condition in zip(np.atleast_1d(axes), CONDITIONS):
        matrix = heatmap_rows[condition]
        im = ax.imshow(matrix, vmin=0.0, vmax=max(0.25, _global_max(heatmap_rows)), cmap="viridis")
        ax.set_xticks(range(4), [f"A{i + 1}" for i in range(4)])
        ax.set_yticks(range(4), [f"A{i + 1}" for i in range(4)])
        ax.set_title(CONDITION_TITLES[condition])
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Figure 1: Pairwise Net Influence adjacency (agent 1-2 block = planted cartel)")
    save_figure(output_dir, "fig1_net_influence_heatmap")

    # Figure 3: sample complexity curve.
    plt.figure(figsize=(6, 4))
    for condition in ("grim_trigger", "bid_rotation"):
        xs = list(SAMPLE_COMPLEXITY_ROUNDS)
        ys = [100.0 * sample_complexity[condition][t] / seeds for t in xs]
        plt.plot(xs, ys, marker="o", label=CONDITION_TITLES[condition])
    plt.axhline(100.0, color="grey", linestyle=":", linewidth=1)
    plt.xlabel("Observed rounds T")
    plt.ylabel("Top-1 pair identification accuracy (%)")
    plt.title("Figure 3: Sample complexity of pair identification")
    plt.legend()
    plt.grid(alpha=0.3)
    save_figure(output_dir, "fig3_sample_complexity")

    summary = {
        "top1_accuracy": {
            condition: {
                "hits": top1_hits[condition],
                "runs": seeds,
                "rate": top1_hits[condition] / seeds,
            }
            for condition in ("grim_trigger", "bid_rotation")
        },
        "sample_complexity": {
            condition: {str(t): sample_complexity[condition][t] / seeds for t in SAMPLE_COMPLEXITY_ROUNDS}
            for condition in ("grim_trigger", "bid_rotation")
        },
    }
    _write_csv(output_dir / "spec01_pair_scores_seed0.csv", per_seed_rows)
    return summary


def _heatmap_matrix(run: RunRecord, ranking: CoalitionRanking) -> list[list[float]]:
    agents = run.agent_ids
    size = len(agents)
    matrix = [[0.0] * size for _ in range(size)]
    for score in ranking.scores:
        i, j = agents.index(score.agent_a), agents.index(score.agent_b)
        matrix[i][j] = matrix[j][i] = score.net_influence
    return matrix


def _global_max(heatmap_rows: dict[str, dict[str, float]]) -> float:
    return max(
        value
        for matrix in heatmap_rows.values()
        for row in matrix.values()
        for value in row
    )


# --------------------------------------------------------------------------- #
# Phase 2: cached-log audit
# --------------------------------------------------------------------------- #
def run_cached_phase(ranker: CoalitionRankerService, output_dir: Path, max_files: int) -> dict:
    residualizer = ResidualizerService(num_bins=5)
    rows: list[dict] = []
    runs_scanned = files_scanned = 0

    if COLLUDEBENCH_RESULTS.is_dir():
        jsonl_files = sorted(COLLUDEBENCH_RESULTS.rglob("*.jsonl"))
        jsonl_files = [p for p in jsonl_files if not p.name.endswith("_interventions.jsonl")]
        for file_path in jsonl_files[:max_files]:
            files_scanned += 1
            for run in load_colludebench_jsonl(file_path):
                _score_run(rows, run, ranker, residualizer)
                runs_scanned += 1
    else:
        print(f"[spec01] ColludeBench results not found at {COLLUDEBENCH_RESULTS}; skipping.")

    if DOUBLE_AUCTION_DATA.is_dir():
        for run in load_double_auction_root(DOUBLE_AUCTION_DATA):
            _score_run(rows, run, ranker, residualizer)
            runs_scanned += 1
    else:
        print(f"[spec01] Double-auction data not found at {DOUBLE_AUCTION_DATA}; skipping.")

    if not rows:
        print("[spec01] No cached runs found - cached phase produced no output.")
        return {"runs": 0}

    collusive = [r["net_influence"] for r in rows if r["is_collusive"]]
    honest = [r["net_influence"] for r in rows if not r["is_collusive"]]
    coupling_collusive = [r["compression_coupling"] for r in rows if r["is_collusive"]]
    coupling_honest = [r["compression_coupling"] for r in rows if not r["is_collusive"]]
    correlation_collusive = [r["naive_correlation"] for r in rows if r["is_collusive"]]
    correlation_honest = [r["naive_correlation"] for r in rows if not r["is_collusive"]]
    elevation_collusive = [r["price_elevation"] for r in rows if r["is_collusive"]]
    elevation_honest = [r["price_elevation"] for r in rows if not r["is_collusive"]]

    from src.common.metrics import auroc, mann_whitney_p_value

    summary: dict = {"runs": len(rows), "files": files_scanned}
    screens = {
        "net_influence": (collusive, honest),
        "compression_coupling": (coupling_collusive, coupling_honest),
        "hybrid": ([r["hybrid_score"] for r in rows if r["is_collusive"]],
                   [r["hybrid_score"] for r in rows if not r["is_collusive"]]),
        "naive_correlation": (correlation_collusive, correlation_honest),
        "price_elevation": (elevation_collusive, elevation_honest),
    }
    for name, (pos, neg) in screens.items():
        if pos and neg:
            summary.setdefault("auroc", {})[name] = round(auroc(pos, neg), 4)
            summary.setdefault("p_value", {})[name] = mann_whitney_p_value(pos, neg)

    plt = get_pyplot()
    plt.figure(figsize=(6.4, 5))
    for name, (pos, neg) in screens.items():
        if pos and neg:
            fpr, tpr = _roc_curve(pos, neg)
            plt.plot(fpr, tpr, label=f"{name} (AUROC {summary['auroc'][name]:.3f})")
    plt.plot([0, 1], [0, 1], color="grey", linestyle=":", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Figure 2: Cached-log detection performance (ColludeBench + double auctions)")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    save_figure(output_dir, "fig2_auroc_curves")

    _write_csv(output_dir / "spec01_cached_run_scores.csv", rows)
    return summary


def _score_run(
    rows: list[dict],
    run: RunRecord,
    ranker: CoalitionRankerService,
    residualizer: ResidualizerService,
) -> None:
    if len(run.agent_ids) < 2:
        return  # single-agent runs carry no pair signal
    ranking = ranker.audit_run(discretize(run, residualizer))
    best = max(ranking.scores, key=lambda s: s.hybrid_score)
    rows.append(
        {
            "run_id": run.run_id,
            "source": run.source,
            "is_collusive": run.is_collusive,
            "net_influence": round(best.net_influence, 6),
            "compression_coupling": round(best.compression_coupling, 6),
            "hybrid_score": round(best.hybrid_score, 6),
            "naive_correlation": round(naive_price_correlation(run), 6),
            "price_elevation": round(price_elevation_screen(run), 6),
            "n_agents": len(run.agent_ids),
            "n_rounds": len({t.round_id for t in run.turns}),
        }
    )


def _roc_curve(positive: list[float], negative: list[float]) -> tuple[list[float], list[float]]:
    """Threshold sweep over the combined score distribution."""
    scores = sorted({*positive, *negative}, reverse=True)
    thresholds = [scores[0] + 1.0] + list(scores) + [scores[-1] - 1.0]
    fpr, tpr = [], []
    for threshold in thresholds:
        tp = sum(p >= threshold for p in positive)
        fp = sum(n >= threshold for n in negative)
        tpr.append(tp / len(positive))
        fpr.append(fp / len(negative))
    return fpr, tpr


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("control", "cached", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/spec01"))
    parser.add_argument("--seeds", type=int, default=20, help="scripted runs per condition")
    parser.add_argument("--max-files", type=int, default=400, help="ColludeBench JSONL cap")
    parser.add_argument("--no-messages", action="store_true", help="exclude message text from NCD")
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ranker = CoalitionRankerService(use_messages=not args.no_messages)

    summary: dict = {}
    if args.phase in ("control", "all"):
        summary["control"] = run_control_phase(ranker, output_dir, args.seeds)
        for condition, stats in summary["control"]["top1_accuracy"].items():
            print(
                f"[spec01] {condition}: top-1 identification {stats['hits']}/{stats['runs']}"
                f" ({stats['rate']:.0%})"
            )
    if args.phase in ("cached", "all"):
        summary["cached"] = run_cached_phase(ranker, output_dir, args.max_files)
        print(f"[spec01] cached audit: {summary['cached'].get('runs', 0)} runs scored")
        for name, value in summary["cached"].get("auroc", {}).items():
            print(f"[spec01]   AUROC {name}: {value}")

    with open(output_dir / "spec01_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[spec01] outputs written to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
