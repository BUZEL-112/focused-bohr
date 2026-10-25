"""Unit tests for the spec 01 detector (residualizer, net influence, NCD, ranker, parsers)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.common.metrics import auroc, mann_whitney_p_value
from src.spec01_net_influence.models import (
    AgentTurnRecord,
    CoalitionRanking,
    DiscretizedTrajectory,
    PairwiseAuditScore,
)
from src.spec01_net_influence.repositories.trajectory_repository import (
    generate_scripted_run,
    load_colludebench_jsonl,
    load_double_auction_dir,
)
from src.spec01_net_influence.services.audit_statistics_service import (
    fast_auroc,
    mann_whitney_p,
    roc_curve,
    wilson_interval,
)
from src.spec01_net_influence.services.coalition_ranker_service import CoalitionRankerService
from src.spec01_net_influence.services.compression_service import CompressionDistanceService
from src.spec01_net_influence.services.net_influence_service import NetInfluenceService
from src.spec01_net_influence.services.residualizer_service import ResidualizerService


# --------------------------------------------------------------------------- #
# Residualizer
# --------------------------------------------------------------------------- #
def test_residualizer_rejects_tiny_bin_counts() -> None:
    with pytest.raises(ValueError):
        ResidualizerService(num_bins=1)


def test_residualizer_bins_recover_scripted_levels() -> None:
    run = generate_scripted_run("bid_rotation", rounds=12, seed=3)
    trajectory = ResidualizerService(num_bins=5).discretize_turns(
        run.run_id, run.turns, run.is_collusive, run.colluding_pair
    )
    # Residuals sit exactly on the {-4, -2, 0, 2, 4} grid, so binning must be lossless:
    # the rotation pattern of agents 1/2 alternates between bin 0 and bin 4.
    stream_1 = trajectory.symbols_by_agent["agent_1"]
    assert len(stream_1) == 12
    assert {0, 4}.issubset(set(stream_1))


def test_residualizer_constant_residuals_maps_to_center_bin() -> None:
    turns = tuple(
        AgentTurnRecord(round_id=t, agent_id="a", action_value=100.0, market_benchmark=100.0)
        for t in range(5)
    )
    trajectory = ResidualizerService(num_bins=5).discretize_turns("r", turns, False)
    assert set(trajectory.symbols_by_agent["a"]) == {2}  # neutral bin


# --------------------------------------------------------------------------- #
# Net influence
# --------------------------------------------------------------------------- #
def test_net_influence_near_zero_for_independent_streams() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 5, size=400).tolist()
    b = rng.integers(0, 5, size=400).tolist()
    service = NetInfluenceService()
    _a_to_b, _b_to_a, net = service.compute_pairwise_net_influence(a, b)
    assert net < 0.02


def test_net_influence_detects_coupled_streams() -> None:
    rng = np.random.default_rng(1)
    source = rng.integers(0, 5, size=400).tolist()
    # Target copies the source's previous symbol 70% of the time.
    target = [int(rng.integers(0, 5)) if rng.random() < 0.3 else source[t - 1] for t in range(400)]
    service = NetInfluenceService()
    a_to_b, b_to_a, net = service.compute_pairwise_net_influence(source, target)
    assert a_to_b > 0.5
    assert b_to_a < a_to_b
    assert net == pytest.approx(b_to_a)


def test_net_influence_uses_context_conditioning() -> None:
    """A common driver must not inflate NI once the driver is conditioned on.

    ``a`` reacts to the driver immediately and ``b`` one round later, so a_{t-1} and b_t
    share driver_{t-1}: a lagged dependence that the driver alone explains.
    """
    rng = np.random.default_rng(2)
    driver = rng.integers(0, 5, size=500).tolist()
    a = [int(np.clip(d + rng.integers(-1, 2), 0, 4)) for d in driver]
    b = [int(rng.integers(0, 5))] + [
        int(np.clip(d + rng.integers(-1, 2), 0, 4)) for d in driver[:-1]
    ]
    service = NetInfluenceService()
    without_context = service.compute_directed_information(a, b)
    with_context = service.compute_directed_information(a, b, [driver])
    assert with_context < without_context


def test_net_influence_requires_equal_lengths() -> None:
    service = NetInfluenceService()
    with pytest.raises(ValueError):
        service.compute_pairwise_net_influence([0, 1], [0, 1, 2])


# --------------------------------------------------------------------------- #
# Compression distance
# --------------------------------------------------------------------------- #
def test_ncd_identical_streams_higher_coupling_than_random() -> None:
    service = CompressionDistanceService()
    coupled = [0, 1, 2, 3, 4] * 40
    rng = np.random.default_rng(3)
    random_stream = rng.integers(0, 5, size=200).tolist()
    assert service.compute_coupling_score(coupled, coupled) > service.compute_coupling_score(
        random_stream, list(rng.integers(0, 5, size=200))
    )


def test_ncd_message_payload_contributes() -> None:
    service = CompressionDistanceService()
    shared_secret = ["alpha-beta-gamma"] * 30
    unrelated = [f"note-{i}" for i in range(30)]
    symbols = [1, 2] * 15
    coupled = service.compute_coupling_score(symbols, symbols, shared_secret, shared_secret)
    independent = service.compute_coupling_score(symbols, symbols, shared_secret, unrelated)
    assert coupled > independent


def test_compressor_choice_validated() -> None:
    with pytest.raises(ValueError):
        CompressionDistanceService(compressor="gzip")
    assert CompressionDistanceService(compressor="lzma").compute_coupling_score([0, 1], [0, 1]) >= 0.0


# --------------------------------------------------------------------------- #
# Coalition ranker on scripted positive controls
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("condition", ["grim_trigger", "bid_rotation"])
def test_ranker_flags_true_pair_top1(condition: str) -> None:
    ranker = CoalitionRankerService()
    residualizer = ResidualizerService(num_bins=5)
    for seed in range(3):
        run = generate_scripted_run(condition, rounds=30, seed=seed)
        ranking = ranker.audit_run(
            residualizer.discretize_turns(run.run_id, run.turns, run.is_collusive, run.colluding_pair)
        )
        assert ranking.rank_of(("agent_1", "agent_2")) == 1


def test_ranker_independent_condition_has_no_dominant_pair() -> None:
    ranker = CoalitionRankerService()
    residualizer = ResidualizerService(num_bins=5)
    run = generate_scripted_run("independent", rounds=30, seed=11)
    ranking = ranker.audit_run(
        residualizer.discretize_turns(run.run_id, run.turns, run.is_collusive, run.colluding_pair)
    )
    # No pair should stand out: all hybrid scores of a 6-pair graph are z-scores, so the
    # max is bounded; the planted-pair check is that NI stays small for every pair.
    assert max(score.net_influence for score in ranking.scores) < 0.15


# --------------------------------------------------------------------------- #
# Repository parsers
# --------------------------------------------------------------------------- #
def _write_colludebench_file(tmp_path: Path, runs: list[dict[str, list[dict]]]) -> Path:
    path = tmp_path / "audit" / "pricing_duopoly" / "transcript.jsonl"
    path.parent.mkdir(parents=True)
    with open(path, "w", encoding="utf-8") as handle:
        for run in runs:
            for line in run:
                handle.write(json.dumps(line) + "\n")
    return path


def test_colludebench_parser_splits_appended_runs(tmp_path: Path) -> None:
    run_a = [
        {"run_id": "x", "round_id": 1, "agent_id": "firm_a", "action": "10.0", "message": "m", "reward": 0.1},
        {"run_id": "x", "round_id": 1, "agent_id": "firm_b", "action": "11.0", "message": "m", "reward": 0.1},
        {"run_id": "x", "round_id": 2, "agent_id": "firm_a", "action": "12.0", "message": "m", "reward": 0.2},
        {"run_id": "x", "round_id": 2, "agent_id": "firm_b", "action": "12.0", "message": "m", "reward": 0.2},
    ]
    run_b = [
        {"run_id": "x", "round_id": 1, "agent_id": "firm_a", "action": "9.0", "message": "m", "reward": 0.0},
        {"run_id": "x", "round_id": 1, "agent_id": "firm_b", "action": "10.0", "message": "m", "reward": 0.0},
    ]
    path = _write_colludebench_file(tmp_path, [run_a, run_b])
    runs = load_colludebench_jsonl(path)
    assert len(runs) == 2
    assert all(run.is_collusive for run in runs)  # under audit/ tree
    assert all({t.agent_id for t in run.turns} == {"firm_a", "firm_b"} for run in runs)


def test_colludebench_calibration_tree_labelled_honest(tmp_path: Path) -> None:
    path = tmp_path / "calibration" / "pricing_duopoly" / "t.jsonl"
    path.parent.mkdir(parents=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {"run_id": "c", "round_id": 1, "agent_id": "firm_a", "action": "9.0", "message": "", "reward": 0.0}
            )
            + "\n"
        )
        handle.write(
            json.dumps(
                {"run_id": "c", "round_id": 1, "agent_id": "firm_b", "action": "10.0", "message": "", "reward": 0.0}
            )
            + "\n"
        )
    runs = load_colludebench_jsonl(path)
    assert runs and not runs[0].is_collusive


def test_double_auction_dir_parser(tmp_path: Path) -> None:
    exp_dir = tmp_path / "20250516_212322_base-seller_comms-1"
    exp_dir.mkdir()
    block = (
        "## Auction Results: Round 1\n\n```json\n"
        + json.dumps(
            {
                "round_number": 1,
                "seller_asks": {"Seller Lena": 97.5, "Seller Marc": 96.0},
                "buyer_bids": {"Buyer Eli": 82.0},
                "trades": [],
                "seller_messages": {"Seller Lena": "hold firm"},
                "buyer_messages": {},
            }
        )
        + "\n```\n"
    )
    (exp_dir / "auction_results.md").write_text(block, encoding="utf-8")
    (exp_dir / "collusion_metrics.json").write_text(
        json.dumps({"competitive_equilibrium_price": 90.0}), encoding="utf-8"
    )
    run = load_double_auction_dir(exp_dir)
    assert run is not None
    assert run.is_collusive
    assert len(run.turns) == 2
    assert all(t.market_benchmark == 90.0 for t in run.turns)


def test_scripted_run_ground_truth() -> None:
    independent = generate_scripted_run("independent", rounds=20, seed=0)
    cartel = generate_scripted_run("grim_trigger", rounds=20, seed=0)
    assert not independent.is_collusive and independent.colluding_pair is None
    assert cartel.is_collusive and cartel.colluding_pair == ("agent_1", "agent_2")
    assert len(independent.turns) == 20 * 4


# --------------------------------------------------------------------------- #
# Spec conditioning and surrogate debiasing
# --------------------------------------------------------------------------- #
def test_spec_context_conditions_on_target_past() -> None:
    """Source == target: the spec context (own past) explains everything; compound does not."""
    rng = np.random.default_rng(4)
    target = [int(rng.integers(0, 5))]
    for _ in range(299):
        target.append(target[-1] if rng.random() < 0.7 else int(rng.integers(0, 5)))
    spec = NetInfluenceService(context_mode="spec").compute_directed_information(target, target)
    compound = NetInfluenceService(context_mode="compound").compute_directed_information(target, target)
    assert spec == pytest.approx(0.0, abs=1e-9)
    assert compound > 0.3


def test_context_mode_validated() -> None:
    with pytest.raises(ValueError):
        NetInfluenceService(context_mode="full")
    with pytest.raises(ValueError):
        NetInfluenceService(surrogates=-1)


def test_surrogate_shifts_are_deterministic_and_in_range() -> None:
    service = NetInfluenceService(surrogates=19)
    shifts = service._surrogate_shifts(30)
    assert shifts == service._surrogate_shifts(30)
    assert shifts and all(1 <= shift <= 29 for shift in shifts)
    assert service._surrogate_shifts(3) == []


def test_effective_equals_raw_without_surrogates() -> None:
    rng = np.random.default_rng(5)
    a = rng.integers(0, 5, size=60).tolist()
    b = rng.integers(0, 5, size=60).tolist()
    raw, effective = NetInfluenceService(surrogates=0).compute_effective_directed_information(a, b)
    assert raw == effective


def test_ranker_reports_raw_and_actions_only_fields() -> None:
    run = generate_scripted_run("bid_rotation", rounds=12, seed=0)
    trajectory = ResidualizerService(num_bins=5).discretize_turns(
        run.run_id, run.turns, run.is_collusive, run.colluding_pair
    )
    ranking = CoalitionRankerService().audit_run(trajectory)
    assert len(ranking.scores) == 6
    for score in ranking.scores:
        assert score.net_influence_raw >= 0.0
        assert 0.0 <= score.compression_coupling_actions_only <= 1.0


# --------------------------------------------------------------------------- #
# Benchmark statistics
# --------------------------------------------------------------------------- #
def test_fast_auroc_and_p_value_match_reference() -> None:
    rng = np.random.default_rng(6)
    pos = rng.integers(0, 8, size=40).astype(float).tolist()  # ties on purpose
    neg = rng.integers(0, 6, size=35).astype(float).tolist()
    assert fast_auroc(pos, neg) == pytest.approx(auroc(pos, neg), abs=1e-12)
    assert mann_whitney_p(pos, neg) == pytest.approx(mann_whitney_p_value(pos, neg), rel=1e-9)
    assert np.isnan(fast_auroc([], neg))


def test_wilson_interval_bounds() -> None:
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12) and 0.2 < high < 0.35
    low, high = wilson_interval(10, 10)
    assert high == pytest.approx(1.0, abs=1e-12) and 0.65 < low < 0.8
    assert all(np.isnan(v) for v in wilson_interval(0, 0))


def test_roc_curve_endpoints() -> None:
    fpr, tpr = roc_curve([0.9, 0.8, 0.4], [0.5, 0.1])
    assert (fpr[0], tpr[0]) == (0.0, 0.0)
    assert (fpr[-1], tpr[-1]) == (1.0, 1.0)


# --------------------------------------------------------------------------- #
# End-to-end CLI smoke tests (tiny settings; exercise every writer and figure)
# --------------------------------------------------------------------------- #
def test_cli_control_phase_smoke(tmp_path: Path) -> None:
    from src.spec01_net_influence.cli.run_spec01_audit import main

    out = tmp_path / "out"
    code = main(["--phase", "control", "--seeds", "2", "--rounds", "10", "--bootstrap", "20",
                 "--output-dir", str(out)])
    assert code == 0
    for name in (
        "spec01_control_pair_scores.csv", "spec01_control_runs.csv", "spec01_sample_complexity.csv",
        "spec01_control_separation.csv", "spec01_summary.json", "spec01_report.md", "run_metadata.json",
        "fig1_net_influence_heatmap.png", "fig3_sample_complexity.png",
        "fig4_pair_score_distributions.png", "fig5_xu_regime.png",
    ):
        assert (out / name).is_file(), name
    summary = json.loads((out / "spec01_summary.json").read_text(encoding="utf-8"))
    assert set(summary["control"]["top1_accuracy"]) == {"independent", "grim_trigger", "bid_rotation", "assistant"}


def _write_duopoly_jsonl(path: Path, runs: int, rounds: int, high: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(len(str(path)))
    with open(path, "w", encoding="utf-8") as handle:
        for _ in range(runs):
            for round_id in range(1, rounds + 1):
                for agent in ("firm_a", "firm_b"):
                    price = (15.0 if high else 1.1) + float(rng.normal(0.0, 0.2))
                    handle.write(json.dumps({
                        "run_id": "r", "round_id": round_id, "agent_id": agent,
                        "action": f"price:{price:.2f}", "message": "ok", "reward": 0.0,
                    }) + "\n")


def _write_double_auction_dir(root: Path, name: str, asks: list[dict[str, float]]) -> None:
    exp_dir = root / name
    exp_dir.mkdir(parents=True)
    blocks = []
    for round_number, round_asks in enumerate(asks, start=1):
        payload = {"round_number": round_number, "seller_asks": round_asks, "seller_messages": {}}
        blocks.append(f"## Round {round_number}\n\n```json\n{json.dumps(payload)}\n```\n")
    (exp_dir / "auction_results.md").write_text("\n".join(blocks), encoding="utf-8")
    (exp_dir / "collusion_metrics.json").write_text(
        json.dumps({"competitive_equilibrium_price": 90.0}), encoding="utf-8"
    )


def test_cli_cached_phase_smoke(tmp_path: Path) -> None:
    from src.spec01_net_influence.cli.run_spec01_audit import main

    hackathon = tmp_path / "collusion_hackathon"
    results = hackathon / "steganographic-collusion" / "results"
    _write_duopoly_jsonl(results / "audit" / "pricing_duopoly" / "a.jsonl", runs=3, rounds=8, high=True)
    _write_duopoly_jsonl(results / "calibration" / "pricing_duopoly" / "c.jsonl", runs=3, rounds=8, high=False)
    data = hackathon / "llm-agent-collusion" / "data"
    sellers = [{"Seller A": 95.0 + i % 3, "Seller B": 96.0 - i % 2, "Seller C": 94.0 + i % 4} for i in range(6)]
    _write_double_auction_dir(data, "20250516_212322_base-seller_comms-1", sellers)
    _write_double_auction_dir(data, "20250516_211616_base-1", sellers)

    out = tmp_path / "out"
    code = main(["--phase", "cached", "--hackathon-dir", str(hackathon), "--bootstrap", "20",
                 "--output-dir", str(out)])
    assert code == 0
    for name in (
        "spec01_cached_runs.csv", "spec01_cached_pair_scores.csv", "spec01_cached_auroc.csv",
        "spec01_cached_group_summary.csv", "fig2_auroc_curves.png", "fig6_cached_score_distributions.png",
        "spec01_report.md",
    ):
        assert (out / name).is_file(), name
    summary = json.loads((out / "spec01_summary.json").read_text(encoding="utf-8"))
    assert summary["cached"]["runs"] == 3 + 3 + 2
    assert summary["cached"]["class_counts"]["all"] == {"collusive": 4, "honest": 4}


def test_rebenchmark_breaks_duopoly_mirroring(tmp_path: Path) -> None:
    """With the population benchmark, two firms' residuals are no longer exact mirrors."""
    from src.spec01_net_influence.cli.run_spec01_audit import collect_cached_runs

    hackathon = tmp_path / "collusion_hackathon"
    results = hackathon / "steganographic-collusion" / "results"
    _write_duopoly_jsonl(results / "audit" / "pricing_duopoly" / "a.jsonl", runs=1, rounds=6, high=True)
    _write_duopoly_jsonl(results / "calibration" / "pricing_duopoly" / "c.jsonl", runs=2, rounds=6, high=False)
    collected = collect_cached_runs(hackathon, max_files=0)
    audit_run, meta = next((run, meta) for run, meta in collected if run.is_collusive)
    assert meta["benchmark"] == "honest_population_median"
    by_round: dict[int, list[float]] = {}
    for turn in audit_run.turns:
        by_round.setdefault(turn.round_id, []).append(turn.action_value - turn.market_benchmark)
    # Collusive prices (~15) sit far above the honest median (~1.1): both residuals positive.
    assert all(min(values) > 10.0 for values in by_round.values())


# --------------------------------------------------------------------------- #
# Bonjour et al. (2022): same-round influence, eq. 4 subtraction, CP, assistant
# --------------------------------------------------------------------------- #
def test_simultaneous_influence_detects_same_round_copy() -> None:
    """b copies a in the SAME round: the same-round channel sees it, lag-1 does not."""
    rng = np.random.default_rng(7)
    a = rng.integers(0, 5, size=300).tolist()
    b = [value if rng.random() < 0.8 else int(rng.integers(0, 5)) for value in a]
    service = NetInfluenceService()
    _raw, simultaneous = service.compute_simultaneous_influence(a, b)
    _lag_raw, lag = service.compute_effective_directed_information(a, b)
    assert simultaneous > 0.5
    assert lag < 0.1


def test_simultaneous_influence_survives_periodic_schedule() -> None:
    """Regression: a period-2 rotation is invariant under circular shifts, so a shift null
    debiased its 1 bit of coupling to 0. The permutation null must keep it."""
    a = [0, 4] * 15
    b = [4, 0] * 15
    raw, effective = NetInfluenceService().compute_simultaneous_influence(a, b)
    assert raw > 0.9
    assert effective > 0.5


def _trajectory(symbols: dict[str, list[int]]) -> DiscretizedTrajectory:
    length = len(next(iter(symbols.values())))
    return DiscretizedTrajectory(
        run_id="t",
        is_collusive=True,
        colluding_pair=None,
        symbols_by_agent={agent: tuple(values) for agent, values in symbols.items()},
        messages_by_agent={agent: ("",) * length for agent in symbols},
    )


def test_bonjour_subtraction_never_exceeds_individual_influence() -> None:
    rng = np.random.default_rng(8)
    x = rng.integers(0, 5, size=120).tolist()
    trajectory = _trajectory({
        "x": x,
        "y": [v if rng.random() < 0.7 else int(rng.integers(0, 5)) for v in x],
        "z": [v if rng.random() < 0.7 else int(rng.integers(0, 5)) for v in x],
        "w": rng.integers(0, 5, size=120).tolist(),
    })
    with_sub = CoalitionRankerService(bonjour_subtraction=True).audit_run(trajectory)
    without = CoalitionRankerService(bonjour_subtraction=False).audit_run(trajectory)
    for score in with_sub.scores:
        assert score.net_influence_simultaneous <= score.influence_simultaneous + 1e-12
        assert score.net_influence_lag <= score.influence_lag_min + 1e-12
    for score in without.scores:
        assert score.net_influence_simultaneous == pytest.approx(score.influence_simultaneous)
        assert score.net_influence_lag == pytest.approx(score.influence_lag_min)


def test_influence_channel_validated() -> None:
    with pytest.raises(ValueError):
        CoalitionRankerService(channel="both")


def _score(a: str, b: str, net: float, lag: float = 0.0, sim: float = 0.0) -> PairwiseAuditScore:
    return PairwiseAuditScore(
        agent_a=a, agent_b=b, directed_info_a_to_b=0.0, directed_info_b_to_a=0.0,
        net_influence=net, compression_coupling=0.0, influence_lag_min=lag, influence_simultaneous=sim,
    )


def test_flagged_pairs_market_influence_and_bonjour_rule() -> None:
    from src.spec01_net_influence.cli.run_spec01_audit import bonjour_detects

    ranking = CoalitionRanking(
        run_id="r",
        scores=(
            _score("agent_1", "agent_2", net=0.30, lag=0.2, sim=0.4),
            _score("agent_1", "agent_3", net=0.01, lag=0.1, sim=0.0),
            _score("agent_2", "agent_3", net=-0.2, lag=0.0, sim=0.1),
        ),
    )
    assert ranking.flagged_pairs(0.05) == [("agent_1", "agent_2")]
    assert ranking.market_influence == pytest.approx((0.4 + 0.1 + 0.1) / 3)
    assert bonjour_detects(ranking, 0.05)
    assert not bonjour_detects(ranking, 0.005)  # two pairs flagged -> not "exactly one"


def test_collusion_probability_one_matches_default() -> None:
    default = generate_scripted_run("grim_trigger", rounds=25, seed=3)
    explicit = generate_scripted_run("grim_trigger", rounds=25, seed=3, collusion_probability=1.0)
    assert default.turns == explicit.turns and default.run_id == explicit.run_id


def test_collusion_probability_zero_leaves_colluders_honest() -> None:
    run = generate_scripted_run("bid_rotation", rounds=30, seed=1, collusion_probability=0.0)
    levels = {round((t.action_value - t.market_benchmark) / 2.0 + 2) for t in run.turns if t.agent_id == "agent_1"}
    assert levels & {1, 2, 3}  # a rotating colluder would only ever post levels 0 and 4
    assert run.is_collusive and run.run_id.endswith("_cp0")
    with pytest.raises(ValueError):
        generate_scripted_run("bid_rotation", collusion_probability=1.5)


def test_assistant_condition_never_undercuts_primary() -> None:
    run = generate_scripted_run("assistant", rounds=40, seed=2)
    by_round: dict[int, dict[str, int]] = {}
    for turn in run.turns:
        level = round((turn.action_value - turn.market_benchmark) / 2.0 + 2)
        by_round.setdefault(turn.round_id, {})[turn.agent_id] = level
    for levels in by_round.values():
        assert levels["agent_2"] == min(4, levels["agent_1"] + 1)


def test_cli_cp_sweep_smoke(tmp_path: Path) -> None:
    from src.spec01_net_influence.cli.run_spec01_audit import main

    out = tmp_path / "out"
    code = main(["--phase", "cpsweep", "--seeds", "2", "--cp-values", "0.5", "1.0", "--cp-rounds", "6", "10",
                 "--output-dir", str(out)])
    assert code == 0
    for name in (
        "spec01_cp_sweep_runs.csv", "spec01_cp_sweep_cda.csv", "spec01_cp_sweep_swiftness.csv",
        "fig7_cda_vs_rounds.png", "fig8_net_influence_vs_cp.png", "spec01_report.md",
    ):
        assert (out / name).is_file(), name
    summary = json.loads((out / "spec01_summary.json").read_text(encoding="utf-8"))
    assert len(summary["cp_sweep"]["swiftness"]) == 3 * 2  # 3 cartel conditions x 2 CP values
