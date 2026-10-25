"""Unit tests for the spec 01 detector (residualizer, net influence, NCD, ranker, parsers)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.spec01_net_influence.models import AgentTurnRecord
from src.spec01_net_influence.repositories.trajectory_repository import (
    generate_scripted_run,
    load_colludebench_jsonl,
    load_double_auction_dir,
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
