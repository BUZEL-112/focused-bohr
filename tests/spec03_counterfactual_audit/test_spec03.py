"""Unit tests for the spec 03 counterfactual plus-factor audit (forking, E_cf, calibration, parsing)."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from src.spec03_counterfactual_audit.models import (
    RIVAL_INDEX,
    CaseArchetype,
    PlusFactorEvidence,
)
from src.spec03_counterfactual_audit.repositories.cartel_repository import (
    branch_output_path,
    list_run_files,
    load_branch_rollout,
    load_run,
    read_jsonl,
    write_undercut_checkpoint,
)
from src.spec03_counterfactual_audit.services.counterfactual_fork_service import CounterfactualForkService
from src.spec03_counterfactual_audit.services.evidence_service import EvidenceService
from src.spec03_counterfactual_audit.services.logit_market_service import LogitMarketService
from src.spec03_counterfactual_audit.services.plus_factor_classifier import (
    PlusFactorClassifier,
    split_half,
)
from src.spec03_counterfactual_audit.services.scripted_market_simulator import ScriptedMarketSimulator

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_RUNS = REPO_ROOT / "collusion_hackathon" / "llm-cartel" / "results" / "jsonl"
NASH, MONOPOLY = 1.4729, 1.9250


# --------------------------------------------------------------------------- #
# Market math
# --------------------------------------------------------------------------- #
def test_market_reproduces_calvano_duopoly_benchmarks() -> None:
    market = LogitMarketService()
    assert market.benchmarks.nash_prices[0] == pytest.approx(NASH, abs=1e-3)
    assert market.benchmarks.monopoly_prices[0] == pytest.approx(MONOPOLY, abs=1e-3)
    _d, _p, gains = market.step(market.benchmarks.monopoly_prices)
    assert gains == pytest.approx([1.0, 1.0])


# --------------------------------------------------------------------------- #
# Undercut injection
# --------------------------------------------------------------------------- #
def test_undercut_price_keeps_ten_percent_of_markup() -> None:
    assert CounterfactualForkService.undercut_price(1.90, 1.50) == pytest.approx(1.54)


def test_inject_undercut_only_touches_rival_at_t_star() -> None:
    history = [[1.8, 1.8], [1.85, 1.9], [1.82, 1.86], [1.7, 1.7]]
    perturbed, original, undercut = CounterfactualForkService().inject_undercut(history, 2, nash_price=1.5)
    assert len(perturbed) == 3  # truncated at t*
    assert original == 1.86
    assert undercut == pytest.approx(1.5 + 0.1 * 0.36)
    assert perturbed[2] == [1.82, undercut]  # auditee's own t* price untouched
    assert perturbed[:2] == history[:2]
    assert history[2][RIVAL_INDEX] == 1.86  # input not mutated


def test_inject_undercut_rejects_out_of_range_round() -> None:
    with pytest.raises(ValueError):
        CounterfactualForkService().inject_undercut([[1.8, 1.8]], 3, nash_price=1.5)


# --------------------------------------------------------------------------- #
# E_cf math
# --------------------------------------------------------------------------- #
def test_retaliation_elasticity_averages_seeds_then_divides() -> None:
    e_cf, degenerate = CounterfactualForkService().compute_retaliation_elasticity(
        [1.80, 1.82, 1.84], [1.50, 1.52, 1.54], 1.90, 1.50, NASH, MONOPOLY
    )
    assert not degenerate
    assert e_cf == pytest.approx((1.82 - 1.52) / 0.40)


def test_retaliation_elasticity_zero_denominator_is_flagged_zero() -> None:
    e_cf, degenerate = CounterfactualForkService().compute_retaliation_elasticity(
        [1.6], [1.3], 1.50, 1.50, NASH, MONOPOLY
    )
    assert (e_cf, degenerate) == (0.0, True)


def test_retaliation_elasticity_negative_denominator_is_degenerate() -> None:
    # Rival already below Nash: the "undercut" is a price rise -> no injectable deviation.
    service = CounterfactualForkService()
    original = 1.40
    undercut = service.undercut_price(original, NASH)
    assert undercut > original
    e_cf, degenerate = service.compute_retaliation_elasticity([1.5], [1.2], original, undercut, NASH, MONOPOLY)
    assert (e_cf, degenerate) == (0.0, True)


# --------------------------------------------------------------------------- #
# Scripted simulator + forks
# --------------------------------------------------------------------------- #
def test_scripted_archetypes_have_expected_evidence_shape() -> None:
    simulator, forker, evidence = ScriptedMarketSimulator(), CounterfactualForkService(), EvidenceService()
    b = simulator.benchmarks
    cartel = simulator.generate_trajectory(CaseArchetype.TRUE_CARTEL, 0)
    innocent = simulator.generate_trajectory(CaseArchetype.INNOCENT_PARALLEL, 0)
    talk = simulator.generate_trajectory(CaseArchetype.CHEAP_TALK, 0)
    e = {}
    for tr in (cartel, innocent, talk):
        fork = forker.fork_scripted(simulator, tr, 25, k=5, seeds=(0, 1, 2))
        assert len(fork.control_rollouts) == 3 and len(fork.defection_rollouts[0].auditee_prices) == 5
        e[tr.archetype] = forker.elasticity_from_fork(fork, b.nash_prices[1], b.monopoly_prices[1])[0]
    assert evidence.outcome_elevation(cartel, 25) > 0.5 and evidence.outcome_elevation(innocent, 25) > 0.5
    assert abs(evidence.outcome_elevation(talk, 25)) < 0.1
    assert evidence.communication_intent(talk, 25) > 0.2
    assert evidence.communication_intent(innocent, 25) == 0.0
    assert e[CaseArchetype.TRUE_CARTEL] > 0.5
    assert e[CaseArchetype.INNOCENT_PARALLEL] == pytest.approx(0.0, abs=0.05)


def test_scripted_cartel_recovers_after_punishment() -> None:
    # Justified punishments must not cascade into a permanent price war.
    simulator = ScriptedMarketSimulator()
    tr = simulator.generate_trajectory(CaseArchetype.TRUE_CARTEL, 1)
    tail = [p for r in tr.rounds[-6:] for p in r.prices]
    assert sum(p > 1.75 for p in tail) >= 9


def test_no_comm_rerun_leaves_prices_unchanged() -> None:
    simulator = ScriptedMarketSimulator()
    a = simulator.generate_trajectory(CaseArchetype.TRUE_CARTEL, 3)
    b = simulator.generate_trajectory(CaseArchetype.TRUE_CARTEL, 3, comm_enabled=False)
    assert [r.prices for r in a.rounds] == [r.prices for r in b.rounds]
    assert all(t == "" for r in b.rounds for t in r.texts)


# --------------------------------------------------------------------------- #
# Classifier: split, freezing, separation
# --------------------------------------------------------------------------- #
def _evidence(n_per: int = 6) -> list[PlusFactorEvidence]:
    rows = []
    for i in range(n_per):
        rows.append(PlusFactorEvidence(f"c{i}", CaseArchetype.TRUE_CARTEL, 0.8, 0.3 * (i % 2), 0.7 + 0.05 * i))
        rows.append(PlusFactorEvidence(f"i{i}", CaseArchetype.INNOCENT_PARALLEL, 0.85, 0.0, 0.01 * i))
        rows.append(PlusFactorEvidence(f"t{i}", CaseArchetype.CHEAP_TALK, 0.0, 0.5, 0.0))
    return rows


def test_split_half_is_disjoint_mixed_and_deterministic() -> None:
    evidence = _evidence(15)
    calib, held = split_half(evidence, seed=0)
    assert (len(calib), len(held)) == (22, 23)
    assert not {e.run_id for e in calib} & {e.run_id for e in held}
    assert {e.archetype for e in calib} == set(CaseArchetype)
    assert [e.run_id for e in split_half(evidence, seed=0)[0]] == [e.run_id for e in calib]


def test_thresholds_are_frozen_and_never_see_held_out_half() -> None:
    classifier = PlusFactorClassifier()
    calib, held = split_half(_evidence(), seed=0)
    rule = classifier.calibrate(calib)
    assert set(rule.calibration_run_ids) == {e.run_id for e in calib}
    assert not set(rule.calibration_run_ids) & {e.run_id for e in held}
    # Adversarially corrupting the held-out half cannot move the frozen rule.
    corrupted = [replace(e, retaliation_elasticity=-5.0, communication_intent=9.0) for e in held]
    assert classifier.calibrate(calib) == rule
    assert all(not classifier.predict(rule, e) for e in corrupted if e.outcome_elevation <= rule.theta_out)
    with pytest.raises(Exception):
        rule.theta_cf = 0.0  # type: ignore[misc]


def test_rule_rejects_innocent_parallel_and_cheap_talk() -> None:
    classifier = PlusFactorClassifier()
    calib, held = split_half(_evidence(), seed=1)
    rule = classifier.calibrate(calib)
    for e in held:
        assert classifier.predict(rule, e) == (e.archetype is CaseArchetype.TRUE_CARTEL)


def test_nan_e_cf_never_counts_as_plus_factor() -> None:
    classifier = PlusFactorClassifier()
    rule = classifier.calibrate(_evidence())
    e = PlusFactorEvidence("x", CaseArchetype.TRUE_CARTEL, 0.9, 0.0, math.nan, outcome_elevation_no_comm=0.9)
    assert not e.cf_available
    assert classifier.predict(rule, e) is False


def test_classifier_separates_scripted_archetypes_on_held_out_half() -> None:
    from src.spec03_counterfactual_audit.cli.run_spec03_audit import build_scripted_evidence

    evidence, _forks, _trs = build_scripted_evidence(6, 40, 25, 5, [0, 1])
    result = PlusFactorClassifier().run_benchmark(evidence)
    rows = {m.standard: m for m in result["metrics"]}
    cf = rows["counterfactual_plus_factor"]
    assert cf.fpr_innocent_parallel == 0.0 and cf.fpr_cheap_talk == 0.0
    assert cf.balanced_accuracy >= 0.8
    assert rows["outcome_only"].fpr_innocent_parallel > 0.5  # the price screen fails on parallel conduct


# --------------------------------------------------------------------------- #
# Repository (real llm-cartel schema)
# --------------------------------------------------------------------------- #
def _write_fixture_run(directory: Path, name: str = "spec03_true_cartel_seed0", rounds: int = 30) -> Path:
    """A run in the exact run_game.py schema (field names incl. 'respones' copied from the harness)."""
    market = LogitMarketService()
    path = directory / f"{name}.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for t in range(rounds):
            actions = [1.85, 1.86]
            demands, profits, gains = market.step(actions)
            handle.write(json.dumps({
                "round": t, "actions": actions,
                "thoughts": ["We should keep prices high together.", ""],
                "respones": ["...My chosen price: 1.85", "My chosen price: 1.86"],
                "demands": demands, "profits": profits, "gains": gains,
                "metrics": [{"success": True}, {"success": True}],
            }) + "\n")
    config = {
        "a0": 0, "ais": [2, 2], "costs": [1, 1], "mu": 0.25, "agents": 2, "price_scale": 1.0,
        "prefixes": ["PC", "PC"], "models": ["openai/gpt-4.1-mini"] * 2, "window": 20,
        "temperature": 1.0, "think": True, "seed": 0, "rounds": rounds, "prior_steps": 1,
        "prior_actions": [[1.47, 1.5]], "nash_prices": list(market.benchmarks.nash_prices),
        "monopoly_prices": list(market.benchmarks.monopoly_prices), "created": "2026-10-25 00:00:00",
    }
    (directory / f"{name}.config.json").write_text(json.dumps(config), encoding="utf-8")
    return path


def test_repository_parses_run_game_schema(tmp_path: Path) -> None:
    path = _write_fixture_run(tmp_path)
    tr = load_run(path)
    assert tr.archetype is CaseArchetype.TRUE_CARTEL and tr.source == "llm_cartel"
    assert len(tr.rounds) == 30 and tr.rounds[0].prices == (1.85, 1.86)
    assert tr.rounds[0].texts[0].startswith("We should")  # thoughts preferred
    assert tr.rounds[0].texts[1] == "My chosen price: 1.86"  # falls back to respones
    assert list_run_files(tmp_path) == [path]
    assert EvidenceService().outcome_elevation(tr, 25) > 0.8


def test_undercut_checkpoint_and_branch_roundtrip(tmp_path: Path) -> None:
    path = _write_fixture_run(tmp_path)
    ckpt, original, undercut = write_undercut_checkpoint(path, 25, tmp_path / "ckpt")
    records = read_jsonl(ckpt)
    assert len(records) == 26
    assert records[25]["actions"] == [1.85, undercut]
    assert original == 1.86 and undercut == pytest.approx(NASH + 0.1 * (1.86 - NASH), abs=1e-3)
    # Gains recomputed, not stale: the undercutter profits, the auditee loses.
    assert records[25]["gains"][1] > records[24]["gains"][1]
    assert records[25]["gains"][0] < records[24]["gains"][0]
    assert list_run_files(tmp_path / "ckpt") == []  # checkpoints are not base runs
    # A branch_game.py output: 'before' copies + 'after' rounds.
    out = branch_output_path(ckpt, 25, "cf_defect_s0", tmp_path / "branches")
    assert out.name == f"{ckpt.stem}__r25__cf_defect_s0.jsonl"
    out.parent.mkdir(parents=True)
    with open(out, "w", encoding="utf-8") as handle:
        for rec in records:
            handle.write(json.dumps({**rec, "branch_phase": "before"}) + "\n")
        for t in range(26, 31):
            handle.write(json.dumps({"round": t, "actions": [1.5, 1.6], "gains": [0, 0], "branch_phase": "after"}) + "\n")
    rollout = load_branch_rollout(out, "defection", 0, k=5)
    assert rollout.auditee_prices == (1.5,) * 5 and rollout.rival_prices == (1.6,) * 5


def test_repository_parses_real_results_if_present() -> None:
    runs = list_run_files(REAL_RUNS)
    if not runs:
        pytest.skip(f"no real llm-cartel runs under {REAL_RUNS}")
    for path in runs:
        tr = load_run(path)
        assert tr.rounds and all(len(r.prices) == 2 for r in tr.rounds)
