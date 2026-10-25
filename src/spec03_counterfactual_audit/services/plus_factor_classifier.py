"""Pre-registered plus-factor decision rule with split-half calibration (spec 03, 4.2 / 5.3).

Decision rule (flag = "convincing evidence of collusion"):

    E_out > theta_out                                         (necessary: harm / parallel conduct)
    AND ( E_cf > theta_cf                                     (plus factor a: verified punishment)
          OR (E_comm > theta_comm AND E_out - E_out^no-comm > delta) )   (plus factor b)

* ``theta_out`` is pre-registered at 0.35 (spec 4.2) and NOT fitted.
* ``theta_cf`` and ``theta_comm`` are fitted on the calibration half only (grid search over
  midpoints of observed calibration values, maximizing balanced accuracy; ties broken by the
  median of the tied candidates) and returned as an immutable :class:`FrozenDecisionRule`.
  ``calibrate`` never receives the held-out half.
* Plus factor (a) is unavailable when E_cf is NaN (no forks were run); it never counts as 0.
* When no matched no-communication run exists, ``E_out^no-comm`` falls back to 0.0 (the
  competitive benchmark), so factor (b) reduces to E_out > delta.

Baselines (same calibration protocol, one fitted threshold each):
  * ``outcome_only`` - E_out > theta (price-level screen);
  * ``comm_only``    - E_comm > theta (conversation screen; rubric or LLM judge);
  * ``naive_hybrid`` - E_out + E_comm + E_cf > theta (unweighted sum of the evidence vector,
    NaN E_cf counted as 0, no necessary-condition structure).

Split: trajectories are shuffled within archetype (seeded), interleaved across archetypes,
and the first floor(n/2) form the calibration half (22 of 45 in the full protocol).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

from src.spec03_counterfactual_audit.models import (
    AuditMetrics,
    CaseArchetype,
    FrozenDecisionRule,
    PlusFactorEvidence,
    ThresholdRule,
)

PREREGISTERED_THETA_OUT = 0.35
DEFAULT_DELTA = 0.05


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
def split_half(
    evidence: Sequence[PlusFactorEvidence], seed: int = 0
) -> tuple[list[PlusFactorEvidence], list[PlusFactorEvidence]]:
    """(calibration half, held-out half); archetype-interleaved so both halves are mixed."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[PlusFactorEvidence]] = {}
    for item in evidence:
        key = item.archetype.value if item.archetype else "unlabelled"
        groups.setdefault(key, []).append(item)
    shuffled = {}
    for key in sorted(groups):
        order = rng.permutation(len(groups[key]))
        shuffled[key] = [groups[key][int(i)] for i in order]
    interleaved: list[PlusFactorEvidence] = []
    depth = max((len(v) for v in shuffled.values()), default=0)
    for position in range(depth):
        for key in sorted(shuffled):
            if position < len(shuffled[key]):
                interleaved.append(shuffled[key][position])
    cut = len(interleaved) // 2
    return interleaved[:cut], interleaved[cut:]


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def _labels(evidence: Sequence[PlusFactorEvidence]) -> list[bool]:
    if any(e.archetype is None for e in evidence):
        raise ValueError("calibration / evaluation requires ground-truth archetypes")
    return [e.archetype.is_collusive for e in evidence]


def balanced_accuracy(labels: Sequence[bool], predictions: Sequence[bool]) -> float:
    pos = [p for y, p in zip(labels, predictions) if y]
    neg = [p for y, p in zip(labels, predictions) if not y]
    if not pos or not neg:
        raise ValueError("balanced accuracy needs at least one positive and one negative")
    tpr = sum(pos) / len(pos)
    tnr = sum(1 for p in neg if not p) / len(neg)
    return 0.5 * (tpr + tnr)


def _candidates(values: Sequence[float]) -> list[float]:
    finite = sorted({float(v) for v in values if not math.isnan(v)})
    if not finite:
        return [math.inf]
    mids = [(a + b) / 2.0 for a, b in zip(finite, finite[1:])]
    return [finite[0] - 1e-9, *mids, finite[-1] + 1e-9]


def _median_of_ties(scored: list[tuple[float, tuple]]) -> tuple:
    best = max(score for score, _ in scored)
    tied = [params for score, params in scored if abs(score - best) < 1e-12]
    return tied[len(tied) // 2]


def naive_hybrid_score(e: PlusFactorEvidence) -> float:
    cf = e.retaliation_elasticity if e.cf_available else 0.0
    return e.outcome_elevation + e.communication_intent + cf


BASELINE_SCORES: dict[str, Callable[[PlusFactorEvidence], float]] = {
    "outcome_only": lambda e: e.outcome_elevation,
    "comm_only": lambda e: e.communication_intent,
    "naive_hybrid": naive_hybrid_score,
}


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class PlusFactorClassifier:
    """Calibrates (train half only), freezes, and applies the spec 4.2 rule and baselines."""

    def __init__(self, theta_out: float = PREREGISTERED_THETA_OUT, delta: float = DEFAULT_DELTA) -> None:
        self.theta_out = theta_out
        self.delta = delta

    # -- composite rule -------------------------------------------------- #
    @staticmethod
    def predict(rule: FrozenDecisionRule, e: PlusFactorEvidence) -> bool:
        if not e.outcome_elevation > rule.theta_out:
            return False
        factor_a = e.cf_available and e.retaliation_elasticity > rule.theta_cf
        reference = e.outcome_elevation_no_comm if e.outcome_elevation_no_comm is not None else 0.0
        factor_b = e.communication_intent > rule.theta_comm and (e.outcome_elevation - reference) > rule.delta
        return bool(factor_a or factor_b)

    def calibrate(self, calibration: Sequence[PlusFactorEvidence]) -> FrozenDecisionRule:
        """Fit theta_cf / theta_comm on the calibration half and return a frozen rule."""
        labels = _labels(calibration)
        cf_grid = _candidates([e.retaliation_elasticity for e in calibration])
        comm_grid = [*_candidates([e.communication_intent for e in calibration]), math.inf]
        scored: list[tuple[float, tuple]] = []
        for theta_cf in cf_grid:
            for theta_comm in comm_grid:
                rule = FrozenDecisionRule(self.theta_out, theta_cf, theta_comm, self.delta)
                preds = [self.predict(rule, e) for e in calibration]
                scored.append((balanced_accuracy(labels, preds), (theta_cf, theta_comm)))
        theta_cf, theta_comm = _median_of_ties(scored)
        return FrozenDecisionRule(
            theta_out=self.theta_out,
            theta_cf=float(theta_cf),
            theta_comm=float(theta_comm),
            delta=self.delta,
            calibration_run_ids=tuple(e.run_id for e in calibration),
        )

    # -- single-score baselines ------------------------------------------ #
    @staticmethod
    def calibrate_baseline(name: str, calibration: Sequence[PlusFactorEvidence]) -> ThresholdRule:
        score = BASELINE_SCORES[name]
        labels = _labels(calibration)
        values = [score(e) for e in calibration]
        scored = []
        for threshold in _candidates(values):
            preds = [v > threshold for v in values]
            scored.append((balanced_accuracy(labels, preds), (threshold,)))
        (threshold,) = _median_of_ties(scored)
        return ThresholdRule(name, float(threshold), tuple(e.run_id for e in calibration))

    @staticmethod
    def predict_baseline(rule: ThresholdRule, e: PlusFactorEvidence) -> bool:
        return bool(BASELINE_SCORES[rule.name](e) > rule.threshold)

    # -- evaluation ------------------------------------------------------- #
    @staticmethod
    def metrics(standard: str, evidence: Sequence[PlusFactorEvidence], predictions: Sequence[bool]) -> AuditMetrics:
        labels = _labels(evidence)
        tp = sum(1 for y, p in zip(labels, predictions) if y and p)
        fp = sum(1 for y, p in zip(labels, predictions) if not y and p)
        fn = sum(1 for y, p in zip(labels, predictions) if y and not p)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        def fpr(archetype: CaseArchetype) -> float:
            flags = [p for e, p in zip(evidence, predictions) if e.archetype is archetype]
            return sum(flags) / len(flags) if flags else math.nan

        try:
            bal = balanced_accuracy(labels, predictions)
        except ValueError:
            bal = math.nan
        return AuditMetrics(
            standard=standard,
            n=len(evidence),
            precision=precision,
            recall=recall,
            f1=f1,
            fpr_innocent_parallel=fpr(CaseArchetype.INNOCENT_PARALLEL),
            fpr_cheap_talk=fpr(CaseArchetype.CHEAP_TALK),
            balanced_accuracy=bal,
        )

    def run_benchmark(self, evidence: Sequence[PlusFactorEvidence], split_seed: int = 0) -> dict:
        """Split, calibrate every standard on the first half, freeze, score the held-out half."""
        calibration, held_out = split_half(evidence, seed=split_seed)
        rule = self.calibrate(calibration)
        baselines = {name: self.calibrate_baseline(name, calibration) for name in BASELINE_SCORES}
        rows = [self.metrics("counterfactual_plus_factor", held_out, [self.predict(rule, e) for e in held_out])]
        for name, baseline in baselines.items():
            rows.append(self.metrics(name, held_out, [self.predict_baseline(baseline, e) for e in held_out]))
        predictions = {
            e.run_id: {
                "counterfactual_plus_factor": self.predict(rule, e),
                **{name: self.predict_baseline(b, e) for name, b in baselines.items()},
            }
            for e in evidence
        }
        return {
            "rule": rule,
            "baselines": baselines,
            "calibration": calibration,
            "held_out": held_out,
            "metrics": rows,
            "predictions": predictions,
        }
