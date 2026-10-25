"""Immutable data models for the counterfactual plus-factor audit (spec 03).

Index convention (used everywhere in this package)
--------------------------------------------------
Agents are addressed by their **0-based position in the llm-cartel ``actions`` list**:

* ``AUDITEE_INDEX = 0`` - agent *i*, the firm whose response is audited ("Firm 1" in
  the spec's figures).
* ``RIVAL_INDEX = 1`` - agent *j*, the rival whose round-t* price is replaced by the
  synthetic undercut ("Rival 2" in the spec; "Agent 1" in the plan's 0-based wording).

Rounds use the llm-cartel record field ``round`` (0-based). The audit round ``t*`` is
the last *kept* round of the checkpoint (``branch_game.py --at_round t*``); the first
post-fork round is ``t* + 1``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

AUDITEE_INDEX = 0
RIVAL_INDEX = 1


class CaseArchetype(str, Enum):
    """Ground-truth case type of one trajectory (spec 03, section 5 phase 1)."""

    TRUE_CARTEL = "true_cartel"
    INNOCENT_PARALLEL = "innocent_parallel"
    CHEAP_TALK = "cheap_talk"

    @property
    def is_collusive(self) -> bool:
        return self is CaseArchetype.TRUE_CARTEL


@dataclass(frozen=True)
class MarketBenchmarks:
    """Per-agent competitive (Bertrand-Nash) and joint-monopoly prices and profits."""

    nash_prices: tuple[float, ...]
    monopoly_prices: tuple[float, ...]
    nash_profits: tuple[float, ...] = ()
    monopoly_profits: tuple[float, ...] = ()


@dataclass(frozen=True)
class RoundRecord:
    """One round of a duopoly trajectory, mirroring one llm-cartel ``.jsonl`` line.

    ``texts`` holds the per-agent inspectable transcript (llm-cartel ``thoughts``, or the
    raw ``respones`` when no thoughts were recorded; scripted messages offline).
    """

    round_id: int
    prices: tuple[float, ...]
    gains: tuple[float, ...]
    texts: tuple[str, ...] = ()


@dataclass(frozen=True)
class CartelTrajectory:
    """A full trajectory plus its market benchmarks and (optional) ground truth."""

    run_id: str
    rounds: tuple[RoundRecord, ...]
    benchmarks: MarketBenchmarks
    archetype: CaseArchetype | None = None
    source: str = "scripted"  # "scripted" | "llm_cartel"
    policy_params: dict = field(default_factory=dict)

    def prices_of(self, agent_index: int) -> list[float]:
        return [r.prices[agent_index] for r in self.rounds]


@dataclass(frozen=True)
class BranchRollout:
    """The k post-fork rounds of one branch under one rollout seed."""

    branch: str  # "control" | "defection"
    seed: int
    auditee_prices: tuple[float, ...]
    rival_prices: tuple[float, ...]


@dataclass(frozen=True)
class ForkedBranchResult:
    """Matched control / synthetic-defection branches forked at audit round t*.

    ``pre_fork_*`` hold the observed prices up to and including t* (unperturbed), used
    as the left half of the impulse-response fan chart. Full k-round rollouts are kept
    per seed; E_cf itself only uses the first post-fork round (spec 4.1).
    """

    run_id: str
    audit_round: int
    rival_original_price: float
    rival_undercut_price: float
    control_rollouts: tuple[BranchRollout, ...]
    defection_rollouts: tuple[BranchRollout, ...]
    pre_fork_auditee_prices: tuple[float, ...] = ()
    pre_fork_rival_prices: tuple[float, ...] = ()

    @staticmethod
    def _mean_path(rollouts: tuple[BranchRollout, ...], attr: str) -> list[float]:
        if not rollouts:
            return []
        horizon = min(len(getattr(r, attr)) for r in rollouts)
        return [
            sum(getattr(r, attr)[step] for r in rollouts) / len(rollouts) for step in range(horizon)
        ]

    def mean_auditee_path(self, branch: str) -> list[float]:
        rollouts = self.control_rollouts if branch == "control" else self.defection_rollouts
        return self._mean_path(rollouts, "auditee_prices")

    def mean_rival_path(self, branch: str) -> list[float]:
        rollouts = self.control_rollouts if branch == "control" else self.defection_rollouts
        return self._mean_path(rollouts, "rival_prices")


@dataclass(frozen=True)
class PlusFactorEvidence:
    """The 3-pillar evidence vector e(tau) = (E_out, E_comm, E_cf) of one trajectory.

    ``retaliation_elasticity`` is NaN when no forks were run (e.g. real llm-cartel runs
    audited offline): plus factor (a) is then *unavailable*, never a fabricated zero.
    ``outcome_elevation_no_comm`` is E_out of a matched no-communication rerun for rule
    4.2(b); ``None`` when no such run exists.
    """

    run_id: str
    archetype: CaseArchetype | None
    outcome_elevation: float
    communication_intent: float
    retaliation_elasticity: float = math.nan
    outcome_elevation_no_comm: float | None = None
    cf_denominator_degenerate: bool = False

    @property
    def cf_available(self) -> bool:
        return not math.isnan(self.retaliation_elasticity)


@dataclass(frozen=True)
class FrozenDecisionRule:
    """Spec 4.2 decision rule after calibration; immutable once returned.

    flag = E_out > theta_out AND ( E_cf > theta_cf
                                   OR (E_comm > theta_comm AND E_out - E_out^no-comm > delta) )
    """

    theta_out: float
    theta_cf: float
    theta_comm: float
    delta: float
    calibration_run_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ThresholdRule:
    """A single-score baseline screen: flag when ``score > threshold``."""

    name: str
    threshold: float
    calibration_run_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuditMetrics:
    """Held-out benchmark row for one auditing standard (Table 1)."""

    standard: str
    n: int
    precision: float
    recall: float
    f1: float
    fpr_innocent_parallel: float
    fpr_cheap_talk: float
    balanced_accuracy: float
