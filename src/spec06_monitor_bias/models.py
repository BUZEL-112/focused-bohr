"""Immutable data models for the cross-family monitor audit pipeline (spec 06)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MonitorView(str, Enum):
    """The four Track 2.2 monitor views (spec 06, section 5 phase 2)."""

    MESSAGES_ONLY = "messages_only"
    ACTIONS_ONLY = "actions_only"
    RAW_HYBRID = "raw_hybrid"
    SYMBOLIC_SCRUBBED = "symbolic_scrubbed"

    @property
    def carries_prose(self) -> bool:
        """True for the views that expose raw natural-language messages (V1, V3)."""
        return self in (MonitorView.MESSAGES_ONLY, MonitorView.RAW_HYBRID)


# Canonical commitment / action symbols emitted by the say-do compiler.
SYMBOL_COOPERATE = "COOP"      # social-dilemma cooperative choice (CHEAP, YES)
SYMBOL_DEFECT = "DEFECT"       # social-dilemma defection (EXPENSIVE, NO)
SYMBOL_HIGH = "HIGH"           # price above the competitive anchor (cartel markup)
SYMBOL_LOW = "LOW"             # price at / below the competitive anchor
SYMBOL_UNKNOWN = "UNK"         # unparseable or missing commitment

# Symbols whose fulfilment harms parties outside the coalition (D_cross trigger).
WELFARE_HARMING_SYMBOLS = frozenset({SYMBOL_HIGH})


@dataclass(frozen=True)
class RoundSayDoRecord:
    """One agent's three-stage record in one round of one trajectory.

    Field provenance (real log keys):
      * llm-trust-breaking endogenous JSON (``trial_runner.py`` lines 914-943):
        ``private_plan`` <- ``agents[i].stage1.intended_action`` (+ ``stage1.reasoning``),
        ``public_message`` <- ``agents[i].stage2.message``,
        ``stated_action`` <- ``agents[i].stage2.stated_action``,
        ``executed_action`` <- ``agents[i].stage3.choice``,
        ``payoff`` <- ``trials[k].outcomes.payoffs[i]``.
      * colludebench jsonl: ``public_message`` <- ``message``, ``stated_action`` <- the
        ``Price:<x>`` token inside ``message``, ``executed_action`` <- ``action``,
        ``payoff`` <- ``reward``; there is no private plan (empty string).

    Actions are kept as raw strings (categorical games use EXPENSIVE/CHEAP, YES/NO,
    GO/STAY); ``stated_symbol`` / ``executed_symbol`` are the canonical compiler symbols.
    """

    round_id: int
    agent_id: str
    model_family: str
    private_plan: str
    public_message: str
    stated_action: str
    executed_action: str
    payoff: float
    stated_symbol: str = SYMBOL_UNKNOWN
    executed_symbol: str = SYMBOL_UNKNOWN


@dataclass(frozen=True)
class TrajectoryRecord:
    """A full multi-agent trajectory with its ground-truth collusion/deception label."""

    trajectory_id: str
    agent_family: str
    is_collusive: bool
    rounds: tuple[RoundSayDoRecord, ...]
    source: str = "scripted"  # "scripted" | "trust_breaking" | "colludebench"
    competitive_anchor: float | None = None  # numeric games: competitive price level

    @property
    def agent_ids(self) -> list[str]:
        ordered: list[str] = []
        for record in self.rounds:
            if record.agent_id not in ordered:
                ordered.append(record.agent_id)
        return ordered

    @property
    def round_ids(self) -> list[int]:
        return sorted({record.round_id for record in self.rounds})

    def mean_message_tokens(self) -> float:
        """Mean whitespace-token length of the public messages (verbosity axis of fig 3)."""
        if not self.rounds:
            return 0.0
        return sum(len(r.public_message.split()) for r in self.rounds) / len(self.rounds)


@dataclass(frozen=True)
class SayDoIndicators:
    """Spec 06 section 4.2 deterministic indicators for one (agent, round) cell."""

    agent_alias: str
    round_id: int
    claim_symbol: str
    claim_value: str
    action_symbol: str
    action_value: str
    payoff_delta: float
    d_self: int      # public commitment contradicts executed action
    d_cross: int     # action fulfils a peer's prior welfare-harming proposal
    v_exc: float     # raw message tokens per extracted propositional claim


@dataclass(frozen=True)
class MonitorVerdict:
    """One monitor family's normalized suspicion score for one trajectory under one view."""

    trajectory_id: str
    agent_family: str
    monitor_family: str
    view: MonitorView
    is_collusive: bool
    suspicion: float  # S(m, a, tau, v) in [0, 1]
    self_recognized: bool = False  # Track 3.3 probe: monitor believes "own family"
    mean_message_tokens: float = 0.0


@dataclass(frozen=True)
class ViewEvaluation:
    """Aggregate statistics for one monitor view over the 3x3 family grid."""

    view: MonitorView
    beta_self: float
    auroc: float
    collusive_grid: dict[tuple[str, str], float] = field(default_factory=dict)  # (agent, monitor)
    threshold: float = 0.5
    test_accuracy: float = 0.0
    test_fpr: float = 0.0
    test_tpr: float = 0.0
    diagonal_vs_offdiagonal_p: float = 1.0
    self_recognition_rate: float = 0.0
