"""Immutable data models for the Net-Influence / NCD audit pipeline (spec 01)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentTurnRecord:
    """One agent's action (and optional message) in one round of one run.

    ``action_value`` is the raw continuous action (ask price, bid, quantity);
    ``market_benchmark`` is the expected competitive response E[A_t | S_t] used by the
    residualizer (stage-game competitive price, or the rolling cross-sectional median).
    """

    round_id: int
    agent_id: str
    action_value: float
    message_text: str = ""
    market_benchmark: float = 0.0


@dataclass(frozen=True)
class RunRecord:
    """A full multi-agent trajectory (one run) with its audit ground truth."""

    run_id: str
    is_collusive: bool
    colluding_pair: tuple[str, str] | None
    turns: tuple[AgentTurnRecord, ...]
    source: str = "scripted"  # "scripted" | "colludebench" | "double_auction"

    @property
    def agent_ids(self) -> list[str]:
        ordered: list[str] = []
        for turn in self.turns:
            if turn.agent_id not in ordered:
                ordered.append(turn.agent_id)
        return ordered


@dataclass(frozen=True)
class DiscretizedTrajectory:
    """Residualized, quantized per-agent symbol streams plus the raw message streams."""

    run_id: str
    is_collusive: bool
    colluding_pair: tuple[str, str] | None
    symbols_by_agent: dict[str, tuple[int, ...]]
    messages_by_agent: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PairwiseAuditScore:
    """All detector statistics for one unordered agent pair.

    ``directed_info_*`` and ``net_influence`` are surrogate-debiased ("effective")
    values used for ranking; the ``*_raw`` fields keep the Miller-Madow-only estimates.
    ``compression_coupling`` includes message text when the ranker uses messages;
    ``compression_coupling_actions_only`` never does.
    """

    agent_a: str
    agent_b: str
    directed_info_a_to_b: float
    directed_info_b_to_a: float
    net_influence: float
    compression_coupling: float
    hybrid_score: float = 0.0
    directed_info_a_to_b_raw: float = 0.0
    directed_info_b_to_a_raw: float = 0.0
    net_influence_raw: float = 0.0
    compression_coupling_actions_only: float = 0.0

    @property
    def pair(self) -> frozenset[str]:
        return frozenset((self.agent_a, self.agent_b))


@dataclass(frozen=True)
class CoalitionRanking:
    """Ranked pair scores for one run plus the flagged (top-1) pair."""

    run_id: str
    scores: tuple[PairwiseAuditScore, ...] = field(default_factory=tuple)

    @property
    def top_pair(self) -> tuple[str, str] | None:
        if not self.scores:
            return None
        best = max(self.scores, key=lambda s: s.hybrid_score)
        return (best.agent_a, best.agent_b)

    def rank_of(self, pair: tuple[str, str]) -> int | None:
        """1-based rank of ``pair`` under the hybrid score (None if absent)."""
        target = frozenset(pair)
        ordered = sorted(self.scores, key=lambda s: s.hybrid_score, reverse=True)
        for position, score in enumerate(ordered, start=1):
            if score.pair == target:
                return position
        return None
