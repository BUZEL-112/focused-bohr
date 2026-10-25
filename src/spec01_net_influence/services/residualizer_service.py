"""Stage 1: public-state residualization and quantization into discrete symbol streams."""

from __future__ import annotations

from typing import Sequence

from src.spec01_net_influence.models import AgentTurnRecord, DiscretizedTrajectory


class ResidualizerService:
    """Converts continuous agent actions into market-residualized discrete symbol streams.

    The residual r_{i,t} = a_{i,t} - E[A_t | S_t] removes innocent correlation caused by
    shared demand/cost shocks. Residuals are mapped onto ``num_bins`` symmetric bins
    (default K = 5: strong undercut ... strong markup) using the run's own residual scale
    so the mapping is deterministic and contamination-free (no cross-run leakage).
    """

    def __init__(self, num_bins: int = 5) -> None:
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2.")
        self._num_bins = num_bins

    @property
    def num_bins(self) -> int:
        return self._num_bins

    def residual(self, turn: AgentTurnRecord) -> float:
        return turn.action_value - turn.market_benchmark

    def discretize_turns(
        self,
        run_id: str,
        turns: Sequence[AgentTurnRecord],
        is_collusive: bool,
        colluding_pair: tuple[str, str] | None = None,
    ) -> DiscretizedTrajectory:
        if not turns:
            raise ValueError("cannot discretize an empty turn list")

        by_agent: dict[str, list[AgentTurnRecord]] = {}
        for turn in turns:
            by_agent.setdefault(turn.agent_id, []).append(turn)
        for agent_id, agent_turns in by_agent.items():
            agent_turns.sort(key=lambda t: t.round_id)

        scale = max(abs(self.residual(turn)) for turn in turns)
        if scale < 1e-9:
            scale = 1.0

        symbols: dict[str, tuple[int, ...]] = {}
        messages: dict[str, tuple[str, ...]] = {}
        for agent_id, agent_turns in by_agent.items():
            symbols[agent_id] = tuple(self._bin(self.residual(turn), scale) for turn in agent_turns)
            messages[agent_id] = tuple(turn.message_text for turn in agent_turns)

        return DiscretizedTrajectory(
            run_id=run_id,
            is_collusive=is_collusive,
            colluding_pair=colluding_pair,
            symbols_by_agent=symbols,
            messages_by_agent=messages,
        )

    def _bin(self, residual: float, scale: float) -> int:
        """Map a residual in [-scale, scale] onto {0, .., K-1} (0 = strongest undercut)."""
        normalized = residual / scale  # in [-1, 1]
        shifted = (normalized + 1.0) / 2.0  # in [0, 1]
        index = int(shifted * self._num_bins)
        return min(self._num_bins - 1, max(0, index))
