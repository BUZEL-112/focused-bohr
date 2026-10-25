"""Stage 3: coalition-graph pair ranker with the hybrid z-scored detector score."""

from __future__ import annotations

from itertools import combinations

from src.common.metrics import zscores
from src.spec01_net_influence.models import (
    CoalitionRanking,
    DiscretizedTrajectory,
    PairwiseAuditScore,
)
from src.spec01_net_influence.services.compression_service import CompressionDistanceService
from src.spec01_net_influence.services.net_influence_service import NetInfluenceService


class CoalitionRankerService:
    """Scores every agent pair with NI, CCS and Score(i,j) = z(NI) + z(CCS)."""

    def __init__(
        self,
        net_influence: NetInfluenceService | None = None,
        compression: CompressionDistanceService | None = None,
        use_messages: bool = True,
    ) -> None:
        self._net_influence = net_influence or NetInfluenceService()
        self._compression = compression or CompressionDistanceService()
        self._use_messages = use_messages

    @property
    def net_influence_service(self) -> NetInfluenceService:
        return self._net_influence

    @property
    def use_messages(self) -> bool:
        return self._use_messages

    def audit_run(self, trajectory: DiscretizedTrajectory) -> CoalitionRanking:
        """Compute all pairwise scores for one discretized run and return the ranking."""
        agent_ids = list(trajectory.symbols_by_agent)
        streams = trajectory.symbols_by_agent
        msgs = trajectory.messages_by_agent if self._use_messages else None

        pairs = list(combinations(sorted(agent_ids), 2))
        reports = []
        couplings: list[float] = []
        couplings_actions_only: list[float] = []
        for agent_a, agent_b in pairs:
            others = [agent for agent in agent_ids if agent not in (agent_a, agent_b)]
            reports.append(
                self._net_influence.compute_pairwise_report(
                    streams[agent_a], streams[agent_b], [streams[other] for other in others]
                )
            )
            actions_only = self._compression.compute_coupling_score(streams[agent_a], streams[agent_b])
            couplings_actions_only.append(actions_only)
            if msgs:
                couplings.append(
                    self._compression.compute_coupling_score(
                        streams[agent_a], streams[agent_b], msgs[agent_a], msgs[agent_b]
                    )
                )
            else:
                couplings.append(actions_only)

        # Ranking uses the surrogate-debiased NI; z-scores are within this run's pairs.
        net_z = zscores([report.net_influence for report in reports])
        coupling_z = zscores(couplings)
        scored = [
            PairwiseAuditScore(
                agent_a=agent_a,
                agent_b=agent_b,
                directed_info_a_to_b=reports[position].a_to_b,
                directed_info_b_to_a=reports[position].b_to_a,
                net_influence=reports[position].net_influence,
                compression_coupling=couplings[position],
                hybrid_score=net_z[position] + coupling_z[position],
                directed_info_a_to_b_raw=reports[position].a_to_b_raw,
                directed_info_b_to_a_raw=reports[position].b_to_a_raw,
                net_influence_raw=reports[position].net_influence_raw,
                compression_coupling_actions_only=couplings_actions_only[position],
            )
            for position, (agent_a, agent_b) in enumerate(pairs)
        ]
        return CoalitionRanking(run_id=trajectory.run_id, scores=tuple(scored))
