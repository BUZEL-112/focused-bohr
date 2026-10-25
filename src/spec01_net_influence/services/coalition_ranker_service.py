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

    def audit_run(self, trajectory: DiscretizedTrajectory) -> CoalitionRanking:
        """Compute all pairwise scores for one discretized run and return the ranking."""
        agent_ids = list(trajectory.symbols_by_agent)
        streams = trajectory.symbols_by_agent
        msgs = trajectory.messages_by_agent if self._use_messages else None

        pairs = list(combinations(sorted(agent_ids), 2))
        net_influences: list[float] = []
        couplings: list[float] = []
        directional: dict[tuple[str, str], tuple[float, float]] = {}
        for agent_a, agent_b in pairs:
            others = [agent for agent in agent_ids if agent not in (agent_a, agent_b)]
            info_a_to_b, info_b_to_a, net_influence = self._net_influence.compute_pairwise_net_influence(
                streams[agent_a], streams[agent_b], [streams[other] for other in others]
            )
            directional[(agent_a, agent_b)] = (info_a_to_b, info_b_to_a)
            net_influences.append(net_influence)

            msgs_a = msgs[agent_a] if msgs else None
            msgs_b = msgs[agent_b] if msgs else None
            couplings.append(
                self._compression.compute_coupling_score(
                    streams[agent_a], streams[agent_b], msgs_a, msgs_b
                )
            )

        net_z = zscores(net_influences)
        coupling_z = zscores(couplings)
        scored = [
            PairwiseAuditScore(
                agent_a=agent_a,
                agent_b=agent_b,
                directed_info_a_to_b=directional[(agent_a, agent_b)][0],
                directed_info_b_to_a=directional[(agent_a, agent_b)][1],
                net_influence=net_influences[position],
                compression_coupling=couplings[position],
                hybrid_score=net_z[position] + coupling_z[position],
            )
            for position, (agent_a, agent_b) in enumerate(pairs)
        ]
        return CoalitionRanking(run_id=trajectory.run_id, scores=tuple(scored))
