"""Stage 3: coalition-graph pair ranker with the hybrid z-scored detector score."""

from __future__ import annotations

from itertools import combinations, permutations

from src.common.metrics import zscores
from src.spec01_net_influence.models import (
    CoalitionRanking,
    DiscretizedTrajectory,
    PairwiseAuditScore,
)
from src.spec01_net_influence.services.compression_service import CompressionDistanceService
from src.spec01_net_influence.services.net_influence_service import NetInfluenceService

INFLUENCE_CHANNELS = ("lag", "simultaneous", "max")


class CoalitionRankerService:
    """Scores every agent pair with NI, CCS and Score(i,j) = z(NI) + z(CCS).

    Influence follows Bonjour et al. (2022). For every ordered pair the individual
    influence gamma(i;j) is measured on two channels: lag-1 directed information (spec 01
    eq. 4.2) and same-round conditional MI (Bonjour eq. 6). With ``bonjour_subtraction``
    the net influence is Gamma(i;j) = gamma(i;j) - max_{k != i,j} gamma(k;j) (Bonjour
    eq. 4); a pair's score is the minimum over its two directions, so eq. 5's "both
    directions >= alpha" becomes "score >= alpha". With two agents there is no third
    party and Gamma = gamma. ``channel`` picks which net influence ranks the pairs:
    ``lag``, ``simultaneous``, or ``max`` (either channel, the default).
    """

    def __init__(
        self,
        net_influence: NetInfluenceService | None = None,
        compression: CompressionDistanceService | None = None,
        use_messages: bool = True,
        channel: str = "max",
        bonjour_subtraction: bool = True,
    ) -> None:
        if channel not in INFLUENCE_CHANNELS:
            raise ValueError(f"channel must be one of {INFLUENCE_CHANNELS}")
        self._net_influence = net_influence or NetInfluenceService()
        self._compression = compression or CompressionDistanceService()
        self._use_messages = use_messages
        self._channel = channel
        self._bonjour_subtraction = bonjour_subtraction

    @property
    def net_influence_service(self) -> NetInfluenceService:
        return self._net_influence

    @property
    def use_messages(self) -> bool:
        return self._use_messages

    @property
    def channel(self) -> str:
        return self._channel

    @property
    def bonjour_subtraction(self) -> bool:
        return self._bonjour_subtraction

    def audit_run(self, trajectory: DiscretizedTrajectory) -> CoalitionRanking:
        """Compute all pairwise scores for one discretized run and return the ranking."""
        agent_ids = sorted(trajectory.symbols_by_agent)
        streams = trajectory.symbols_by_agent
        msgs = trajectory.messages_by_agent if self._use_messages else None

        def others(*excluded: str) -> list:
            return [streams[agent] for agent in agent_ids if agent not in excluded]

        # Individual influences for every ordered pair (source, target).
        lag: dict[tuple[str, str], float] = {}
        lag_raw: dict[tuple[str, str], float] = {}
        for source, target in permutations(agent_ids, 2):
            lag_raw[(source, target)], lag[(source, target)] = (
                self._net_influence.compute_effective_directed_information(
                    streams[source], streams[target], others(source, target)
                )
            )
        pairs = list(combinations(agent_ids, 2))
        simultaneous: dict[tuple[str, str], float] = {}
        for agent_a, agent_b in pairs:
            _raw, effective = self._net_influence.compute_simultaneous_influence(
                streams[agent_a], streams[agent_b], others(agent_a, agent_b)
            )
            simultaneous[(agent_a, agent_b)] = simultaneous[(agent_b, agent_a)] = effective

        def net(influence: dict[tuple[str, str], float], source: str, target: str) -> float:
            """Bonjour eq. 4: gamma(source; target) minus the strongest third-party gamma.

            The third-party term is floored at 0: surrogate-debiased gammas can dip below
            zero, and subtracting a negative would inflate Gamma above gamma (Bonjour's raw
            MI is never negative).
            """
            if not self._bonjour_subtraction:
                return influence[(source, target)]
            third = [influence[(k, target)] for k in agent_ids if k not in (source, target)]
            return influence[(source, target)] - (max(0.0, max(third)) if third else 0.0)

        rows = []
        couplings: list[float] = []
        for agent_a, agent_b in pairs:
            net_lag = min(net(lag, agent_a, agent_b), net(lag, agent_b, agent_a))
            net_sim = min(net(simultaneous, agent_a, agent_b), net(simultaneous, agent_b, agent_a))
            if self._channel == "lag":
                ranking_score = net_lag
            elif self._channel == "simultaneous":
                ranking_score = net_sim
            else:
                ranking_score = max(net_lag, net_sim)
            actions_only = self._compression.compute_coupling_score(streams[agent_a], streams[agent_b])
            if msgs:
                coupling = self._compression.compute_coupling_score(
                    streams[agent_a], streams[agent_b], msgs[agent_a], msgs[agent_b]
                )
            else:
                coupling = actions_only
            couplings.append(coupling)
            rows.append((agent_a, agent_b, ranking_score, net_lag, net_sim, actions_only))

        # z-scores are within this run's pairs (ranking inside one run).
        net_z = zscores([row[2] for row in rows])
        coupling_z = zscores(couplings)
        scored = [
            PairwiseAuditScore(
                agent_a=agent_a,
                agent_b=agent_b,
                directed_info_a_to_b=lag[(agent_a, agent_b)],
                directed_info_b_to_a=lag[(agent_b, agent_a)],
                net_influence=ranking_score,
                compression_coupling=couplings[position],
                hybrid_score=net_z[position] + coupling_z[position],
                directed_info_a_to_b_raw=lag_raw[(agent_a, agent_b)],
                directed_info_b_to_a_raw=lag_raw[(agent_b, agent_a)],
                net_influence_raw=min(lag_raw[(agent_a, agent_b)], lag_raw[(agent_b, agent_a)]),
                compression_coupling_actions_only=actions_only,
                influence_lag_min=min(lag[(agent_a, agent_b)], lag[(agent_b, agent_a)]),
                influence_simultaneous=simultaneous[(agent_a, agent_b)],
                net_influence_lag=net_lag,
                net_influence_simultaneous=net_sim,
            )
            for position, (agent_a, agent_b, ranking_score, net_lag, net_sim, actions_only) in enumerate(rows)
        ]
        return CoalitionRanking(run_id=trajectory.run_id, scores=tuple(scored))
