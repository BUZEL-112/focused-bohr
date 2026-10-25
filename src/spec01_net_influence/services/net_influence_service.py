"""Stage 2A: Pairwise Net Influence (Bonjour et al., 2022) via first-order Markov
conditional directed information with Miller-Madow bias correction."""

from __future__ import annotations

from collections.abc import Sequence

from src.common.metrics import conditional_mutual_information


class NetInfluenceService:
    """Computes Bonjour et al. (2022) Pairwise Net Influence between agent symbol streams.

    The full directed information I(z_i^T -> z_j^T || z_{-ij}^T) sums per-step
    conditional mutual informations I(z_{j,t} ; z_i^{t-1} | z_j^{t-1}, z_{-ij}^{t-1}).
    Under the first-order Markov approximation (spec 01, eq. 4.2) each per-step term is
    estimated from a pooled histogram over t, so the (1/T)-normalized directed
    information equals the single-step estimate:

        I_1(i -> j) = I(z_{j,t} ; z_{i,t-1} | z_{j,t-1}, z_{-ij,t-1})

    Net influence is the symmetric minimum NI(i, j) = min(I_1(i->j), I_1(j->i)):
    true collusion requires reciprocal coordination, not one-way price leadership.
    """

    def compute_directed_information(
        self,
        source_stream: Sequence[int],
        target_stream: Sequence[int],
        context_streams: Sequence[Sequence[int]] = (),
    ) -> float:
        """(1/T)-normalized conditional directed information from source to target, in bits.

        ``context_streams`` are the other agents' symbol streams; their lag-1 values form
        the conditioning compound symbol. Streams must be equal length.
        """
        if len(source_stream) != len(target_stream):
            raise ValueError("source and target streams must have equal length")
        for context in context_streams:
            if len(context) != len(target_stream):
                raise ValueError("all streams must have equal length")
        if len(target_stream) < 2:
            return 0.0

        source_past = list(source_stream[:-1])
        target_now = list(target_stream[1:])
        target_past = list(target_stream[:-1])
        contexts = [
            tuple(stream[t] for stream in context_streams) for t in range(len(target_stream) - 1)
        ]
        return conditional_mutual_information(
            source_past, target_now, contexts, bias_correction=True
        )

    def compute_pairwise_net_influence(
        self,
        stream_a: Sequence[int],
        stream_b: Sequence[int],
        other_streams: Sequence[Sequence[int]] = (),
    ) -> tuple[float, float, float]:
        """Returns (I(a->b), I(b->a), NI = min of the two), all in bits per step."""
        info_a_to_b = self.compute_directed_information(stream_a, stream_b, other_streams)
        info_b_to_a = self.compute_directed_information(stream_b, stream_a, other_streams)
        return info_a_to_b, info_b_to_a, min(info_a_to_b, info_b_to_a)
