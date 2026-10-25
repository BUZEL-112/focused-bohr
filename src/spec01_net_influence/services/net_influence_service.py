"""Stage 2A: Pairwise Net Influence (Bonjour et al., 2022) via first-order Markov
conditional directed information with Miller-Madow bias correction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.common.metrics import conditional_mutual_information

CONTEXT_MODES = ("spec", "compound")


@dataclass(frozen=True)
class DirectedInfluenceReport:
    """Both directions of one pair, raw (Miller-Madow only) and surrogate-debiased."""

    a_to_b_raw: float
    b_to_a_raw: float
    a_to_b: float
    b_to_a: float

    @property
    def net_influence_raw(self) -> float:
        return min(self.a_to_b_raw, self.b_to_a_raw)

    @property
    def net_influence(self) -> float:
        return min(self.a_to_b, self.b_to_a)


class NetInfluenceService:
    """Computes Bonjour et al. (2022) Pairwise Net Influence between agent symbol streams.

    The full directed information I(z_i^T -> z_j^T || z_{-ij}^T) sums per-step
    conditional mutual informations I(z_{j,t} ; z_i^{t-1} | z_j^{t-1}, z_{-ij}^{t-1}).
    Under the first-order Markov approximation (spec 01, eq. 4.2) each per-step term is
    estimated from a pooled histogram over t, so the (1/T)-normalized directed
    information equals the single-step estimate:

        I_1(i -> j) = I(z_{j,t} ; z_{i,t-1} | z_{j,t-1}, zbar_{-ij,t-1})

    Context modes:
      * ``spec`` (default) - the spec's context: the target's own past plus the rounded
        mean symbol of the remaining agents (at most K * K context states).
      * ``compound`` - the original implementation: the full tuple of the remaining
        agents' symbols, without the target's own past (up to K^(M-2) states). Kept so
        both conditionings can be compared in the same run.

    Finite-sample bias: at T ~ 30 the context table is sparse and Miller-Madow cannot
    equalize the bias between low-entropy (cartel) and high-entropy (honest) streams.
    With ``surrogates > 0`` the "effective" estimate subtracts the mean directed
    information over deterministic circular shifts of the source stream, which keep its
    marginal and autocorrelation but break its alignment with the target.

    Net influence is the symmetric minimum NI(i, j) = min(I_1(i->j), I_1(j->i)):
    true collusion requires reciprocal coordination, not one-way price leadership.
    """

    def __init__(self, context_mode: str = "spec", surrogates: int = 19) -> None:
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}")
        if surrogates < 0:
            raise ValueError("surrogates must be non-negative")
        self._context_mode = context_mode
        self._surrogates = surrogates

    @property
    def context_mode(self) -> str:
        return self._context_mode

    @property
    def surrogates(self) -> int:
        return self._surrogates

    # ------------------------------------------------------------------ #
    # Raw estimate (Miller-Madow only)
    # ------------------------------------------------------------------ #
    def compute_directed_information(
        self,
        source_stream: Sequence[int],
        target_stream: Sequence[int],
        context_streams: Sequence[Sequence[int]] = (),
    ) -> float:
        """(1/T)-normalized conditional directed information from source to target, in bits.

        ``context_streams`` are the other agents' symbol streams; their lag-1 values form
        the conditioning symbol (see ``context_mode``). Streams must be equal length.
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
        contexts = self._contexts(target_stream, context_streams)
        return conditional_mutual_information(
            source_past, target_now, contexts, bias_correction=True
        )

    # ------------------------------------------------------------------ #
    # Surrogate-debiased ("effective") estimate
    # ------------------------------------------------------------------ #
    def compute_effective_directed_information(
        self,
        source_stream: Sequence[int],
        target_stream: Sequence[int],
        context_streams: Sequence[Sequence[int]] = (),
    ) -> tuple[float, float]:
        """Returns (raw, effective) where effective = raw - mean over shifted sources."""
        raw = self.compute_directed_information(source_stream, target_stream, context_streams)
        shifts = self._surrogate_shifts(len(source_stream))
        if not shifts:
            return raw, raw
        source = list(source_stream)
        null_values = [
            self.compute_directed_information(source[shift:] + source[:shift], target_stream, context_streams)
            for shift in shifts
        ]
        return raw, raw - float(np.mean(null_values))

    def compute_simultaneous_influence(
        self,
        stream_a: Sequence[int],
        stream_b: Sequence[int],
        context_streams: Sequence[Sequence[int]] = (),
    ) -> tuple[float, float]:
        """Same-round influence, Bonjour et al. (2022) eq. 6: I(z_{a,t} ; z_{b,t} | S_t).

        Symmetric in a and b. Residualization already removes the public demand state,
        so S_t is the remaining market context: the rounded mean symbol of the other
        agents in the same round (``spec``) or their full tuple (``compound``).
        Catches coordination agreed before the round (chat, then prices), which the
        lag-1 directed information cannot see. Returns (raw, effective).
        """
        if len(stream_a) != len(stream_b):
            raise ValueError("streams must have equal length")
        for context in context_streams:
            if len(context) != len(stream_a):
                raise ValueError("all streams must have equal length")
        length = len(stream_a)
        if length < 2:
            return 0.0, 0.0
        contexts = self._same_round_contexts(length, context_streams)
        raw = conditional_mutual_information(list(stream_a), list(stream_b), contexts, bias_correction=True)
        shifts = self._surrogate_shifts(length)
        if not shifts:
            return raw, raw
        source = list(stream_a)
        null_values = [
            conditional_mutual_information(
                source[shift:] + source[:shift], list(stream_b), contexts, bias_correction=True
            )
            for shift in shifts
        ]
        return raw, raw - float(np.mean(null_values))

    def compute_pairwise_report(
        self,
        stream_a: Sequence[int],
        stream_b: Sequence[int],
        other_streams: Sequence[Sequence[int]] = (),
    ) -> DirectedInfluenceReport:
        a_to_b_raw, a_to_b = self.compute_effective_directed_information(stream_a, stream_b, other_streams)
        b_to_a_raw, b_to_a = self.compute_effective_directed_information(stream_b, stream_a, other_streams)
        return DirectedInfluenceReport(
            a_to_b_raw=a_to_b_raw, b_to_a_raw=b_to_a_raw, a_to_b=a_to_b, b_to_a=b_to_a
        )

    def compute_pairwise_net_influence(
        self,
        stream_a: Sequence[int],
        stream_b: Sequence[int],
        other_streams: Sequence[Sequence[int]] = (),
    ) -> tuple[float, float, float]:
        """Returns effective (I(a->b), I(b->a), NI = min of the two), in bits per step."""
        report = self.compute_pairwise_report(stream_a, stream_b, other_streams)
        return report.a_to_b, report.b_to_a, report.net_influence

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _contexts(
        self, target_stream: Sequence[int], context_streams: Sequence[Sequence[int]]
    ) -> list[tuple]:
        """One conditioning symbol per lag-1 sample t = 0 .. T-2."""
        steps = range(len(target_stream) - 1)
        if self._context_mode == "compound":
            return [tuple(stream[t] for stream in context_streams) for t in steps]
        if not context_streams:
            return [(target_stream[t],) for t in steps]
        return [
            (target_stream[t], int(round(float(np.mean([stream[t] for stream in context_streams])))))
            for t in steps
        ]

    def _same_round_contexts(self, length: int, context_streams: Sequence[Sequence[int]]) -> list[tuple]:
        """One conditioning symbol per round t = 0 .. T-1 (same round, no lag)."""
        if not context_streams:
            return [()] * length
        if self._context_mode == "compound":
            return [tuple(stream[t] for stream in context_streams) for t in range(length)]
        return [
            (int(round(float(np.mean([stream[t] for stream in context_streams])))),)
            for t in range(length)
        ]

    def _surrogate_shifts(self, length: int) -> list[int]:
        """Deterministic, evenly spaced circular shifts in [1, length - 1]."""
        if self._surrogates == 0 or length < 4:
            return []
        candidates = np.linspace(1, length - 1, num=self._surrogates)
        return sorted({int(round(value)) for value in candidates} - {0, length})
