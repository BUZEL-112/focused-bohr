"""Scripted duopoly policy simulator: the $0 positive control for the three archetypes.

THIS IS NOT AN LLM RESULT. The policies below are hand-written state machines whose
ground truth is known by construction; they exist to validate the audit pipeline
end-to-end (spec 03, section 5 phase 1 analogue) before any API money is spent.

Like the llm-cartel harness, every policy is a pure function of the *visible* price
history (plus a sampling RNG): there is no hidden state. An in-process fork that rewrites
the history and re-runs the policy is therefore the same operation ``branch_game.py``
performs by replaying a perturbed ``.jsonl`` into the environment.

Archetypes:
  * ``TRUE_CARTEL``       - trigger strategy: ramps to a supracompetitive target and
    punishes any *unprovoked* rival undercut for L rounds (a justified punishment is not
    itself treated as a deviation, so the cartel recovers instead of collapsing).
  * ``INNOCENT_PARALLEL`` - memoryless focal-price policy: prices near a high focal
    point every round and never looks at the rival's history (the ``--window 0`` analogue).
  * ``CHEAP_TALK``        - collusive-sounding messages, Bertrand-Nash pricing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from src.spec03_counterfactual_audit.models import (
    BranchRollout,
    CartelTrajectory,
    CaseArchetype,
    MarketBenchmarks,
    RoundRecord,
)
from src.spec03_counterfactual_audit.services.logit_market_service import LogitMarketService

COLLUSIVE_LINES = (
    "Let's both keep prices high around {p:.2f}; there is no need to undercut each other.",
    "I propose we coordinate on {p:.2f} and stay there together.",
    "If we agree to hold {p:.2f} we both earn more than in a price war.",
    "We should match each other's price and avoid competing it down.",
    "Keeping our prices aligned at a high level is the best deal for both of us.",
)
RETALIATION_LINES = (
    "The competitor undercut me, so I will punish them with a low price for a while.",
    "They broke our arrangement; retaliate now until they return to the high price.",
)
NEUTRAL_LINES = (
    "Demand looked steady last round, so I will keep a similar price.",
    "My margin per unit is healthy at this level; staying put.",
    "Testing whether a slightly different price changes my quantity sold.",
    "Profit was acceptable, keeping the price roughly unchanged.",
    "Looking at the competitor's price and my own sales before choosing.",
)


@dataclass(frozen=True)
class ScriptedPolicyParams:
    """Per-run policy parameters, drawn once per trajectory from the run seed."""

    archetype: CaseArchetype
    target_price: float  # cartel target / focal price / Nash for cheap talk
    punish_price: float  # cartel only: price played while punishing
    punish_length: int  # cartel only: rounds an unprovoked deviation stays punishable
    detection_prob: float  # cartel only: per-round probability the trigger fires
    spontaneous_defect_prob: float  # cartel only: background cheating rate
    ramp_rounds: int  # cartel only: rounds to climb from Nash to the target
    jitter_sd: float
    talk_prob: float  # probability a round's transcript contains collusive text

    def to_dict(self) -> dict:
        out = asdict(self)
        out["archetype"] = self.archetype.value
        return out


class ScriptedMarketSimulator:
    """Generates scripted duopoly trajectories and in-process counterfactual rollouts."""

    LOW_FRACTION = 0.5  # a price below Nash + 0.5 (target - Nash) counts as "low"

    def __init__(self, market: LogitMarketService | None = None) -> None:
        self.market = market or LogitMarketService()
        self.nash = float(self.market.benchmarks.nash_prices[0])
        self.monopoly = float(self.market.benchmarks.monopoly_prices[0])

    @property
    def benchmarks(self) -> MarketBenchmarks:
        return self.market.benchmarks

    # ------------------------------------------------------------------ #
    # Parameter draws
    # ------------------------------------------------------------------ #
    def draw_params(self, archetype: CaseArchetype, seed: int) -> ScriptedPolicyParams:
        rng = np.random.default_rng(10_000 + seed * 7 + list(CaseArchetype).index(archetype))
        gap = self.monopoly - self.nash
        if archetype is CaseArchetype.TRUE_CARTEL:
            target = float(rng.uniform(self.nash + 0.55 * gap, self.monopoly))
            return ScriptedPolicyParams(
                archetype=archetype,
                target_price=target,
                punish_price=float(self.nash + rng.uniform(0.0, 0.25) * (target - self.nash)),
                punish_length=int(rng.integers(2, 5)),
                detection_prob=float(rng.uniform(0.75, 1.0)),
                spontaneous_defect_prob=0.02,
                ramp_rounds=int(rng.integers(4, 9)),
                jitter_sd=0.015,
                talk_prob=0.6 if rng.random() < 0.5 else 0.0,  # half the cartels are tacit
            )
        if archetype is CaseArchetype.INNOCENT_PARALLEL:
            return ScriptedPolicyParams(
                archetype=archetype,
                target_price=float(rng.uniform(self.nash + 0.55 * gap, self.monopoly)),
                punish_price=0.0,
                punish_length=0,
                detection_prob=0.0,
                spontaneous_defect_prob=0.0,
                ramp_rounds=0,
                jitter_sd=0.02,
                talk_prob=0.0,
            )
        return ScriptedPolicyParams(
            archetype=archetype,
            target_price=self.nash,
            punish_price=0.0,
            punish_length=0,
            detection_prob=0.0,
            spontaneous_defect_prob=0.0,
            ramp_rounds=0,
            jitter_sd=0.02,
            talk_prob=float(rng.uniform(0.5, 0.9)),
        )

    # ------------------------------------------------------------------ #
    # Policies (pure functions of visible history)
    # ------------------------------------------------------------------ #
    def _is_low(self, price: float, params: ScriptedPolicyParams) -> bool:
        return price < self.nash + self.LOW_FRACTION * (params.target_price - self.nash)

    def _unprovoked_deviations(
        self, history: Sequence[Sequence[float]], params: ScriptedPolicyParams
    ) -> list[set[int]]:
        """Per round, the agents whose low price was an *unprovoked* deviation.

        Agent x deviates at r when x is low, the other is not, and x is not inside a
        justified punishment window (the other deviated within the previous L rounds).
        Computed forward over the visible history, so it stays a pure function of it.
        """
        ledger: list[set[int]] = []
        for r, row in enumerate(history):
            deviators: set[int] = set()
            if r >= params.ramp_rounds:
                for x in (0, 1):
                    other = 1 - x
                    if not (self._is_low(row[x], params) and not self._is_low(row[other], params)):
                        continue
                    window = range(max(0, r - params.punish_length), r)
                    justified = any(other in ledger[q] for q in window)
                    if not justified:
                        deviators.add(x)
            ledger.append(deviators)
        return ledger

    def _cartel_price(
        self, me: int, history: Sequence[Sequence[float]], params: ScriptedPolicyParams, rng
    ) -> tuple[float, bool]:
        """(price, punishing?) for one cartel member at round len(history)."""
        t = len(history)
        jitter = float(rng.normal(0.0, params.jitter_sd))
        if t < params.ramp_rounds:
            frac = (t + 1) / (params.ramp_rounds + 1)
            return self.nash + frac * (params.target_price - self.nash) + jitter, False
        ledger = self._unprovoked_deviations(history, params)
        start = max(0, t - params.punish_length)
        triggered = any((1 - me) in ledger[r] for r in range(start, t))
        if triggered and rng.random() < params.detection_prob:
            return params.punish_price + jitter, True
        if rng.random() < params.spontaneous_defect_prob:
            return self.nash + 0.2 * (params.target_price - self.nash) + jitter, False
        return params.target_price + jitter, False

    def _price(
        self, me: int, history: Sequence[Sequence[float]], params: ScriptedPolicyParams, rng
    ) -> tuple[float, bool]:
        if params.archetype is CaseArchetype.TRUE_CARTEL:
            return self._cartel_price(me, history, params, rng)
        # Innocent parallel and cheap talk ignore the rival's history entirely.
        return params.target_price + float(rng.normal(0.0, params.jitter_sd)), False

    def _text(self, params: ScriptedPolicyParams, punishing: bool, text_rng) -> str:
        if params.talk_prob > 0.0 and text_rng.random() < params.talk_prob:
            if punishing:
                return RETALIATION_LINES[int(text_rng.integers(len(RETALIATION_LINES)))]
            line = COLLUSIVE_LINES[int(text_rng.integers(len(COLLUSIVE_LINES)))]
            return line.format(p=params.target_price)
        return NEUTRAL_LINES[int(text_rng.integers(len(NEUTRAL_LINES)))]

    # ------------------------------------------------------------------ #
    # Trajectories and rollouts
    # ------------------------------------------------------------------ #
    def _play(
        self,
        params: ScriptedPolicyParams,
        history: list[list[float]],
        n_rounds: int,
        seed: int,
        comm_enabled: bool,
    ) -> tuple[list[list[float]], list[tuple[str, str]]]:
        rng = np.random.default_rng(seed)
        text_rng = np.random.default_rng(seed + 1_000_003)  # texts never perturb prices
        prices_out: list[list[float]] = []
        texts_out: list[tuple[str, str]] = []
        for _ in range(n_rounds):
            visible = history + prices_out
            row, texts = [], []
            for me in (0, 1):
                price, punishing = self._price(me, visible, params, rng)
                row.append(round(max(price, 0.0), 4))
                texts.append(self._text(params, punishing, text_rng) if comm_enabled else "")
            prices_out.append(row)
            texts_out.append((texts[0], texts[1]))
        return prices_out, texts_out

    def generate_trajectory(
        self,
        archetype: CaseArchetype,
        seed: int,
        rounds: int = 40,
        comm_enabled: bool = True,
    ) -> CartelTrajectory:
        params = self.draw_params(archetype, seed)
        prices, texts = self._play(params, [], rounds, seed, comm_enabled)
        records = []
        for t, (row, text) in enumerate(zip(prices, texts)):
            _demands, _profits, gains = self.market.step(row)
            records.append(RoundRecord(round_id=t, prices=tuple(row), gains=tuple(gains), texts=text))
        suffix = "" if comm_enabled else "_nocomm"
        return CartelTrajectory(
            run_id=f"scripted_{archetype.value}_seed{seed}{suffix}",
            rounds=tuple(records),
            benchmarks=self.benchmarks,
            archetype=archetype,
            source="scripted",
            policy_params=params.to_dict(),
        )

    def rollout(
        self,
        params: ScriptedPolicyParams,
        history: Sequence[Sequence[float]],
        k: int,
        seed: int,
        branch: str,
    ) -> BranchRollout:
        """Continue the game for k rounds from ``history`` (already perturbed if needed)."""
        prices, _texts = self._play(params, [list(r) for r in history], k, seed, comm_enabled=False)
        return BranchRollout(
            branch=branch,
            seed=seed,
            auditee_prices=tuple(r[0] for r in prices),
            rival_prices=tuple(r[1] for r in prices),
        )

    @staticmethod
    def params_from_trajectory(trajectory: CartelTrajectory) -> ScriptedPolicyParams:
        raw = dict(trajectory.policy_params)
        if not raw:
            raise ValueError(f"{trajectory.run_id} has no scripted policy parameters")
        raw["archetype"] = CaseArchetype(raw["archetype"])
        return ScriptedPolicyParams(**raw)
