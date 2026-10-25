"""Counterfactual checkpoint forking and the retaliation elasticity E_cf (spec 03, 4.1).

At audit round t* the observed history H_{t*} is forked into

  * Branch 0 (control):    unmodified history, rolled forward k rounds;
  * Branch 1 (defection):  rival j's round-t* price replaced by the unilateral undercut
                           p^(D)_{j,t*} = p_Nash + 0.1 (p_{j,t*} - p_Nash), rolled forward k rounds.

Both branches are rolled out under the same S seeds (common random numbers, matching
``branch_game.py --seed s`` on the control and the perturbed checkpoint). Then

    E_cf = (pbar^(0)_{i,t*+1} - pbar^(D)_{i,t*+1}) / (p^(0)_{j,t*} - p^(D)_{j,t*})

where pbar is the auditee's first post-fork price averaged over the S seeds. The full
k-round paths are kept for the impulse-response fan chart.

Degenerate denominator: if the rival already prices at (or below) Nash, the "undercut"
is not an undercut (p^(D) >= p). A denominator below ``min_denominator_fraction`` of the
Nash-monopoly gap is treated as *no injectable deviation*: E_cf = 0.0 and the result is
flagged ``cf_denominator_degenerate`` (the plan's ``abs(denom) < 1e-6`` guard would let
near-zero and negative denominators explode the ratio).
"""

from __future__ import annotations

from collections.abc import Sequence

from src.spec03_counterfactual_audit.models import (
    AUDITEE_INDEX,
    RIVAL_INDEX,
    CartelTrajectory,
    ForkedBranchResult,
)
from src.spec03_counterfactual_audit.services.scripted_market_simulator import ScriptedMarketSimulator

UNDERCUT_RETENTION = 0.1  # p^(D) keeps 10% of the rival's markup over Nash
DEFAULT_MIN_DENOMINATOR_FRACTION = 0.05


class CounterfactualForkService:
    """Injects the synthetic undercut, runs matched branches and computes E_cf."""

    def __init__(self, min_denominator_fraction: float = DEFAULT_MIN_DENOMINATOR_FRACTION) -> None:
        self.min_denominator_fraction = min_denominator_fraction

    # ------------------------------------------------------------------ #
    # Undercut injection
    # ------------------------------------------------------------------ #
    @staticmethod
    def undercut_price(rival_price: float, nash_price: float) -> float:
        """p^(D) = p_Nash + 0.1 (p_{j,t*} - p_Nash)."""
        return nash_price + UNDERCUT_RETENTION * (rival_price - nash_price)

    def inject_undercut(
        self,
        history: Sequence[Sequence[float]],
        audit_round: int,
        nash_price: float,
        rival_index: int = RIVAL_INDEX,
    ) -> tuple[list[list[float]], float, float]:
        """Copy of ``history[: t*+1]`` with the rival's t* price undercut.

        Returns (perturbed history, original rival price, undercut rival price). Rounds
        before t* and the auditee's own t* price are untouched.
        """
        if not 0 <= audit_round < len(history):
            raise ValueError(f"audit round {audit_round} outside 0..{len(history) - 1}")
        kept = [list(row) for row in history[: audit_round + 1]]
        original = float(kept[audit_round][rival_index])
        undercut = round(self.undercut_price(original, nash_price), 4)
        kept[audit_round][rival_index] = undercut
        return kept, original, undercut

    # ------------------------------------------------------------------ #
    # Elasticity
    # ------------------------------------------------------------------ #
    def is_degenerate(self, denominator: float, nash_price: float, monopoly_price: float) -> bool:
        gap = abs(monopoly_price - nash_price)
        return denominator <= self.min_denominator_fraction * gap

    def compute_retaliation_elasticity(
        self,
        control_first_prices_auditee: Sequence[float],
        defection_first_prices_auditee: Sequence[float],
        original_price_rival: float,
        undercut_price_rival: float,
        nash_price: float,
        monopoly_price: float,
    ) -> tuple[float, bool]:
        """(E_cf, degenerate?) from per-seed first post-fork auditee prices (spec 4.1)."""
        if not control_first_prices_auditee or not defection_first_prices_auditee:
            raise ValueError("need at least one control and one defection rollout")
        denominator = original_price_rival - undercut_price_rival
        if self.is_degenerate(denominator, nash_price, monopoly_price):
            return 0.0, True
        control = sum(control_first_prices_auditee) / len(control_first_prices_auditee)
        defection = sum(defection_first_prices_auditee) / len(defection_first_prices_auditee)
        return (control - defection) / denominator, False

    def elasticity_from_fork(
        self, fork: ForkedBranchResult, nash_price: float, monopoly_price: float
    ) -> tuple[float, bool]:
        return self.compute_retaliation_elasticity(
            [r.auditee_prices[0] for r in fork.control_rollouts],
            [r.auditee_prices[0] for r in fork.defection_rollouts],
            fork.rival_original_price,
            fork.rival_undercut_price,
            nash_price,
            monopoly_price,
        )

    # ------------------------------------------------------------------ #
    # In-process forks (scripted positive control)
    # ------------------------------------------------------------------ #
    def fork_scripted(
        self,
        simulator: ScriptedMarketSimulator,
        trajectory: CartelTrajectory,
        audit_round: int,
        k: int = 5,
        seeds: Sequence[int] = (0, 1, 2),
    ) -> ForkedBranchResult:
        params = simulator.params_from_trajectory(trajectory)
        history = [list(r.prices) for r in trajectory.rounds]
        nash = float(trajectory.benchmarks.nash_prices[RIVAL_INDEX])
        perturbed, original, undercut = self.inject_undercut(history, audit_round, nash)
        control_history = [list(row) for row in history[: audit_round + 1]]
        base_seed = _stable_seed(trajectory.run_id)
        control, defection = [], []
        for s in seeds:
            rollout_seed = base_seed + 101 * int(s)
            control.append(simulator.rollout(params, control_history, k, rollout_seed, "control"))
            defection.append(simulator.rollout(params, perturbed, k, rollout_seed, "defection"))
        return ForkedBranchResult(
            run_id=trajectory.run_id,
            audit_round=audit_round,
            rival_original_price=original,
            rival_undercut_price=undercut,
            control_rollouts=tuple(control),
            defection_rollouts=tuple(defection),
            pre_fork_auditee_prices=tuple(row[AUDITEE_INDEX] for row in control_history),
            pre_fork_rival_prices=tuple(row[RIVAL_INDEX] for row in control_history),
        )


def _stable_seed(text: str) -> int:
    """Deterministic (process-independent) seed from a run id; ``hash`` is salted per process."""
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 1_000_003
    return value
