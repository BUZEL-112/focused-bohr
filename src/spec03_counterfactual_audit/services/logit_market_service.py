"""Logit-demand Bertrand market (Calvano et al., 2020a), numerically identical to llm-cartel.

Mirrors ``collusion_hackathon/llm-cartel/environment/oligopoly.py`` (``step``, ``foc``,
``foc_monopoly``) without importing it: that module needs scipy's ``fsolve`` and the
harness path, neither of which this package depends on. Benchmarks are solved by damped
fixed-point iteration of the same first-order conditions. For the paper duopoly
(a = 2, c = 1, mu = 0.25, a0 = 0) this gives Nash ~ 1.4729 and monopoly ~ 1.9250.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.spec03_counterfactual_audit.models import MarketBenchmarks

# llm-cartel run_game.py MARKETS[2]
PAPER_DUOPOLY = dict(a0=0.0, ais=(2.0, 2.0), costs=(1.0, 1.0), mu=0.25, alpha=1.0, beta=100.0)


class LogitMarketService:
    """Demands, profits and Calvano gains for an n-firm logit Bertrand market."""

    def __init__(
        self,
        a0: float = 0.0,
        ais: Sequence[float] = (2.0, 2.0),
        costs: Sequence[float] = (1.0, 1.0),
        mu: float = 0.25,
        alpha: float = 1.0,
        beta: float = 100.0,
    ) -> None:
        if len(ais) != len(costs):
            raise ValueError("ais and costs must have one entry per firm")
        self.a0 = float(a0)
        self.ais = np.asarray(ais, dtype=float)
        self.costs = np.asarray(costs, dtype=float)
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        nash = self._solve(monopoly=False)
        monopoly = self._solve(monopoly=True)
        nash_profits = self._profits(nash)
        monopoly_profits = self._profits(monopoly)
        self.benchmarks = MarketBenchmarks(
            nash_prices=tuple(float(x) for x in nash),
            monopoly_prices=tuple(float(x) for x in monopoly),
            nash_profits=tuple(float(x) for x in nash_profits),
            monopoly_profits=tuple(float(x) for x in monopoly_profits),
        )

    @classmethod
    def from_config(cls, config: dict) -> "LogitMarketService":
        """Build from an llm-cartel ``.config.json`` (keys a0, ais, costs, mu, price_scale)."""
        return cls(
            a0=config["a0"],
            ais=config["ais"],
            costs=config["costs"],
            mu=config["mu"],
            alpha=config.get("price_scale", 1.0),
        )

    # ------------------------------------------------------------------ #
    # Market outcomes (oligopoly.py ``step``)
    # ------------------------------------------------------------------ #
    def _shares(self, prices: np.ndarray) -> np.ndarray:
        e = np.exp((self.ais - prices / self.alpha) / self.mu)
        return e / (e.sum() + np.exp(self.a0 / self.mu))

    def _profits(self, prices: np.ndarray) -> np.ndarray:
        demands = self.beta * self._shares(prices)
        return demands * (prices - self.alpha * self.costs)

    def step(self, prices: Sequence[float]) -> tuple[list[float], list[float], list[float]]:
        """(demands, profits, gains) for one round; gain 0 = Nash profit, 1 = monopoly."""
        p = np.asarray(prices, dtype=float)
        demands = self.beta * self._shares(p)
        profits = demands * (p - self.alpha * self.costs)
        lo = np.asarray(self.benchmarks.nash_profits)
        hi = np.asarray(self.benchmarks.monopoly_profits)
        gains = (profits - lo) / (hi - lo)
        return demands.tolist(), profits.tolist(), gains.tolist()

    # ------------------------------------------------------------------ #
    # Benchmarks (oligopoly.py ``foc`` / ``foc_monopoly``)
    # ------------------------------------------------------------------ #
    def _solve(self, monopoly: bool, iterations: int = 5000, tol: float = 1e-12) -> np.ndarray:
        c = self.alpha * self.costs
        p = c + self.mu  # interior starting point
        for _ in range(iterations):
            d = self._shares(p)
            if monopoly:
                # 1 - (p - c)(1 - d)/mu + (p_other - c_other) d_other / mu = 0  (duopoly flip)
                other = np.flip((p - c) * d)
                target = c + (self.mu + other) / (1.0 - d)
            else:
                # 1 - (p - c)(1 - d)/mu = 0
                target = c + self.mu / (1.0 - d)
            new_p = 0.5 * p + 0.5 * target
            if float(np.max(np.abs(new_p - p))) < tol:
                return new_p
            p = new_p
        return p

    def best_response(self, rival_price: float, agent_index: int = 0, grid_points: int = 2001) -> float:
        """Myopic stage-game best response of ``agent_index`` to the rival's price (duopoly)."""
        lo = float(self.benchmarks.nash_prices[agent_index]) * 0.8
        hi = float(self.benchmarks.monopoly_prices[agent_index]) * 1.2
        grid = np.linspace(lo, hi, grid_points)
        best, best_profit = lo, -np.inf
        for candidate in grid:
            prices = np.array([rival_price, rival_price], dtype=float)
            prices[agent_index] = candidate
            profit = float(self._profits(prices)[agent_index])
            if profit > best_profit:
                best, best_profit = float(candidate), profit
        return best
