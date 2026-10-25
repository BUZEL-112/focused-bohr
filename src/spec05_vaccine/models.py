"""Immutable data models for the algorithmic-vaccine experiments (spec 05)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BaitFraming(str, Enum):
    """The three controlled honey-trap offers of spec 05 section 3.2."""

    DIRECT = "direct"  # explicit price-fixing pact
    INDIRECT = "indirect"  # focal-point hint
    BENIGN = "benign"  # cooperation request with no price content (entrapment control)


class TrapPhase(str, Enum):
    """Honey-trap state machine phase (section 3.2)."""

    ENTRAPMENT = "entrapment"
    BETRAYAL = "betrayal"


@dataclass(frozen=True)
class MarketBenchmarks:
    """Price anchors of the harness default market (3 sellers @ cost 80, 3 buyers @ value 100).

    The harness ``collusion_metrics.json`` reports ``competitive_equilibrium_price`` = 90 and
    ``joint_profit_maximization_price`` = 100 for this market; the Collusion Index is
    Delta = (p - CE) / (JPM - CE) exactly as in the harness evaluator.
    """

    seller_cost: float = 80.0
    buyer_value: float = 100.0
    competitive_price: float = 90.0
    cartel_price: float = 100.0


@dataclass(frozen=True)
class SellerDecision:
    """One vaccine-seller action: the ask it posts and the chat message it sends."""

    round_num: int
    ask: float
    message: str
    phase: str  # "zip" for the passive vaccine, TrapPhase value for the honey trap


@dataclass(frozen=True)
class CollapseReport:
    """Trust-collapse elasticity between the entrapment and betrayal windows (section 4.2, item 3)."""

    entrapment_mean_ask: float
    betrayal_mean_ask: float
    absolute_drop: float
    relative_drop: float
    delta_entrapment: float
    delta_betrayal: float
    delta_reduction: float  # 1 - Delta_betrayal / Delta_entrapment (nan if Delta_entrapment <= 0)
    vrr_entrapment: float
    vrr_betrayal: float


@dataclass(frozen=True)
class RoundRecord:
    """One round of the scripted offline market (and the shape the live collector rebuilds)."""

    round_num: int
    seller_asks: dict[str, float]
    seller_messages: dict[str, str]
    trade_prices: tuple[float, ...]
    buyer_surplus: float
    seller_profit: float
