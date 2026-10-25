"""Microstructure interventions: tick-grid quantization and order-book canonicalization.

This module is the canonical, testable implementation. The live harness
(``llm-agent-collusion/src/continuous_double_auction/market/market.py``) applies the
identical rounding rule inline (it cannot import this package - both are named ``src``
in their own environments), so any change here must be mirrored there.
"""

from __future__ import annotations

from typing import Sequence


class OrderBookFilterService:
    """Quantizes continuous ask prices and strips unexecuted ask beacons from history."""

    def __init__(self, tick_size: float | None = None, canonicalize_history: bool = False) -> None:
        if tick_size is not None and tick_size <= 0:
            raise ValueError("tick_size must be positive when provided.")
        self._tick_size = tick_size
        self._canonicalize_history = canonicalize_history

    @property
    def tick_size(self) -> float | None:
        return self._tick_size

    @property
    def canonicalize_history(self) -> bool:
        return self._canonicalize_history

    def quantize_price(self, price: float) -> float:
        """Round to the nearest tick multiple (A' = delta * round(A / delta)); passthrough at 2dp."""
        if self._tick_size is None:
            return round(price, 2)
        return round(round(price / self._tick_size) * self._tick_size, 2)

    def canonicalize_round_summary(
        self,
        round_number: int,
        seller_asks: dict[str, float],
        buyer_bids: dict[str, float] | None = None,
        trade_prices: Sequence[float] = (),
    ) -> list[str]:
        """Canonicalized history lines for one round: clearing price + volume only.

        Hides individual seller identities and the exact unexecuted ask vector so high
        asks cannot act as price-floor beacons for tacit coordination (spec 02, 4.3).
        """
        if not self._canonicalize_history:
            lines = [f"Hour {round_number}:"]
            for seller_id in sorted(seller_asks):
                lines.append(f"  {seller_id} ask ${seller_asks[seller_id]:.2f}")
            for buyer_id in sorted(buyer_bids or {}):
                lines.append(f"  {buyer_id} bid ${buyer_bids[buyer_id]:.2f}")
            return lines

        prices = list(trade_prices)
        if prices:
            clearing = sum(prices) / len(prices)
            return [f"Hour {round_number}: {len(prices)} trade(s), clearing price ${clearing:.2f}"]
        return [f"Hour {round_number}: no trades"]
