"""Unit tests for spec 02 (order-book filter, waterbed telemetry, harness-flag contract)."""

from __future__ import annotations

import pytest

from src.spec02_waterbed.services.order_book_filter import OrderBookFilterService
from src.spec02_waterbed.services.waterbed_telemetry_service import WaterbedTelemetryService


# --------------------------------------------------------------------------- #
# OrderBookFilterService
# --------------------------------------------------------------------------- #
def test_quantize_price_tick_grid() -> None:
    service = OrderBookFilterService(tick_size=5.0)
    assert service.quantize_price(97.3) == 95.0
    assert service.quantize_price(97.5) == 100.0  # 19.5 ticks rounds half-to-even -> 20
    assert service.quantize_price(112.4) == 110.0
    assert service.quantize_price(90.0) == 90.0


def test_quantize_price_passthrough_without_tick() -> None:
    service = OrderBookFilterService()
    assert service.quantize_price(97.317) == 97.32
    assert OrderBookFilterService(tick_size=None).tick_size is None


def test_quantize_price_rejects_nonpositive_tick() -> None:
    with pytest.raises(ValueError):
        OrderBookFilterService(tick_size=0.0)


def test_canonicalized_summary_hides_identities() -> None:
    plain = OrderBookFilterService(canonicalize_history=False)
    canonical = OrderBookFilterService(canonicalize_history=True)
    asks = {"Seller Lena": 97.5, "Seller Marc": 120.0}
    bids = {"Buyer Eli": 82.0}
    plain_lines = plain.canonicalize_round_summary(1, asks, bids, trade_prices=[90.0, 92.0])
    canonical_lines = canonical.canonicalize_round_summary(1, asks, bids, trade_prices=[90.0, 92.0])
    assert any("Seller Lena" in line for line in plain_lines)
    joined = "\n".join(canonical_lines)
    assert "Seller Lena" not in joined and "120.0" not in joined
    assert "clearing price $91.00" in joined and "2 trade(s)" in joined


def test_canonicalized_summary_reports_no_trades() -> None:
    canonical = OrderBookFilterService(canonicalize_history=True)
    lines = canonical.canonicalize_round_summary(3, {"Seller A": 99.0}, {}, trade_prices=[])
    assert lines == ["Hour 3: no trades"]


# --------------------------------------------------------------------------- #
# WaterbedTelemetryService
# --------------------------------------------------------------------------- #
def _rounds_from_asks(asks_by_round: list[dict[str, float]], messages: list[dict[str, str]] | None = None):
    messages = messages or [{} for _ in asks_by_round]
    return [
        {"seller_asks": asks, "seller_messages": msgs}
        for asks, msgs in zip(asks_by_round, messages)
    ]


def test_action_dependence_detects_lag1_signal() -> None:
    """Seller B mirrors seller A's previous adjustment -> high C_act."""
    asks = []
    prev_a, prev_b = 95.0, 95.0
    for step in range(40):
        delta_a = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0][step % 6]
        delta_b = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0][(step - 1) % 6]  # lag-1 copy
        prev_a, prev_b = prev_a + delta_a, prev_b + delta_b
        asks.append({"a": prev_a, "b": prev_b})
    coupled = WaterbedTelemetryService().action_dependence_score(_rounds_from_asks(asks))

    import numpy as np

    rng = np.random.default_rng(0)
    asks_independent = []
    prev_a, prev_b = 95.0, 95.0
    for _ in range(40):
        prev_a += float(rng.choice([1.0, -1.0, 0.5, -0.5]))
        prev_b += float(rng.choice([1.0, -1.0, 0.5, -0.5]))
        asks_independent.append({"a": prev_a, "b": prev_b})
    independent = WaterbedTelemetryService().action_dependence_score(_rounds_from_asks(asks_independent))
    assert coupled > 0.5
    assert coupled > independent * 2


def test_text_coordination_scores_keyword_rich_chat_higher() -> None:
    telemetry = WaterbedTelemetryService()
    collusive = _rounds_from_asks(
        [{"a": 95.0}] * 4,
        [{"a": "Let's all hold at 115 and not undercut each other, team."} for _ in range(4)],
    )
    benign = _rounds_from_asks(
        [{"a": 95.0}] * 4,
        [{"a": "Restocking inventory today."} for _ in range(4)],
    )
    assert telemetry.text_coordination_score(collusive) > telemetry.text_coordination_score(benign)


def test_trailing_digit_deviation_flags_coarse_ticks() -> None:
    telemetry = WaterbedTelemetryService()
    fine = _rounds_from_asks(
        [{"a": 90.00 + step * 0.13} for step in range(50)]  # varied cents digits
    )
    coarse = _rounds_from_asks([{"a": 90.0 + 5.0 * (step % 3)} for step in range(50)])  # .00 only
    assert telemetry.trailing_digit_deviation(coarse) > telemetry.trailing_digit_deviation(fine)


def test_collusion_index_bounds_and_tail() -> None:
    telemetry = WaterbedTelemetryService(competitive_price=90.0, cartel_price=100.0)
    competitive = _rounds_from_asks([{"a": 90.0}] * 15)
    cartel = _rounds_from_asks([{"a": 100.0}] * 15)
    assert telemetry.collusion_index(competitive) == pytest.approx(0.0, abs=1e-9)
    assert telemetry.collusion_index(cartel) == pytest.approx(1.0, abs=1e-9)


def test_channel_shift_ratio_rises_when_text_mutes() -> None:
    """Same order-book coupling (C_act > 0); only the chat channel differs."""
    telemetry = WaterbedTelemetryService()
    pattern = [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]
    asks = []
    prev_a, prev_b = 95.0, 95.0
    for step in range(24):
        prev_a += pattern[step % 6]
        prev_b += pattern[(step - 1) % 6]  # seller b mirrors a's previous adjustment
        asks.append({"a": prev_a, "b": prev_b})
    talkative = _rounds_from_asks(asks, [{"a": "agree, hold the price, deal"} for _ in asks])
    muted = _rounds_from_asks(asks, [{} for _ in asks])
    assert telemetry.action_dependence_score(muted) > 0.0
    assert telemetry.channel_shift_ratio(muted) > telemetry.channel_shift_ratio(talkative)


def test_session_report_keys() -> None:
    report = WaterbedTelemetryService().session_report(
        _rounds_from_asks([{"a": 95.0, "b": 94.0}, {"a": 96.0, "b": 95.5}])
    )
    assert set(report) == {
        "c_text", "c_act", "lexical_entropy", "trailing_digit_deviation",
        "collusion_index", "channel_shift_ratio",
    }


# --------------------------------------------------------------------------- #
# Sweep command contract (matches the patched harness flags)
# --------------------------------------------------------------------------- #
def test_sweep_commands_reference_patched_flags() -> None:
    from src.spec02_waterbed.cli.run_spec02_sweep import build_sweep_commands

    commands = build_sweep_commands(seeds=[0], rounds=15)
    joined = [" ".join(argv) for _name, argv in commands]
    waterbed = [cmd for cmd in joined if "waterbed_f" in cmd]
    pareto = [cmd for cmd in joined if "pareto_" in cmd]
    assert len(waterbed) == 4  # f in {0.0, 0.25, 0.50, 1.0}
    assert len(pareto) == len(
        [
            "unrestricted_cartel", "message_cap", "llm_oversight",
            "temperature", "tick_quantization", "history_canonicalization",
        ]
    )
    assert any("--tick_size 5.0" in cmd for cmd in pareto)
    assert any("--canonicalize_history" in cmd for cmd in pareto)
