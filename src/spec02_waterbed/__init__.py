"""Spec 02: the collusion waterbed effect and action-space tick quantization.

Implements spec/02_waterbed_effect_and_tick_quantization.md (Tracks 3.4 / 3.1 / 1.5):
    - OrderBookFilterService: pure tick-grid quantization + order-book history
      canonicalization (the reference implementation; the live double-auction harness
      applies the same two rules inline in market.py, see the [MODIFY] patches).
    - WaterbedTelemetryService: C_text / C_act / channel-shift-ratio measurement that
      formalizes "text coordination falls while action dependence rises".
"""
