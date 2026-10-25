"""Spec 01: Information-theoretic collusion detection via Pairwise Net Influence and residual NCD.

Implements spec/01_pairwise_net_influence_and_ncd.md (Tracks 1.4 / 1.3):
    - Market-state residualization of raw actions (Stage 1)
    - First-order Markov conditional directed information with Miller-Madow
      correction -> Pairwise Net Influence (Bonjour et al., 2022) (Stage 2A)
    - Conditional Normalized Compression Distance on interleaved
      residual-and-message streams via zlib (Stage 2B)
    - Coalition graph pair ranking with a hybrid z-scored score (Stage 3)
"""
