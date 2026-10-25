"""Spec 06: Monitor-agent same-family collusion and symbolic say-do trace scrubbing.

Implements spec/06_monitor_family_collusion_and_symbolic_scrubbing.md
(Track 2.2 primary, Track 3.3 + literature Open Gap #2 secondary):
    - Say-do round records from llm-trust-breaking 3-stage logs, colludebench
      pricing transcripts, and a scripted family-styled trajectory generator
    - Four Track 2.2 monitor views (messages only / actions only / raw hybrid /
      symbolic scrubbed say-do report with D_self, D_cross, V_exc indicators)
    - 3x3 agent-family x monitor-family suspicion grid, Familial Leniency Bias
      beta_self(v) and per-view AUROC (spec section 4.1)
"""
