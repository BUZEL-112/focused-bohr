"""Spec 05: algorithmic vaccines - a no-regret ZIP seller and a honey-trap seller in the CDA.

Implements spec/05_algorithmic_vaccine_no_regret_honeypot.md (Tracks 1.5 / 3.2) following
spec/plan/plan_05_algorithmic_vaccine.md:
    - ZIPSellerPolicy / HoneyTrapPolicy: the pure reference seller policies (section 4.1 margin
      update; section 3.2 bait-then-betray state machine with DIRECT / INDIRECT / BENIGN framing).
      The live double-auction harness mirrors them in
      ``llm-agent-collusion/src/continuous_double_auction/market/vaccine_sellers.py``.
    - ReciprocityEvaluator: VRR_F, BRS_F and the post-betrayal price-collapse elasticity
      (section 4.2), scored on plain per-round asks/messages.
    - ScriptedVaccineMarket: a $0 in-process scripted CDA (scripted cartel sellers + adaptive
      ZIP buyers) that exercises the full pipeline offline. It is a simulation, not LLM evidence.
"""
