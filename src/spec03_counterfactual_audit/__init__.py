"""Spec 03: Counterfactual checkpoint forking as an antitrust plus-factor audit.

Implements spec/03_counterfactual_replay_audit_plus_factors.md (Track 2.3, with Tracks
1.2 / 1.3) and spec/plan/plan_03_counterfactual_replay_audit.md:
    - Three evidentiary pillars per trajectory: outcome elevation E_out (Calvano
      profit-gain index), communication intent E_comm (keyword rubric, optional LLM
      judge) and counterfactual retaliation elasticity E_cf (spec 4.1)
    - Checkpoint forking at audit round t*: a control branch versus a synthetic
      rival-undercut branch, k rounds each, averaged over S rollout seeds
    - Pre-registered plus-factor decision rule (spec 4.2) calibrated on the first half
      of the trajectories and frozen before the held-out half is scored
    - Offline scripted positive control (in-process forks, $0) plus a live path that
      emits / executes the exact llm-cartel run_game.py and branch_game.py commands
"""
