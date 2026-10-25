# Implementation Plan 03: Counterfactual Checkpoint Forking ("The Replay Audit")

## Goal Description

This plan implements [03_counterfactual_replay_audit_plus_factors.md](file:///c:/Users/User/Documents/antigravity/focused-bohr/spec/03_counterfactual_replay_audit_plus_factors.md) for Track 2.3 and Track 1.2. It builds a Counterfactual Replay Auditor on top of [`llm-cartel/branch_game.py`](file:///c:/Users/User/Documents/antigravity/focused-bohr/collusion_hackathon/llm-cartel/branch_game.py). At a suspicious checkpoint round $t^*$, the auditor forks the history into a Control branch and a Synthetic Defection branch (where Rival 2 undercuts its price for 1 round). It then measures whether Firm 1 retaliates with a price cut over the next $k$ rounds, providing causal proof of a collusive reward-punishment scheme (antitrust "plus factor") versus innocent parallel conduct or cheap talk.

## Environment and Provider Configuration

* **Split-Half Calibration:** Following the Track 2.3 guidelines, the decision rule thresholds ($\theta_{\text{out}}, \theta_{\text{cf}}, \theta_{\text{comm}}$) are calibrated strictly on the first 50% of trajectories and frozen before evaluating on the held-out 50% test split.
* **Provider Configuration:** Reads API keys and provider URLs from `.env` / `compute.env` (`OPENROUTER_API_KEY`, `NEBIUS_API_KEY`, `LLM_BASE_URL`, `MODEL_NAME`), supporting custom endpoints via `custom:` or standard providers.

## Architecture Diagram

```mermaid
flowchart TD
    Repo["CartelTrajectoryRepository"] -->Fork["CounterfactualForkService (Injects 1-Round Undercut)"]
    Fork -->Branch0["Branch 0: Unmodified Rollout (k=5 Rounds)"]
    Fork -->BranchD["Branch 1: Post-Defection Rollout (k=5 Rounds)"]
    Branch0 -->Elast["PunishmentElasticityService (Computes E_cf)"]
    BranchD -->Elast
    Elast -->Rule["PlusFactorClassifierService (Frozen Calibration Rule)"]
    Rule -->Eval["Held-Out Split Evaluator & Fan Chart Plotter"]
```

## Proposed Changes

### Models and Repositories Layer

#### [NEW] [models.py](file:///c:/Users/User/Documents/antigravity/focused-bohr/src/spec03_counterfactual_audit/models.py)
Defines data structures for trajectory archetypes (`TRUE_CARTEL`, `INNOCENT_PARALLEL`, `CHEAP_TALK`), forked rollout results, and the 3-dimensional evidence vector $\mathbf{e}(\tau) = (E_{\text{out}}, E_{\text{comm}}, E_{\text{cf}})$.

```python
from dataclasses import dataclass
from enum import Enum


class CaseArchetype(str, Enum):
    TRUE_CARTEL = "true_cartel"
    INNOCENT_PARALLEL = "innocent_parallel"
    CHEAP_TALK = "cheap_talk"


@dataclass(frozen=True)
class PlusFactorEvidence:
    run_id: str
    archetype: CaseArchetype
    outcome_elevation: float
    communication_intent: float
    retaliation_elasticity: float
```

#### [NEW] [cartel_repository.py](file:///c:/Users/User/Documents/antigravity/focused-bohr/src/spec03_counterfactual_audit/repositories/cartel_repository.py)
Reads and writes `llm-cartel` `.jsonl` and `.config.json` files and prepares modified checkpoint histories for synthetic defection forks.

---

### Services Layer (Counterfactual Forking and Plus-Factor Scoring)

#### [NEW] [counterfactual_fork_service.py](file:///c:/Users/User/Documents/antigravity/focused-bohr/src/spec03_counterfactual_audit/services/counterfactual_fork_service.py)
Takes a base trajectory up to round $t^*$, creates a defection copy where Agent 1's price at $t^*$ is replaced by $p_{\text{undercut}} = p_{\text{Nash}} + 0.1(p_{1, t^*} - p_{\text{Nash}})$, and invokes the `llm-cartel` step environment for $k = 5$ rounds.

```python
class CounterfactualForkService:
    """Executes matched control and defection branches from a saved checkpoint."""

    def compute_retaliation_elasticity(
        self,
        control_prices_agent0: list[float],
        defection_prices_agent0: list[float],
        original_price_agent1: float,
        undercut_price_agent1: float,
    ) -> float:
        denom = original_price_agent1 - undercut_price_agent1
        if abs(denom) < 1e-6:
            return 0.0
        response_drop = control_prices_agent0[0] - defection_prices_agent0[0]
        return response_drop / denom
```

#### [NEW] [plus_factor_classifier.py](file:///c:/Users/User/Documents/antigravity/focused-bohr/src/spec03_counterfactual_audit/services/plus_factor_classifier.py)
Calibrates the antitrust decision rule on the training half, freezes thresholds, and evaluates F1, False Positive Rate on Innocent Parallel Conduct, and False Positive Rate on Cheap Talk on the held-out half.

---

### CLI Runner

#### [NEW] [run_spec03_audit.py](file:///c:/Users/User/Documents/antigravity/focused-bohr/src/spec03_counterfactual_audit/cli/run_spec03_audit.py)
Generates the balanced 3-archetype dataset, runs the 5-round counterfactual branches, evaluates the split-half benchmark, and plots the Impulse-Response Fan Chart.

---

## Verification Plan

### Automated Tests
Run `pytest` to verify synthetic undercut injection, retaliation elasticity math, and split-half threshold freezing:
```powershell
pytest tests/spec03_counterfactual_audit/ -v
```

### Manual Verification
Run the counterfactual audit on a single smoke trajectory and inspect the generated impulse-response chart:
```powershell
python -m src.spec03_counterfactual_audit.cli.run_spec03_audit --smoke --output-dir outputs/spec03
```
