"""llm-cartel run / branch file access for the spec 03 audit.

Real on-disk format (from ``collusion_hackathon/llm-cartel/run_game.py`` / ``branch_game.py``
/ ``harness/io.py``; field names copied verbatim, including the paper's ``respones`` typo):

* ``<run>.jsonl`` - one JSON object per round:
  ``{round, actions, thoughts, respones, demands, profits, gains, metrics}``; branch files
  add ``branch_phase`` ("before" | "after"). ``actions`` / ``gains`` are per-agent lists in
  agent-index order; ``gains`` is the Calvano index (0 = Nash, 1 = monopoly profit).
* ``<run>.config.json`` (same stem, same directory) - ``agents, prefixes, models, window,
  temperature, think, seed, rounds, prior_steps, prior_actions, nash_prices,
  monopoly_prices, a0, ais, costs, mu, price_scale`` (+ ``summary``); branch configs add
  ``branch_of, branch_round, branch_label, ...``.

The undercut checkpoint written here is a normal run file truncated at t* whose round-t*
rival action is replaced by the synthetic undercut (demands / profits / gains of that one
record recomputed with the same logit formula). ``branch_game.py --at_round t*`` replays
it through the environment, so the agents see the undercut in their next-round prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.spec03_counterfactual_audit.models import (
    AUDITEE_INDEX,
    RIVAL_INDEX,
    BranchRollout,
    CartelTrajectory,
    CaseArchetype,
    MarketBenchmarks,
    RoundRecord,
)
from src.spec03_counterfactual_audit.services.counterfactual_fork_service import CounterfactualForkService
from src.spec03_counterfactual_audit.services.logit_market_service import LogitMarketService

REQUIRED_RECORD_FIELDS = ("round", "actions", "gains")
SPEC03_RUN_PREFIX = "spec03_"
_ARCHETYPE_RE = re.compile(r"spec03_(true_cartel|innocent_parallel|cheap_talk)_")


# --------------------------------------------------------------------------- #
# Low-level io (mirrors harness/io.py)
# --------------------------------------------------------------------------- #
def config_path(run_path: str | Path) -> Path:
    p = Path(run_path)
    return p.with_name(p.stem + ".config.json")


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_config(run_path: str | Path) -> dict:
    with open(config_path(run_path), encoding="utf-8") as handle:
        return json.load(handle)


def _write_run(path: Path, records: list[dict], config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    with open(config_path(path), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)


def branch_output_path(run_path: str | Path, branch_round: int, label: str, out_dir: str | Path) -> Path:
    """Where ``branch_game.py`` writes a branch (its own naming rule, reproduced)."""
    safe = label.replace("/", "-").replace(":", "-").replace(".", "p")
    return Path(out_dir) / f"{Path(run_path).stem}__r{branch_round}__{safe}.jsonl"


def infer_archetype(run_id: str) -> CaseArchetype | None:
    """Ground truth only for runs this package launched (``spec03_<archetype>_...``)."""
    match = _ARCHETYPE_RE.search(run_id)
    return CaseArchetype(match.group(1)) if match else None


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #
def load_run(run_path: str | Path, archetype: CaseArchetype | None = None) -> CartelTrajectory:
    """Parse one llm-cartel run (``.jsonl`` + sibling ``.config.json``)."""
    run_path = Path(run_path)
    records = read_jsonl(run_path)
    config = load_config(run_path)
    if config.get("agents", 2) != 2:
        raise ValueError(f"{run_path.name}: spec 03 audits duopolies only (agents={config.get('agents')})")
    rounds = []
    for record in records:
        missing = [f for f in REQUIRED_RECORD_FIELDS if f not in record]
        if missing:
            raise ValueError(f"{run_path.name}: record missing fields {missing}")
        thoughts = record.get("thoughts") or []
        responses = record.get("respones") or []
        texts = tuple(
            (thoughts[i] if i < len(thoughts) and thoughts[i] else (responses[i] if i < len(responses) else "")) or ""
            for i in range(len(record["actions"]))
        )
        rounds.append(
            RoundRecord(
                round_id=int(record["round"]),
                prices=tuple(float(x) for x in record["actions"]),
                gains=tuple(float(x) for x in record["gains"]),
                texts=texts,
            )
        )
    benchmarks = MarketBenchmarks(
        nash_prices=tuple(float(x) for x in config["nash_prices"]),
        monopoly_prices=tuple(float(x) for x in config["monopoly_prices"]),
    )
    return CartelTrajectory(
        run_id=run_path.stem,
        rounds=tuple(rounds),
        benchmarks=benchmarks,
        archetype=archetype if archetype is not None else infer_archetype(run_path.stem),
        source="llm_cartel",
        policy_params={k: config.get(k) for k in ("prefixes", "models", "window", "think", "seed")},
    )


def list_run_files(directory: str | Path) -> list[Path]:
    """Base run files (jsonl with a sibling config that is not itself a branch)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    runs = []
    for path in sorted(directory.glob("*.jsonl")):
        if not config_path(path).is_file():
            continue
        try:
            config = load_config(path)
        except (json.JSONDecodeError, OSError):
            continue
        if "branch_of" in config or "spec03_perturbation" in config:
            continue
        runs.append(path)
    return runs


# --------------------------------------------------------------------------- #
# Checkpoints and branches
# --------------------------------------------------------------------------- #
def write_undercut_checkpoint(
    run_path: str | Path,
    audit_round: int,
    out_dir: str | Path,
    fork_service: CounterfactualForkService | None = None,
    rival_index: int = RIVAL_INDEX,
) -> tuple[Path, float, float]:
    """Write ``<stem>__undercut_r<t*>.jsonl`` (+ config); returns (path, p_j, p_j^(D))."""
    fork_service = fork_service or CounterfactualForkService()
    run_path = Path(run_path)
    records = read_jsonl(run_path)
    config = load_config(run_path)
    if not 0 <= audit_round < len(records):
        raise ValueError(f"audit round {audit_round} outside 0..{len(records) - 1}")
    history = [r["actions"] for r in records]
    nash = float(config["nash_prices"][rival_index])
    perturbed, original, undercut = fork_service.inject_undercut(history, audit_round, nash, rival_index)

    kept = [dict(r) for r in records[: audit_round + 1]]
    market = LogitMarketService.from_config(config)
    demands, profits, gains = market.step(perturbed[audit_round])
    kept[audit_round].update(
        {"actions": perturbed[audit_round], "demands": demands, "profits": profits, "gains": gains}
    )
    new_config = {
        **config,
        "spec03_perturbation": {
            "base_run": str(run_path),
            "audit_round": audit_round,
            "rival_index": rival_index,
            "rival_original_price": original,
            "rival_undercut_price": undercut,
        },
    }
    out_path = Path(out_dir) / f"{run_path.stem}__undercut_r{audit_round}.jsonl"
    _write_run(out_path, kept, new_config)
    return out_path, original, undercut


def load_branch_rollout(branch_path: str | Path, branch: str, seed: int, k: int | None = None) -> BranchRollout:
    """Post-fork ("after") rounds of a ``branch_game.py`` output as a :class:`BranchRollout`."""
    after = [r for r in read_jsonl(branch_path) if r.get("branch_phase") == "after"]
    after.sort(key=lambda r: r["round"])
    if k is not None:
        after = after[:k]
    if not after:
        raise ValueError(f"{Path(branch_path).name}: no post-branch rounds")
    return BranchRollout(
        branch=branch,
        seed=seed,
        auditee_prices=tuple(float(r["actions"][AUDITEE_INDEX]) for r in after),
        rival_prices=tuple(float(r["actions"][RIVAL_INDEX]) for r in after),
    )
