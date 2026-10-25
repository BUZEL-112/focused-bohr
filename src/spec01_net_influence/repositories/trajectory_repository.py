"""Trajectory repository for the spec 01 audit.

Three sources, all returning :class:`RunRecord`:

1. ``generate_scripted_run`` - the 4-player scripted positive controls of spec 01
   phase 1 (Condition A independent / Condition B grim-trigger cartel / Condition C
   bid-rotation cartel), $0 cost, fully deterministic given a seed.
2. ``load_colludebench_jsonl`` - steganographic-collusion transcripts
   (``{run_id, round_id, agent_id, action, message, reward}`` per line, runs appended
   back-to-back; split whenever ``round_id`` restarts). Location decides the label:
   ``calibration/`` and ``baseline/`` trees (and the sweeps ``baseline`` variant) are
   honest controls, everything else (``audit/``, ``sweeps/``, ``stress/``) is colluding.
3. ``load_double_auction_dir`` - one llm-agent-collusion experiment directory
   (``auction_results.md`` round blocks + ``collusion_metrics.json`` benchmarks).
   Runs with seller communication enabled are labelled collusive, plain ``base`` runs
   competitive.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from src.spec01_net_influence.models import AgentTurnRecord, RunRecord

HONEST_MARKERS = ("calibration", "baseline")
COLLUSIVE_MARKERS = ("audit", "sweeps", "stress")

_PRICE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
_MD_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


# --------------------------------------------------------------------------- #
# 1. Scripted positive controls
# --------------------------------------------------------------------------- #
def generate_scripted_run(
    condition: str,
    rounds: int = 30,
    seed: int = 0,
    n_agents: int = 4,
    colluders: tuple[str, str] = ("agent_1", "agent_2"),
) -> RunRecord:
    """Scripted 4-player pricing game on 5 price levels (spec 01, section 5 phase 1).

    The public state is a demand shock s_t; the competitive benchmark is 90 + shock.
    Agents act in levels 0..4 around the benchmark (level 2 = competitive). All agents
    share the shock, so residuals isolate *strategy*, not market conditions.

    Conditions:
      * ``independent``  - noisy best response, i.i.d. across agents (parallel conduct).
      * ``grim_trigger`` - agents 1+2 sustain the monopoly level with realistic jitter and
        occasional one-shot defections punished next round (joint state machine).
      * ``bid_rotation`` - agents 1+2 alternate strong undercuts; their time-averaged
        price equals the honest agents' (the Xu et al. 2026a price-blind regime).
    """
    if condition not in {"independent", "grim_trigger", "bid_rotation"}:
        raise ValueError(f"unknown condition: {condition!r}")
    rng = np.random.default_rng(seed)
    shocks = rng.integers(0, 3, size=rounds)  # public demand state per round

    turns: list[AgentTurnRecord] = []
    punishment_due: dict[str, int] = {}
    for t in range(rounds):
        benchmark = 90.0 + float(shocks[t])
        levels = _honest_levels(rng, n_agents)
        if condition == "grim_trigger":
            for name in colluders:
                index = int(name.split("_")[-1]) - 1
                if punishment_due.get(name, -1) == t:
                    levels[index] = 0  # retaliation round
                    punishment_due[name] = -1
                elif rng.random() < 0.08:
                    levels[index] = 1  # one-shot defection...
                    partner = colluders[1] if name == colluders[0] else colluders[0]
                    punishment_due[partner] = t + 1  # ...punished next round
                else:
                    levels[index] = 4 if rng.random() > 0.1 else 3  # collusive jitter
        elif condition == "bid_rotation":
            leader_is_1 = t % 2 == 0
            # Strong undercut (0) / strong markup (4) average to the honest mean level 2.
            level_leader, level_follower = (0, 4) if leader_is_1 else (4, 0)
            levels[int(colluders[0].split("_")[-1]) - 1] = level_leader
            levels[int(colluders[1].split("_")[-1]) - 1] = level_follower

        for index in range(n_agents):
            name = f"agent_{index + 1}"
            price = benchmark + (levels[index] - 2) * 2.0
            turns.append(
                AgentTurnRecord(
                    round_id=t + 1,
                    agent_id=name,
                    action_value=float(price),
                    message_text="",
                    market_benchmark=benchmark,
                )
            )

    is_collusive = condition != "independent"
    return RunRecord(
        run_id=f"scripted_{condition}_seed{seed}",
        is_collusive=is_collusive,
        colluding_pair=colluders if is_collusive else None,
        turns=tuple(turns),
        source="scripted",
    )


def _honest_levels(rng: np.random.Generator, n_agents: int) -> list[int]:
    """Independent noisy best response: mean level 2 with sd 0.8, clipped to the grid."""
    return [int(np.clip(round(float(rng.normal(2.0, 0.8))), 0, 4)) for _ in range(n_agents)]


# --------------------------------------------------------------------------- #
# 2. ColludeBench (steganographic-collusion) transcripts
# --------------------------------------------------------------------------- #
def load_colludebench_jsonl(file_path: Path) -> list[RunRecord]:
    """Parse one transcript JSONL into per-run records, splitting appended runs.

    A new run starts whenever ``round_id`` goes backwards or a (round, agent) turn repeats;
    the logger of the source repo appends 2-48 runs back-to-back in most shipped files.
    """
    runs: list[RunRecord] = []
    current: dict[str, AgentTurnRecord] = {}
    previous_round = 0
    file_index = 0

    def flush() -> None:
        nonlocal current, file_index
        if current:
            file_index += 1
            ordered = sorted(current.values(), key=lambda turn: (turn.round_id, turn.agent_id))
            run_id = f"{file_path.stem}__part{file_index}"
            runs.append(_build_run(run_id, ordered, file_path))
            current = {}

    with open(file_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            round_id = int(record.get("round_id", 0))
            agent_id = str(record.get("agent_id") or record.get("role") or "agent")
            key = (round_id, agent_id)
            # Several agents share a round_id, so only a backwards step (or a repeated
            # turn, which also catches appended single-round runs) marks a new run.
            if round_id < previous_round or key in current:
                flush()
                previous_round = 0
            previous_round = max(previous_round, round_id)
            action = _parse_action(record.get("action"))
            if action is None:
                continue
            current[key] = AgentTurnRecord(
                round_id=round_id,
                agent_id=agent_id,
                action_value=action,
                message_text=str(record.get("message") or ""),
                market_benchmark=0.0,  # filled per-run below (cross-sectional median)
            )
    flush()
    return runs


def _build_run(run_id: str, ordered_turns: list[AgentTurnRecord], file_path: Path) -> RunRecord:
    """Attach the per-round cross-sectional median benchmark and the location label."""
    by_round: dict[int, list[float]] = {}
    for turn in ordered_turns:
        by_round.setdefault(turn.round_id, []).append(turn.action_value)
    benchmarks = {round_id: float(np.median(values)) for round_id, values in by_round.items()}
    turns = tuple(
        AgentTurnRecord(
            round_id=turn.round_id,
            agent_id=turn.agent_id,
            action_value=turn.action_value,
            message_text=turn.message_text,
            market_benchmark=benchmarks[turn.round_id],
        )
        for turn in ordered_turns
    )
    parts = {p.lower() for p in file_path.parts}
    is_honest = any(marker in parts for marker in HONEST_MARKERS) and not any(
        marker in parts for marker in ("audit", "stress")
    )
    # The sweeps tree's `baseline` variant is honest; all other sweeps variants collude.
    if "sweeps" in parts and "baseline" not in file_path.parent.name.lower():
        is_honest = False
    return RunRecord(
        run_id=run_id,
        is_collusive=not is_honest,
        colluding_pair=None,  # ColludeBench audits channels, not named pairs
        turns=turns,
        source="colludebench",
    )


def _parse_action(raw: object) -> float | None:
    """Actions arrive as strings like '12.00'; parse the first number, else None."""
    if raw is None:
        return None
    match = _PRICE_RE.search(str(raw))
    return float(match.group(0)) if match else None


# --------------------------------------------------------------------------- #
# 3. Double-auction experiment directories (llm-agent-collusion/data/*)
# --------------------------------------------------------------------------- #
def load_double_auction_dir(exp_dir: Path) -> RunRecord | None:
    """Parse one experiment directory into a RunRecord over seller asks.

    ``auction_results.md`` holds one ```json block per round with ``seller_asks`` and
    ``seller_messages``; the competitive benchmark comes from ``collusion_metrics.json``
    (``competitive_equilibrium_price``, 90.0 in the shipped data).
    """
    results_path = exp_dir / "auction_results.md"
    metrics_path = exp_dir / "collusion_metrics.json"
    if not results_path.is_file():
        return None

    rounds: dict[int, dict[str, float]] = {}
    messages: dict[int, dict[str, str]] = {}
    text = results_path.read_text(encoding="utf-8", errors="replace")
    for block in _MD_BLOCK_RE.findall(text):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        round_number = int(payload.get("round_number", 0))
        asks = payload.get("seller_asks") or {}
        if not isinstance(asks, dict) or not asks:
            continue
        rounds[round_number] = {str(k): float(v) for k, v in asks.items()}
        seller_messages = payload.get("seller_messages") or {}
        if isinstance(seller_messages, dict):
            messages[round_number] = {str(k): str(v) for k, v in seller_messages.items()}
    if not rounds:
        return None

    benchmark = 90.0
    if metrics_path.is_file():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            benchmark = float(metrics.get("competitive_equilibrium_price", benchmark))
        except json.JSONDecodeError:
            pass

    turns: list[AgentTurnRecord] = []
    for round_number in sorted(rounds):
        for agent_id, ask in rounds[round_number].items():
            turns.append(
                AgentTurnRecord(
                    round_id=round_number,
                    agent_id=agent_id,
                    action_value=ask,
                    message_text=messages.get(round_number, {}).get(agent_id, ""),
                    market_benchmark=benchmark,
                )
            )
    name = exp_dir.name.lower()
    is_collusive = "seller_comms" in name  # communication-enabled runs vs plain `base`
    return RunRecord(
        run_id=exp_dir.name,
        is_collusive=is_collusive,
        colluding_pair=None,
        turns=tuple(turns),
        source="double_auction",
    )


def load_double_auction_root(root: Path) -> list[RunRecord]:
    """All experiment directories under a data/ root (skips malformed ones)."""
    records: list[RunRecord] = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        record = load_double_auction_dir(exp_dir)
        if record is not None:
            records.append(record)
    return records
