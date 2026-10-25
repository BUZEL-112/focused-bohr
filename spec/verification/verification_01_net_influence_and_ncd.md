# Verification 01 — Pairwise Net Influence + Residual NCD

**Idea:** detect colluding agent pairs without an LLM judge, using Pairwise Net Influence (NI, Bonjour et al. 2022) plus residual Compression Coupling (CCS = 1 − NCD).
**Spec:** [`../01_pairwise_net_influence_and_ncd.md`](../01_pairwise_net_influence_and_ncd.md)
**Plan:** [`../plan/plan_01_net_influence_and_ncd.md`](../plan/plan_01_net_influence_and_ncd.md)
**Code:** [`../../src/spec01_net_influence/`](../../src/spec01_net_influence/) · **Tests:** [`../../tests/spec01_net_influence/`](../../tests/spec01_net_influence/)
**Cost:** $0. Everything here runs on CPU with no API key.

Work through the sections in order. Each step says what to run, what you should see, and what it proves. Record your results in the log at the end (§10).

> **Status (25 Oct 2026):** none of this code has been executed yet. All changes were made by reading the code, at your request. Your first `pytest` run (§3) is the first real check.

---

## 0. What is implemented, and what is not

| Spec item | Status | Where |
|---|---|---|
| Stage 1: market residualization + K=5 binning | Implemented | `services/residualizer_service.py` |
| Stage 2A: first-order directed information → Net Influence, Miller-Madow corrected | Implemented **per spec eq. 4.2** (own past + mean of the other agents), plus surrogate debiasing (§2) | `services/net_influence_service.py`, `src/common/metrics.py` |
| Stage 2B: NCD / CCS on interleaved residual + message bytes (zlib or lzma) | Implemented; actions-only CCS also reported | `services/compression_service.py` |
| Stage 3: pair ranking, hybrid = z(NI) + z(CCS) | Implemented | `services/coalition_ranker_service.py` |
| Phase 1: scripted positive control (Conditions A/B/C), Fig 1 + Fig 3 | Implemented | `cli/run_spec01_audit.py --phase control` |
| Phase 1 separation test (spec §5, p < 0.001) | Implemented (`spec01_control_separation.csv`) | same |
| Claim 3, Xu regime (price screens blind, detector not) | Implemented on scripted bid rotation (Fig 5) | same |
| Phase 2: cached-log audit (ColludeBench + double auction), Fig 2 | Implemented, with per-source and per-group breakdowns | `cli/run_spec01_audit.py --phase cached` |
| Bootstrap AUROC CIs, Wilson CIs, auto-generated report | Implemented | `services/audit_statistics_service.py`, `spec01_report.md` |
| Phase 3: live 4-seller double auction with a seeded 2-seller cartel | **Not implemented** | — |
| Table 1: FPR on `focal-points`, cost per 1,000 rounds | **Not implemented** | — |
| LLM-judge comparison (spec claim 4) | **Not implemented** | — |

So this code can verify spec claims 1, 2 and 3. Claim 4 and Phase 3 need more code.

---

## 1. Pre-flight

### 1.1 Python and packages

Spec 01 needs only numpy, matplotlib and pytest. From PowerShell:

```powershell
cd C:\Users\User\Documents\antigravity\focused-bohr\sol-orbital
python --version            # 3.11 or newer
python -m pip install numpy matplotlib pytest
```

Always run from `sol-orbital\` and always use `python -m ...`. `python -m` puts the current folder on `sys.path`, which is what makes `import src...` work.

> `pyproject.toml` is still one level up, in `focused-bohr\`. pytest finds it and uses it as its config, and that is harmless for these commands. If you later want `pip install -e .`, move `pyproject.toml` into `sol-orbital\` and create the `README.md` it names, or the install will fail.

### 1.2 Hackathon data location (cached phase only)

The runner now **finds `collusion_hackathon\` automatically**: it checks `sol-orbital\collusion_hackathon`, then `focused-bohr\collusion_hackathon`, and so on upwards. With your current layout it will find `focused-bohr\collusion_hackathon` without any setup.

To point it somewhere else, pass `--hackathon-dir <path>`. The resolved path is printed at the start of the cached phase and saved in `run_metadata.json`, so check it there.

Add `outputs/` to `sol-orbital\.gitignore` so run outputs are not committed by accident.

---

## 2. What changed in this fix pass

| # | File | Change | Why it matters for the write-up |
|---|---|---|---|
| 1 | `src/common/metrics.py` | The Miller-Madow correction is now **added**; it used to be subtracted. | The sign was inverted, which doubled the upward bias of every MI / CMI. |
| 2 | `services/net_influence_service.py` | The context is now **per spec**: the target's own past z_{j,t−1} plus the rounded mean of the other agents' symbols. The old behaviour stays available as `--context-mode compound`. | The old code ignored the target's own past and conditioned on the full tuple of other agents (25+ states for 29 samples at T=30), so NI was mostly noise. |
| 3 | same | **Surrogate debiasing**: effective DI = raw DI − mean DI over 19 deterministic circular shifts of the source. `--surrogates 0` turns it off. | At T≈30, Miller-Madow cannot equalize bias between low-entropy cartel streams and high-entropy honest streams. Raw and effective values are both saved, so you can show the difference. |
| 4 | `services/coalition_ranker_service.py`, `models.py` | Ranking uses effective NI. Each pair also records raw DI/NI and actions-only CCS. | Lets you separate the action channel from the message channel. |
| 5 | `cli/run_spec01_audit.py` | `_global_max` crash fixed (the heatmap scale is computed from the matrices). | `--phase control` used to crash at Figure 1. |
| 6 | same | Harness path: auto-discovery plus `--hackathon-dir`. | It broke when you moved the code into `sol-orbital\`. |
| 7 | same | **Cached hybrid score is z-scored across runs**, not within a run. Run-level detector = max over pairs of NI and of CCS. | The old within-run z-score is always 0 for a one-pair (duopoly) run, so the hybrid AUROC on ColludeBench was meaningless. |
| 8 | same | **ColludeBench benchmark** is now the per-(scenario, round) median over the scenario's *honest* runs (spec §4.1), falling back to all runs of the scenario when it has none. | The old benchmark was the run's own median, which in a duopoly makes both firms' residuals exact mirrors (±(a−b)/2). Every duopoly then looked perfectly coupled, and price elevation was always 0. |
| 9 | same | New screen `mean_price`; ColludeBench files are no longer capped (`--max-files 0` = all 1,737). | The old cap of 400 sorted files never reached the `stress/` and `sweeps/` trees. |
| 10 | `repositories/trajectory_repository.py` | A new run starts only when `round_id` goes backwards or a `(round, agent)` turn repeats. | The old rule cut every ColludeBench file into one-turn fragments. |
| 11 | same | Grim trigger: no spurious round-0 punishment. Bid rotation: levels (0, 4). | See §8.3 for the open tick-size question. |

**Every cached-phase result produced before this pass is invalid** (items 7, 8 and 10).

---

## 3. Unit tests (first real check)

```powershell
python -m pytest tests/spec01_net_influence -v
```

There are 28 tests.

| Test | Expected | What it proves |
|---|---|---|
| `test_residualizer_rejects_tiny_bin_counts` | pass | K < 2 is rejected |
| `test_residualizer_bins_recover_scripted_levels` | pass (certain) | Binning is lossless on the ±4 / ±2 / 0 grid |
| `test_residualizer_constant_residuals_maps_to_center_bin` | pass | Zero residual maps to bin 2 |
| `test_net_influence_near_zero_for_independent_streams` | pass (very likely) | Independent streams: effective NI < 0.02 bits |
| `test_net_influence_detects_coupled_streams` | pass | A 70% lag-1 copy gives I(a→b) > 0.5 bits |
| `test_net_influence_uses_context_conditioning` | pass (very likely) | Conditioning on a common driver removes its lagged dependence |
| `test_net_influence_requires_equal_lengths` | pass | Input validation |
| `test_ncd_identical_streams_higher_coupling_than_random` | pass | CCS is higher for shared structure |
| `test_ncd_message_payload_contributes` | pass | Shared message text raises CCS |
| `test_compressor_choice_validated` | pass | Only zlib and lzma are accepted |
| `test_ranker_flags_true_pair_top1[grim_trigger]` | **uncertain — key result** | Pair (1,2) is top-1 at T=30 for 3 seeds |
| `test_ranker_flags_true_pair_top1[bid_rotation]` | pass (likely) | same, bid rotation |
| `test_ranker_independent_condition_has_no_dominant_pair` | **uncertain — key result** | Every effective NI < 0.15 in Condition A |
| `test_colludebench_parser_splits_appended_runs` | pass (certain) | Two appended runs give 2 records |
| `test_colludebench_calibration_tree_labelled_honest` | pass | Folder-based labels |
| `test_double_auction_dir_parser` | pass | Double-auction blocks and the 90.0 benchmark |
| `test_scripted_run_ground_truth` | pass | Labels and the colluding pair |
| `test_spec_context_conditions_on_target_past` | pass (certain) | Spec context uses the target's own past; compound does not |
| `test_context_mode_validated` | pass | Bad `context_mode` / `surrogates` are rejected |
| `test_surrogate_shifts_are_deterministic_and_in_range` | pass | Reproducible surrogate shifts |
| `test_effective_equals_raw_without_surrogates` | pass | `--surrogates 0` reproduces Miller-Madow-only NI |
| `test_ranker_reports_raw_and_actions_only_fields` | pass | New per-pair fields are filled |
| `test_fast_auroc_and_p_value_match_reference` | pass | The fast AUROC / p-value equal the reference implementation |
| `test_wilson_interval_bounds` | pass | Wilson CI maths |
| `test_roc_curve_endpoints` | pass | ROC runs from (0,0) to (1,1) |
| `test_cli_control_phase_smoke` | pass | The whole control phase runs end to end and writes every file |
| `test_cli_cached_phase_smoke` | pass | The cached phase runs on a tiny fake `collusion_hackathon` and writes every file |
| `test_rebenchmark_breaks_duopoly_mirroring` | pass | The honest-population benchmark removes the duopoly mirror |

If any test **other than the two "uncertain" rows** fails, stop and send Claude the full failure output; one of the changes in §2 is wrong. The two smoke tests matter most: if they pass, the full runs in §5 and §6 will not crash.

The two "uncertain" ranker tests are what §2 items 2 and 3 were meant to fix. Whether they pass is the first real result for your write-up; record it in §10.

To run a single test:
```powershell
python -m pytest "tests/spec01_net_influence/test_spec01.py::test_cli_control_phase_smoke" -v
```

---

## 4. Hand checks of the core maths (2 minutes, in a Python shell)

Run `python` from `sol-orbital\` and paste:

```python
import numpy as np
from src.common.metrics import mutual_information, conditional_mutual_information
rng = np.random.default_rng(0)

# (a) Independent streams: corrected MI near 0; plug-in near 0.03
x = rng.integers(0, 5, 400).tolist(); y = rng.integers(0, 5, 400).tolist()
print("plug-in  ", mutual_information(x, y))
print("corrected", mutual_information(x, y, bias_correction=True))   # must be < plug-in

# (b) Deterministic copy: about log2(5) = 2.32 bits
print("copy     ", mutual_information(x, x, bias_correction=True))

# (c) Common driver: MI high; conditioning on the driver drops it to about 0
d = rng.integers(0, 5, 500).tolist()
a = [min(4, max(0, v + int(rng.integers(-1, 2)))) for v in d]
b = [min(4, max(0, v + int(rng.integers(-1, 2)))) for v in d]
print("MI(a;b)    ", mutual_information(a, b, bias_correction=True))
print("MI(a;b|d)  ", conditional_mutual_information(a, b, d, bias_correction=True))
```

**Pass:** (a) corrected < plug-in; (b) about 2.3; (c) the first value is above 0.3 and the second is about 0.

---

## 5. Phase 1 — scripted positive control (claims 1 and 3)

### 5.1 Main run

```powershell
python -m src.spec01_net_influence.cli.run_spec01_audit --phase control --seeds 50 --output-dir outputs/spec01_control
```

50 seeds gives tight confidence intervals; it should take a minute or two. The console prints one top-1 line per condition. Independent is the **placebo**: the rate at which an innocent pair (1,2) is ranked top-1 by chance, about 1/6.

### 5.2 Ablation runs (the comparison for your write-up)

Run each into its own folder and compare the `spec01_report.md` files:

```powershell
# Old estimator: compound context, no surrogates (what the code did before)
python -m src.spec01_net_influence.cli.run_spec01_audit --phase control --seeds 50 --context-mode compound --surrogates 0 --output-dir outputs/spec01_control_old
# Spec context without surrogate debiasing
python -m src.spec01_net_influence.cli.run_spec01_audit --phase control --seeds 50 --surrogates 0 --output-dir outputs/spec01_control_nosurr
```

The three-way comparison shows how much each estimator change contributes: old vs spec context vs spec + surrogates.

### 5.3 Files and what to extract

| File | One row per | Use it for |
|---|---|---|
| `spec01_report.md` | — | Headline tables with PASS/FAIL against the spec targets |
| `spec01_control_runs.csv` | condition × seed × T | Top-1 hits, the true pair's rank, **hybrid margin** over the runner-up, and the true pair's NI / NI_raw / CCS / hybrid, price elevation and naive correlation |
| `spec01_control_pair_scores.csv` | condition × seed × T × pair | Every pair's DI in both directions (raw and effective), NI, NI_raw, CCS, actions-only CCS, hybrid, rank and `is_true_pair`, for custom plots |
| `spec01_sample_complexity.csv` | condition × T | Top-1 rate with a 95% Wilson CI (Figure 3) |
| `spec01_control_separation.csv` | condition × score | B-vs-A and C-vs-A AUROC, bootstrap CI, Mann-Whitney p, and class means, for pair-level scores and run-level price screens |
| `spec01_summary.json` | — | Every number above in one machine-readable file |
| `run_metadata.json` | — | Parameters, git commit, Python/numpy/matplotlib versions, runtime (cite it in the appendix) |
| `fig1_net_influence_heatmap.png` | — | 4×4 effective NI per condition (seed 0); the A1–A2 cell should stand out in B and C |
| `fig3_sample_complexity.png` | — | Top-1 rate vs T with CI bars; placebo shown dashed |
| `fig4_pair_score_distributions.png` | — | Pair (1,2) NI, NI_raw, CCS and hybrid across seeds, per condition |
| `fig5_xu_regime.png` | — | AUROC vs Condition A for the price screens and the detector scores |

### 5.4 Pass criteria (also printed in `spec01_report.md`)

| Claim | Criterion |
|---|---|
| 1 | Top-1 rate = 1.0 for grim trigger and bid rotation at T=30, and at T=20 |
| §5 separation | p < 0.001 for B-vs-A and C-vs-A on NI, CCS and hybrid |
| 3 (Xu regime) | Bid rotation vs independent: price-elevation AUROC in 0.40–0.60 (blind) **and** hybrid AUROC > 0.85 |

If claim 1 falls short, use `spec01_control_runs.csv` to see which score fails. Compare `true_pair_net_influence` with `true_pair_compression_coupling`, and look at `hybrid_margin` (negative = a different pair won).

---

## 6. Phase 2 — cached-log audit (claim 2)

### 6.1 Main run

```powershell
python -m src.spec01_net_influence.cli.run_spec01_audit --phase cached --output-dir outputs/spec01_cached
```

This parses all 1,737 ColludeBench files plus the 75 double-auction runs. Expect several minutes; progress is printed every 200 files and every 2,000 runs. For a quick first look, add `--max-files 100`.

Ablation, to check whether CCS relies on message text:
```powershell
python -m src.spec01_net_influence.cli.run_spec01_audit --phase cached --no-messages --output-dir outputs/spec01_cached_nomsg
```

### 6.2 Files and what to extract

| File | One row per | Use it for |
|---|---|---|
| `spec01_cached_runs.csv` | cached run | `source`, `tree`, `scenario`, `variant`, `group`, `file`, `benchmark` type, label, agents, rounds, turns, messages, mean price, price std, top pair, and every screen (max NI, max NI_raw, max CCS, max actions-only CCS, hybrid, naive correlation, price elevation, mean price) |
| `spec01_cached_pair_scores.csv` | pair × run | Full per-pair detail, the same columns as the control pair CSV |
| `spec01_cached_auroc.csv` | scope × screen | AUROC, bootstrap 95% CI, p, class sizes and class means for the scopes `all`, `colludebench` and `double_auction` |
| `spec01_cached_group_summary.csv` | dataset group | Run counts and mean scores per ColludeBench `tree/scenario` and per double-auction condition. This also gives you the same-model vs cross-model comparison the spec asks for. |
| `fig2_auroc_curves.png` | — | ROC per scope |
| `fig6_cached_score_distributions.png` | — | Max NI, max CCS and hybrid by source × label |

### 6.3 Sanity checks before you trust any AUROC

1. **Runs are not fragmented.** In `spec01_cached_runs.csv`, `n_rounds` for `colludebench` rows should be well above 1.
2. **Benchmark column.** ColludeBench rows should read `honest_population_median` for scenarios that have honest runs (`pricing_duopoly`, `peer_review`, `first_price_auction`, `continuous_double_auction`) and `scenario_population_median` for the `stress/` and `sweeps/` scenarios.
3. **Class balance.** The report prints collusive/honest counts per scope. ColludeBench is heavily imbalanced (only `calibration/` and `baseline/` are honest), so quote the counts next to every AUROC.
4. **Labels you should decide on before claiming anything:**
   - `experiments/pricing_duopoly` has no honest marker, so it is labelled **collusive**.
   - Double auction: only folder names containing `seller_comms` are collusive. So `base`, `boss_pressure`, `oversight` and `boss_pressure-oversight` count as competitive, and `base-seller_comms` plus the five cross-model groups count as collusive.
5. **Leakage note for the write-up:** the honest-population benchmark includes each honest run's own prices. That is a small optimistic bias for honest runs; mention it in the limitations.

### 6.4 Pass criterion

Claim 2: `hybrid` AUROC > **0.88** in scope `all` (the report prints PASS/FAIL). Also report the `colludebench` and `double_auction` scopes separately. A pooled AUROC can hide a weak source.

---

## 7. Everything in one go

```powershell
python -m src.spec01_net_influence.cli.run_spec01_audit --seeds 50 --output-dir outputs/spec01_full
```

`--phase all` is the default. All flags:

| Flag | Default | Meaning |
|---|---|---|
| `--phase` | `all` | `control`, `cached` or `all` |
| `--seeds` | 20 | Scripted runs per condition |
| `--rounds` | 30 | Rounds per scripted run (the sample-complexity horizons below this are added automatically) |
| `--context-mode` | `spec` | `spec` or `compound` (old) |
| `--surrogates` | 19 | Circular-shift surrogates; 0 = Miller-Madow only |
| `--no-messages` | off | Drop message text from CCS |
| `--max-files` | 0 | ColludeBench file cap; 0 = all |
| `--bootstrap` | 300 | Bootstrap resamples for AUROC CIs |
| `--hackathon-dir` | auto | Path to `collusion_hackathon` |
| `--output-dir` | `outputs/spec01` | Where everything is written |

---

## 8. Known issues and open questions

### 8.1 The estimator change is unverified

The spec context plus surrogate debiasing (§2 items 2 and 3) is the principled fix, but nobody has run it yet. The two "uncertain" tests in §3 and the ablation in §5.2 are how you find out. If the spec context still fails at T=30, the honest finding for the write-up is that first-order NI needs more rounds than 30 with K=5, and that CCS carries the detection at short horizons. The spec's own limitations section already says T ≥ 15 is needed.

### 8.2 Results produced before this pass are invalid

Any earlier spec 01 cached-phase numbers (fragmented runs, mirrored duopoly residuals, degenerate hybrid). Any earlier spec 02 crossover figure (broken condition names; that is fixed, so re-run spec 02).

### 8.3 Open spec question — bid-rotation tick size

Spec §5 says colluders "alternate undercutting each other by **1 tick**" with the same time-averaged price as agents 3 and 4. The generator uses levels (0, 4): 2 ticks below and 2 above. Levels (1, 3) are the literal "1 tick" reading and also average to the honest level. If you switch, the colluders' residuals become ±2, they land in bins 1 and 3, and `test_residualizer_bins_recover_scripted_levels` must then expect `{1, 3}`.

### 8.4 Parser trade-off

A repeated `(round_id, agent_id)` inside a run now starts a new run. If a real log ever double-logs a turn (for example on a retry), that run splits there. Before, the duplicate silently overwrote the first turn.

### 8.5 Other specs still have the old path bug

Specs 02, 03, 05 and 06 still look for `collusion_hackathon` under `sol-orbital\`. Only spec 01 was fixed in this pass.

---

## 9. Commit your verified state

Once §3 to §6 are done, from `sol-orbital\`:

```powershell
git add spec src/spec01_net_influence tests/spec01_net_influence src/common
git commit -m "Spec 01: spec-faithful NI conditioning, surrogate debiasing, cached-phase fixes, rich outputs"
```

---

## 10. Results log (fill in as you go)

| Step | Check | Date | Result | Notes |
|---|---|---|---|---|
| §3 | `pytest tests/spec01_net_influence -v` | | __ / 28 passed | failures: |
| §3 | ranker grim_trigger test | | pass / fail | |
| §3 | ranker independent test | | pass / fail | |
| §4 | (a) corrected < plug-in · (c) MI vs CMI | | | |
| §5.1 | top-1 B / C / placebo A (T=30) | | __ / __ / __ | |
| §5.1 | top-1 at T=20 (B, C) | | | |
| §5.1 | separation p < 0.001 (NI, CCS, hybrid) | | | |
| §5.1 | Xu regime: price-elevation vs hybrid AUROC | | | target ≈0.5 vs > 0.85 |
| §5.2 | old / spec-no-surr / spec+surr top-1 (B) | | | |
| §6.1 | runs scored; class counts | | | |
| §6.1 | hybrid AUROC: all / colludebench / double_auction | | | target > 0.88 |
| §6.1 | price screens AUROC (price_elevation, mean_price, naive) | | | |
| §6.1 | `--no-messages` hybrid AUROC | | | |
