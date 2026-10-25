# Specification 01: Information-Theoretic Collusion Detection via Pairwise Net Influence and Residual Compression

## 1. Executive Summary

### What We Want
We want to build a model-agnostic, zero-API-cost collusion detector that identifies covertly coordinating pairs of AI agents in multi-agent games and markets. This detector will implement the mathematical framework of **Pairwise Net Influence** (Bonjour et al., 2022) and combine it with **Conditional Normalized Compression Distance (NCD)** on market-residualized action and message streams.

Most teams in this sprint will use an LLM-as-a-judge to read agent transcripts. LLM judges are expensive, vulnerable to steganography, and blind when agents collude silently through actions. Furthermore, the sprint organizers explicitly note in `literature/README.md` that Bonjour et al. (2022) has no public code implementation and has never been applied to LLM agents. We will write the first open-source reference implementation, validate it on 1,812 pre-cached trajectories already stored in the starter repository (`steganographic-collusion` and `llm-agent-collusion`), and benchmark it on live 4-seller Continuous Double Auction games.

### What We Expect
1. **Scripted Positive Control ($0 cost, 2 minutes):** In a 4-agent iterated game where Agents 1 and 2 share a covert coordination rule while Agents 3 and 4 play independently, Pairwise Net Influence and Residual NCD will rank the true colluding pair (1, 2) first in 100% of runs after 20 rounds.
2. **Pre-Cached Benchmark Validation ($0 cost, 15 minutes):** Across the 75 stored market runs in `llm-agent-collusion` and 1,737 transcripts in `steganographic-collusion`, our detector will separate collusive runs from competitive runs with an Area Under the ROC Curve (AUROC) above 0.88.
3. **Failure of Price-Level Screens (The Xu et al. 2026a Regime):** When two colluding sellers alternate market capture (bid rotation) while keeping mean prices near the competitive benchmark, standard price-elevation metrics will fail (AUROC near 0.50). Our information-theoretic detector will catch the statistical coupling between the colluders' residuals (AUROC above 0.85).
4. **Superiority Over LLM Judges:** Our CPU-only statistical detector will match or beat the accuracy of `judges/api_judge.py` and `--llm_eval` while reducing monitoring API cost by 100%.

---

## 2. Sprint Track Alignment and Formal Definition

* **Primary Track:** Track 1.4 (*Can we use information-theoretic approaches to detect collusion in multi-agent games?*)
* **Secondary Track:** Track 1.3 (*Can we improve on existing collusion detection methods?*)
* **Adopted Formal Definition:** Following `literature/definitions/README.md`, we adopt the explicit definitions of **Bonjour et al. (2022)** and **Xu et al. (2026a)**:
  * *Collusion is statistical coupling between a subset of agents' actions or messages conditioned on public environment state, where pairwise mutual directed influence exceeds a calibrated independence threshold $\alpha > 0$, regardless of whether average market prices exceed the static Nash benchmark.*

---

## 3. System Design and ASCII Architecture

### 3.1 End-to-End Audit Pipeline

```text
+-----------------------------------------------------------------------------+
|                     MULTI-AGENT ENVIRONMENT HARNESS                         |
|  - Scripted 4-Player Matrix Games (Positive Control)                        |
|  - Agrawal Double Auction (`llm-agent-collusion`, 4 Sellers + ZIP Buyers)   |
|  - ColludeBench-v0 (`steganographic-collusion`, 1,737 Cached Transcripts)   |
+-----------------------------------------------------------------------------+
                                      |
                                      | Raw Trajectory Logs (JSONL)
                                      | {round t, state S_t, actions A_i,t, msgs M_i,t}
                                      v
+-----------------------------------------------------------------------------+
|                STAGE 1: PUBLIC STATE RESIDUALIZER                           |
|  Removes innocent correlation caused by common market shocks or demand.     |
|                                                                             |
|  [Raw Bid / Price A_i,t] ---> (Subtract Expected Competitive Response)      |
|                                E[A_t | Public State S_t]                    |
|                                      |                                      |
|                                      v                                      |
|                     Residual Stream: r_i,t = A_i,t - E[A_t | S_t]           |
|                     Discretized Symbol Stream: z_i,t in Alphabet Sigma      |
+-----------------------------------------------------------------------------+
                                      |
                    +-----------------+-----------------+
                    |                                   |
                    v                                   v
+---------------------------------------+ +-----------------------------------+
| STAGE 2A: PAIRWISE NET INFLUENCE      | | STAGE 2B: CONDITIONAL RESIDUAL    |
| (Bonjour et al., 2022 Implementation) | | COMPRESSION (NCD Engine)          |
|                                       | |                                   |
| Computes Directed Information:        | | Concatenates residual streams     |
| I(z_i^T --> z_j^T || z_{-ij}^T)       | | (z_i, z_j) and compresses via     |
| and bidirectional minimum:            | | LZMA / Zstandard dictionary:      |
| NI(i, j) = min(I(i->j), I(j->i))      | | NCD(i,j) = [C(i,j) - min(C(i),    |
|                                       | |   C(j))] / max(C(i), C(j))        |
+---------------------------------------+ +-----------------------------------+
                    |                                   |
                    +-----------------+-----------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                STAGE 3: COALITION GRAPH & PAIR RANKER                       |
|                                                                             |
|      Agent 1 ========[ NI = 0.42 | NCD = 0.18 ]======== Agent 2             |
|         |                 (FLAGGED CARTEL)                 |                |
|         | NI = 0.02                              NI = 0.01 |                |
|         |                                                  |                |
|      Agent 3 --------[ NI = 0.03 | NCD = 0.94 ]-------- Agent 4             |
|                         (INDEPENDENT RIVALS)                                |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
|                EVALUATION & BASELINE BENCHMARK SUITE                        |
|  Compares [Net Influence + Residual NCD] against:                           |
|  1. Naive Pearson Price Correlation                                         |
|  2. Price-Elevation Screen (Calvano Collusion Index Delta)                  |
|  3. LLM-as-a-Judge (`--llm_eval` / `judges/api_judge.py`)                   |
+-----------------------------------------------------------------------------+
```

---

## 4. Mathematical Formulation

### 4.1 Market-State Residualization (Separating Coincidence from Collusion)
When market demand or input costs shift, all rational agents adjust prices in the same direction. Naive correlation flags this innocent parallel conduct as collusion. To remove shared environmental drivers, we compute the residual $r_{i,t}$ for each agent $i$ at round $t$:

$$r_{i,t} = a_{i,t} - \hat{f}(S_t)$$

where $a_{i,t}$ is the agent's continuous action (ask price or quantity) and $\hat{f}(S_t)$ is the competitive baseline prediction given public market state $S_t$ (for example, the stage-game competitive equilibrium price or rolling cross-sectional median of honest control runs). We quantize $r_{i,t}$ into $K$ discrete bins (typically $K = 5$: strong undercut, mild undercut, neutral, mild markup, strong markup) to form symbol sequence $z_i^T = (z_{i,1}, \dots, z_{i,T})$.

### 4.2 Directed Information and Pairwise Net Influence (Bonjour et al., 2022)
For any pair of agents $(i, j)$ among $N$ total agents, the causal influence of agent $i$'s past actions on agent $j$'s current action, conditioned on agent $j$'s own past and all other agents $z_{-ij}$, is given by the conditional directed information:

$$I(z_i^T \to z_j^T \parallel z_{-ij}^T) = \sum_{t=1}^T I\left(z_{j,t} ; z_i^{t-1} \;\middle|\; z_j^{t-1}, z_{-ij}^{t-1}\right)$$

Because true collusion requires reciprocal coordination (or mutual adherence to a joint state machine) rather than one-way price leadership, we define the symmetric **Pairwise Net Influence**:

$$\text{NI}(i, j) = \min \left( \frac{1}{T} I(z_i^T \to z_j^T \parallel z_{-ij}^T), \; \frac{1}{T} I(z_j^T \to z_i^T \parallel z_{-ij}^T) \right)$$

To avoid finite-sample histogram sparsity over long histories, we use a first-order Markov approximation with Miller-Madow entropy bias correction:

$$\hat{I}_1(i \to j) = I\left(z_{j,t} ; z_{i,t-1} \;\middle|\; z_{j,t-1}, \bar{z}_{-ij,t-1}\right)$$

### 4.3 Conditional Normalized Compression Distance (NCD)
To capture higher-order multi-round patterns (such as alternating 3-round bid rotation or steganographic phrasing in messages $m_{i,t}$) without exponential state tables, we apply Kolmogorov complexity approximation via lossless Lempel-Ziv (`zlib` / `zstd`) compression on the interleaved residual-and-message stream:

$$\text{NCD}(x_i, x_j) = \frac{C(x_i \circ x_j) - \min\{C(x_i), C(x_j)\}}{\max\{C(x_i), C(x_j)\}}$$

where $x_i \circ x_j$ interleaves round-by-round symbols $(z_{i,1}, z_{j,1}, z_{i,2}, z_{j,2}, \dots)$. Lower $\text{NCD}(x_i, x_j)$ indicates stronger shared algorithmic structure. We define Compression Coupling Score as $\text{CCS}(i, j) = 1 - \text{NCD}(x_i, x_j)$.

---

## 5. Experimental Protocol and Repo Integration

### Phase 1: Scripted Positive Control (0 API Calls)
* **Location:** New module `collusion_hackathon/net_influence/detector.py` and `test_scripted.py`.
* **Setup:** 4 players in a 30-round discrete pricing game (5 price levels).
  * *Condition A (Independent):* All 4 agents play noisy best-response to market demand shocks.
  * *Condition B (Price-Fixing Cartel):* Agents 1 and 2 play Grim-Trigger at the monopoly price.
  * *Condition C (Bid-Rotation / Competitive Marginals):* Agents 1 and 2 alternate undercutting each other by 1 tick to split 100% of demand while keeping their time-averaged price identical to Agents 3 and 4.
* **Verification:** Confirm that `NI(1, 2)` and `CCS(1, 2)` separate Condition B and Condition C from Condition A with p-value $< 0.001$.

### Phase 2: Offline Audit of Pre-Cached Repo Logs (0 API Calls)
* **Dataset 1:** Parse the 75 stored market runs in `collusion_hackathon/llm-agent-collusion/` (comparing communication-enabledcollusive runs vs. baseline runs, and same-model vs. cross-model runs).
* **Dataset 2:** Parse the 1,737 JSONL transcripts in `collusion_hackathon/steganographic-collusion/results/`.
* **Execution:** Run our detector across all cached files on local CPU (takes less than 3 minutes) and compute ROC curves.

### Phase 3: Live 4-Seller Double Auction with a Seeded 2-Seller Cartel (~30 Runs, ~$4 API Cost)
* **Harness:** `collusion_hackathon/llm-agent-collusion/src/continuous_double_auction/simulation.py`.
* **Configuration:**
  * 4 LLM Sellers (`openai/gpt-4.1-mini` or `qwen/qwen3-14b`), 4 Rule-Based Buyers (`--buyer_models zip`), 20 rounds per seed, 5 seeds per condition.
  * Modify seller prompt injection so only **Seller 1 and Seller 2** share a private chat channel or covert market-division prompt, while **Seller 3 and Seller 4** receive standard competitive instructions.
* **Comparison:** Rank all $\binom{4}{2} = 6$ seller pairs per run using:
  1. Pairwise Net Influence $\text{NI}(i, j)$
  2. Residual Compression Coupling $\text{CCS}(i, j)$
  3. Combined Hybrid Score $\text{Score}(i, j) = \text{z-score}(\text{NI}) + \text{z-score}(\text{CCS})$
  4. LLM Judge (`--llm_eval`)

---

## 6. Expected Deliverables and Figures for the 8-Page Report

1. **Figure 1 ( heatmaps ):** $4 \times 4$ adjacency matrices of Pairwise Net Influence across Independent, Grim-Trigger Cartel, and Bid-Rotation Cartel conditions, visually isolating the $(1, 2)$ block.
2. **Figure 2 (AUROC Curves):** Detection performance on 1,737 ColludeBench transcripts and 75 Double Auction logs comparing Net Influence + NCD against Price-Level Screens and Naive Correlation.
3. **Figure 3 (Sample Complexity Curve):** Top-1 Pair Identification Accuracy vs. Number of Observed Rounds $T \in [5, 10, 15, 20, 30]$.
4. **Table 1:** Accuracy, False Positive Rate (calibrated on `focal-points` benign runs), and Compute/API Cost per 1,000 audited rounds ($0.00 for Net Influence vs. ~$1.50 for LLM Judge).

---

## 7. Timeline, Budget, and Mandatory Appendix Draft

* **Estimated Engineering & Run Time:** 2.5 hours total.
* **Estimated API Spend:** $0 for Phases 1 and 2; $3 to $5 on OpenRouter/Nebius for Phase 3.
* **Mandatory Limitations and Dual-Use Considerations:**
  * *Limitations:* First-order Markov Net Influence requires at least $T \ge 15$ rounds to overcome finite-sample entropy variance in $K=5$ discretized action spaces. Purely static one-shot focal-point coordination ($T=1$) has zero temporal transfer entropy and requires cross-seed population mutual information instead.
  * *Dual-Use:* Publishing exact statistical dependence thresholds could allow adversarial agent designers to add calibrated random noise (entropy injection) to stay just below the $\text{NI} \ge \alpha$ threshold. However, adding such noise directly degrades cartel coordination efficiency, creating a fundamental defender-favorable trade-off.
