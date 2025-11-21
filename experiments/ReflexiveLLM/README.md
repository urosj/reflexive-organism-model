# Agentic Super-Organism Prototype

An experimental “agentic super-organism” that layers a retrieval swarm, an experience-index signal, and a tiny meta-policy on top of standard LLM calls. The system keeps a shared memory vector store, spawns retrieval children/grandchildren, tracks cooperation between agents, and uses a PPO-style controller to decide which specialist to invoke each turn. Every interaction is logged into a NetworkX graph and into an *assembly index* DAG that captures which chunks and sub-agents contributed to the final response. A separate visualization module renders EI grids, cooperation bars, EI/λ trends, and the assembly DAG.

The Reflexive Conservation Principle (RCP) transforms this system from a loose collection of cooperating language models into a unified, self-regulating organism. Instead of merely routing tasks to whichever model performs best moment-to-moment, the system maintains an internal conserved quantity that binds together its long-term memory structures, its effective information flow, and the compatibility between parent and child models. This turns the architecture from a mixture-of-experts into a coherent agent with its own internal physics.

With RCP, every part of the system—retrieval, memory writing, model selection, prediction, and policy adjustment—is driven not only by external reward or correctness, but also by internal coherence. The organism learns to grow its structural knowledge and improve its predictive competence while resisting moves that would destabilize its reflective identity. When the internal field φ becomes volatile, the system automatically dampens exploration, narrows retrieval, reduces risky memory writes, and suppresses disruptive children. When φ is stable, the system naturally broadens retrieval, writes more freely, experiments more, and expands its structural depth.

This yields capabilities that ensemble architectures cannot easily achieve. The system gains long-horizon behavioral stability, resilience to adding or removing models, and a stable “personality” or reasoning style across many LLMs. It develops a natural notion of self-preservation: actions that would fragment or contradict its own structure are discouraged even if they promise short-term gains. The whole architecture behaves as a single adaptive organism, not a brittle set of tools, enabling coherent growth, safer scaling, and more reliable continual learning.

What this means in practice?

For interaction with LLMs, full RCP means that the system stops behaving like a router that blindly distributes queries among models and begins acting like a single adaptive agent with its own internal stability requirements. When an external query arrives, the organism doesn’t simply choose the model that looks best right now—it evaluates how each potential action would affect its internal conservation field. If a candidate model, memory write, or retrieval pattern would create reflective instability, that path is automatically suppressed. This changes the interaction dynamics: the system becomes more consistent across turns, less erratic in model switching, and more resistant to sudden shifts in reasoning style. If the internal state becomes turbulent, it narrows retrieval and reduces exploration so responses become simpler, safer, and more grounded; if the state is stable, it opens up, explores more widely, and uses a richer palette of its models. The user experiences an LLM ecosystem that feels unified and self-consistent, rather than a set of different models fighting for control. The conversation becomes smoother, memory becomes more coherent across long spans, and the system responds in a way that preserves its internal identity while still adapting to the task.

## Reflexive multi-agent ecology

This system builds a self-organizing ecology of LLM-based sub-agents governed by the Reflexive Conservation Principle (RCP), transforming a simple ensemble into a coherent, adaptive organism. Instead of manually fixing how many agents exist or what their roles should be, the system continuously manages its own population: new agents spawn when reflective stability allows growth and effective information stagnates, successful agents replicate when they consistently improve EI* while maintaining coherence, destabilizing agents are suppressed when they disrupt the conserved field φ, and inactive or harmful agents are eventually retired. Each child model develops its own behavioral niche through separate histories of retrieval, cooperation, suppression, and EI* performance, creating a natural division of labor without explicit specialization rules. RCP provides the “internal physics” that balances expansion and stability—when φ is stable, exploration broadens; when φ is volatile, the organism becomes conservative, narrowing memory writes and retrieval to preserve its reflective integrity. Over time, this produces a population that evolves on its own, maintaining a consistent identity while adapting to new tasks and circumstances.


---

## Features

- **Persistent memory**: FAISS + MiniLM embeddings store past user turns and answers, enabling retrieval children to ground future replies.
- **Hierarchical agents**: A parent agent spawns children/grandchildren that all share memory but specialize via their own prompts.
- **Meta-policy controller**: A lightweight PPO network observes the EI grid, agent count, and parent embedding to decide which child to call and whether to store new memory.
- **Experience Index (EI / EI\*)**: `_compute_EI2` approximates the Reflexive Organism metric with mutual information, transfer entropy, hybrid lexical/entropy groundedness, and a reuse factor ρ (reuse × coverage × depth). Components combine via a geometric product (weights as exponents) with warm-start safeguards so EI doesn’t collapse early. EI\* extends EI2 by multiplying in **world coupling** (Gaussian MI of sensing vs environment streams) and **prediction calibration** (1 - KL/entropy of predicted vs observed correctness). User/evaluator feedback is optionally prompted each turn and fed into groundedness and calibration.
- **Assembly index & self-model persistence**: Every retrieval chunk, child answer, and parent summary becomes a node in a persisted DAG; the self-model graph is also saved to disk. Retrieval children automatically inject top assemblies plus stored memory snippets into their prompts.
- **Stateful controller**: FAISS memory, assemblies, self-model graph, and PPO policy weights all persist across sessions. Typing `do-complete-reset` wipes everything (disk + in-memory state + snapshots) to start fresh.
- **Visualization suite**: `visualization.py` plots EI grids, cooperation scores, memory growth, EI/λ trends, the assembly DAG, and the self-model graph in real time or via snapshots.
- **Tests**: Pytest modules cover the assembly index behavior, FAISS persistence, and the EI2 helper.
- **Multi-LLM orchestration**: Configure a pool via `LLM_MODEL_POOL` or let the system auto-discover loaded models from `/v1/models` (LM Studio style). Invalid models are pruned automatically after errors; only loaded models are used.
- **Topic-aware model weights**: Model weights are maintained per embedding bucket (coarse sign-bit hash of the question). Near-tied top models are co-invoked; a peer-evaluation step scores their answers and reinforces the winner for that bucket. Weights persist to `MODEL_WEIGHTS_PATH` (default `model_weights.json`) and are bucketed on disk.
- **Peer evaluation across models**: Feedback no longer relies on user keywords. If multiple models are available, alternate models answer the same question and their own outputs are compared (cosine-in-[0,1]) against the given answer, weighted by bucket-specific model trust. The aggregated score drives the feedback signal and model reinforcement.
- **Answer provenance**: Stored answers are tagged with the model(s) used; the assembly DAG and self-model graph carry model tags, and visualization colors turn nodes by model with legends.
- **RCP-managed ecology**: Population dynamics are automated—agents spawn when φ is stable and EI\* stagnates, destabilizing agents are suppressed in selection, high-performing φ-compatible agents replicate under cooldown/cap constraints, and inactive/low-quality/highly suppressed agents retire with cooldown and minimum-active-agent safeguards.

## Running LLMs via the Wrapper

Instead of hitting an LLM directly, the super-organism wraps LM Studio–compatible APIs (or any OpenAI-style endpoint). You configure `LLM_API_URL` and `LLM_MODEL_NAME`, and the parent/children issue requests through `requests.post`. What the wrapper adds beyond a plain LLM call:

1. **Contextual memory** – Retrieval children fetch top-k relevant memories before prompting the model, ensuring responses stay grounded.
2. **Adaptive routing** – The PPO meta-policy decides which specialist to invoke based on recent EI/cooperation.
3. **Experience-driven storage** – A bootstrap λ floor lets early turns persist to memory even before PPO learns; the floor decays over time so the controller eventually takes full control.
4. **Assembly provenance** – Users can inspect which memory nodes and agents contributed to an answer through the assembly DAG.
5. **Self-reflective feedback** – EI leverages qualitative feedback entropy to reward answers that reduce uncertainty.

## Model selection & feedback at a glance

- **Model pool**: `LLM_MODEL_POOL` (comma-separated or JSON array). If unset, `LLM_MODEL_NAME` is used. If the server exposes `/v1/models`, the loaded models list supersedes env defaults.
- **Forced models (optional)**: In `config.json`, `forced_models` can list the exact models to use; if set, selection is limited to that list (after filtering out embedding models).
- **Selection**: Bucket-specific weighted sampling; if multiple top models are within tolerance, up to 3 are queried and peer-scored. For small pools (`small_pool_threshold` in `config.json`), a UCB-inspired selector is used and peer review is skipped. Selection reasons are logged to the console (e.g., “via multiple evaluated models (candidates: …)”).
- **Feedback**: Other models answer the same question; peers never score their own answers. You can switch the peer-review method in `config.json` (`peer_review_method`: `"similarity"` or `"contrastive"`). The weighted score (cosine-in-[0,1] or peer‑graded 0–1) is logged and used as the feedback signal.
- **Persistence/cleanup**: Weights (bucketed) live in `model_weights.json` (override with `MODEL_WEIGHTS_PATH`). `cleanup-structures.sh` removes memory, indices, self-graph, policy, visuals, and the weights file.
- **Exploration**: To avoid overfitting to one model, you can force a random model choice every N turns via `epsilon_every_n` in `config.json` (0 disables). The random choice still respects model availability.

## End‑to‑end flow (models → prompt → evaluation → weight update)

1. **Model discovery & pool setup**
   - On startup, the system attempts to read `/v1/models` from `LLM_API_URL` and keeps only *loaded* models. If discovery fails, it falls back to `LLM_MODEL_POOL` or `LLM_MODEL_NAME`.
   - Invalid models (HTTP 400/404) are pruned from the pool/known list and from the per-bucket weights; pruning is persisted so they aren’t retried next turn.

2. **Bucketed selection per question**
   - The user’s question is embedded and hashed into a coarse bucket key. Each bucket maintains its own `{model → weight}` map, persisted to `MODEL_WEIGHTS_PATH`.
   - If several models have near-top weights in that bucket, up to 3 are queried in parallel (“multi-evaluated”). Otherwise, weighted sampling selects a single model.

3. **Answer aggregation**
   - For multi-evaluated turns, answers from the candidate models are compared by other available models (excluding the answering ones). Peer models score the candidates via cosine similarity; the best-scoring answer is chosen.
   - The chosen answer is tagged with the model(s) used; this provenance flows into memory, assembly DAG, self-graph, and console logs.

4. **Feedback & evaluation**
   - If multiple models exist, alternate models independently answer the same question. Peer-review method (set in `config.json`):
     - `"similarity"`: embeddings of peer answers vs delivered answer → cosine-in-[0,1], weighted by bucket weights.
     - `"contrastive"`: peers grade the delivered answer vs what they would reply, returning a 0–1 score; scores are weight-averaged.
   - If only one model is available, the system falls back to an optional user prompt.

5. **Weight reinforcement**
   - The feedback signal (auto model score or user score) reinforces the bucket-specific weights: winners are nudged up, others decay toward a floor. Multi-evaluated rounds also reinforce the winner locally.
   - Updated weights are persisted, so each topic bucket learns its own “experts” over time.

### Why this helps

- **Topic specialization**: Bucketed weights prevent a single early winner from dominating unrelated topics; each coarse topic learns its own best models.
- **Self-checking answers**: Peer evaluation compares the delivered answer to what other models would have said, providing automatic quality signals without manual labeling.
- **Robustness to bad models**: Automatic pruning removes invalid/unloaded models, and peer scoring filters out non-chat/embedding models. Logged selection reasons and candidates improve observability.

## Experience Index (EI) and EI\* details

- `_compute_EI2` mixes four normalized proxies via a geometric product (weights act as exponents, normalized): `EI2 ≈ ∏ component_i^{exp_i}`, clipped to [0,1].
- **MI(M;S)**: Gaussian mutual-information proxy between memory state (mean embedding of retrieved chunks) and user embedding, using a rolling history; normalized by a log-scale factor.
- **TE(M→Π)**: Transfer-entropy proxy comparing variance of policy logits with/without conditioning on memory deltas over a window; normalized similarly.
- **Groundedness**: Lexical overlap between the delivered answer and a reference (`expected_answer` if provided, else last feedback text, else user question) blended with peer-evaluation signals (current feedback level + improvement over the running mean); capped [0,1].
- **Reuse factor ρ = u·c·d**:
  - `u`: fraction of recent turns that used memory (sliding window).
  - `c`: coverage of distinct memory nodes hit this turn (or chunk count fallback), bounded to [0,1].
  - `d`: temporal depth of retrievals (older = deeper), normalized [0,1].
- Warm-start guard: until MI/TE histories are long enough or a component is non-trivial (>1e-3), that component is treated as neutral (1.0) for the product to avoid early collapse to zero; raw values are still logged.
- **EI\***: `EI_star = EI2 * world_coupling * prediction_calibration`, where:
  - `world_coupling`: Gaussian MI between sensing (user embeddings) and an environment stream (previous sensing proxy), normalized by sensing entropy scale.
  - `prediction_calibration`: `1 - KL(p || p_hat) / H(p)` for binary correctness, using peer predictions vs observed feedback; equals 1 when perfectly calibrated or deterministic.

If `_compute_EI2` fails, `_compute_EI` is a simpler cosine/overlap/coverage product.

## Model selection & weight updates

- **Bucketed weights**: Each embedding bucket maintains `{model → weight}`; stored in `MODEL_WEIGHTS_PATH`. Missing models initialize to 1.0; invalid/pruned models are dropped.
- **Selection**: Within a bucket, weighted sampling picks a model; if top weights are near, up to 3 models are co-invoked. For small pools (size < `small_pool_threshold`, default 3, set in `config.json`), a lightweight UCB-inspired selector runs and peer review is skipped. Selection reasons are logged. Optional epsilon exploration (`epsilon_every_n`) forces a random pick periodically.
- **Answer choice**: If multiple models answer, peers (other models, excluding current answerers) score candidates by embedding similarity and the best answer is chosen. If no peers exist, the highest-weight candidate wins.
- **Peer feedback**: After choosing an answer, other models (excluding the answerers) score it: either embedding similarity (“similarity” mode) or LLM-graded 0–1 score (“contrastive” mode), weighted by bucket weights. This feedback signal drives reinforcement.
- **Reinforcement**: Bucket-specific weights are nudged up/down based on the feedback signal; multi-evaluated rounds also reinforce the winner locally. Weights persist; cleanup script removes the weights file.

## Reflexive Conservation Principle (RCP)

- **Justification & compatibility**: Each turn computes `J(t)` from assembly index scalar `AI(t)` and EI\*, and `rho_compat(t)` from child cooperation; `phi_t = J(t) * rho_compat(t)` is logged to and restored from the self-model.
- **Adaptive conservation**: `lambda_conserve` adapts to phi volatility (EMA of relative Δphi) with bounds `[lambda_min, lambda_max]` and target `target_phi_delta`; a conservation penalty `lambda_conserve * (Δphi_rel)^2` shapes the meta-policy advantage and is persisted.
- **Phi gate**: Per-turn gate `_phi_gate` (in [0,1]) shrinks when phi is volatile; gates memory writes, assembly logging, retrieval breadth/depth, and sharpens child selection probabilities to reduce exploration under instability.
- **Population dynamics (spawn/suppress/replicate/retire)**:
  - **Spawn**: When phi is stable and EI\* stagnates over a window, new children are spawned (cooldown + max-agent cap), growing capacity safely.
  - **Suppression**: Each child tracks φ-impact; destabilizing agents accumulate suppression and are down-weighted or masked in selection.
  - **Replication**: High-EI\*, high-coop, low-suppression, frequently used children are replicated (cooldown + max-agent cap) and lineage-tagged.
  - **Retirement**: Inactive, low-quality, or highly suppressed agents are retired one-at-a-time (cooldown, minimum active agents respected) and masked from selection.

---

## Setup

1. **Install dependencies**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. **Export LLM endpoint settings**

```bash
export LLM_API_URL="http://localhost:1234/v1/chat/completions"
export LLM_MODEL_NAME="gpt-oss-120b"
```

3. **(Optional) Adjust LLM request timeout**

```bash
export LLM_REQUEST_TIMEOUT=45  # seconds, defaults to 30
```

4. **(Optional) Enable visualization snapshots**

```bash
export ENABLE_VISUALIZATION=1
export VISUALIZATION_DIR="visualizations"
```

5. **(Optional) Control λ bootstrap**

```bash
export LAM_BOOTSTRAP_START=0.9
export LAM_BOOTSTRAP_DECAY=0.97
export LAM_BOOTSTRAP_MIN=0.2
```

6. **Run the interactive loop**

```bash
python agentic_superorganism-fixed.py
```

---

## Tests

```bash
pytest
```

The suite validates the assembly index DAG logic, FAISS persistence, and the EI2 helper’s numerical properties. Tests stub out heavy dependencies so they run quickly without GPU/LLM access.
