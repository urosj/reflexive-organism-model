# Agentic Super-Organism Prototype

An experimental “agentic super-organism” that layers a retrieval swarm, an experience-index signal, and a tiny meta-policy on top of standard LLM calls. The system keeps a shared memory vector store, spawns retrieval children/grandchildren, tracks cooperation between agents, and uses a PPO-style controller to decide which specialist to invoke each turn. Every interaction is logged into a NetworkX graph and into an *assembly index* DAG that captures which chunks and sub-agents contributed to the final response. A separate visualization module renders EI grids, cooperation bars, EI/λ trends, and the assembly DAG.

## Features

- **Persistent memory**: FAISS + MiniLM embeddings store past user turns and answers, enabling retrieval children to ground future replies.
- **Hierarchical agents**: A parent agent spawns children/grandchildren that all share memory but specialize via their own prompts.
- **Meta-policy controller**: A lightweight PPO network observes the EI grid, agent count, and parent embedding to decide which child to call and whether to store new memory.
- **Experience Index (EI)**: `_compute_EI2` approximates the Reflexive Organism metric with mutual information, transfer entropy, a hybrid lexical/entropy groundedness term, and assembly coverage. User/evaluator feedback is optionally prompted each turn and fed into groundedness.
- **Assembly index & self-model persistence**: Every retrieval chunk, child answer, and parent summary becomes a node in a persisted DAG; the self-model graph is also saved to disk. Retrieval children automatically inject top assemblies plus stored memory snippets into their prompts.
- **Stateful controller**: FAISS memory, assemblies, self-model graph, and PPO policy weights all persist across sessions. Typing `do-complete-reset` wipes everything (disk + in-memory state + snapshots) to start fresh.
- **Visualization suite**: `visualization.py` plots EI grids, cooperation scores, memory growth, EI/λ trends, the assembly DAG, and the self-model graph in real time or via snapshots.
- **Tests**: Pytest modules cover the assembly index behavior, FAISS persistence, and the EI2 helper.

## Running LLMs via the Wrapper

Instead of hitting an LLM directly, the super-organism wraps LM Studio–compatible APIs (or any OpenAI-style endpoint). You configure `LLM_API_URL` and `LLM_MODEL_NAME`, and the parent/children issue requests through `requests.post`. What the wrapper adds beyond a plain LLM call:

1. **Contextual memory** – Retrieval children fetch top-k relevant memories before prompting the model, ensuring responses stay grounded.
2. **Adaptive routing** – The PPO meta-policy decides which specialist to invoke based on recent EI/cooperation.
3. **Experience-driven storage** – A bootstrap λ floor lets early turns persist to memory even before PPO learns; the floor decays over time so the controller eventually takes full control.
4. **Assembly provenance** – Users can inspect which memory nodes and agents contributed to an answer through the assembly DAG.
5. **Self-reflective feedback** – EI leverages qualitative feedback entropy to reward answers that reduce uncertainty.

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

## Tests

```bash
pytest
```

The suite validates the assembly index DAG logic, FAISS persistence, and the EI2 helper’s numerical properties. Tests stub out heavy dependencies so they run quickly without GPU/LLM access.
