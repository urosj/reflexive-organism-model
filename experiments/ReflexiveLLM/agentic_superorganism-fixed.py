#!/usr/bin/env python3
# --------------------------------------------------------------
#  Agentic Super‑Organism (LM Studio + hierarchical RL)
#  – single‑file prototype
# --------------------------------------------------------------

import json, time, random, math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict, fields

import numpy as np
import faiss                     # vector store (Citation 1 & 2)
import torch                      # PPO meta-policy (Citation 4)
import torch.nn as nn
import torch.optim as optim
import requests                   # LM Studio HTTP client
from sentence_transformers import SentenceTransformer
import networkx as nx             # self-model graph
from networkx.readwrite import json_graph
import os

try:
    from visualization import VisualizationSystem
except Exception:
    VisualizationSystem = None

# --------------------------------------------------------------
# 1️⃣  Core utilities
# --------------------------------------------------------------

def now() -> float:
    """simple time helper used for timestamps."""
    return time.time()


# --------------------------------------------------------------
# 2️⃣  Vector store M(t) – persistent memory
# --------------------------------------------------------------

class MemoryStore:
    """
    FAISS + Sentence‑Transformer wrapper.
    Stores (embedding, text) pairs and can retrieve the top‑k most similar chunks.
    """

    def __init__(self, dim: int = 384, storage_dir: Optional[Path] = None):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)          # flat L2 index
        self.texts: List[str] = []                    # parallel list of raw strings
        self.storage_dir = Path(storage_dir) if storage_dir else Path(os.getenv("MEM_STORE_DIR", "memory_store"))
        self.index_path = self.storage_dir / "store.faiss"
        self.texts_path = self.storage_dir / "texts.json"
        try:
            self.emb_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
        except Exception as e:
            print(f"Failed to initialize embedding model: {e}")
            raise
        self._load_from_disk()

    def add(self, txt: str):
        try:
            emb = self.emb_model.encode([txt], normalize_embeddings=True)
            self.index.add(emb.astype(np.float32))
            self.texts.append(txt)
        except Exception as e:
            print(f"Failed to add text to memory: {e}")

    def save(self):
        try:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
            self.texts_path.write_text(json.dumps(self.texts))
        except Exception as e:
            print(f"Failed to persist memory store: {e}")

    def search(self, query: str, k: int = 4) -> List[str]:
        results = self.search_with_indices(query, k=k)
        return [txt for _, txt in results]

    def search_with_indices(self, query: str, k: int = 4) -> List[Tuple[int, str]]:
        try:
            q_emb = self.emb_model.encode([query], normalize_embeddings=True)
            D, I = self.index.search(q_emb.astype(np.float32), k)
            hits = []
            for idx in I[0]:
                if 0 <= idx < len(self.texts):
                    hits.append((int(idx), self.texts[idx]))
            return hits
        except Exception as e:
            print(f"Failed to search memory: {e}")
            return []

    def _load_from_disk(self):
        if not self.storage_dir.exists():
            return
        try:
            if self.index_path.exists() and self.texts_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                self.texts = json.loads(self.texts_path.read_text())
        except Exception as e:
            print(f"Failed to load memory store: {e}")


# --------------------------------------------------------------
# 3️⃣  Experience index (EI) and spatial EI field
# --------------------------------------------------------------

class ExperienceField:
    """
    2‑D grid that accumulates stochastic boosts proportional to
    cooperation scores.  Acts as the "spatial EI sensor" (Citation 5).
    """

    def __init__(self, width: int = 20, height: int = 20):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width), dtype=np.float32)

    def boost(self, x: int, y: int, amount: float):
        """Add a stochastic bump at (x,y)."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] += amount * random.random()

    def mean(self) -> float:
        return float(np.mean(self.grid))

    def as_vector(self) -> np.ndarray:
        """Flattened version useful for the meta‑policy."""
        return self.grid.flatten()


# --------------------------------------------------------------
# 4️⃣  Assembly index – DAG of reusable chunks
# --------------------------------------------------------------

@dataclass
class AssemblyNode:
    node_id: str
    level: int
    content: str
    sources: List[str]
    activation_count: int = 0
    last_used_turn: Optional[str] = None
    quality: float = 0.0
    tags: List[str] = field(default_factory=list)


class AssemblyIndex:
    """Tracks assemblies (memory chunks, child outputs, parent summaries) as a DAG."""

    def __init__(self):
        self.nodes: Dict[str, AssemblyNode] = {}
        self.graph = nx.DiGraph()

    def ensure_memory_node(self, mem_idx: int, content: str) -> str:
        node_id = f"mem_{mem_idx}"
        if node_id not in self.nodes:
            self.nodes[node_id] = AssemblyNode(
                node_id=node_id,
                level=0,
                content=content,
                sources=[],
                tags=["memory"],
            )
            self.graph.add_node(node_id, level=0, text=content, kind="memory")
        return node_id

    def create_composite_node(
        self,
        node_id: str,
        level: int,
        content: str,
        sources: List[str],
        tags: Optional[List[str]] = None,
    ) -> str:
        if node_id not in self.nodes:
            self.nodes[node_id] = AssemblyNode(
                node_id=node_id,
                level=level,
                content=content,
                sources=list(sources),
                tags=list(tags or []),
            )
            self.graph.add_node(node_id, level=level, text=content, kind="assembly")
        else:
            existing = self.nodes[node_id]
            existing.content = content
            existing.sources = list(sources)
            if tags:
                existing.tags = list(tags)
            self.graph.nodes[node_id]["text"] = content
        for src in sources:
            self.graph.add_edge(src, node_id)
        return node_id

    def register_activation(self, node_id: str, turn_id: str, quality: float):
        node = self.nodes.get(node_id)
        if not node:
            return
        node.activation_count += 1
        node.last_used_turn = turn_id
        node.quality = 0.8 * node.quality + 0.2 * quality
        self.graph.add_node(turn_id, kind="turn", ei=quality)
        self.graph.add_edge(node_id, turn_id, weight=quality)
        return turn_id

    def retrieve_context(self, limit: int = 3, min_level: int = 1) -> List[str]:
        candidates = [
            node
            for node in self.nodes.values()
            if node.level >= min_level and node.content
        ]
        candidates.sort(
            key=lambda n: (
                round(n.quality, 6),
                n.activation_count,
                float(n.last_used_turn[1:]) if n.last_used_turn and n.last_used_turn.startswith("t") else 0.0,
            ),
            reverse=True,
        )
        snippets = []
        for node in candidates[:limit]:
            snippets.append(f"[{node.node_id}] {node.content}")
        return snippets

    def save(self, path: Path):
        path = Path(path)
        payload = {
            "nodes": [asdict(node) for node in self.nodes.values()],
            "graph_nodes": [
                {"id": n, "attrs": data} for n, data in self.graph.nodes(data=True)
            ],
            "graph_edges": [
                {"source": u, "target": v, "attrs": data} for u, v, data in self.graph.edges(data=True)
            ],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    @classmethod
    def load_from_file(cls, path: Path):
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return cls()

        instance = cls()
        node_fields = {f.name for f in fields(AssemblyNode)}
        for entry in payload.get("nodes", []):
            filtered = {k: entry[k] for k in node_fields if k in entry}
            try:
                node = AssemblyNode(**filtered)
            except TypeError:
                continue
            instance.nodes[node.node_id] = node

        graph = nx.DiGraph()
        for node_entry in payload.get("graph_nodes", []):
            graph.add_node(node_entry.get("id"), **node_entry.get("attrs", {}))
        for edge_entry in payload.get("graph_edges", []):
            graph.add_edge(
                edge_entry.get("source"),
                edge_entry.get("target"),
                **edge_entry.get("attrs", {}),
            )
        instance.graph = graph
        return instance


# --------------------------------------------------------------
# 5️⃣  Cooperation scoring (Citation 1)
# --------------------------------------------------------------

def compute_cooperation(parent, child) -> float:
    """
    Returns a scalar in [0,1] that mixes three terms:
      * calibration   – how well child's past messages match parent expectation,
      * causal usefulness – transfer‑entropy like influence on parent's decisions,
      * mutual information between child and other children.
    The concrete formulas are placeholders; they illustrate the idea.
    """
    # ----- calibration (simple cosine similarity of latest embeddings) -----
    if child.recent_emb is None or parent.recent_emb is None:
        cal = 0.0
    else:
        try:
            cal = float(
                np.dot(child.recent_emb, parent.recent_emb)
                / (np.linalg.norm(child.recent_emb) * np.linalg.norm(parent.recent_emb) + 1e-8)
            )
        except Exception:
            cal = 0.0

    # ----- causal usefulness (how often child's output appears in parent's next turn) -----
    caus = child.last_used_in_parent

    # ----- mutual information among siblings (average pairwise similarity) -----
    mi = 0.0
    if parent.children:
        try:
            sims = [
                np.dot(child.recent_emb, sib.recent_emb)
                / (np.linalg.norm(child.recent_emb) * np.linalg.norm(sib.recent_emb) + 1e-8)
                for sib in parent.children if sib is not child and sib.recent_emb is not None
            ]
            mi = float(np.mean(sims)) if sims else 0.0
        except Exception:
            mi = 0.0

    # weighted sum (weights are arbitrary but can be learned later)
    coop = 0.4 * cal + 0.3 * caus + 0.3 * mi
    return max(0.0, min(1.0, coop))


def update_mutual_info(parent):
    """
    Refreshes each child's `recent_emb` with the latest embedding of its output.
    (Called after every turn – see Citation 3.)
    """
    for child in parent.children:
        if child.last_output is not None and hasattr(child, 'mem'):
            try:
                emb = parent.mem.emb_model.encode([child.last_output], normalize_embeddings=True)[0]
                child.recent_emb = emb
            except Exception as e:
                print(f"Failed to update mutual info for {child.name}: {e}")


# --------------------------------------------------------------
# 5️⃣  Agent classes (parent + children + grandchildren)
# --------------------------------------------------------------

class RetrievalChild:
    """
    A lightweight "retrieval" agent that can be called by the parent.
    It owns its own memory slice (shared with ancestors) and produces
    a text answer based on the retrieved chunks plus the user prompt.
    """

    _id_counter = 0

    def __init__(self, mem: MemoryStore, assembly_index: Optional[AssemblyIndex] = None, name: Optional[str] = None):
        self.mem = mem
        self.name = name or f"child_{RetrievalChild._id_counter}"
        RetrievalChild._id_counter += 1

        # runtime state
        self.last_output: Optional[str] = None
        self.recent_emb: Optional[np.ndarray] = None   # updated by `update_mutual_info`
        self.last_used_in_parent: float = 0.0       # causal usefulness estimate
        self.last_retrieved_indices: List[int] = []
        self.last_retrieved_texts: List[str] = []
        self.assembly_index = assembly_index

    def answer(self, user_msg: str) -> str:
        """Retrieve relevant chunks, prepend a short system prompt and call the LLM."""
        try:
            retrieved = self.mem.search_with_indices(user_msg, k=4)
            self.last_retrieved_indices = [idx for idx, _ in retrieved]
            self.last_retrieved_texts = [txt for _, txt in retrieved]
            sections: List[str] = []
            if self.assembly_index:
                assemblies = self.assembly_index.retrieve_context(limit=2)
                if assemblies:
                    sections.append("Assembly context:\n" + "\n".join(assemblies))

            if self.last_retrieved_texts:
                sections.append("Memory snippets:\n" + "\n".join(self.last_retrieved_texts))

            context = "\n\n".join(sections).strip()
            if not context:
                context = "No explicit context available."

            system_prompt = (
                "You are a specialist retrieval sub‑agent. "
                "Use the following pieces of knowledge to answer concisely."
            )

            # Use environment variable for model name
            model_name = os.getenv("LLM_MODEL_NAME", "gpt-oss-120b")

            timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "600"))

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"{context}\n\nQuestion: {user_msg}"},
                ],
                "temperature": 0.7,
            }

            # Use environment variable for API endpoint
            api_url = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")

            resp = requests.post(
                api_url,
                json=payload,
                timeout=timeout  # configurable timeout
            )
            resp.raise_for_status()  # Raise exception for bad status codes

            answer = resp.json()["choices"][0]["message"]["content"]
            self.last_output = answer
            return answer
        except Exception as e:
            print(f"Error in RetrievalChild.answer: {e}")
            return f"Error processing query: {str(e)}"


class SuperAgent:
    """
    The *parent* organism.  It owns:
      • the global memory store `M(t)`,
      • a list of children (depth‑1) and grandchildren (depth‑2),
      • the spatial EI field,
      • a tiny PPO meta‑policy that decides actions each turn.
    """

    def __init__(self):
        # ---- core components -------------------------------------------------
        try:
            mem_dir = Path(os.getenv("MEM_STORE_DIR", "memory_store"))
            self.mem = MemoryStore(storage_dir=mem_dir)
        except Exception as e:
            print(f"Failed to initialize memory store: {e}")
            raise

        self.ei_grid = ExperienceField(width=20, height=20)
        self.assembly_path = Path(os.getenv("ASSEMBLY_INDEX_PATH", "assembly_index.json"))
        try:
            self.assembly_index = AssemblyIndex.load_from_file(self.assembly_path)
        except Exception as e:
            print(f"Failed to load assembly index: {e}")
            self.assembly_index = AssemblyIndex()

        # ---- hierarchical agents ---------------------------------------------
        self.children: List[RetrievalChild] = []
        self.grandchildren: List[RetrievalChild] = []   # created by hierarchical spawn

        # ---- self-model graph ------------------------------------------------
        self.graph_path = Path(os.getenv("SELF_MODEL_PATH", "self_model.json"))
        self.graph = self._load_self_model()

        # ---- meta‑policy (tiny PPO) -----------------------------------------
        try:
            self.policy_path = Path(os.getenv("POLICY_PATH", "meta_policy.pt"))
            self.meta_policy = MetaPolicyNetwork(
                n_children=0,          # will be resized after first child is added
                grid_dim=self.ei_grid.width * self.ei_grid.height,
            )
            self.optimizer = optim.Adam(self.meta_policy.parameters(), lr=1e-3)
            self._load_policy()
        except Exception as e:
            print(f"Failed to initialize meta-policy: {e}")
            raise

        # ---- bookkeeping -----------------------------------------------------
        self.recent_emb: Optional[np.ndarray] = None   # parent's own embedding of last turn
        self.last_user_msg: Optional[str] = None
        self._last_ei: float = 0.0
        self._last_direct_retrieval: Optional[List[Tuple[int, str]]] = None
        self.enable_feedback_prompt: bool = True
        self._last_feedback_text: Optional[str] = None
        self._feedback_history: List[int] = []  # binary signals
        self._graph_dirty: bool = False
        self._turn_count: int = 0
        self._lam_floor = float(os.getenv("LAM_BOOTSTRAP_START", "0.9"))
        self._lam_floor_decay = float(os.getenv("LAM_BOOTSTRAP_DECAY", "0.97"))
        self._lam_floor_min = float(os.getenv("LAM_BOOTSTRAP_MIN", "0.2"))
        self.visualization_enabled = os.getenv("ENABLE_VISUALIZATION", "0") == "1"
        self.visualizer = None
        self._visualization_dir = Path(os.getenv("VISUALIZATION_DIR", "visualizations"))
        if self.visualization_enabled and VisualizationSystem:
            try:
                self.visualizer = VisualizationSystem(self)
                self._visualization_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Failed to initialize visualization system: {e}")
                self.visualizer = None

    # ------------------------------------------------------------------
    # 5️⃣ a  Hierarchical spawning (Citation 3)
    # ------------------------------------------------------------------

    def _hierarchical_spawn(self, depth: int = 2):
        """
        Create a child and optionally a grand‑child.
        Grand‑children inherit the *same* memory store (`self.mem`) so they
        automatically see everything the parent has written.
        """
        try:
            child = RetrievalChild(self.mem, assembly_index=self.assembly_index)
            self.children.append(child)

            if depth > 1:
                # create a grand‑child that lives under `child`
                gc = RetrievalChild(self.mem, assembly_index=self.assembly_index, name=f"{child.name}_gc")
                self.grandchildren.append(gc)

            # update meta‑policy input size (number of selectable agents)
            total_agents = len(self.children) + len(self.grandchildren)
            self.meta_policy.resize_action_space(total_agents)

        except Exception as e:
            print(f"Failed to spawn hierarchical agent: {e}")
        else:
            self._persist_policy()

    # ------------------------------------------------------------------
    # 5️⃣ b  Meta‑policy forward & training step (Citation 4)
    # ------------------------------------------------------------------

    def _select_actions(self, user_msg: str) -> Tuple[List[int], float, Optional[np.ndarray]]:
        """
        Returns:
          * a list of indices of agents to call this turn,
          * the altruism mixing coefficient λ(t).
        The policy receives as input:
          – flattened EI grid,
          – current number of children,
          – (optionally) a tiny embedding of the user message.
        """
        try:
            # ----- prepare observation -----------------------------------------
            if self.recent_emb is None:
                # first turn: embed the raw user text
                try:
                    self.recent_emb = self.mem.emb_model.encode([user_msg], normalize_embeddings=True)[0]
                except Exception as e:
                    print(f"Failed to encode initial user message: {e}")
                    self.recent_emb = np.zeros(384, dtype=np.float32)

            obs = np.concatenate(
                [
                    self.ei_grid.as_vector(),
                    np.array([len(self.children) + len(self.grandchildren)], dtype=np.float32),
                    self.recent_emb,
                ]
            )
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)  # (1, dim)

            with torch.no_grad():
                probs, lam = self.meta_policy(obs_tensor)

            probs_np = probs.squeeze(0).detach().cpu().numpy()
            if probs_np.ndim == 0:
                probs_np = np.array([float(probs_np)])

            # Ensure we don't sample from invalid indices
            valid_indices = len(probs_np)
            if valid_indices > 0:
                chosen = np.random.choice(valid_indices, p=probs_np)

                # we interpret the chosen index as "call this agent".
                # If the index is out of range (possible early in training) we fall back to no call.
                if 0 <= chosen < len(self.children) + len(self.grandchildren):
                    return [chosen], float(lam.item()), probs_np
                else:
                    return [], float(lam.item()), probs_np
            else:
                return [], 0.5, probs_np

        except Exception as e:
            print(f"Error in _select_actions: {e}")
            return [], 0.5, None

    def _policy_update(self, log_prob: torch.Tensor, advantage: float):
        """Simple REINFORCE‑style PPO step (single‑step)."""
        try:
            loss = -log_prob * advantage
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.meta_policy.parameters(), max_norm=1.0)

            self.optimizer.step()
        except Exception as e:
            print(f"Error in _policy_update: {e}")

    # ------------------------------------------------------------------
    # 5️⃣ c  One interaction turn
    # ------------------------------------------------------------------

    def run_one_turn(self, user_msg: str) -> str:
        """
        Main loop for a single user turn:
          1. decide which agents to call & λ(t);
          2. collect their answers (if any);
          3. possibly store the user‑turn in memory;
          4. compute EI and update the spatial field;
          5. log everything into the self‑model graph;
          6. perform a tiny policy gradient step.
        """

        # ---- 1️⃣ decide actions -------------------------------------------
        self._last_direct_retrieval = None
        lam = 0.5
        lam_for_storage = max(lam, self._lam_floor)
        try:
            chosen_idxs, lam, policy_probs = self._select_actions(user_msg)
            lam_for_storage = max(lam, self._lam_floor)

            # split index space: first N children, then M grandchildren
            n_child = len(self.children)
            answers = []
            used_children = []

            for idx in chosen_idxs:
                if idx < n_child:
                    agent = self.children[idx]
                elif idx - n_child < len(self.grandchildren):
                    agent = self.grandchildren[idx - n_child]
                else:
                    continue  # Skip invalid indices

                try:
                    ans = agent.answer(user_msg)
                    answers.append(ans)
                    used_children.append(agent)
                except Exception as e:
                    print(f"Failed to get answer from agent {idx}: {e}")

            # If no child was called we fall back to a direct LLM call (the "parent" answer)
            if not answers:
                parent_answer = self._call_llm_direct(user_msg)
            else:
                # simple majority vote / concatenation – you can replace with any aggregation rule
                parent_answer = " ".join(answers)
                self._last_direct_retrieval = None

        except Exception as e:
            print(f"Error during turn execution: {e}")
            return f"System error: {str(e)}"

        print(f"\n🤖 Agent: {parent_answer}")
        expected_answer = self._prompt_feedback()
        self._last_feedback_text = expected_answer
        feedback_signal = self._quantify_feedback(expected_answer)
        if feedback_signal is not None:
            self._feedback_history.append(feedback_signal)
            self._feedback_history = self._feedback_history[-100:]

        turn_id = f"t{int(now())}"
        retrieved_indices: List[int] = []
        for child in used_children:
            retrieved_indices.extend(getattr(child, "last_retrieved_indices", []))
        if not retrieved_indices and self._last_direct_retrieval:
            retrieved_indices.extend(idx for idx, _ in self._last_direct_retrieval)

        # ---- 2️⃣ optionally store user turn (meta‑policy decides via λ) ---
        try:
            if lam_for_storage > 0.5:                     # bootstrap-adjusted threshold
                self.mem.add(user_msg)
                self.mem.add(parent_answer)
                self.mem.save()
        except Exception as e:
            print(f"Failed to store in memory: {e}")

        # ---- 3️⃣ compute EI ------------------------------------------------
        try:
            ei = self._compute_EI2(
                user_msg,
                parent_answer,
                expected_answer=expected_answer,
                retrieved_indices=retrieved_indices or None,
                policy_logits=policy_probs,
            )
        except Exception as e:
            print(f"Error computing EI2: {e}")
            try:
                ei = self._compute_EI(user_msg, parent_answer)
            except Exception as inner_e:
                print(f"Fallback EI computation failed: {inner_e}")
                ei = 0.0
        self._last_ei = ei

        # Record assemblies once EI is known
        try:
            self._record_assemblies(turn_id, used_children, parent_answer, ei)
        except Exception as e:
            print(f"Error recording assemblies: {e}")
        else:
            self._persist_assembly_index()
            self._persist_self_model()
            self._refresh_visualization(turn_id)

        # boost a random location in the spatial field proportionally to EI (Citation 5)
        try:
            x, y = random.randint(0, self.ei_grid.width - 1), random.randint(0, self.ei_grid.height - 1)
            self.ei_grid.boost(x, y, amount=ei)
        except Exception as e:
            print(f"Error boosting EI grid: {e}")

        # ---- 4️⃣ update cooperation scores for every child that participated
        try:
            for child in used_children:
                coop = compute_cooperation(self, child)          # Citation 1
                child.last_used_in_parent = coop                  # stores causal usefulness

            # also refresh mutual‑info embeddings (Citation 3)
            update_mutual_info(self)
        except Exception as e:
            print(f"Error updating cooperation scores: {e}")

        # ---- 5️⃣ log turn into self‑model graph ---------------------------
        try:
            self.graph.add_node(
                turn_id,
                user=user_msg,
                answer=parent_answer,
                ei=ei,
                lam=lam,
                lam_floor=self._lam_floor,
                lam_effective=lam_for_storage,
                mem_size=len(self.mem.texts),
            )
            self.graph.add_edge("parent", turn_id)

            for child in used_children:
                coop = compute_cooperation(self, child)
                self.graph.add_edge(child.name, turn_id, coop=coop)
            self._graph_dirty = True
        except Exception as e:
            print(f"Error logging to graph: {e}")

        # ---- 6️⃣ tiny policy-gradient step ---------------------------------
        try:
            # reward = EI (the internal signal we want to maximise)
            advantage = ei - self.ei_grid.mean()          # simple baseline
            if not chosen_idxs:  # No action taken
                log_prob = None
            else:
                # retrieve log‑prob of the action that was actually taken
                obs = np.concatenate(
                    [
                        self.ei_grid.as_vector(),
                        np.array([len(self.children) + len(self.grandchildren)], dtype=np.float32),
                        self.recent_emb if self.recent_emb is not None else np.zeros(384, dtype=np.float32)
                    ]
                )
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
                probs, _ = self.meta_policy(obs_tensor)
                probs_vec = probs.squeeze(0)
                if probs_vec.ndim == 0:
                    probs_vec = probs_vec.unsqueeze(0)

                # Handle case where we might have invalid index
                if 0 <= chosen_idxs[0] < probs_vec.shape[0]:
                    log_prob = torch.log(probs_vec[chosen_idxs[0]])
                else:
                    log_prob = None

            if log_prob is not None and log_prob.requires_grad:
                self._policy_update(log_prob, advantage)
                self._persist_policy()
        except Exception as e:
            print(f"Error in policy update: {e}")

        # ---- housekeeping -------------------------------------------------
        try:
            self.last_user_msg = user_msg
            if parent_answer is not None:
                self.recent_emb = self.mem.emb_model.encode([parent_answer], normalize_embeddings=True)[0]
        except Exception as e:
            print(f"Error in housekeeping: {e}")

        self._turn_count += 1
        self._lam_floor = max(self._lam_floor * self._lam_floor_decay, self._lam_floor_min)

        return parent_answer

    # ------------------------------------------------------------------
    # Helper methods used inside the turn
    # ------------------------------------------------------------------

    def _call_llm_direct(self, msg: str) -> str:
        """Fallback when no child is selected – direct call to LM‑Studio."""
        try:
            self._last_direct_retrieval = self.mem.search_with_indices(msg, k=4)
            model_name = os.getenv("LLM_MODEL_NAME", "gpt-oss-120b")
            timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "600"))

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are the parent superorganism."},
                    {"role": "user",   "content": msg},
                ],
                "temperature": 0.7,
            }

            api_url = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")

            memory_context = "\n".join(txt for _, txt in self._last_direct_retrieval) if self._last_direct_retrieval else ""
            assembly_context = "\n".join(self._get_assembly_context(limit=3))
            context_sections = []
            if assembly_context:
                context_sections.append("Assembly context:\n" + assembly_context)
            if memory_context:
                context_sections.append("Memory snippets:\n" + memory_context)
            if context_sections:
                payload["messages"].insert(
                    1,
                    {"role": "system", "content": "\n\n".join(context_sections)},
                )

            resp = requests.post(
                api_url,
                json=payload,
                timeout=timeout
            )
            resp.raise_for_status()

            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in direct LLM call: {e}")
            return f"Direct LLM error: {str(e)}"

    def _record_assemblies(
        self,
        turn_id: str,
        used_children: List[RetrievalChild],
        parent_answer: str,
        ei: float,
    ):
        """Create/update assembly DAG nodes for this turn."""
        if not hasattr(self, "assembly_index"):
            return

        child_nodes: List[str] = []

        for child in used_children:
            mem_nodes: List[str] = []
            for idx, txt in zip(
                getattr(child, "last_retrieved_indices", []),
                getattr(child, "last_retrieved_texts", []),
            ):
                mem_nodes.append(self.assembly_index.ensure_memory_node(idx, txt))

            child_node_id = f"{child.name}_{turn_id}"
            child_node_id = self.assembly_index.create_composite_node(
                node_id=child_node_id,
                level=1,
                content=child.last_output or "",
                sources=mem_nodes,
                tags=[child.name],
            )
            self.assembly_index.register_activation(child_node_id, turn_id, quality=ei)
            child_nodes.append(child_node_id)

        parent_node_id = f"parent_{turn_id}"
        parent_node_id = self.assembly_index.create_composite_node(
            node_id=parent_node_id,
            level=2,
            content=parent_answer,
            sources=child_nodes,
            tags=["parent"],
        )
        self.assembly_index.register_activation(parent_node_id, turn_id, quality=ei)

        # If no children fired, still bind any memory nodes touched directly.
        if not child_nodes and getattr(self, "_last_direct_retrieval", None):
            mem_nodes = []
            for idx, txt in self._last_direct_retrieval:
                mem_nodes.append(self.assembly_index.ensure_memory_node(idx, txt))
            for node_id in mem_nodes:
                self.assembly_index.graph.add_edge(node_id, parent_node_id)

    def _prompt_feedback(self) -> Optional[str]:
        """Ask the user for qualitative feedback to feed EI groundedness."""
        if not self.enable_feedback_prompt:
            return None
        try:
            feedback = input("📝 Quick feedback on the answer? (press Enter to skip): ").strip()
            return feedback or None
        except EOFError:
            return None

    def _quantify_feedback(self, feedback: Optional[str]) -> Optional[int]:
        if not feedback:
            return None
        normalized = feedback.lower().strip()
        positive_markers = ["good", "great", "expected", "correct", "in line", "ok", "yes"]
        negative_markers = ["not", "bad", "incorrect", "wrong", "miss", "no", "off"]
        if any(marker in normalized for marker in positive_markers) and not any(
            marker in normalized for marker in negative_markers
        ):
            return 1
        if any(marker in normalized for marker in negative_markers) and not any(
            marker in normalized for marker in positive_markers
        ):
            return 0
        return None

    def _get_assembly_context(self, limit: int = 3) -> List[str]:
        if hasattr(self, "assembly_index"):
            return self.assembly_index.retrieve_context(limit=limit)
        return []

    def _persist_assembly_index(self):
        if not getattr(self, "assembly_index", None):
            return
        try:
            self.assembly_index.save(self.assembly_path)
        except Exception as e:
            print(f"Failed to persist assembly index: {e}")

    def _persist_self_model(self, force: bool = False):
        if not getattr(self, "graph", None):
            return
        if not force and not self._graph_dirty:
            return
        try:
            data = json_graph.node_link_data(self.graph, edges="links")
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
            self.graph_path.write_text(json.dumps(data))
            self._graph_dirty = False
        except Exception as e:
            print(f"Failed to persist self-model graph: {e}")

    def _load_self_model(self) -> nx.DiGraph:
        path = getattr(self, "graph_path", Path("self_model.json"))
        if path.exists():
            try:
                data = json.loads(path.read_text())
                graph = json_graph.node_link_graph(data)
                if "parent" not in graph:
                    graph.add_node("parent")
                return graph
            except Exception as e:
                print(f"Failed to load self-model graph: {e}")
        graph = nx.DiGraph()
        graph.add_node("parent")
        return graph

    def _persist_policy(self):
        if not hasattr(self, "meta_policy") or not hasattr(self, "optimizer"):
            return
        try:
            payload = {
                "model_state": self.meta_policy.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
            }
            self.policy_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, self.policy_path)
        except Exception as e:
            print(f"Failed to persist policy: {e}")

    def _load_policy(self):
        if not hasattr(self, "policy_path") or not self.policy_path.exists():
            return
        try:
            payload = torch.load(self.policy_path, map_location="cpu")
            model_state = payload.get("model_state")
            if model_state:
                head_weight = model_state.get("policy_head.weight")
                if head_weight is not None:
                    target_size = head_weight.shape[0]
                    self.meta_policy.resize_action_space(target_size)
                self.meta_policy.load_state_dict(model_state, strict=False)
            opt_state = payload.get("optimizer_state")
            if opt_state:
                self.optimizer.load_state_dict(opt_state)
        except Exception as e:
            print(f"Failed to load policy: {e}")

    def _refresh_visualization(self, turn_id: Optional[str] = None):
        if not self.visualizer:
            return
        try:
            self.visualizer.visualize_all_components()
            filename = self._visualization_dir / (f"turn_{turn_id}.png" if turn_id else "snapshot.png")
            self.visualizer.save_snapshot(str(filename))
        except Exception as e:
            print(f"Error updating visualization: {e}")

    def shutdown(self):
        """Flush persistent artifacts and capture a final visualization."""
        try:
            if hasattr(self, "mem"):
                self.mem.save()
            self._persist_assembly_index()
            self._persist_self_model(force=True)
            self._persist_policy()
            self._refresh_visualization("final")
        except Exception as e:
            print(f"Error during shutdown: {e}")

    def _reset_persistent_state(self):
        try:
            for path in [self.mem.index_path, self.mem.texts_path, self.assembly_path, self.graph_path, self.policy_path]:
                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass
            if self._visualization_dir.exists():
                for img in self._visualization_dir.glob("*.png"):
                    try:
                        img.unlink()
                    except Exception:
                        pass
            self.mem = MemoryStore(storage_dir=self.mem.storage_dir)
            self.assembly_index = AssemblyIndex()
            self.graph = nx.DiGraph()
            self.graph.add_node("parent")
            self.meta_policy = MetaPolicyNetwork(
                n_children=0,
                grid_dim=self.ei_grid.width * self.ei_grid.height,
            )
            self.optimizer = optim.Adam(self.meta_policy.parameters(), lr=1e-3)
            self.children.clear()
            self.grandchildren.clear()
            self.recent_emb = None
            self.last_user_msg = None
            self._last_ei = 0.0
            self._last_direct_retrieval = None
            self._last_feedback_text = None
            self._feedback_history = []
            self._graph_dirty = False
            self._turn_count = 0
            self._lam_floor = float(os.getenv("LAM_BOOTSTRAP_START", "0.9"))
            self._ms_history = []
            self._policy_history = []
            self._mem_state_history = []
            if self.visualizer:
                self.visualizer.turn_history.clear()
                self.visualizer._seen_turn_nodes = set()
        except Exception as e:
            print(f"Failed to reset persistent state: {e}")


    def _compute_EI2(
        self,
        user_msg: str,
        answer: str,
        expected_answer: Optional[str] = None,
        retrieved_indices: Optional[List[int]] = None,
        policy_logits: Optional[np.ndarray] = None,
    ) -> float:
        """
        Approximate the Reflexive Organism EI using information-theoretic proxies.

        Components:
          * I(M;S): mutual information between memory state and sensing.
          * T_{M→Π}: transfer entropy from memory to policy logits.
          * Prediction groundedness: BLEU-style overlap with an expected answer.
          * Assembly coverage: fraction of distinct memory nodes consulted.

        All sub-scores are normalized to [0,1] then combined via learned weights.
        """
        eps = 1e-6

        # Ensure history buffers exist for running covariance/entropy estimates.
        if not hasattr(self, "_ms_history"):
            self._ms_history: List[Tuple[np.ndarray, np.ndarray]] = []
        if not hasattr(self, "_policy_history"):
            self._policy_history: List[np.ndarray] = []
        if not hasattr(self, "_mem_state_history"):
            self._mem_state_history: List[np.ndarray] = []

        try:
            user_emb = self.mem.emb_model.encode([user_msg], normalize_embeddings=True)[0]
        except Exception:
            user_emb = np.zeros(384, dtype=np.float32)

        # Build an aggregate memory state from the chunks retrieved for this turn.
        retrieved_chunks: List[str] = []
        try:
            retrieved_chunks = self.mem.search(user_msg, k=4)
            if retrieved_chunks:
                mem_embs = self.mem.emb_model.encode(retrieved_chunks, normalize_embeddings=True)
                mem_state = np.mean(mem_embs, axis=0)
            else:
                mem_state = np.zeros_like(user_emb)
        except Exception:
            mem_state = np.zeros_like(user_emb)

        # Store recent samples (bounded buffer to control conditioning).
        self._ms_history.append((mem_state, user_emb))
        self._mem_state_history.append(mem_state)
        max_hist = 256
        if len(self._ms_history) > max_hist:
            self._ms_history = self._ms_history[-max_hist:]
        if len(self._mem_state_history) > max_hist:
            self._mem_state_history = self._mem_state_history[-max_hist:]

        # Helper: Gaussian mutual information estimate using covariance matrices.
        def _gaussian_mi(history: List[Tuple[np.ndarray, np.ndarray]]) -> float:
            if len(history) < 5:
                return 0.0
            mem_samples = np.stack([m for m, _ in history])
            sense_samples = np.stack([s for _, s in history])
            joint_samples = np.concatenate([mem_samples, sense_samples], axis=1)

            cov_m = np.cov(mem_samples, rowvar=False) + eps * np.eye(mem_samples.shape[1])
            cov_s = np.cov(sense_samples, rowvar=False) + eps * np.eye(sense_samples.shape[1])
            cov_joint = np.cov(joint_samples, rowvar=False) + eps * np.eye(joint_samples.shape[1])

            sign_m, logdet_m = np.linalg.slogdet(cov_m)
            sign_s, logdet_s = np.linalg.slogdet(cov_s)
            sign_joint, logdet_joint = np.linalg.slogdet(cov_joint)

            if sign_m <= 0 or sign_s <= 0 or sign_joint <= 0:
                return 0.0

            mi_val = 0.5 * (logdet_m + logdet_s - logdet_joint)
            return max(0.0, float(mi_val))

        mi_ms = _gaussian_mi(self._ms_history)
        mi_scale = math.log(len(user_emb) + 1)
        mi_norm = max(0.0, min(1.0, mi_ms / max(mi_scale, eps)))

        # Policy logits default to the current meta-policy probabilities.
        if policy_logits is None:
            obs = np.concatenate(
                [
                    self.ei_grid.as_vector(),
                    np.array([len(self.children) + len(self.grandchildren)], dtype=np.float32),
                    self.recent_emb if self.recent_emb is not None else np.zeros(384, dtype=np.float32),
                ]
            )
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            with torch.no_grad():
                probs, _ = self.meta_policy(obs_tensor)
            policy_vec = probs.squeeze(0).detach().cpu().numpy()
            if np.isscalar(policy_vec):
                policy_vec = np.array([policy_vec], dtype=np.float32)
        else:
            policy_vec = np.asarray(policy_logits, dtype=np.float32)
            if policy_vec.ndim == 0:
                policy_vec = np.array([policy_vec], dtype=np.float32)

        self._policy_history.append(policy_vec)
        if len(self._policy_history) > max_hist:
            self._policy_history = self._policy_history[-max_hist:]

        # Transfer entropy proxy comparing conditional variance reductions.
        def _transfer_entropy(
            policy_hist: List[np.ndarray],
            mem_hist: List[np.ndarray],
            window: int = 8,
        ) -> float:
            if len(policy_hist) <= window + 1 or len(mem_hist) <= window + 1:
                return 0.0

            recent_policy = np.stack(policy_hist[-(window + 1):])
            recent_mem = np.stack(mem_hist[-(window + 1):])

            baseline = np.var(recent_policy[1:] - recent_policy[:-1], axis=0).mean()
            conditioned = np.var(
                (recent_policy[1:] - recent_policy[:-1]) - (recent_mem[1:] - recent_mem[:-1]),
                axis=0,
            ).mean()

            te_val = math.log((baseline + eps) / (conditioned + eps))
            return max(0.0, te_val)

        te_mp = _transfer_entropy(self._policy_history, self._mem_state_history)
        te_scale = math.log(len(policy_vec) + 1)
        te_norm = max(0.0, min(1.0, te_mp / max(te_scale, eps)))

        # Prediction groundedness via hybrid lexical + feedback entropy proxy.
        reference = expected_answer or getattr(self, "_last_feedback_text", None) or getattr(self, "_expected_answer", None) or user_msg
        ref_tokens = reference.lower().split()
        ans_tokens = answer.lower().split()
        lexical_grounded = 0.0
        if ref_tokens and ans_tokens:
            ref_counts = {}
            for tok in ref_tokens:
                ref_counts[tok] = ref_counts.get(tok, 0) + 1
            overlap = 0
            consumed = {}
            for tok in ans_tokens:
                cnt = consumed.get(tok, 0)
                if tok in ref_counts and cnt < ref_counts[tok]:
                    overlap += 1
                    consumed[tok] = cnt + 1
            lexical_grounded = overlap / len(ans_tokens)

        entropy_grounded = 0.0
        if len(self._feedback_history) >= 1:
            feedback_arr = np.array(self._feedback_history, dtype=np.float32)
            p_current = feedback_arr[-1]
            prev_mean = np.mean(feedback_arr[:-1]) if len(feedback_arr) > 1 else p_current
            delta = p_current - prev_mean
            entropy_grounded = max(0.0, delta)

        grounded = max(0.0, min(1.0, 0.6 * lexical_grounded + 0.4 * entropy_grounded))

        # Assembly coverage derived from distinct memory hits.
        total_memory = max(1, len(self.mem.texts))
        if retrieved_indices:
            coverage = len(set(retrieved_indices)) / total_memory
        else:
            coverage = min(1.0, len(retrieved_chunks) / total_memory) if retrieved_chunks else 0.0

        coverage = max(0.0, min(1.0, coverage))

        # Weighted combination (weights sum to 1.0)
        weights = getattr(self, "_ei_weights", (0.35, 0.25, 0.25, 0.15))
        w1, w2, w3, w4 = weights
        ei_score = w1 * mi_norm + w2 * te_norm + w3 * grounded + w4 * coverage
        return max(0.0, min(1.0, float(ei_score)))

    def _compute_EI(self, user_msg: str, answer: str) -> float:
        """
        Very lightweight proxy for the full EI formula (Citation 3).
        We combine four normalized factors:
          * memory‑to‑sensing mutual information,
          * transfer entropy from memory to policy (approximated by cosine similarity),
          * prediction groundedness (BLEU‑like overlap between answer and expected answer),
          * reuse/coverage (fraction of memory accessed in this turn).
        The result is clipped to [0,1].
        """
        try:
            # 1️⃣ MI(M;S) – we approximate with embedding similarity
            user_emb = self.mem.emb_model.encode([user_msg], normalize_embeddings=True)[0]

            if self.last_user_msg is not None:
                mem_emb = self.mem.emb_model.encode([self.last_user_msg], normalize_embeddings=True)[0]
            else:
                # Create a dummy embedding for first turn
                mem_emb = np.zeros_like(user_emb)

            mi_ms = float(np.dot(mem_emb, user_emb) / (np.linalg.norm(mem_emb) * np.linalg.norm(user_emb) + 1e-8))

            # 2️⃣ TE(M→Π) – approximated by similarity between memory embedding and answer embedding
            ans_emb = self.mem.emb_model.encode([answer], normalize_embeddings=True)[0]
            te_mp = float(np.dot(mem_emb, ans_emb) / (np.linalg.norm(mem_emb) * np.linalg.norm(ans_emb) + 1e-8))

            # 3️⃣ Prediction groundedness – simple token overlap ratio
            set_user = set(user_msg.lower().split())
            set_ans = set(answer.lower().split())
            if len(set_user | set_ans) > 0:
                grounded = len(set_user & set_ans) / len(set_user | set_ans)
            else:
                grounded = 0.0

            # 4️⃣ Reuse/coverage – fraction of memory entries accessed in this turn
            try:
                reuse_count = (len(self.mem.search(user_msg)) + len(self.mem.search(answer)))
                reuse = min(1.0, reuse_count / 8.0)
            except Exception:
                reuse = 0.0

            ei_raw = mi_ms * te_mp * grounded * reuse
            return max(0.0, min(1.0, ei_raw))
        except Exception as e:
            print(f"Error computing EI: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Interactive REPL – runs until the user types "exit"
    # ------------------------------------------------------------------

    def interactive_loop(self):
        print("=== Agentic Retrieval Super‑Organism (type 'exit' to quit) ===")
        try:
            while True:
                try:
                    usr = input("\n🧑‍💻 You: ").strip()
                except EOFError:
                    break
                if usr.lower() in {"exit", "quit"}:
                    break

                if usr.lower() == "do-complete-reset":
                    self._reset_persistent_state()
                    print("[info] persistent state reset")
                    continue

                start = time.time()
                answer = self.run_one_turn(usr)
                elapsed = time.time() - start

                try:
                    ei = getattr(self, "_last_ei", None)
                    if ei is None:
                        ei = self._compute_EI2(
                            usr,
                            answer,
                            expected_answer=getattr(self, "_last_feedback_text", None),
                        )
                    print(f"[info] turn took {elapsed:.2f}s, EI≈{ei:.3f}")
                except Exception as e:
                    print(f"[info] turn took {elapsed:.2f}s, EI computation failed: {e}")
        finally:
            self.shutdown()


# --------------------------------------------------------------
# 6️⃣  Tiny PPO meta‑policy network (Citation 4)
# --------------------------------------------------------------

class MetaPolicyNetwork(nn.Module):
    """
    Input:   vector = flatten(EI_grid) + #agents + parent embedding
    Output:
        * a categorical distribution over *which* child to call,
        * a scalar λ(t) ∈ [0,1] that mixes altruism vs. selfishness.
    The network can be resized on‑the‑fly when new children appear.
    """

    def __init__(self, n_children: int, grid_dim: int):
        super().__init__()
        self.n_children = max(1, n_children)          # at least one dummy action
        self.grid_dim = grid_dim

        hidden = 128
        self.fc1 = nn.Linear(grid_dim + 1 + 384, hidden)   # +1 for count, +384 for embedding
        self.fc2 = nn.Linear(hidden, hidden)

        # policy head (softmax over agents)
        self.policy_head = nn.Linear(hidden, self.n_children)

        # λ‑head (sigmoid to keep it in [0,1])
        self.lambda_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            h = torch.relu(self.fc1(x))
            h = torch.relu(self.fc2(h))

            logits = self.policy_head(h)
            probs = torch.softmax(logits, dim=-1)

            lam = torch.sigmoid(self.lambda_head(h)).squeeze(-1)   # altruism mixing coefficient
            return probs, lam
        except Exception as e:
            print(f"Error in MetaPolicyNetwork forward: {e}")
            # Return defaults to prevent crashes
            dummy_probs = torch.ones(self.n_children) / self.n_children
            dummy_lam = torch.tensor(0.5)
            return dummy_probs, dummy_lam

    def resize_action_space(self, new_n: int):
        """Replace the policy head with a larger one when more agents appear."""
        try:
            if new_n == self.n_children:
                return
            self.n_children = max(1, new_n)
            # Store current weights for initialization of new layer
            old_weights = self.policy_head.weight.data.clone()
            old_bias = self.policy_head.bias.data.clone()

            # Create new policy head with larger size
            self.policy_head = nn.Linear(self.fc2.out_features, self.n_children)

            # Initialize the new weights using existing ones as much as possible
            if len(old_weights) >= self.n_children:
                # Copy existing weights and pad with zeros for new dimensions
                self.policy_head.weight.data[:self.n_children] = old_weights[:self.n_children]
                self.policy_head.bias.data[:self.n_children] = old_bias[:self.n_children]
        except Exception as e:
            print(f"Error resizing action space: {e}")


# --------------------------------------------------------------
# 7️⃣  Entry point
# --------------------------------------------------------------

if __name__ == "__main__":
    try:
        agent = SuperAgent()

        # Bootstrap: create a few children so the meta‑policy has something to select.
        # You can change the numbers or call `_hierarchical_spawn` later from the REPL.
        for _ in range(2):          # start with 2 depth‑1 children
            agent._hierarchical_spawn(depth=2)   # each also spawns a grand‑child

        agent.interactive_loop()
    except Exception as e:
        print(f"Critical error in main execution: {e}")
