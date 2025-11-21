#!/usr/bin/env python3
# --------------------------------------------------------------
#  Agentic Super‑Organism (LM Studio + hierarchical RL)
#  – single‑file prototype
# --------------------------------------------------------------

import json, time, random, math, re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, asdict, fields
from collections import defaultdict

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
# LLM model selection helpers
# --------------------------------------------------------------


def _parse_llm_pool() -> List[str]:
    """
    Returns the configured pool of LLM model names.
    Supports:
      • LLM_MODEL_POOL as comma‑separated list or JSON array.
      • fallback to single LLM_MODEL_NAME.
    """
    pool_env = os.getenv("LLM_MODEL_POOL")
    models: List[str] = []
    if pool_env:
        try:
            if pool_env.strip().startswith("["):
                models = [m for m in json.loads(pool_env) if isinstance(m, str)]
            else:
                models = [m.strip() for m in pool_env.split(",") if m.strip()]
        except Exception:
            models = []

    if not models:
        fallback = os.getenv("LLM_MODEL_NAME", "gpt-oss-120b")
        models = [fallback]
    return models


def choose_llm_model(
    model_pool: Optional[List[str]] = None,
    available: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[str, str]:
    """
    Pick a model, preferring ones that are known to be available on the server.
    If weights are provided, sample proportional to weight (fallback is uniform).
    """
    pool = model_pool if model_pool is not None else _parse_llm_pool()
    candidates = []
    if available:
        candidates = [m for m in pool if m in available] if pool else list(available)
    if not candidates:
        candidates = pool
    if not candidates:
        return os.getenv("LLM_MODEL_NAME", "gpt-oss-120b"), "fallback_env"

    if weights:
        w = [max(0.0, float(weights.get(m, 0.0))) for m in candidates]
        if any(val > 0 for val in w):
            total = sum(w)
            r = random.random() * total
            cumulative = 0.0
            for m, val in zip(candidates, w):
                cumulative += val
                if r <= cumulative:
                    return m, "weighted"
    return random.choice(candidates), "random"


def _top_models_by_weight(
    pool: List[str],
    weights: Optional[Dict[str, float]],
    k: int = 3,
    tolerance: float = 0.1,
) -> List[str]:
    """
    Return up to k models whose weights are near the top weight (within tolerance).
    If weights are missing, returns empty.
    """
    if not weights:
        return []
    scored = [(m, weights.get(m, 0.0)) for m in pool]
    if not scored:
        return []
    scored.sort(key=lambda x: x[1], reverse=True)
    top_weight = scored[0][1]
    candidates = [m for m, w in scored if w >= top_weight - tolerance]
    return candidates[:k]


def _bucket_for_text(text: str, emb_model, bits: int = 12) -> str:
    """Hash an embedding into a coarse bucket using sign bits (topic proxy)."""
    try:
        emb = emb_model.encode([text], normalize_embeddings=True)[0]
        signs = (emb[:bits] > 0).astype(np.int8)
        return "b:" + "".join(map(str, signs.tolist()))
    except Exception:
        return "b:global"


def _ensure_bucket_weights(
    weight_map: Dict[str, Dict[str, float]],
    bucket: str,
    models: List[str],
    default: float = 1.0,
) -> Dict[str, float]:
    """Ensure a bucket weight dict exists and is initialized for given models."""
    if bucket not in weight_map:
        weight_map[bucket] = {}
    bucket_weights = weight_map[bucket]
    for m in models:
        if m not in bucket_weights:
            bucket_weights[m] = default
    return bucket_weights


def _load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load optional JSON config; return defaults on failure."""
    cfg_path = Path(path or os.getenv("CONFIG_PATH", "config.json"))
    if not cfg_path.exists():
        return {}
    try:
        data = json.loads(cfg_path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"Failed to load config {cfg_path}: {e}")
        return {}


def _filter_embed_models(models: Optional[List[str]]) -> List[str]:
    """Remove embedding-only models by heuristic substring match."""
    if not models:
        return []
    return [m for m in models if "embed" not in m.lower()]
    try:
        return json.loads(cfg_path.read_text())
    except Exception as e:
        print(f"Failed to load config {cfg_path}: {e}")
        return {}


def _prune_model(
    model_name: str,
    pool: Optional[List[str]],
    weights: Optional[Dict[str, float]],
    known: Optional[List[str]] = None,
):
    """Remove a model from selection pool/weights/known list when server rejects it."""
    if pool is not None and model_name in pool:
        try:
            pool[:] = [m for m in pool if m != model_name]
            print(f"[model pruning] removed '{model_name}' from pool after server rejection")
        except Exception:
            pass
    if known is not None and model_name in known:
        try:
            known[:] = [m for m in known if m != model_name]
            print(f"[model pruning] removed '{model_name}' from known models after server rejection")
        except Exception:
            pass
    if weights is not None:
        try:
            # Handle nested bucketed weights
            if all(isinstance(v, dict) for v in weights.values()):
                for bucket, wmap in weights.items():
                    if model_name in wmap:
                        wmap.pop(model_name, None)
                        print(f"[model pruning] removed '{model_name}' from bucket '{bucket}' weights after server rejection")
            elif model_name in weights:
                weights.pop(model_name, None)
                print(f"[model pruning] removed '{model_name}' from weights after server rejection")
        except Exception:
            pass


def _extract_http_error(err: Exception) -> str:
    """Return a short description including response body if available."""
    try:
        resp = getattr(err, "response", None)
        if resp is None:
            return str(err)
        body = ""
        try:
            body = resp.text[:500]
        except Exception:
            body = "<unreadable>"
        return f"{resp.status_code} {resp.reason} – {body}"
    except Exception:
        return str(err)


def _discover_models(api_url: Optional[str]) -> List[str]:
    """
    Try to fetch available models from the LLM server.
    Only returns models that are marked as loaded if the server reports that flag.
    Returns an empty list on failure.
    """
    if not api_url:
        return []
    base = api_url.rstrip("/")
    # Common LM-Studio style: /v1/chat/completions -> /v1/models
    if base.endswith("/chat/completions"):
        models_endpoint = base.rsplit("/chat/completions", 1)[0] + "/models"
    else:
        models_endpoint = f"{base}/models"
    try:
        resp = requests.get(models_endpoint, timeout=float(os.getenv("LLM_REQUEST_TIMEOUT", "600")))
        resp.raise_for_status()
        data = resp.json()
        ids = []
        if isinstance(data, dict):
            items = data.get("data") or data.get("models") or []
            for item in items:
                mid = item.get("id") if isinstance(item, dict) else None
                if isinstance(mid, str):
                    if isinstance(item, dict) and "loaded" in item and not item.get("loaded"):
                        continue
                    ids.append(mid)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str):
                    ids.append(item)
                elif isinstance(item, dict) and isinstance(item.get("id"), str):
                    if "loaded" in item and not item.get("loaded"):
                        continue
                    ids.append(item["id"])
        return ids
    except Exception as e:
        print(f"Could not list models from server ({models_endpoint}): {e}")
        return []

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
            if not isinstance(payload, dict):
                return cls()
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

    def compute_scalar_ai(self) -> float:
        """
        Crude scalar AI(t) ≈ average depth (level) of non-memory assemblies,
        normalized to [0, 1].
        """
        if not self.nodes:
            return 0.0

        levels = [n.level for n in self.nodes.values() if getattr(n, "level", 0) > 0]
        if not levels:
            return 0.0

        avg_level = float(np.mean(levels))
        scale = 10.0  # saturation level: ~10 levels => AI ≈ 1
        ai = avg_level / scale
        if ai < 0.0:
            ai = 0.0
        if ai > 1.0:
            ai = 1.0
        return ai


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

    def __init__(
        self,
        mem: MemoryStore,
        assembly_index: Optional[AssemblyIndex] = None,
        name: Optional[str] = None,
        model_pool: Optional[List[str]] = None,
        available_models: Optional[List[str]] = None,
        small_pool_threshold: int = 3,
    ):
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
        self.model_pool = _filter_embed_models(model_pool)
        self.available_models = _filter_embed_models(available_models)
        # Shared bucket -> weights map
        self.model_weights: Optional[Dict[str, Dict[str, float]]] = None
        self.model_usage_counts: Dict[str, int] = {}
        self.last_model_used: Optional[str] = None
        self.last_model_reason: Optional[str] = None
        self.last_model_candidates: Optional[List[str]] = None
        self.current_bucket: Optional[str] = None
        self.last_model_scores: Optional[Dict[str, float]] = None
        self.small_pool_threshold = max(1, int(small_pool_threshold))
        # Reflexive suppression state
        self.phi_impact_ema: float = 0.0
        self.suppression_level: float = 0.0
        # RCP-based replication statistics
        self.ei_star_ema: float = 0.0
        self.coop_ema: float = 0.0
        self.usage_count: int = 0
        self.lineage_tag: Optional[str] = None
        # RCP-based retirement metadata
        self.last_used_turn: int = 0
        self.retired: bool = False
        # RCP-based replication statistics
        self.ei_star_ema: float = 0.0
        self.coop_ema: float = 0.0
        self.usage_count: int = 0
        self.lineage_tag: Optional[str] = None

    def answer(self, user_msg: str) -> str:
        """Retrieve relevant chunks, prepend a short system prompt and call the LLM."""
        try:
            gate_source = getattr(self, "parent", self)
            phi_gate = getattr(gate_source, "_phi_gate", 1.0)
            base_k = 4
            scale_factor = 0.5 + 0.5 * phi_gate
            k_adj = int(round(base_k * scale_factor))
            if k_adj < 2:
                k_adj = 2
            retrieved = self.mem.search_with_indices(user_msg, k=k_adj)
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
            # Choose one or more models if top weights are close (bucketed by topic)
            self.current_bucket = _bucket_for_text(user_msg, self.mem.emb_model)
            pool = self.available_models or self.model_pool or []
            bucket_weights = _ensure_bucket_weights(self.model_weights, self.current_bucket, pool)
            small_pool_size = getattr(self, "small_pool_threshold", 3)
            if len(pool) < small_pool_size:
                selected_models, selection_reason = self._select_small_pool_models(pool, bucket_weights)
            else:
                selected_models = None
                selection_reason = None

            if selected_models is None:
                top_candidates = _top_models_by_weight(pool, bucket_weights, k=3, tolerance=0.1)
                selected_models = []
                selection_reason = "random"
                if top_candidates and len(top_candidates) > 1:
                    # Query multiple models for this question
                    selected_models = top_candidates
                    selection_reason = "multi-weighted"
                else:
                    model_name, reason = choose_llm_model(self.model_pool, available=self.available_models, weights=bucket_weights)
                    selected_models = [model_name]
                    selection_reason = reason

            answers = []
            for m in selected_models:
                try:
                    ans = self._query_model(system_prompt, context, user_msg, m)
                    answers.append((m, ans))
                except Exception as e:
                    print(f"Failed query for model {m}: {e}")
            if not answers:
                return "Error processing query: no answers"

            # Evaluate answers with other models (peer models not used in answering)
            best_model = selected_models[0]
            best_answer = answers[0][1]
            if len(answers) > 1:
                best_model, best_answer = self._score_answers_with_peers(user_msg, answers)
                # Reinforce winning model locally for this child
                self._reinforce_local_weights(best_model, answers)
            self.last_model_used = best_model
            # Make the reason clearer for multi-model evaluation
            if selection_reason == "multi-weighted":
                self.last_model_reason = "multi-evaluated (weighted cluster)"
            else:
                self.last_model_reason = selection_reason
            self.last_model_candidates = selected_models
            self.last_output = best_answer
            return best_answer

        except requests.HTTPError as e:
            print(f"Error in RetrievalChild.answer: {_extract_http_error(e)}")
            if self.last_model_used:
                _prune_model(self.last_model_used, self.model_pool, self.model_weights, getattr(self, "available_models", None))
            return f"Error processing query: {_extract_http_error(e)}"
        except Exception as e:
            print(f"Error in RetrievalChild.answer: {e}")
            return f"Error processing query: {str(e)}"

    def _select_small_pool_models(self, pool, bucket_weights):
        """
        Small-pool model selection for len(pool) < 3.

        Uses a simple UCB-style score:
            score = weight + lam / sqrt(1 + usage_count)
        and returns either 1 or 2 models depending on how close the
        top two scores are.
        """
        import math

        # Ensure usage counts exist for each model in the pool
        for m in pool:
            self.model_usage_counts.setdefault(m, 0)

        lam = 0.2  # exploration bonus strength
        scored = []
        for m in pool:
            w = bucket_weights.get(m, 1.0)
            u = self.model_usage_counts.get(m, 0)
            bonus = lam / math.sqrt(1.0 + u)
            scored.append((m, w + bonus))

        # Sort by descending score
        scored.sort(key=lambda x: x[1], reverse=True)

        if len(scored) == 0:
            # Fallback: no models, should not happen, return empty selection
            return [], "small-pool-empty"

        if len(scored) == 1:
            top_models = [scored[0][0]]
            reason = "small-pool-single"
        else:
            # len(scored) == 2 because we only enter this method when len(pool) < 3
            top_score = scored[0][1]
            second_score = scored[1][1]
            if abs(top_score - second_score) < 0.05:
                # Scores are similar: query both models
                top_models = [scored[0][0], scored[1][0]]
                reason = "small-pool-multi"
            else:
                # Clear winner: query only the top model
                top_models = [scored[0][0]]
                reason = "small-pool-single"

        # Update usage counts for selected models
        for m in top_models:
            self.model_usage_counts[m] = self.model_usage_counts.get(m, 0) + 1

        return top_models, reason

    def _query_model(self, system_prompt: str, context: str, user_msg: str, model_name: str) -> str:
        timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "600"))
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"{context}\n\nQuestion: {user_msg}"},
            ],
            "temperature": 0.7,
        }
        api_url = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
        resp = requests.post(api_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _score_answers_with_peers(self, user_msg: str, answers: List[Tuple[str, str]]) -> Tuple[str, str]:
        """
        Use other models (not in answers) to rate the candidate answers.
        Returns best (model, answer) based on average embedding agreement.
        """
        try:
            pool = self.available_models or self.model_pool or []
            small_pool_size = getattr(self, "small_pool_threshold", 3)
            if len(pool) < small_pool_size:
                # Small pool: skip peer-model scoring and choose by bucket weights
                bucket_weights = _ensure_bucket_weights(self.model_weights, self.current_bucket, pool)
                best_answer = None
                best_score = float("-inf")
                for ans in answers:
                    model_name = ans[0]
                    score = bucket_weights.get(model_name, 1.0)
                    if score > best_score:
                        best_score = score
                        best_answer = ans
                return best_answer

            used_models = {m for m, _ in answers}
            bucket = self.current_bucket or "b:global"
            bw = self.model_weights.get(bucket, {}) if self.model_weights else {}

            peer_models = [m for m in pool if m not in used_models and "embed" not in m.lower()]
            peer_models = peer_models[:2]  # primary peer set cap

            if not peer_models:
                # Fallback: pick highest-weight answer.
                # NOTE: use the *bucket-local* weights (bw) if they exist.
                if bw:
                    # bw contains weights for the current bucket.
                    answers.sort(key=lambda x: bw.get(x[0], 0.0), reverse=True)
                    return answers[0]

                # If this bucket has no weights, optionally fall back to a
                # global bucket (e.g. "b:global") if available.
                global_bw = self.model_weights.get("b:global", {}) if self.model_weights else {}
                if global_bw:
                    answers.sort(key=lambda x: global_bw.get(x[0], 0.0), reverse=True)
                    return answers[0]

                # If no weights at all, there is no justified "best" answer.
                # Use a random choice instead of always returning the first.
                return random.choice(answers)

            # Collect peer judgments via embeddings
            candidates_text = [ans for _, ans in answers]
            cand_embs = self.mem.emb_model.encode(candidates_text, normalize_embeddings=True)

            peer_scores = [0.0 for _ in answers]
            for peer in peer_models:
                try:
                    peer_ans = self._call_llm_for_feedback(user_msg, peer)
                    if not peer_ans:
                        continue
                    peer_emb = self.mem.emb_model.encode([peer_ans], normalize_embeddings=True)[0]
                    for idx, cemb in enumerate(cand_embs):
                        num = float(np.dot(peer_emb, cemb))
                        den = float(np.linalg.norm(peer_emb) * np.linalg.norm(cemb) + 1e-8)
                        peer_scores[idx] += num / den
                    try:
                        norm_scores = [max(0.0, min(1.0, s)) for s in peer_scores]
                        avg_score = float(np.mean(norm_scores)) if norm_scores else None
                        if avg_score is not None:
                            self._last_peer_prediction = avg_score
                        else:
                            self._last_peer_prediction = None
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Peer scoring failed for {peer}: {e}")

            best_idx = int(np.argmax(peer_scores))
            return answers[best_idx]
        except Exception as e:
            print(f"Failed to score answers with peers: {e}")
            return answers[0]

    def _call_llm_for_feedback(self, msg: str, model_name: str) -> Optional[str]:
        """Lightweight LLM call for model comparison; avoids mutating main state."""
        # Skip obvious embedding-only models
        if "embed" in model_name.lower():
            return None
        try:
            timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "600"))
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a peer reviewer model. Provide a concise answer."},
                    {"role": "user", "content": msg},
                ],
                "temperature": 0.2,
            }
            api_url = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
            resp = requests.post(api_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            print(f"Feedback LLM call failed for model {model_name}: {_extract_http_error(e)}")
            _prune_model(model_name, self.model_pool, self.model_weights, getattr(self, "available_models", None))
            return None
        except Exception as e:
            print(f"Feedback LLM call failed for model {model_name}: {e}")
            return None

    def _reinforce_local_weights(self, best_model: str, answers: List[Tuple[str, str]]):
        """Simple local reinforcement: bump winner, decay losers."""
        if not self.model_weights:
            return
        bucket = self.current_bucket or "b:global"
        pool = self.available_models or self.model_pool or []
        bucket_weights = _ensure_bucket_weights(self.model_weights, bucket, pool)
        alpha = 0.15
        floor = 0.1
        models_in_round = [m for m, _ in answers]
        max_w = max((bucket_weights.get(m, 1.0) for m in models_in_round), default=1.0)
        for m in models_in_round:
            prev = bucket_weights.get(m, 1.0)
            if m == best_model:
                updated = prev + alpha * (1.0 - prev)
            else:
                updated = prev - alpha * (prev - floor)
            bucket_weights[m] = max(floor, min(1.5 * max_w, updated))


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
        self.config = _load_config()
        forced = self.config.get("forced_models")
        if forced and isinstance(forced, list):
            self.model_pool = _filter_embed_models([m for m in forced if isinstance(m, str)])
        else:
            self.model_pool = _filter_embed_models(_parse_llm_pool())
        self._known_models = _filter_embed_models(_discover_models(os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")))
        if forced and isinstance(forced, list):
            self._known_models = [m for m in self._known_models if m in forced]
            self.model_pool = [m for m in self.model_pool if m in forced]
        if self._known_models:
            # Prefer discovered models over env defaults to avoid invalid selections.
            self.model_pool = list(self._known_models)
        self.model_weights: Dict[str, Dict[str, float]] = {}
        self.model_weights_path = Path(os.getenv("MODEL_WEIGHTS_PATH", "model_weights.json"))
        self._init_model_weights()
        self._load_model_weights()
        # Exploration: every N steps, force a random model choice (per bucket)
        self.epsilon_every_n = int(self.config.get("epsilon_every_n", 0))
        self.small_pool_threshold = int(self.config.get("small_pool_threshold", 3))
        self._step_counter = 0

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
        self._memory_use_history: List[bool] = []
        self._retrieval_depth_history: List[float] = []
        self._graph_dirty: bool = False
        self._turn_count: int = 0
        self._lam_floor = float(os.getenv("LAM_BOOTSTRAP_START", "0.9"))
        self._lam_floor_decay = float(os.getenv("LAM_BOOTSTRAP_DECAY", "0.97"))
        self._lam_floor_min = float(os.getenv("LAM_BOOTSTRAP_MIN", "0.2"))
        self._last_model_used: Optional[str] = None
        self._last_model_reason: Optional[str] = None
        self._last_model_candidates: Optional[List[str]] = None
        self._last_model_bucket: Optional[str] = None
        self.peer_review_method: str = str(self.config.get("peer_review_method", "similarity")).lower()
        self.visualization_enabled = os.getenv("ENABLE_VISUALIZATION", "0") == "1"
        self.visualizer = None
        self._visualization_dir = Path(os.getenv("VISUALIZATION_DIR", "visualizations"))
        self.global_turn_idx: int = 0
        if self.visualization_enabled and VisualizationSystem:
            try:
                self.visualizer = VisualizationSystem(self)
                self._visualization_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                print(f"Failed to initialize visualization system: {e}")
                self.visualizer = None
        # Last computed φ(t) = J(t) * ρ_compat(t) for this turn.
        self._last_phi = None
        # Last justification scalar J(t), coupling AI and EI*.
        self._last_J = None
        # Last compatibility term ρ_compat(t) between parent and children.
        self._last_rho_compat = None
        # Previous φ(t−1), used to measure relative change Δφ/φ.
        self._prev_phi = None
        # Restore any persisted conservation state from disk if available.
        self._restore_conservation_state()
        # Rolling history of sensing embeddings S_t (user messages) for world-coupling MI.
        self._sense_history: List[np.ndarray] = []
        # Rolling history of environment embeddings E_t (context / previous messages).
        self._env_history: List[np.ndarray] = []
        # History of (peer_pred, actual_outcome) pairs for prediction calibration.
        self._prediction_history: List[Tuple[float, float]] = []
        # Embedding of the last user message, reused as a simple proxy for E_{t+1}.
        self._last_user_emb: Optional[np.ndarray] = None
        # Last peer model prediction of correctness (0..1) for calibration.
        self._last_peer_prediction: Optional[float] = None
        
        # ---- Adaptive RCP control parameters ------
        # Current strength of conservation penalty on the policy advantage.
        self.lambda_conserve = 0.1
        # Lower and upper bounds for lambda_conserve.
        self.lambda_min = 0.01
        self.lambda_max = 1.0
        # Target relative change |Δφ_rel| per turn; above this, conservation tightens.
        self.target_phi_delta = 0.05
        # EMA of |Δφ_rel|, used as a φ volatility measure.
        self._phi_delta_ema = 0.0
        # Smoothing factor for φ volatility EMA.
        self._phi_delta_ema_beta = 0.9
        # Per-turn gate in [0,1] used to modulate memory writes, retrieval, and exploration.
        self._phi_gate = 1.0

        # ---- RCP-driven spawning thresholds ------
        # Max |Δφ_rel| to treat φ as stable enough to spawn new agents.
        self.spawn_phi_stability_thresh = 0.05
        # Number of recent turns used to detect EI* stagnation.
        self.spawn_ei_stagnation_window = 5
        # Minimum EI* improvement over the window; below this counts as stagnation.
        self.spawn_ei_min_improvement = 0.01
        # Minimum number of turns between spawn events.
        self.spawn_cooldown = 10
        # Countdown until the next spawn is allowed.
        self._spawn_cooldown_counter = 0
        # History of EI* values used for stagnation detection.
        self._ei_history_for_spawning: List[float] = []

        # ---- RCP-driven suppression hyperparameters ------
        # |Δφ_rel| above this is treated as destabilizing when attributing impact to children.
        self.suppression_phi_thresh = 0.10
        # EMA smoothing factor for each child's φ impact.
        self.suppression_ema_beta = 0.8
        # Clamp range for per-child suppression_level.
        self.suppression_max = 1.0
        self.suppression_min = 0.0
        # Strength of suppression when adjusting selection logits (higher => stronger down-weight).
        self.suppression_alpha = 0.8
        # Above this suppression_level, children are effectively hard-masked from selection.
        self.suppression_hard_cutoff = 0.95

        # ---- RCP-driven retirement hyperparameters ------
        # Do not retire agents if total active agents would drop below this.
        self.retire_min_agents = 4
        # If an agent is unused for this many turns, it is considered inactive.
        self.retire_usage_inactive_turns = 50
        # EI* EMA below this marks an agent as low-quality.
        self.retire_ei_thresh = 0.3
        # Cooperation EMA below this marks an agent as poorly compatible.
        self.retire_coop_thresh = 0.3
        # Suppression_level above this marks an agent as highly destabilizing.
        self.retire_supp_min = 0.7
        # Minimum number of turns between retirement actions.
        self.retire_cooldown = 20
        # Countdown until the next retirement is allowed.
        self._retire_cooldown_counter = 0

        # ---- RCP-driven replication hyperparameters ------
        # EI* EMA must be at least this high for an agent to be replicated.
        self.replicate_ei_thresh = 0.7
        # Cooperation EMA must be at least this high to be considered φ-compatible.
        self.replicate_coop_thresh = 0.7
        # Suppression_level must be below this threshold to be eligible to clone.
        self.replicate_supp_max = 0.3
        # Agent must have been used at least this many times before replication.
        self.replicate_min_usage = 10
        # Minimum number of turns between replication events.
        self.replicate_cooldown = 20
        # Countdown until the next replication is allowed.
        self._replicate_cooldown_counter = 0
        # Hard cap on total agents (children + grandchildren) in the ecology.
        self.max_agents = 16

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
            child = RetrievalChild(
                self.mem,
                assembly_index=self.assembly_index,
                model_pool=self.model_pool,
                available_models=self._known_models,
                small_pool_threshold=self.small_pool_threshold,
                name=None,
            )
            child.parent = self
            child.model_weights = self.model_weights
            self.children.append(child)

            if depth > 1:
                # create a grand‑child that lives under `child`
                gc = RetrievalChild(
                    self.mem,
                    assembly_index=self.assembly_index,
                    name=f"{child.name}_gc",
                    model_pool=self.model_pool,
                    available_models=self._known_models,
                    small_pool_threshold=self.small_pool_threshold,
                )
                gc.parent = self
                gc.model_weights = self.model_weights
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

            # Gate exploration via phi stability by sharpening distribution
            gate = getattr(self, "_phi_gate", 1.0)
            tau_min = 0.5
            tau = tau_min + (1.0 - tau_min) * gate
            if tau > 0 and probs_np.size > 0:
                sharpened = probs_np ** (1.0 / max(tau, 1e-6))
                denom = sharpened.sum()
                if denom > 0:
                    probs_np = sharpened / denom

            # Apply suppression to logits/probabilities for children/grandchildren
            agents = list(self.children) + list(self.grandchildren)
            if len(probs_np) == len(agents) and len(probs_np) > 0:
                adjusted_logits = []
                alpha = getattr(self, "suppression_alpha", 0.8)
                min_scale = 0.1
                hard_cut = getattr(self, "suppression_hard_cutoff", 0.95)
                for idx, base_prob in enumerate(probs_np):
                    agent = agents[idx]
                    if getattr(agent, "retired", False):
                        adjusted_logits.append(-1e9)
                        continue
                    supp = float(getattr(agent, "suppression_level", 0.0))
                    scale = 1.0 - alpha * supp
                    if scale < min_scale:
                        scale = min_scale
                    if supp >= hard_cut:
                        adjusted_logits.append(-1e9)
                    else:
                        adjusted_logits.append(math.log(max(base_prob, 1e-12)) + math.log(scale))
                adjusted_logits = np.array(adjusted_logits, dtype=np.float32)
                exps = np.exp(adjusted_logits - np.max(adjusted_logits))
                denom = exps.sum()
                if denom > 0:
                    probs_np = exps / denom

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

        self.global_turn_idx += 1

        # ---- 1️⃣ decide actions -------------------------------------------
        self._last_direct_retrieval = None
        lam = 0.5
        lam_for_storage = max(lam, self._lam_floor)
        try:
            bucket_for_turn = _bucket_for_text(user_msg, self.mem.emb_model)
            self._last_model_bucket = bucket_for_turn

            self._step_counter += 1
            force_random = self.epsilon_every_n > 0 and (self._step_counter % self.epsilon_every_n == 0)
            chosen_idxs, lam, policy_probs = self._select_actions(user_msg)
            lam_for_storage = max(lam, self._lam_floor)

            # Exploration: override policy selection with a random agent when configured
            random_direct = False
            if force_random:
                total_agents = len(self.children) + len(self.grandchildren)
                if total_agents > 0:
                    rand_idx = random.randrange(total_agents)
                    chosen_idxs = [rand_idx]
                    print(f"[info] epsilon exploration: forcing random agent index {rand_idx} of {total_agents}")
                else:
                    random_direct = True

            # split index space: first N children, then M grandchildren
            n_child = len(self.children)
            answers = []
            used_children = []
            parent_models_used: List[str] = []

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
                    if getattr(agent, "last_model_used", None):
                        parent_models_used.append(agent.last_model_used)
                except Exception as e:
                    print(f"Failed to get answer from agent {idx}: {e}")

            # If no child was called we fall back to a direct LLM call (the "parent" answer)
            if random_direct and not answers:
                try:
                    # Randomly pick a model from known/pool and call directly
                    pool = self._known_models or self.model_pool or []
                    if pool:
                        override_model = random.choice(pool)
                    else:
                        override_model = None
                    parent_answer = self._call_llm_direct(user_msg, override_model=override_model)
                    if getattr(self, "_last_model_used", None):
                        parent_models_used = [self._last_model_used]
                    model_selection_info = [(self._last_model_used, self._last_model_reason, getattr(self, "_last_model_candidates", None))]
                    answers = []
                    used_children = []
                    print(f"[info] epsilon exploration: forced random direct model -> {self._last_model_used}")
                except Exception as e:
                    print(f"Epsilon exploration direct call failed: {e}")

            if not answers:
                parent_answer = self._call_llm_direct(user_msg)
                if getattr(self, "_last_model_used", None):
                    parent_models_used = [self._last_model_used]
                model_selection_info = [(self._last_model_used, self._last_model_reason, getattr(self, "_last_model_candidates", None))]
            else:
                # simple majority vote / concatenation – you can replace with any aggregation rule
                parent_answer = " ".join(answers)
                self._last_direct_retrieval = None
                if parent_models_used:
                    # Keep a representative model for quick reference
                    self._last_model_used = parent_models_used[0]
                model_selection_info = []
                for child in used_children:
                    if getattr(child, "last_model_used", None):
                        model_selection_info.append(
                            (
                                child.last_model_used,
                                getattr(child, "last_model_reason", None),
                                getattr(child, "last_model_candidates", None),
                            )
                        )

        except Exception as e:
            print(f"Error during turn execution: {e}")
            return f"System error: {str(e)}"

        if 'model_selection_info' in locals() and model_selection_info:
            info_lines = []
            for entry in model_selection_info:
                if len(entry) == 3:
                    m, r, cands = entry
                else:
                    m, r, cands = entry[0], entry[1], None
                if not m:
                    continue
                # Human-friendly reason text
                if r == "multi-weighted" or (r or "").startswith("multi-evaluated"):
                    suffix = " via multiple evaluated models"
                    if cands and len(cands) > 1:
                        suffix += f" (candidates: {', '.join(cands)})"
                elif r:
                    suffix = f" via {r}"
                else:
                    suffix = ""
                info_lines.append(f"{m}{suffix}")
            if info_lines:
                print(f"[model selection] {'; '.join(info_lines)}")

        print(f"\n🤖 Agent: {parent_answer}")
        expected_answer, feedback_signal = self._collect_feedback(user_msg, parent_answer, used_children, parent_models_used)
        self._last_feedback_text = expected_answer
        if feedback_signal is not None:
            self._feedback_history.append(feedback_signal)
            self._feedback_history = self._feedback_history[-100:]
            try:
                peer_pred = getattr(self, "_last_peer_prediction", None)
                if peer_pred is None:
                    fb_arr = np.array(self._feedback_history, dtype=np.float32)
                    if len(fb_arr) > 0:
                        peer_pred = float(fb_arr.mean())
                    else:
                        peer_pred = 0.5
                actual_outcome = float(feedback_signal)
                if hasattr(self, "_prediction_history"):
                    self._prediction_history.append((peer_pred, actual_outcome))
                    if len(self._prediction_history) > 256:
                        self._prediction_history.pop(0)
            except Exception as e:
                print(f"Failed to store prediction calibration entry: {e}")
        self._reinforce_models(parent_models_used, feedback_signal, bucket_for_turn)

        turn_id = f"t{int(now())}"
        retrieved_indices: List[int] = []
        for child in used_children:
            retrieved_indices.extend(getattr(child, "last_retrieved_indices", []))
        if not retrieved_indices and self._last_direct_retrieval:
            retrieved_indices.extend(idx for idx, _ in self._last_direct_retrieval)

        # ---- 2️⃣ optionally store user turn (meta‑policy decides via λ) ---
        try:
            if lam_for_storage > 0.5:                     # bootstrap-adjusted threshold
                gate = getattr(self, "_phi_gate", 1.0)
                if gate >= 1.0 or random.random() <= gate:
                    self.mem.add(user_msg)
                    annotated_answer = self._annotate_answer_with_models(parent_answer, parent_models_used)
                    self.mem.add(annotated_answer)
                    self.mem.save()
        except Exception as e:
            print(f"Failed to store in memory: {e}")

        # ---- 3️⃣ compute EI ------------------------------------------------
        try:
            print(f"[info] computing EI2 with indices={retrieved_indices or None}, expected={bool(expected_answer)}, policy_probs={'set' if policy_probs is not None else 'none'}")
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
                print("[info] computing fallback EI")
                ei = self._compute_EI(user_msg, parent_answer)
            except Exception as inner_e:
                print(f"Fallback EI computation failed: {inner_e}")
                ei = 0.0
        self._last_ei = ei

        # Record assemblies once EI is known
        try:
            self._record_assemblies(turn_id, used_children, parent_answer, ei, parent_models_used)
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

        # ---- Update last_used_turn for used children -------------------------
        try:
            for child in used_children:
                if not hasattr(child, "last_used_turn"):
                    child.last_used_turn = self.global_turn_idx
                else:
                    child.last_used_turn = self.global_turn_idx
                if not hasattr(child, "usage_count"):
                    child.usage_count = 0
        except Exception as e:
            print(f"[RCP] Error updating last_used_turn: {e}")

        # ---- 5️⃣ log turn into self‑model graph ---------------------------
        try:
            self.graph.add_node(
                turn_id,
                user=user_msg,
                answer=parent_answer,
                models=parent_models_used,
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

        # ---- Reflexive justification & conservation update -------------------
        try:
            # Compute EI* using EI2 plus world-coupling and calibration
            ei_star_t = self._compute_EI_star(ei)

            # Scalar AI from assembly graph (if available)
            ai_t = 0.0
            if hasattr(self, "assembly_index") and self.assembly_index is not None:
                ai_t = float(self.assembly_index.compute_scalar_ai())

            # Compatibility across participating children
            rho_compat_t = self._compute_rho_compat(used_children)

            # Justification and conserved-like product
            J_t = self._compute_justification(ai_t, ei_star_t, rho_compat_t)
            phi_t = J_t * rho_compat_t

            # Cache for later use and logging
            self._last_J = J_t
            self._last_rho_compat = rho_compat_t
            self._last_phi = phi_t

            # ---- Update phi volatility and adaptive lambda_conserve --------------
            try:
                prev_phi = self._prev_phi
                curr_phi = self._last_phi

                if prev_phi is not None and curr_phi is not None:
                    eps_phi = 1e-6
                    rel_change = (curr_phi - prev_phi) / (abs(prev_phi) + eps_phi)
                    abs_rel_change = abs(rel_change)

                    beta = self._phi_delta_ema_beta
                    self._phi_delta_ema = beta * self._phi_delta_ema + (1.0 - beta) * abs_rel_change

                    delta_target = self.target_phi_delta
                    if self._phi_delta_ema > delta_target:
                        self.lambda_conserve *= 1.0 + 0.1 * (
                            (self._phi_delta_ema - delta_target) / (delta_target + eps_phi)
                        )
                    else:
                        self.lambda_conserve *= 1.0 - 0.05 * (
                            (delta_target - self._phi_delta_ema) / (delta_target + eps_phi)
                        )

                    if self.lambda_conserve < self.lambda_min:
                        self.lambda_conserve = self.lambda_min
                    if self.lambda_conserve > self.lambda_max:
                        self.lambda_conserve = self.lambda_max

                    gamma = 2.0
                    min_gate = 0.2
                    phi_gate = 1.0 - gamma * abs_rel_change
                    if phi_gate < min_gate:
                        phi_gate = min_gate
                    if phi_gate > 1.0:
                        phi_gate = 1.0
                    self._phi_gate = phi_gate
                else:
                    self._phi_delta_ema = 0.0
                    self.lambda_conserve = max(self.lambda_conserve, self.lambda_min)
                    self._phi_gate = 1.0
            except Exception as e:
                print(f"Error updating phi volatility / lambda_conserve: {e}")

            # Track EI* history (use EI* if available, EI2 otherwise)
            ei_star_value = ei_star_t if ei_star_t is not None else ei
            self._ei_history_for_spawning.append(ei_star_value)
            if len(self._ei_history_for_spawning) > 100:
                self._ei_history_for_spawning.pop(0)

            # Optionally store into the self-model graph if it exists
            try:
                if getattr(self, "graph", None) is not None:
                    node_id = turn_id if "turn_id" in locals() else None
                    if node_id is not None and node_id in self.graph.nodes:
                        self.graph.nodes[node_id]["J"] = J_t
                        self.graph.nodes[node_id]["rho_compat"] = rho_compat_t
                        self.graph.nodes[node_id]["phi"] = phi_t
                        self._graph_dirty = True
            except Exception:
                pass
        except Exception as e:
            print(f"Error updating justification / conservation: {e}")

        # ---- RCP-driven spawn detection ------------------------------------
        spawn_triggered = False
        try:
            prev_phi = getattr(self, "_prev_phi", None)
            curr_phi = getattr(self, "_last_phi", None)
            eps_phi = 1e-6
            if prev_phi is not None and curr_phi is not None:
                phi_rel_change = abs((curr_phi - prev_phi) / (abs(prev_phi) + eps_phi))
            else:
                phi_rel_change = 0.0

            ei_stagnant = False
            n = getattr(self, "spawn_ei_stagnation_window", 5)
            if len(self._ei_history_for_spawning) >= n + 1:
                recent = self._ei_history_for_spawning[-(n + 1):]
                improvement = recent[-1] - recent[0]
                if improvement < self.spawn_ei_min_improvement:
                    ei_stagnant = True

            if (
                self._spawn_cooldown_counter <= 0
                and phi_rel_change < self.spawn_phi_stability_thresh
                and ei_stagnant
            ):
                spawn_triggered = True

            if self._spawn_cooldown_counter > 0:
                self._spawn_cooldown_counter -= 1
        except Exception as e:
            print(f"Error evaluating RCP spawn conditions: {e}")

        # ---- Update per-child φ impact and suppression -----------------------
        try:
            prev_phi = getattr(self, "_prev_phi", None)
            curr_phi = getattr(self, "_last_phi", None)
            eps_phi = 1e-6
            abs_rel_change = 0.0
            if prev_phi is not None and curr_phi is not None:
                abs_rel_change = abs((curr_phi - prev_phi) / (abs(prev_phi) + eps_phi))

            beta = self.suppression_ema_beta
            phi_thresh = self.suppression_phi_thresh

            for child in used_children:
                if not hasattr(child, "phi_impact_ema"):
                    child.phi_impact_ema = 0.0
                if not hasattr(child, "suppression_level"):
                    child.suppression_level = 0.0

                impact = 0.0
                if abs_rel_change > phi_thresh:
                    impact = abs_rel_change

                child.phi_impact_ema = beta * child.phi_impact_ema + (1.0 - beta) * impact

                if phi_thresh > 0.0:
                    raw_supp = child.phi_impact_ema / (phi_thresh * 2.0)
                else:
                    raw_supp = 0.0

                if raw_supp < self.suppression_min:
                    raw_supp = self.suppression_min
                if raw_supp > self.suppression_max:
                    raw_supp = self.suppression_max

                child.suppression_level = float(raw_supp)

            if used_children:
                try:
                    debug_info = [
                        (
                            getattr(c, "name", f"child_{i}"),
                            getattr(c, "phi_impact_ema", 0.0),
                            getattr(c, "suppression_level", 0.0),
                        )
                        for i, c in enumerate(used_children)
                    ]
                    print(f"[RCP] Suppression update (name, phi_ema, supp): {debug_info}")
                except Exception:
                    pass
        except Exception as e:
            print(f"[RCP] Error updating child suppression: {e}")

        # ---- Update per-child EI* EMA and usage ------------------------------
        try:
            beta_ei = 0.8
            for child in used_children:
                if not hasattr(child, "ei_star_ema"):
                    child.ei_star_ema = 0.0
                if not hasattr(child, "usage_count"):
                    child.usage_count = 0
                child.ei_star_ema = beta_ei * child.ei_star_ema + (1.0 - beta_ei) * float(ei_star_value)
                child.usage_count += 1
        except Exception as e:
            print(f"[RCP] Error updating child EI* stats: {e}")

        # ---- Update per-child cooperation EMA --------------------------------
        try:
            beta_coop = 0.8
            for child in used_children:
                if not hasattr(child, "coop_ema"):
                    child.coop_ema = 0.0
                try:
                    c_val = compute_cooperation(self, child)
                except Exception:
                    c_val = rho_compat_t
                child.coop_ema = beta_coop * child.coop_ema + (1.0 - beta_coop) * float(c_val)
        except Exception as e:
            print(f"[RCP] Error updating child coop_ema: {e}")

        if spawn_triggered:
            try:
                self._hierarchical_spawn(depth=1)
                self._spawn_cooldown_counter = self.spawn_cooldown
                total_agents = len(self.children) + len(self.grandchildren)
                print(f"[RCP] Spawned new child agent(s). Total agents: {total_agents}")
            except Exception as e:
                print(f"[RCP] Error during spawning: {e}")

        # ---- Replication cooldown update -------------------------------------
        if self._replicate_cooldown_counter > 0:
            self._replicate_cooldown_counter -= 1

        # ---- Retirement cooldown update -------------------------------------
        if self._retire_cooldown_counter > 0:
            self._retire_cooldown_counter -= 1

        # ---- RCP-driven replication ------------------------------------------
        replicate_triggered = False
        replicate_source = None
        try:
            if self._replicate_cooldown_counter <= 0:
                candidates = []
                for child in used_children:
                    ei_ema = getattr(child, "ei_star_ema", 0.0)
                    coop_ema = getattr(child, "coop_ema", 0.0)
                    supp = getattr(child, "suppression_level", 0.0)
                    usage = getattr(child, "usage_count", 0)
                    if (
                        usage >= self.replicate_min_usage
                        and ei_ema >= self.replicate_ei_thresh
                        and coop_ema >= self.replicate_coop_thresh
                        and supp <= self.replicate_supp_max
                    ):
                        candidates.append((child, ei_ema, coop_ema, supp, usage))

                total_agents = len(self.children) + len(self.grandchildren)
                if candidates and total_agents < self.max_agents:
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    replicate_source = candidates[0][0]
                    replicate_triggered = True
        except Exception as e:
            print(f"[RCP] Error evaluating replication candidates: {e}")

        if replicate_triggered and replicate_source is not None:
            try:
                before_children = len(self.children)
                before_grandchildren = len(self.grandchildren)
                self._hierarchical_spawn(depth=1)
                after_children = len(self.children)
                after_grandchildren = len(self.grandchildren)
                newly_added = []
                if after_children > before_children:
                    newly_added.extend(self.children[before_children:after_children])
                if after_grandchildren > before_grandchildren:
                    newly_added.extend(self.grandchildren[before_grandchildren:after_grandchildren])
                for new_child in newly_added:
                    try:
                        new_child.lineage_tag = f"clone_of_{getattr(replicate_source, 'lineage_tag', '') or id(replicate_source)}"
                    except Exception:
                        pass
                self._replicate_cooldown_counter = self.replicate_cooldown
                print(f"[RCP] Replicated child: {getattr(replicate_source, 'name', replicate_source)} -> {[getattr(c, 'name', c) for c in newly_added]}")
            except Exception as e:
                print(f"[RCP] Error during replication: {e}")
        try:
            if replicate_source is not None:
                print(
                    "[RCP] Replication candidate:",
                    getattr(replicate_source, "name", str(replicate_source)),
                    "ei_ema=",
                    getattr(replicate_source, "ei_star_ema", 0.0),
                    "coop_ema=",
                    getattr(replicate_source, "coop_ema", 0.0),
                    "supp=",
                    getattr(replicate_source, "suppression_level", 0.0),
                    "usage=",
                    getattr(replicate_source, "usage_count", 0),
                )
        except Exception:
            pass

        # ---- RCP-driven retirement -------------------------------------------
        try:
            retire_candidates: List[Tuple[Any, bool, bool, bool, int]] = []
            if self._retire_cooldown_counter <= 0:
                agents = list(self.children) + list(self.grandchildren)
                for agent in agents:
                    if getattr(agent, "retired", False):
                        continue
                    ei_ema = getattr(agent, "ei_star_ema", 0.0)
                    coop_ema = getattr(agent, "coop_ema", 0.0)
                    supp = getattr(agent, "suppression_level", 0.0)
                    usage = getattr(agent, "usage_count", 0)
                    last_used = getattr(agent, "last_used_turn", 0)
                    turns_since_use = self.global_turn_idx - last_used
                    inactive = turns_since_use >= self.retire_usage_inactive_turns
                    low_quality = (ei_ema < self.retire_ei_thresh and coop_ema < self.retire_coop_thresh)
                    highly_suppressed = supp >= self.retire_supp_min
                    if usage >= self.replicate_min_usage and (inactive or low_quality or highly_suppressed):
                        retire_candidates.append((agent, inactive, low_quality, highly_suppressed, usage))

            total_agents = len(self.children) + len(self.grandchildren)
            if retire_candidates and total_agents > self.retire_min_agents:
                def _retire_score(entry):
                    agent, inactive, low_quality, highly_suppressed, usage = entry
                    score = 0
                    if highly_suppressed:
                        score += 3
                    if low_quality:
                        score += 2
                    if inactive:
                        score += 1
                    score += max(0, 10 - usage) * 0.1
                    return score

                retire_candidates.sort(key=_retire_score, reverse=True)
                agent_to_retire = retire_candidates[0][0]
                self._retire_agent(agent_to_retire)
                self._retire_cooldown_counter = self.retire_cooldown
                try:
                    _, inactive, low_quality, highly_suppressed, usage = retire_candidates[0]
                    print(
                        "[RCP] Retirement decision:",
                        getattr(agent_to_retire, "name", str(agent_to_retire)),
                        "inactive=",
                        inactive,
                        "low_quality=",
                        low_quality,
                        "highly_suppressed=",
                        highly_suppressed,
                        "usage=",
                        usage,
                    )
                except Exception:
                    pass
        except Exception as e:
            print(f"[RCP] Error during retirement evaluation: {e}")

        # ---- 6️⃣ tiny policy-gradient step ---------------------------------
        try:
            # reward = EI (the internal signal we want to maximise)
            advantage = ei - self.ei_grid.mean()          # simple baseline

            # Reflexive conservation penalty on changes in phi = J * rho_compat
            try:
                if self._last_phi is not None:
                    prev_phi = self._prev_phi
                    curr_phi = self._last_phi

                    if prev_phi is not None and curr_phi is not None:
                        eps_phi = 1e-6
                        rel_change = (curr_phi - prev_phi) / (abs(prev_phi) + eps_phi)
                        penalty = rel_change * rel_change
                        advantage = advantage - self.lambda_conserve * penalty

                    self._prev_phi = curr_phi
            except Exception as e:
                print(f"Error applying conservation penalty: {e}")
            if not chosen_idxs:  # No action taken
                log_prob = None
            else:
                log_prob = None
                if policy_probs is not None and len(policy_probs) > chosen_idxs[0]:
                    prob_val = float(policy_probs[chosen_idxs[0]])
                    prob_val = max(prob_val, 1e-8)
                    log_prob = torch.log(torch.tensor(prob_val, dtype=torch.float32))
                else:
                    try:
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

                        if 0 <= chosen_idxs[0] < probs_vec.shape[0]:
                            log_prob = torch.log(probs_vec[chosen_idxs[0]])
                    except Exception as e:
                        print(f"Error recomputing log_prob: {e}")

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

    def _compute_rho_compat(self, used_children):
        """
        Compatibility rho_compat(t): average cooperation score for
        children that participated this turn. If there are no used children,
        return 1.0 by convention (no hierarchy to couple).
        """
        if not used_children:
            return 1.0

        scores = []
        for child in used_children:
            try:
                scores.append(compute_cooperation(self, child))
            except Exception:
                continue

        if not scores:
            return 0.0

        rho = float(np.mean(scores))
        if rho < 0.0:
            rho = 0.0
        if rho > 1.0:
            rho = 1.0
        return rho

    def _compute_justification(self, ai_t: float, ei_star_t: float, rho_compat_t: float) -> float:
        """
        Compute scalar J(t) ≈ AI(t)^{alpha(t)} * EI*(t)^{beta(t)}, squashed to [0,1].
        """
        eps = 1e-6
        ai_clamped = max(ai_t, eps)
        ei_clamped = max(ei_star_t, eps)

        # Simple exponents depending on compatibility and EI*
        k_alpha = 0.5
        k_beta = 0.5

        alpha_t = 1.0 + k_alpha * (rho_compat_t ** 2)
        beta_t = 1.0 + k_beta * (math.log(ei_clamped) ** 2)

        J = (ai_clamped ** alpha_t) * (ei_clamped ** beta_t)
        J_norm = J / (1.0 + J)
        return float(J_norm)

    def _compute_world_coupling(self) -> float:
        """
        Approximate world coupling I(S;E)/H(S) via Gaussian MI between
        sensing embeddings (user messages) and environment embeddings.
        Returns value in [0, 1].
        """
        if not getattr(self, "_sense_history", None) or not getattr(self, "_env_history", None):
            return 0.0

        n = min(len(self._sense_history), len(self._env_history))
        if n < 4:
            return 0.0

        sense = np.stack(self._sense_history[-n:], axis=0)
        env = np.stack(self._env_history[-n:], axis=0)
        joint = np.concatenate([sense, env], axis=1)

        eps = 1e-6
        try:
            cov_s = np.cov(sense, rowvar=False) + eps * np.eye(sense.shape[1])
            cov_e = np.cov(env, rowvar=False) + eps * np.eye(env.shape[1])
            cov_joint = np.cov(joint, rowvar=False) + eps * np.eye(joint.shape[1])

            sign_s, logdet_s = np.linalg.slogdet(cov_s)
            sign_e, logdet_e = np.linalg.slogdet(cov_e)
            sign_j, logdet_j = np.linalg.slogdet(cov_joint)

            if sign_s <= 0 or sign_e <= 0 or sign_j <= 0:
                return 0.0

            mi_se = 0.5 * (logdet_s + logdet_e - logdet_j)
        except Exception:
            return 0.0

        dim_s = float(sense.shape[1])
        h_s_scale = math.log(dim_s + 1.0)
        if h_s_scale <= 0:
            return 0.0

        world_coupling = mi_se / h_s_scale
        world_coupling = max(0.0, min(1.0, world_coupling))
        return float(world_coupling)

    def _compute_prediction_calibration(self) -> float:
        """
        Approximate 1 - KL(p(O) || p(O_hat)) / H(O) for a binary
        correctness variable using prediction_history.
        Returns in [0, 1].
        """
        if not getattr(self, "_prediction_history", None):
            return 0.0

        preds = [x[0] for x in self._prediction_history]
        reals = [x[1] for x in self._prediction_history]

        if not preds or not reals:
            return 0.0

        p_hat = float(np.mean(preds))
        p = float(np.mean(reals))

        eps = 1e-6

        def _bern_kl(p_true, p_model):
            p_true = min(max(p_true, eps), 1.0 - eps)
            p_model = min(max(p_model, eps), 1.0 - eps)
            q_true = 1.0 - p_true
            q_model = 1.0 - p_model
            return (
                p_true * math.log(p_true / p_model)
                + q_true * math.log(q_true / q_model)
            )

        def _bern_ent(p_true):
            p_true = min(max(p_true, eps), 1.0 - eps)
            q_true = 1.0 - p_true
            return -p_true * math.log(p_true) - q_true * math.log(q_true)

        H_p = _bern_ent(p)
        if H_p <= 0.0:
            return 1.0

        kl = _bern_kl(p, p_hat)
        calibration = 1.0 - (kl / H_p)
        calibration = max(0.0, min(1.0, calibration))
        return float(calibration)

    def _compute_EI_star(self, ei2_value: float) -> float:
        """
        Compute EI* = EI2 * world_coupling * prediction_calibration.
        All factors are in [0, 1].
        """
        ei2_clamped = max(0.0, min(1.0, float(ei2_value)))
        world_coupling = self._compute_world_coupling()
        calibration = self._compute_prediction_calibration()
        ei_star = ei2_clamped * world_coupling * calibration
        ei_star = max(0.0, min(1.0, ei_star))
        return float(ei_star)

    def _call_llm_direct(self, msg: str, *, override_model: Optional[str] = None) -> str:
        """Fallback when no child is selected – direct call to LM‑Studio."""
        try:
            gate = getattr(self, "_phi_gate", 1.0)
            base_k = 4
            scale_factor = 0.5 + 0.5 * gate
            k_adj = int(round(base_k * scale_factor))
            if k_adj < 2:
                k_adj = 2
            self._last_direct_retrieval = self.mem.search_with_indices(msg, k=k_adj)
            bucket = _bucket_for_text(msg, self.mem.emb_model)
            pool = self._known_models or self.model_pool or []
            bucket_weights = _ensure_bucket_weights(self.model_weights, bucket, pool)
            model_name, reason = (
                (override_model, "override")
                if override_model
                else choose_llm_model(self.model_pool, available=self._known_models, weights=bucket_weights)
            )
            self._last_model_used = model_name
            self._last_model_reason = reason
            self._last_model_candidates = [model_name]
            self._last_model_bucket = bucket
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
        except requests.HTTPError as e:
            print(f"Error in direct LLM call: {_extract_http_error(e)}")
            if self._last_model_used:
                _prune_model(self._last_model_used, self.model_pool, self.model_weights, self._known_models)
                self._persist_model_weights()
            return f"Direct LLM error: {_extract_http_error(e)}"
        except Exception as e:
            print(f"Error in direct LLM call: {e}")
            return f"Direct LLM error: {str(e)}"

    def _record_assemblies(
        self,
        turn_id: str,
        used_children: List[RetrievalChild],
        parent_answer: str,
        ei: float,
        parent_models: Optional[List[str]] = None,
    ):
        """Create/update assembly DAG nodes for this turn."""
        if not hasattr(self, "assembly_index"):
            return

        gate = getattr(self, "_phi_gate", 1.0)
        if gate < 1.0 and random.random() > gate:
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
        model_tags = []
        if parent_models:
            seen_models = []
            for m in parent_models:
                if m and m not in seen_models:
                    seen_models.append(m)
            model_tags = [f"model:{m}" for m in seen_models]
        parent_node_id = self.assembly_index.create_composite_node(
            node_id=parent_node_id,
            level=2,
            content=parent_answer,
            sources=child_nodes,
            tags=["parent"] + model_tags,
        )
        self.assembly_index.register_activation(parent_node_id, turn_id, quality=ei)

        # If no children fired, still bind any memory nodes touched directly.
        if not child_nodes and getattr(self, "_last_direct_retrieval", None):
            mem_nodes = []
            for idx, txt in self._last_direct_retrieval:
                mem_nodes.append(self.assembly_index.ensure_memory_node(idx, txt))
            for node_id in mem_nodes:
                self.assembly_index.graph.add_edge(node_id, parent_node_id)

    def _retire_agent(self, agent) -> None:
        """
        Retire (deactivate) an agent from the ecology.
        Removes it from children/grandchildren lists and updates the
        meta-policy action space accordingly.
        """
        try:
            try:
                agent.retired = True
            except Exception:
                pass

            if agent in self.children:
                self.children.remove(agent)
            if agent in self.grandchildren:
                self.grandchildren.remove(agent)

            total_agents = len(self.children) + len(self.grandchildren)
            if hasattr(self, "meta_policy") and hasattr(self.meta_policy, "resize_action_space"):
                try:
                    self.meta_policy.resize_action_space(total_agents)
                except Exception as e:
                    print(f"[RCP] Error resizing action space after retirement: {e}")

            print(f"[RCP] Retired agent: {agent}")
        except Exception as e:
            print(f"[RCP] Error in _retire_agent: {e}")

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

    def _peer_model_feedback(self, user_msg: str, parent_answer: str, parent_models: List[str]) -> Tuple[Optional[str], Optional[float]]:
        """
        Call alternative models (if available) to compare answers.
        Uses each peer's *own* answer as the expected output and compares the given answer to it.
        Produces a weighted, normalized score across peers.
        """
        effective_pool = self.model_pool or getattr(self, "_known_models", []) or []
        if len(effective_pool) < 2:
            return None, None
        bucket = getattr(self, "_last_model_bucket", None) or _bucket_for_text(user_msg, self.mem.emb_model)
        bucket_weights = self.model_weights.get(bucket, {}) if isinstance(self.model_weights, dict) else {}
        method = getattr(self, "peer_review_method", "similarity")

        # Choose up to 3 other models different from the ones that answered.
        used = set(m for m in parent_models if m)
        base_pool = self.model_pool or getattr(self, "_known_models", []) or []
        if getattr(self, "_known_models", None):
            base_pool = [m for m in base_pool if m in self._known_models]
        alternatives = [m for m in base_pool if m not in used and "embed" not in m.lower()]
        if bucket_weights:
            alternatives.sort(key=lambda m: bucket_weights.get(m, 0.0), reverse=True)
        if not alternatives:
            # Fall back: allow top-weighted models (excluding used ones) as peers, up to 3.
            alt_pool = [m for m in base_pool if m not in used and "embed" not in m.lower()]
            if bucket_weights:
                alt_pool.sort(key=lambda m: bucket_weights.get(m, 0.0), reverse=True)
            alternatives = alt_pool[:3]
        else:
            alternatives = alternatives[:3]

        if method == "contrastive":
            return self._peer_contrastive(bucket, bucket_weights, user_msg, parent_answer, alternatives)
        else:
            return self._peer_similarity(bucket, bucket_weights, user_msg, parent_answer, alternatives)

    def _call_llm_for_feedback(self, msg: str, model_name: str) -> Optional[str]:
        """Lightweight LLM call for model comparison; avoids mutating main state."""
        if "embed" in model_name.lower():
            return None
        try:
            timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "600"))
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a peer reviewer model. Provide a concise answer."},
                    {"role": "user", "content": msg},
                ],
                "temperature": 0.2,
            }
            api_url = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
            resp = requests.post(api_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except requests.HTTPError as e:
            print(f"Feedback LLM call failed for model {model_name}: {_extract_http_error(e)}")
            _prune_model(model_name, self.model_pool, self.model_weights, self._known_models)
            self._persist_model_weights()
            return None
        except Exception as e:
            print(f"Feedback LLM call failed for model {model_name}: {e}")
            return None

    def _peer_similarity(
        self,
        bucket: str,
        bucket_weights: Dict[str, float],
        user_msg: str,
        parent_answer: str,
        alternatives: List[str],
    ) -> Tuple[Optional[str], Optional[float]]:
        peer_results: List[Tuple[str, str]] = []
        for model_name in alternatives:
            ans = self._call_llm_for_feedback(user_msg, model_name)
            if ans:
                peer_results.append((model_name, ans))

        if not peer_results:
            return None, None

        try:
            given_emb = self.mem.emb_model.encode([parent_answer], normalize_embeddings=True)[0]
            weighted_sum = 0.0
            weight_total = 0.0
            details = []
            for model_name, peer_ans in peer_results:
                peer_emb = self.mem.emb_model.encode([peer_ans], normalize_embeddings=True)[0]
                num = float(np.dot(given_emb, peer_emb))
                den = float(np.linalg.norm(given_emb) * np.linalg.norm(peer_emb) + 1e-8)
                sim = num / den
                # Normalize cosine similarity to [0,1]
                sim_norm = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                w = bucket_weights.get(model_name, 1.0) if bucket_weights else 1.0
                weighted_sum += w * sim_norm
                weight_total += w
                details.append(f"{model_name}:{sim_norm:.2f}")
            if weight_total == 0:
                return None, None
            score = max(0.0, min(1.0, weighted_sum / weight_total))
            signal = score  # keep full [0,1] granularity
            text = f"Auto model feedback (weighted, bucket={bucket}): score {score:.2f} from {', '.join(details)}"
            print(f"[peer scores] {', '.join(details)}")
            try:
                self._last_peer_prediction = score
            except Exception:
                pass
            return text, signal
        except Exception as e:
            print(f"Model peer feedback failed: {e}")
            return None, None

    def _peer_contrastive(
        self,
        bucket: str,
        bucket_weights: Dict[str, float],
        user_msg: str,
        parent_answer: str,
        alternatives: List[str],
    ) -> Tuple[Optional[str], Optional[float]]:
        scores = []
        details = []
        for model_name in alternatives:
            sc = self._llm_score_answer(user_msg, parent_answer, model_name)
            if sc is None:
                continue
            w = bucket_weights.get(model_name, 1.0) if bucket_weights else 1.0
            scores.append((sc, w))
            details.append(f"{model_name}:{sc:.2f}")
        if not scores:
            return None, None
        weighted_sum = sum(s * w for s, w in scores)
        weight_total = sum(w for _, w in scores)
        if weight_total == 0:
            return None, None
        score = max(0.0, min(1.0, weighted_sum / weight_total))
        signal = score  # keep full [0,1] granularity
        text = f"Auto model feedback (contrastive, bucket={bucket}): score {score:.2f} from {', '.join(details)}"
        print(f"[peer scores] {', '.join(details)}")
        try:
            self._last_peer_prediction = score
        except Exception:
            pass
        return text, signal

    def _llm_score_answer(self, user_msg: str, parent_answer: str, model_name: str) -> Optional[float]:
        """
        Ask a peer model to score the given answer vs what it would produce (0-1).
        Returns a float in [0,1], or None on failure.
        """
        if "embed" in model_name.lower():
            return None
        try:
            timeout = float(os.getenv("LLM_REQUEST_TIMEOUT", "600"))
            prompt = (
                "You are a strict reviewer. Given a QUESTION and a PROPOSED ANSWER, "
                "score how well the proposed answer matches what you would reply "
                #"with the following criteria: "
                #"0 means that the proposed answer is completely wrong, " 
                #"0.5 means the proposed answer is expected and considered similar, "
                #"and 1 if the proposed answer provides new and correct information. "
                #"Values can be anywhere between 0 to 1, based on howmuch they match the criteria"
                "Return a single number between 0 and 1. Do not add text."
            )
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"QUESTION:\n{user_msg}\n\nPROPOSED ANSWER:\n{parent_answer}\n\nScore (0-1):"},
                ],
                "temperature": 0.0,
            }
            api_url = os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions")
            resp = requests.post(api_url, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            match = re.search(r"([01](?:\.\d+)?)", content)
            if not match:
                return None
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        except requests.HTTPError as e:
            print(f"Contrastive score failed for {model_name}: {_extract_http_error(e)}")
            _prune_model(model_name, self.model_pool, self.model_weights, self._known_models)
            self._persist_model_weights()
            return None
        except Exception as e:
            print(f"Contrastive score failed for {model_name}: {e}")
            return None

    def _collect_feedback(
        self,
        user_msg: str,
        parent_answer: str,
        used_children: List['RetrievalChild'],
        parent_models: List[str],
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Collect feedback signal:
          • if multiple LLMs are available, compare their answers for agreement;
          • otherwise fall back to user prompt + keyword quantification.
        """
        fb_text, fb_signal = self._peer_model_feedback(user_msg, parent_answer, parent_models)
        if fb_text is not None:
            return fb_text, fb_signal

        # If we had multiple models available but could not gather model feedback, skip user prompt.
        effective_pool = self.model_pool or getattr(self, "_known_models", []) or []
        if len(effective_pool) >= 2:
            return None, None

        feedback = self._prompt_feedback()
        return feedback, self._quantify_feedback(feedback)

    def _init_model_weights(self):
        """Initialize model weights uniformly for discovered models/pool (global bucket)."""
        base_models = self._known_models or self.model_pool or []
        if not base_models:
            return
        _ensure_bucket_weights(self.model_weights, "b:global", base_models)
        # Drop weights for models that are not in known list if known list exists.
        if self._known_models:
            for bucket, weights in list(self.model_weights.items()):
                for m in list(weights.keys()):
                    if m not in self._known_models or "embed" in m.lower():
                        weights.pop(m, None)
            # also trim model_pool to known models
            if self.model_pool:
                self.model_pool = [m for m in self.model_pool if m in self._known_models and "embed" not in m.lower()]

    def _reinforce_models(self, models_used: List[str], feedback_signal: Optional[int], bucket: Optional[str]):
        """Reinforce bucketed model weights with a simple EMA based on feedback (1 good, 0 bad)."""
        if feedback_signal is None:
            return
        bucket_key = bucket or "b:global"
        pool = self._known_models or self.model_pool or []
        bucket_weights = _ensure_bucket_weights(self.model_weights, bucket_key, pool)
        alpha = 0.2
        floor = 0.1
        for m in models_used:
            if not m:
                continue
            prev = bucket_weights.get(m, 1.0)
            target = float(feedback_signal)
            updated = (1 - alpha) * prev + alpha * target
            bucket_weights[m] = max(floor, updated)
        self._persist_model_weights()

    def _persist_model_weights(self):
        if not getattr(self, "model_weights_path", None):
            return
        try:
            self.model_weights_path.parent.mkdir(parents=True, exist_ok=True)
            self.model_weights_path.write_text(json.dumps(self.model_weights))
        except Exception as e:
            print(f"Failed to persist model weights: {e}")

    def _load_model_weights(self):
        if not getattr(self, "model_weights_path", None):
            return
        if not self.model_weights_path.exists():
            return
        try:
            data = json.loads(self.model_weights_path.read_text())
            if isinstance(data, dict):
                # Support old flat format by nesting under global bucket
                if data and all(isinstance(v, (int, float)) for v in data.values()):
                    data = {"b:global": data}
                for bucket, weights in data.items():
                    if not isinstance(weights, dict):
                        continue
                    bucket_weights = self.model_weights.setdefault(bucket, {})
                    for m, w in weights.items():
                        try:
                            bucket_weights[m] = float(w)
                        except Exception:
                            pass
                self._init_model_weights()
        except Exception as e:
            print(f"Failed to load model weights: {e}")

    def _annotate_answer_with_models(self, answer: str, models_used: List[str]) -> str:
        """Append model metadata to the stored answer text."""
        unique_models: List[str] = []
        for m in models_used:
            if m and m not in unique_models:
                unique_models.append(m)
        if not unique_models:
            return answer
        tag = f"[models: {', '.join(unique_models)}]"
        return f"{answer}\n\n{tag}"

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
            try:
                if "parent" not in self.graph:
                    self.graph.add_node("parent")
                parent_node = self.graph.nodes["parent"]
                parent_node["lambda_conserve"] = getattr(self, "lambda_conserve", None)
                parent_node["phi_delta_ema"] = getattr(self, "_phi_delta_ema", None)
                parent_node["phi_gate"] = getattr(self, "_phi_gate", None)
                parent_node["target_phi_delta"] = getattr(self, "target_phi_delta", None)
                parent_node["last_phi"] = getattr(self, "_last_phi", None)
                parent_node["prev_phi"] = getattr(self, "_prev_phi", None)
                parent_node["last_J"] = getattr(self, "_last_J", None)
                parent_node["last_rho_compat"] = getattr(self, "_last_rho_compat", None)
            except Exception as e:
                print(f"Failed to append conservation state to self-model: {e}")
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
                if not data or not isinstance(data, dict):
                    raise ValueError("invalid self-model data")
                graph = json_graph.node_link_graph(data, edges="links")
                if "parent" not in graph:
                    graph.add_node("parent")
                return graph
            except Exception as e:
                print(f"Failed to load self-model graph: {e}")
        graph = nx.DiGraph()
        graph.add_node("parent")
        return graph

    def _restore_conservation_state(self):
        """
        Restore last J, rho_compat, and phi from persisted graph nodes if present.
        """
        try:
            if not getattr(self, "graph", None):
                return
            parent_attrs = self.graph.nodes.get("parent", {}) if hasattr(self, "graph") else {}
            if parent_attrs:
                try:
                    if parent_attrs.get("lambda_conserve") is not None:
                        self.lambda_conserve = float(parent_attrs.get("lambda_conserve"))
                    if parent_attrs.get("phi_delta_ema") is not None:
                        self._phi_delta_ema = float(parent_attrs.get("phi_delta_ema"))
                    if parent_attrs.get("phi_gate") is not None:
                        self._phi_gate = float(parent_attrs.get("phi_gate"))
                    if parent_attrs.get("target_phi_delta") is not None:
                        self.target_phi_delta = float(parent_attrs.get("target_phi_delta"))
                    if parent_attrs.get("last_phi") is not None:
                        self._last_phi = float(parent_attrs.get("last_phi"))
                    if parent_attrs.get("prev_phi") is not None:
                        self._prev_phi = float(parent_attrs.get("prev_phi"))
                    if parent_attrs.get("last_J") is not None:
                        self._last_J = float(parent_attrs.get("last_J"))
                    if parent_attrs.get("last_rho_compat") is not None:
                        self._last_rho_compat = float(parent_attrs.get("last_rho_compat"))
                except Exception as e:
                    print(f"Failed to restore conservation scalars from parent node: {e}")
            turn_nodes = []
            for node_id, data in self.graph.nodes(data=True):
                if isinstance(node_id, str) and node_id.startswith("t"):
                    turn_nodes.append(
                        (
                            node_id,
                            data.get("phi"),
                            data.get("J"),
                            data.get("rho_compat"),
                        )
                    )
            if not turn_nodes:
                return

            def _turn_sort_key(name: str):
                try:
                    return int(name[1:])
                except Exception:
                    return name

            turn_nodes.sort(key=lambda x: _turn_sort_key(x[0]))
            last = turn_nodes[-1]
            if last[1] is not None:
                self._last_phi = float(last[1])
            if last[2] is not None:
                self._last_J = float(last[2])
            if last[3] is not None:
                self._last_rho_compat = float(last[3])
            if len(turn_nodes) >= 2:
                prev = turn_nodes[-2]
                if prev[1] is not None:
                    self._prev_phi = float(prev[1])
        except Exception as e:
            print(f"Failed to restore conservation state: {e}")

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
            if not payload or not isinstance(payload, dict):
                return
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
            self._persist_model_weights()
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
            try:
                if getattr(self, "model_weights_path", None) and self.model_weights_path.exists():
                    self.model_weights_path.unlink()
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
            self._last_model_used = None
            self._last_model_reason = None
            self._last_model_candidates = None
            self._last_model_bucket = None
            self.model_pool = _parse_llm_pool()
            self._known_models = _discover_models(os.getenv("LLM_API_URL", "http://localhost:1234/v1/chat/completions"))
            if self._known_models:
                self.model_pool = list(self._known_models)
            self.model_weights = {}
            self._init_model_weights()
            self._load_model_weights()
            self._ms_history = []
            self._policy_history = []
            self._mem_state_history = []
            self._memory_use_history = []
            self._retrieval_depth_history = []
            self._last_phi = None
            self._last_J = None
            self._last_rho_compat = None
            self._prev_phi = None
            self.lambda_conserve = 0.1
            self._phi_delta_ema = 0.0
            self._phi_gate = 1.0
            self._retire_cooldown_counter = 0
            self._sense_history = []
            self._env_history = []
            self._prediction_history = []
            self._last_user_emb = None
            self._last_peer_prediction = None
            self.spawn_phi_stability_thresh = 0.05
            self.spawn_ei_stagnation_window = 5
            self.spawn_ei_min_improvement = 0.01
            self.spawn_cooldown = 10
            self._spawn_cooldown_counter = 0
            self._ei_history_for_spawning = []
            self.suppression_phi_thresh = 0.10
            self.suppression_ema_beta = 0.8
            self.suppression_max = 1.0
            self.suppression_min = 0.0
            self.suppression_alpha = 0.8
            self.suppression_hard_cutoff = 0.95
            self._replicate_cooldown_counter = 0
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
          * Reuse ρ = u * c * d:
              - u: fraction of recent decisions that accessed memory.
              - c: coverage over memory nodes consulted this turn.
              - d: temporal depth of memory reads (older = deeper).

        All sub-scores are normalized to [0,1] then combined via a geometric product with optional exponents.
        """
        eps = 1e-6

        def clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
            return float(np.clip(val, lo, hi))

        # Ensure history buffers exist for running covariance/entropy estimates.
        if not hasattr(self, "_ms_history"):
            self._ms_history: List[Tuple[np.ndarray, np.ndarray]] = []
        if not hasattr(self, "_policy_history"):
            self._policy_history: List[np.ndarray] = []
        if not hasattr(self, "_mem_state_history"):
            self._mem_state_history: List[np.ndarray] = []
        if not hasattr(self, "_memory_use_history"):
            self._memory_use_history: List[bool] = []
        if not hasattr(self, "_retrieval_depth_history"):
            self._retrieval_depth_history: List[float] = []

        try:
            user_emb = self.mem.emb_model.encode([user_msg], normalize_embeddings=True)[0]
        except Exception:
            user_emb = np.zeros(384, dtype=np.float32)

        # Track sensing embedding history
        if hasattr(self, "_sense_history"):
            self._sense_history.append(user_emb.astype(np.float32))
            if len(self._sense_history) > 256:
                self._sense_history.pop(0)

        # Environment embedding: use previous user embedding as proxy
        if hasattr(self, "_last_user_emb") and self._last_user_emb is not None:
            env_emb = self._last_user_emb
        else:
            env_emb = np.zeros_like(user_emb, dtype=np.float32)
        if hasattr(self, "_env_history"):
            self._env_history.append(env_emb.astype(np.float32))
            if len(self._env_history) > 256:
                self._env_history.pop(0)
        self._last_user_emb = user_emb.astype(np.float32)

        # Build an aggregate memory state from the chunks retrieved for this turn.
        retrieved_chunks: List[str] = []
        try:
            gate = getattr(self, "_phi_gate", 1.0)
            base_k = 4
            scale_factor = 0.5 + 0.5 * gate
            k_adj = int(round(base_k * scale_factor))
            if k_adj < 2:
                k_adj = 2
            retrieved_chunks = self.mem.search(user_msg, k=k_adj)
            if retrieved_chunks:
                mem_embs = self.mem.emb_model.encode(retrieved_chunks, normalize_embeddings=True)
                mem_state = np.mean(mem_embs, axis=0)
            else:
                mem_state = np.zeros_like(user_emb)
        except Exception:
            mem_state = np.zeros_like(user_emb)

        used_memory = bool(retrieved_indices) or bool(retrieved_chunks)
        self._memory_use_history.append(used_memory)
        reuse_hist_max = 128
        if len(self._memory_use_history) > reuse_hist_max:
            self._memory_use_history.pop(0)

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

        # --- Peer-based groundedness from feedback history --------------------
        feedback_level = 0.0      # how good is this answer, 0..1
        feedback_gain = 0.0       # how much better than usual, >= 0

        if hasattr(self, "_feedback_history") and len(self._feedback_history) >= 1:
            feedback_arr = np.array(self._feedback_history, dtype=np.float32)
            p_current = float(feedback_arr[-1])

            # Running average over all previous feedback (including this one)
            p_mean = float(feedback_arr.mean())

            feedback_level = max(0.0, min(1.0, p_current))
            # Only count positive improvement over the running mean
            feedback_gain = max(0.0, p_current - p_mean)

        # Combine level + improvement into a single peer groundedness score
        # peer_grounded in [0,1], with more weight on "how good" than "how much better"
        peer_grounded = 0.0
        if feedback_level > 0.0 or feedback_gain > 0.0:
            w_level = 0.8
            w_gain = 0.2
            peer_grounded = w_level * feedback_level + w_gain * feedback_gain
            if peer_grounded < 0.0:
                peer_grounded = 0.0
            if peer_grounded > 1.0:
                peer_grounded = 1.0

        # Final grounded = mix of lexical similarity and peer evaluation
        # If there is no peer signal yet, grounded falls back to lexical only.
        w_lex = 0.5
        w_peer = 0.5

        if peer_grounded == 0.0 and feedback_level == 0.0:
            grounded = max(0.0, min(1.0, lexical_grounded))
        else:
            grounded = (
                w_lex * lexical_grounded
                + w_peer * peer_grounded
            )
            if grounded < 0.0:
                grounded = 0.0
            if grounded > 1.0:
                grounded = 1.0

        # Reuse factor ρ = u * c * d
        # u: fraction of recent turns that used memory
        reuse_window = 32
        recent_uses = self._memory_use_history[-reuse_window:]
        if recent_uses:
            reuse_u = sum(1 for x in recent_uses if x) / float(len(recent_uses))
        else:
            reuse_u = 1.0 if used_memory else 0.0
        reuse_u = clip(reuse_u)

        # c: coverage over memory nodes consulted this turn
        total_memory = max(1, len(self.mem.texts))
        if retrieved_indices:
            distinct_indices = [idx for idx in set(retrieved_indices) if 0 <= idx < total_memory]
            if distinct_indices:
                coverage_c = len(distinct_indices) / float(total_memory)
            else:
                coverage_c = 0.0
        elif retrieved_chunks:
            coverage_c = min(1.0, len(retrieved_chunks) / float(total_memory))
        else:
            coverage_c = 0.0
        coverage_c = clip(coverage_c)

        # d: temporal depth of retrieved indices (older = deeper)
        if retrieved_indices and total_memory > 1:
            depths = []
            for idx in set(retrieved_indices):
                if 0 <= idx < total_memory:
                    age = (total_memory - 1 - idx) / float(total_memory - 1)
                    depths.append(age)
            if depths:
                depth_d = float(np.mean(depths))
            else:
                depth_d = 0.0
        else:
            depth_d = 0.0
        depth_d = clip(depth_d)
        self._retrieval_depth_history.append(depth_d)
        if len(self._retrieval_depth_history) > reuse_hist_max:
            self._retrieval_depth_history.pop(0)

        rho = reuse_u * coverage_c * depth_d
        rho = clip(rho)

        # Geometric combination of components (product in log space)
        # --- Warm-start logic for EI2 components -------------------------------
        raw_mi = mi_norm
        raw_te = te_norm
        raw_grounded = grounded
        raw_rho = rho

        ready_eps = 1e-3

        # Readiness criteria
        mi_ready = (len(self._ms_history) >= 8) and (raw_mi > ready_eps)
        te_ready = (
            len(self._policy_history) >= 8
            and len(self._mem_state_history) >= 8
            and raw_te > ready_eps
        )
        grounded_ready = raw_grounded > 0.0
        rho_ready = raw_rho > 0.0

        # Effective component values (neutral = 1.0)
        eff_mi = raw_mi if mi_ready else 1.0
        eff_te = raw_te if te_ready else 1.0
        eff_grounded = raw_grounded if grounded_ready else 1.0
        eff_rho = raw_rho if rho_ready else 1.0

        components = np.array(
            [eff_mi, eff_te, eff_grounded, eff_rho],
            dtype=np.float32,
        )
        components = np.clip(components, eps, 1.0)

        weights = getattr(self, "_ei_weights", None)
        if weights is None:
            exponents = np.ones_like(components)
        else:
            exponents = np.array(weights, dtype=np.float32)
            s = float(np.sum(exponents)) or 1.0
            exponents = exponents * (len(components) / s)

        log_ei = (exponents * np.log(components)).sum()
        ei_score = float(np.exp(log_ei))

        ei_score = clip(ei_score)
        print(f"[info] EI2 components mi={mi_norm:.3f}, te={te_norm:.3f}, grounded={grounded:.3f}, rho={rho:.3f} (u={reuse_u:.3f}, c={coverage_c:.3f}, d={depth_d:.3f}), score={ei_score:.3f}")
        return ei_score

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
            print(f"[info] EI fallback components mi={mi_ms:.3f}, te={te_mp:.3f}, grounded={grounded:.3f}, coverage={reuse:.3f}, ei_raw={ei_raw:.3f}")
            return max(0.0, min(1.0, ei_raw))
        except Exception as e:
            print(f"Error computing EI: {e}")
            return 0.0
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
        print(f"[info] peer review method: {agent.peer_review_method}")
        print(f"[info] epsilon_every_n: {agent.epsilon_every_n}")
        print(f"[info] available models: {', '.join(agent.model_pool) if agent.model_pool else 'none'}")

        # Bootstrap: create a few children so the meta‑policy has something to select.
        # You can change the numbers or call `_hierarchical_spawn` later from the REPL.
        for _ in range(2):          # start with 2 depth‑1 children
            agent._hierarchical_spawn(depth=2)   # each also spawns a grand‑child

        agent.interactive_loop()
    except Exception as e:
        import traceback
        print(f"Critical error in main execution: {e}")
        traceback.print_exc()
