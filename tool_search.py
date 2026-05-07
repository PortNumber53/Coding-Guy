"""Semantic tool search engine for Coding Guy.

Provides embedding-based similarity search over the tool registry, with:
  - Multiple embedding backends: API-based (OpenAI/OpenRouter), sentence-transformers,
    or zero-dependency TF-IDF fallback
  - FAISS or numpy vector store for fast similarity search
  - Local embedding cache to avoid recomputation on startup
  - Keyword-based search fallback for low-confidence results
  - Color-coded verbose logging for debugging

Architecture:
  1. On startup, compute embeddings for all tool descriptions (or load from cache)
  2. When a task arrives, embed the task description
  3. Query the vector store for top-K similar tools
  4. If confidence is low, supplement with keyword search
  5. Return ranked list with scores and descriptions

Embedding backends (tried in order):
  1. API-based: call the OpenAI/OpenRouter embeddings endpoint
  2. sentence-transformers: local model (all-MiniLM-L6-v2), if installed
  3. TF-IDF + cosine similarity: pure sklearn-free fallback using numpy
"""

import hashlib
import json
import math
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Color-coded logging
# ---------------------------------------------------------------------------

_COLOR_RESET = "\033[0m"
_COLOR_DIM = "\033[2m"
_COLOR_BRIGHT = "\033[1m"
_COLOR_GREEN = "\033[32m"
_COLOR_YELLOW = "\033[33m"
_COLOR_RED = "\033[31m"
_COLOR_CYAN = "\033[36m"
_COLOR_MAGENTA = "\033[35m"
_COLOR_BLUE = "\033[34m"


def _log(level: str, msg: str, color: str = ""):
    """Color-coded logging to stderr."""
    prefix = f"{color}{level}{_COLOR_RESET}" if color else level
    print(f"[{prefix}] {msg}", file=sys.stderr)


def log_info(msg: str):
    _log("Search", msg, _COLOR_CYAN)


def log_score(msg: str):
    _log("Score", msg, _COLOR_GREEN)


def log_warn(msg: str):
    _log("Search", msg, _COLOR_YELLOW)


def log_debug(msg: str):
    _log("Debug", msg, _COLOR_DIM)


def log_error(msg: str):
    _log("Error", msg, _COLOR_RED)


# ---------------------------------------------------------------------------
# Embedding backends
# ---------------------------------------------------------------------------

class EmbeddingBackend:
    """Base class for embedding backends."""

    name: str = "base"
    dim: int = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def is_available(self) -> bool:
        return False


class APIEmbeddingBackend(EmbeddingBackend):
    """Use OpenAI-compatible embeddings API (OpenRouter, Nvidia, etc.)."""

    name = "api"
    dim = 1536  # text-embedding-3-small dimension

    def __init__(self, api_key: str = "", base_url: str = "",
                 model: str = "text-embedding-3-small"):
        self.api_key = api_key
        self.base_url = base_url or "https://api.openai.com/v1"
        self.model = model
        self._available = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.api_key:
            self._available = False
            return False
        # Try a tiny request to verify
        try:
            import requests
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": ["test"],
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                self.dim = len(data["data"][0]["embedding"])
                self._available = True
                log_info(f"API embedding backend: model={self.model}, dim={self.dim}")
            else:
                log_warn(f"API embedding check failed: {resp.status_code}")
                self._available = False
        except Exception as e:
            log_warn(f"API embedding backend not available: {e}")
            self._available = False
        return self._available

    def embed(self, texts: list[str]) -> list[list[float]]:
        import requests
        all_embeddings = []
        # Batch in groups of 64 (API limit)
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = requests.post(
                f"{self.base_url}/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": batch,
                },
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in sorted(data["data"], key=lambda x: x["index"]):
                all_embeddings.append(item["embedding"])
        return all_embeddings


class SentenceTransformerBackend(EmbeddingBackend):
    """Use sentence-transformers locally (all-MiniLM-L6-v2)."""

    name = "sentence_transformers"
    dim = 384  # all-MiniLM-L6-v2

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def is_available(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
            log_info(f"SentenceTransformer backend: model={self.model_name}, dim={self.dim}")
            return True
        except ImportError:
            log_debug("sentence-transformers not installed, skipping")
            return False
        except Exception as e:
            log_warn(f"SentenceTransformer init failed: {e}")
            return False

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            raise RuntimeError("Model not loaded")
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


class TFIDFBackend(EmbeddingBackend):
    """Zero-dependency TF-IDF embedding using pure Python + numpy.

    This is the fallback when no embedding API or ML library is available.
    Uses a simple tokenization + TF-IDF weighting scheme with numpy
    cosine similarity. No sklearn required.
    """

    name = "tfidf"
    dim = 0  # Set dynamically based on vocabulary

    def __init__(self, max_features: int = 2000):
        self.max_features = max_features
        self._vocabulary: dict[str, int] = {}
        self._idf: list[float] = []
        self._fitted = False

    def is_available(self) -> bool:
        try:
            import numpy  # noqa: F401
            return True
        except ImportError:
            log_warn("numpy not available for TF-IDF backend")
            return False

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric, keep 2+ char tokens."""
        import re
        tokens = re.findall(r'[a-z][a-z0-9_]+', text.lower())
        # Add bigrams for better matching
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def fit(self, texts: list[str]):
        """Build vocabulary and IDF weights from the corpus."""
        import numpy as np

        # Count document frequency for each term
        doc_freq = Counter()
        total_docs = len(texts)

        for text in texts:
            unique_tokens = set(self._tokenize(text))
            for token in unique_tokens:
                doc_freq[token] += 1

        # Select top features by document frequency
        most_common = doc_freq.most_common(self.max_features)
        self._vocabulary = {term: idx for idx, (term, _) in enumerate(most_common)}
        self.dim = len(self._vocabulary)

        # Compute IDF: log((1 + N) / (1 + df)) + 1  (smoothed)
        self._idf = np.zeros(self.dim)
        for term, idx in self._vocabulary.items():
            df = doc_freq[term]
            self._idf[idx] = math.log((1 + total_docs) / (1 + df)) + 1

        self._fitted = True

    def _tfidf_vector(self, text: str):
        """Compute TF-IDF vector for a single text."""
        import numpy as np

        if not self._fitted:
            raise RuntimeError("TF-IDF backend not fitted yet - call fit() first")

        tokens = self._tokenize(text)
        tf = Counter(tokens)

        vec = np.zeros(self.dim)
        for term, count in tf.items():
            if term in self._vocabulary:
                idx = self._vocabulary[term]
                # Normalized TF: 1 + log(tf) if tf > 0
                vec[idx] = (1 + math.log(count)) if count > 0 else 0

        # Multiply by IDF
        vec = vec * self._idf

        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._tfidf_vector(t) for t in texts]


# ---------------------------------------------------------------------------
# Vector store (FAISS or numpy)
# ---------------------------------------------------------------------------

class VectorStore:
    """Simple vector store with cosine similarity search.

    Uses FAISS if available, otherwise falls back to numpy.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._vectors = None  # numpy array or FAISS index
        self._names: list[str] = []
        self._use_faiss = False

    def build(self, names: list[str], vectors: list[list[float]]):
        """Build the index from name-vector pairs."""
        self._names = names

        try:
            import faiss
            import numpy as np
            mat = np.array(vectors, dtype=np.float32)
            # L2 normalize for cosine similarity
            faiss.normalize_L2(mat)
            self._index = faiss.IndexFlatIP(self.dim)  # Inner product = cosine for normalized
            self._index.add(mat)
            self._use_faiss = True
            log_info(f"Vector store: FAISS index built ({len(names)} vectors, dim={self.dim})")
        except ImportError:
            import numpy as np
            self._vectors = np.array(vectors, dtype=np.float32)
            # L2 normalize
            norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self._vectors = self._vectors / norms
            self._use_faiss = False
            log_info(f"Vector store: numpy index built ({len(names)} vectors, dim={self.dim})")

    def search(self, query_vector: list[float], top_k: int = 10) -> list[tuple[str, float]]:
        """Search for the most similar vectors. Returns [(name, score), ...]."""
        import numpy as np

        qvec = np.array([query_vector], dtype=np.float32)

        if self._use_faiss:
            import faiss
            faiss.normalize_L2(qvec)
            scores, indices = self._index.search(qvec, min(top_k, len(self._names)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self._names):
                    results.append((self._names[idx], float(score)))
            return results
        else:
            # Cosine similarity (vectors already normalized)
            norms = np.linalg.norm(qvec)
            if norms > 0:
                qvec = qvec / norms
            similarities = (self._vectors @ qvec.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [(self._names[idx], float(similarities[idx])) for idx in top_indices if idx < len(self._names)]


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "coding-guy", "tool_embeddings")


def _cache_path(backend_name: str, model_name: str, registry_hash: str) -> str:
    """Get the cache file path for a given backend + registry hash."""
    safe_model = model_name.replace("/", "_").replace(":", "_")
    return os.path.join(_CACHE_DIR, f"{backend_name}_{safe_model}_{registry_hash}.pkl")


def _compute_registry_hash(registry_entries: list[dict]) -> str:
    """Compute a hash of the registry contents for cache invalidation."""
    content = json.dumps(
        [{"name": e["name"], "description": e.get("description", "")} for e in registry_entries],
        sort_keys=True,
    )
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def save_embeddings_cache(backend_name: str, model_name: str, registry_hash: str,
                          names: list[str], vectors: list[list[float]], dim: int):
    """Save computed embeddings to disk cache."""
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        path = _cache_path(backend_name, model_name, registry_hash)
        data = {
            "names": names,
            "vectors": vectors,
            "dim": dim,
            "backend": backend_name,
            "model": model_name,
            "timestamp": time.time(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        log_info(f"Embeddings cached to {path}")
    except Exception as e:
        log_warn(f"Failed to cache embeddings: {e}")


def load_embeddings_cache(backend_name: str, model_name: str, registry_hash: str) -> dict | None:
    """Load cached embeddings from disk. Returns None if not found or stale."""
    try:
        path = _cache_path(backend_name, model_name, registry_hash)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Basic sanity check
        if data.get("backend") != backend_name or data.get("model") != model_name:
            return None
        if len(data.get("names", [])) == 0:
            return None
        age_hours = (time.time() - data.get("timestamp", 0)) / 3600
        log_info(f"Loaded cached embeddings ({len(data['names'])} tools, {age_hours:.1f}h old)")
        return data
    except Exception as e:
        log_debug(f"Cache load failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Keyword-based search (fallback / supplement)
# ---------------------------------------------------------------------------

def keyword_search(query: str, registry_entries: list[dict], top_k: int = 10) -> list[tuple[str, float]]:
    """Simple keyword-based search over tool names, descriptions, keywords, and tags.

    Returns [(tool_name, score), ...] sorted by descending score.
    Score is based on term overlap with inverted frequency weighting.
    """
    import re

    # Tokenize the query
    query_tokens = set(re.findall(r'[a-z][a-z0-9_]+', query.lower()))
    # Add bigrams
    q_list = sorted(query_tokens)
    query_bigrams = set(f"{q_list[i]}_{q_list[i+1]}" for i in range(len(q_list) - 1))
    query_all = query_tokens | query_bigrams

    results = []
    for entry in registry_entries:
        score = 0.0
        name = entry["name"]
        name_tokens = set(name.replace("_", " ").split())

        # Exact name match = highest weight
        if query.lower().replace(" ", "_") == name:
            score += 10.0

        # Name token overlap
        name_overlap = query_tokens & name_tokens
        score += len(name_overlap) * 3.0

        # Description overlap
        desc_tokens = set(re.findall(r'[a-z][a-z0-9_]+', entry.get("description", "").lower()))
        desc_overlap = query_tokens & desc_tokens
        score += len(desc_overlap) * 1.5

        # Keyword/tag overlap (higher weight since they're curated)
        kw_set = set(k.lower() for k in entry.get("keywords", []))
        tag_set = set(t.lower() for t in entry.get("capability_tags", []))
        combined = kw_set | tag_set
        kw_overlap = query_all & combined
        score += len(kw_overlap) * 2.0

        # Category match
        category = entry.get("category", "").lower()
        if category in query.lower():
            score += 1.0

        if score > 0:
            results.append((name, score))

    # Normalize scores to 0-1 range
    if results:
        max_score = max(s for _, s in results)
        if max_score > 0:
            results = [(name, score / max_score) for name, score in results]

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ---------------------------------------------------------------------------
# Main SemanticToolSearch class
# ---------------------------------------------------------------------------

class SemanticToolSearch:
    """Semantic tool search engine.

    Usage:
        search = SemanticToolSearch()
        search.initialize()  # builds embeddings, loads cache
        results = search.search("load a sprite sheet for the game")
        # results: [("read_file", 0.89), ("browser_navigate", 0.72), ...]
    """

    def __init__(self, api_key: str = "", api_base_url: str = "",
                 embedding_model: str = "", top_k: int = 10,
                 confidence_threshold: float = 0.4,
                 fallback_threshold: float = 0.3,
                 verbose: bool = False,
                 use_cache: bool = True):
        self.api_key = api_key
        self.api_base_url = api_base_url
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold
        self.fallback_threshold = fallback_threshold
        self.verbose = verbose
        self.use_cache = use_cache

        self._backend: EmbeddingBackend | None = None
        self._store: VectorStore | None = None
        self._registry: list[dict] = []
        self._search_texts: list[str] = []
        self._initialized = False
        self._tool_names: list[str] = []

    def initialize(self) -> bool:
        """Initialize the search engine: select backend, compute/load embeddings, build index.

        Returns True if initialization succeeded.
        """
        from tool_registry import get_registry, build_search_text

        self._registry = get_registry()
        self._search_texts = [build_search_text(e) for e in self._registry]
        self._tool_names = [e["name"] for e in self._registry]

        # Try backends in order
        backend = self._select_backend()
        if backend is None:
            log_error("No embedding backend available — semantic search disabled")
            return False

        self._backend = backend
        registry_hash = _compute_registry_hash(self._registry)

        # Try loading from cache
        cached = None
        if self.use_cache:
            cached = load_embeddings_cache(
                backend.name, self.embedding_model or backend.name, registry_hash
            )

        if cached and len(cached["names"]) == len(self._tool_names):
            # Cache hit — build store from cached vectors
            self._store = VectorStore(cached["dim"])
            self._store.build(cached["names"], cached["vectors"])
            log_info(f"Using cached embeddings ({len(cached['names'])} tools)")
        else:
            # Compute embeddings
            log_info(f"Computing embeddings for {len(self._search_texts)} tools using {backend.name}...")
            try:
                vectors = backend.embed(self._search_texts)
            except Exception as e:
                log_error(f"Embedding computation failed: {e}")
                return False

            self._store = VectorStore(backend.dim)
            self._store.build(self._tool_names, vectors)

            # Cache for next time
            if self.use_cache:
                save_embeddings_cache(
                    backend.name, self.embedding_model or backend.name,
                    registry_hash, self._tool_names, vectors, backend.dim
                )

        self._initialized = True
        log_info(f"Semantic tool search initialized: {len(self._tool_names)} tools indexed")
        return True

    def _select_backend(self) -> EmbeddingBackend | None:
        """Select the best available embedding backend."""
        # 1. Try API-based (if key and URL configured)
        if self.api_key and self.api_base_url:
            model = self.embedding_model or "text-embedding-3-small"
            backend = APIEmbeddingBackend(
                api_key=self.api_key,
                base_url=self.api_base_url,
                model=model,
            )
            if backend.is_available():
                return backend

        # 2. Try sentence-transformers
        model = self.embedding_model or "all-MiniLM-L6-v2"
        backend = SentenceTransformerBackend(model_name=model)
        if backend.is_available():
            return backend

        # 3. Try OpenRouter embeddings (if OPENROUTER_API_KEY set)
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if openrouter_key:
            backend = APIEmbeddingBackend(
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                model=self.embedding_model or "openai/text-embedding-3-small",
            )
            if backend.is_available():
                return backend

        # 4. Try Nvidia API key for embeddings
        nvidia_key = os.getenv("NVIDIA_API_KEY", "")
        if nvidia_key:
            backend = APIEmbeddingBackend(
                api_key=nvidia_key,
                base_url="https://integrate.api.nvidia.com/v1",
                model=self.embedding_model or "nvidia/embed-qa-4",
            )
            if backend.is_available():
                return backend

        # 5. Fallback to TF-IDF (always available if numpy exists)
        backend = TFIDFBackend(max_features=2000)
        if backend.is_available():
            backend.fit(self._search_texts)
            log_info("Using TF-IDF fallback for tool embeddings (install sentence-transformers for better results)")
            return backend

        return None

    def search(self, query: str, top_k: int | None = None,
               include_descriptions: bool = False) -> list[dict]:
        """Search for tools matching the query.

        Args:
            query: Task description or capability query
            top_k: Number of results (default: self.top_k)
            include_descriptions: If True, include tool descriptions in results

        Returns:
            List of dicts: [{"name": ..., "score": ..., "description": ...}, ...]
            Sorted by descending score. May include results from keyword fallback.
        """
        if not self._initialized:
            log_warn("Search engine not initialized, returning empty results")
            return []

        k = top_k or self.top_k
        results = []

        # Semantic search
        if self._backend and self._store:
            try:
                query_vec = self._backend.embed([query])[0]
                semantic_results = self._store.search(query_vec, top_k=k)

                for name, score in semantic_results:
                    entry = self._get_entry(name)
                    result = {"name": name, "score": round(score, 4), "source": "semantic"}
                    if include_descriptions and entry:
                        result["description"] = entry.get("description", "")
                    results.append(result)

                if self.verbose:
                    _log_search_results(query, results, "semantic")

            except Exception as e:
                log_error(f"Semantic search failed: {e}")

        # Check if top result has sufficient confidence
        top_score = results[0]["score"] if results else 0
        if top_score < self.fallback_threshold:
            # Supplement with keyword search
            log_info(f"Low confidence (top={top_score:.3f} < {self.fallback_threshold}), supplementing with keyword search")
            kw_results = keyword_search(query, self._registry, top_k=k)

            # Merge keyword results (boost score for tools already found semantically)
            existing_names = {r["name"] for r in results}
            for name, score in kw_results:
                if name in existing_names:
                    # Boost existing result
                    for r in results:
                        if r["name"] == name:
                            r["score"] = round(min(1.0, r["score"] + score * 0.3), 4)
                            r["source"] = "semantic+keyword"
                else:
                    entry = self._get_entry(name)
                    result = {"name": name, "score": round(score * 0.8, 4), "source": "keyword"}
                    if include_descriptions and entry:
                        result["description"] = entry.get("description", "")
                    results.append(result)

            # Re-sort by score
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:k]

            if self.verbose:
                _log_search_results(query, results, "merged")

        return results

    def search_for_capabilities(self, capabilities: list[str], top_k: int | None = None) -> list[dict]:
        """Search for tools matching a list of required capabilities.

        This is designed for the architect→programmer handoff: the architect
        specifies what capabilities are needed, and this finds tools that match.

        Args:
            capabilities: List of capability descriptions, e.g. ["file editing",
                         "web scraping", "docker management"]
            top_k: Number of results per capability

        Returns:
            Deduplicated list of tool matches, sorted by best score across
            all capabilities.
        """
        all_results = {}
        k = top_k or self.top_k

        for cap in capabilities:
            results = self.search(cap, top_k=k, include_descriptions=True)
            for r in results:
                name = r["name"]
                if name in all_results:
                    # Keep the best score
                    if r["score"] > all_results[name]["score"]:
                        all_results[name]["score"] = r["score"]
                    all_results[name]["matched_capabilities"] = all_results[name].get(
                        "matched_capabilities", []
                    ) + [cap]
                else:
                    r["matched_capabilities"] = [cap]
                    all_results[name] = r

        # Sort by score descending
        sorted_results = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:k * 2]  # Return more since multiple capabilities may match different tools

    def _get_entry(self, name: str) -> dict | None:
        """Get registry entry by tool name."""
        for entry in self._registry:
            if entry["name"] == name:
                return entry
        return None

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def backend_name(self) -> str:
        return self._backend.name if self._backend else "none"

    @property
    def tool_count(self) -> int:
        return len(self._tool_names)


# ---------------------------------------------------------------------------
# Verbose logging helpers
# ---------------------------------------------------------------------------

def _log_search_results(query: str, results: list[dict], source: str):
    """Color-coded display of search results."""
    _log(
        f"{'─' * 60}",
        f"Query: {_COLOR_BRIGHT}{query}{_COLOR_RESET}",
        _COLOR_CYAN,
    )
    print(f"  [{_COLOR_CYAN}Search{_COLOR_RESET}] Query: {_COLOR_BRIGHT}{query}{_COLOR_RESET}", file=sys.stderr)
    print(f"  [{_COLOR_CYAN}Search{_COLOR_RESET}] Source: {source}, Results: {len(results)}", file=sys.stderr)

    for i, r in enumerate(results):
        name = r["name"]
        score = r["score"]
        src = r.get("source", "?")

        # Color-code by confidence
        if score >= 0.7:
            score_color = _COLOR_GREEN
            bar = "████"
        elif score >= 0.4:
            score_color = _COLOR_YELLOW
            bar = "██░░"
        else:
            score_color = _COLOR_RED
            bar = "█░░░"

        desc = r.get("description", "")[:60]
        print(
            f"  [{_COLOR_MAGENTA}Score{_COLOR_RESET}] "
            f"{score_color}{bar} {score:.3f}{_COLOR_RESET} "
            f"{_COLOR_BRIGHT}{name:<25}{_COLOR_RESET} "
            f"{_COLOR_DIM}({src}){_COLOR_RESET} "
            f"{_COLOR_DIM}{desc}{_COLOR_RESET}",
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_global_search: SemanticToolSearch | None = None


def get_tool_search() -> SemanticToolSearch | None:
    """Get the global SemanticToolSearch instance, or None if not initialized."""
    return _global_search


def init_tool_search(api_key: str = "", api_base_url: str = "",
                     embedding_model: str = "", top_k: int = 10,
                     confidence_threshold: float = 0.4,
                     fallback_threshold: float = 0.3,
                     verbose: bool = False,
                     use_cache: bool = True) -> SemanticToolSearch | None:
    """Initialize the global semantic tool search engine.

    Returns the search instance if initialization succeeded, None otherwise.
    """
    global _global_search

    engine = SemanticToolSearch(
        api_key=api_key,
        api_base_url=api_base_url,
        embedding_model=embedding_model,
        top_k=top_k,
        confidence_threshold=confidence_threshold,
        fallback_threshold=fallback_threshold,
        verbose=verbose,
        use_cache=use_cache,
    )

    if engine.initialize():
        _global_search = engine
        return _global_search

    return None
