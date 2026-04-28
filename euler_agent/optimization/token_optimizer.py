"""
Token optimization engine for the Euler multi-agent pipeline.

Implements five complementary strategies:

  1. Query complexity classification
       Classifies queries as FOCUSED / STANDARD / FULL and selects only the
       specialist agents that are actually needed, reducing LLM calls from 11
       down to 3-5 for the majority of real-world tasks.

  2. Relevance-gated context injection
       Memory and semantic-index hits are only injected when their TF-IDF
       cosine similarity exceeds a configurable threshold, eliminating
       low-signal context noise that bloats prompts without helping.

  3. Specialist output compression
       Before aggregating specialist outputs in the arbitrator, each output
       is capped at a configurable character limit so the arbitrator's human
       prompt stays bounded regardless of how verbose a specialist was.

  4. Response caching
       A keyed, TTL-based cache (stored in ~/.euler_agent/response_cache.json)
       returns instant answers for repeated or near-identical queries without
       touching the LLM at all.

  5. Token estimation
       A best-effort token counter (tiktoken when available, word-count
       heuristic otherwise) powers the benchmark and lets callers report
       real savings in tokens rather than just characters.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from euler_agent.analysis.tfidf import cosine_sparse, embed


# ---------------------------------------------------------------------------
# Configuration constants (all overridable via TokenOptimizer constructor)
# ---------------------------------------------------------------------------

_CACHE_FILE = Path.home() / ".euler_agent" / "response_cache.json"
_CACHE_TTL_SECONDS = 3_600          # 1 hour
_RELEVANCE_THRESHOLD = 0.08         # Minimum cosine similarity to include context
_SPECIALIST_OUTPUT_CAP = 1_500      # Max chars per specialist fed to arbitrator
_CACHE_MAX_ENTRIES = 200            # LRU cap for the on-disk cache

# ---------------------------------------------------------------------------
# Keyword signals per specialist
# ---------------------------------------------------------------------------

# Each value is a set of sub-strings.  A specialist is *triggered* when ANY
# signal appears as a sub-string of the lower-cased user query.
_SPECIALIST_SIGNALS: dict[str, frozenset[str]] = {
    "architect": frozenset({
        "architect", "design", "structure", "module", "blueprint",
        "component", "interface", "contract", "microservice",
        "scalab", "layer", "api design", "system design",
    }),
    "coder": frozenset({
        "fix", "implement", "build", "write", "create", "code",
        "function", "class", "method", "bug", "feature", "add",
        "update", "change", "complete", "finish", "solve", "patch",
        "correct", "repair", "edit", "modify", "extend",
    }),
    "tester": frozenset({
        "test", "testing", "unit test", "coverage", "spec",
        "pytest", "jest", "mock", "assertion", "suite", "verify",
        "validate", "edge case",
    }),
    "security": frozenset({
        "security", "vulnerab", "auth", "injection", "owasp",
        "secret", "password", "token", "permission", "xss",
        "sanitiz", "exploit", "attack", "encrypt", "csrf",
    }),
    "devops": frozenset({
        "deploy", "docker", "ci/cd", "ci ", " cd ", "pipeline",
        "kubernetes", "infra", "container", "server", "cloud",
        "github action", "gitlab", "helm", "k8s", "nginx",
        "dockerfile", "compose",
    }),
    "db": frozenset({
        "database", " sql", "schema", "migration", "query",
        " table", "orm", "postgres", "mysql", "mongo", "redis",
        " index", "relation", "join", "transaction", "data layer",
    }),
    "documenter": frozenset({
        "document", "readme", "jsdoc", "docstring",
        "guide", "runbook", "adr", "changelog", "wiki",
    }),
    "refactor": frozenset({
        "refactor", "clean up", "dry ", "simplify", "restructure",
        "rename", "extract", "improve quality", "lint", "format",
        "code quality", "tech debt",
    }),
}

# All specialist keys in canonical order
ALL_SPECIALISTS: list[str] = [
    "architect", "coder", "tester", "security",
    "devops", "db", "documenter", "refactor",
]


# ---------------------------------------------------------------------------
# Enums & data classes
# ---------------------------------------------------------------------------

class QueryComplexity(Enum):
    """
    FOCUSED  — planner + ≤2 specialists + reviewer  (≈3-4 LLM calls).
    STANDARD — planner + 3-5 specialists + arbitrator + reviewer  (≈6-8 calls).
    FULL     — complete 11-call pipeline, no optimisation applied.
    """
    FOCUSED = "focused"
    STANDARD = "standard"
    FULL = "full"


@dataclass
class OptimizationResult:
    """Carries the classification decision and token-saving estimate."""
    complexity: QueryComplexity
    selected_specialists: list[str]
    baseline_specialist_count: int = field(default=8)

    @property
    def specialists_saved(self) -> int:
        return self.baseline_specialist_count - len(self.selected_specialists)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Best-effort LLM token count for *text*.

    Tries tiktoken (cl100k_base, compatible with GPT-4 / Gemini estimates)
    and falls back to a word-count heuristic when not installed.
    The heuristic error is typically within ±15 % for English + code.
    """
    try:
        import tiktoken  # type: ignore[import-untyped]
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # ~1.35 tokens/word is a reliable code-aware estimate
        return max(1, int(len(text.split()) * 1.35))


# ---------------------------------------------------------------------------
# Main optimiser class
# ---------------------------------------------------------------------------

class TokenOptimizer:
    """
    Drop-in optimisation layer for EulerAgent.

    Usage::

        optimizer = TokenOptimizer()

        # 1. Classify query and choose specialists
        result = optimizer.classify_query(user_goal)

        # 2. Filter context by relevance
        good_hits = optimizer.filter_semantic_hits(scored_hits, result.complexity)

        # 3. Compress specialist outputs before arbitration
        compressed = optimizer.compress_specialist_outputs(outputs, result.selected_specialists)

        # 4. Check cache before invoking the LLM pipeline
        if (cached := optimizer.get_cached_response(key)):
            return cached
    """

    def __init__(
        self,
        relevance_threshold: float = _RELEVANCE_THRESHOLD,
        specialist_output_cap: int = _SPECIALIST_OUTPUT_CAP,
        cache_ttl: int = _CACHE_TTL_SECONDS,
        cache_file: Path = _CACHE_FILE,
    ) -> None:
        self.relevance_threshold = relevance_threshold
        self.specialist_output_cap = specialist_output_cap
        self.cache_ttl = cache_ttl
        self.cache_file = cache_file

    # ------------------------------------------------------------------ #
    # 1. Query classification                                              #
    # ------------------------------------------------------------------ #

    def classify_query(self, query: str) -> OptimizationResult:
        """
        Analyse *query* and return an :class:`OptimizationResult` that
        describes which complexity tier and which specialists to invoke.

        Decision logic:
          - Scan for specialist signal sub-strings in the lowercased query.
          - Always include "coder" for any action-oriented query.
          - 1-2 signals  → FOCUSED
          - 3-5 signals  → STANDARD
          - 6+ signals   → FULL  (no point restricting — user wants everything)
        """
        lower = query.lower()

        triggered: list[str] = []
        for specialist, signals in _SPECIALIST_SIGNALS.items():
            if any(sig in lower for sig in signals):
                triggered.append(specialist)

        # Coder is the default workhorse — always include it for action queries
        if "coder" not in triggered:
            triggered.append("coder")

        n = len(triggered)

        if n <= 2:
            complexity = QueryComplexity.FOCUSED
        elif n <= 5:
            complexity = QueryComplexity.STANDARD
        else:
            complexity = QueryComplexity.FULL
            triggered = ALL_SPECIALISTS[:]

        return OptimizationResult(
            complexity=complexity,
            selected_specialists=triggered,
        )

    # ------------------------------------------------------------------ #
    # 2. Relevance-gated context                                           #
    # ------------------------------------------------------------------ #

    def filter_semantic_hits(
        self,
        hits_with_scores: list[tuple[float, dict]],
        complexity: QueryComplexity,
    ) -> list[dict]:
        """
        Drop semantic-index hits whose cosine similarity is below the
        threshold, then cap the list by complexity tier.

        Args:
            hits_with_scores: Pairs of (score, chunk_dict) from the index.
            complexity: Determines the upper count limit.

        Returns:
            Filtered and capped list of chunk dicts ready for context injection.
        """
        limit = {
            QueryComplexity.FOCUSED: 2,
            QueryComplexity.STANDARD: 3,
            QueryComplexity.FULL: 5,
        }[complexity]

        relevant = [
            (score, hit)
            for score, hit in hits_with_scores
            if score >= self.relevance_threshold
        ]
        relevant.sort(key=lambda p: p[0], reverse=True)
        return [h for _, h in relevant[:limit]]

    def filter_memory_hits(
        self,
        entries_with_scores: list[tuple[float, object]],
        complexity: QueryComplexity,
    ) -> list[object]:
        """
        Drop memory entries whose cosine similarity is below the threshold,
        then cap the list by complexity tier.
        """
        limit = {
            QueryComplexity.FOCUSED: 1,
            QueryComplexity.STANDARD: 2,
            QueryComplexity.FULL: 4,
        }[complexity]

        relevant = [
            (score, entry)
            for score, entry in entries_with_scores
            if score >= self.relevance_threshold
        ]
        relevant.sort(key=lambda p: p[0], reverse=True)
        return [e for _, e in relevant[:limit]]

    # ------------------------------------------------------------------ #
    # 3. Specialist output compression                                     #
    # ------------------------------------------------------------------ #

    def compress_specialist_outputs(
        self,
        state: dict,
        selected_specialists: list[str],
    ) -> str:
        """
        Build the arbitrator human-prompt section that contains specialist
        outputs, using only *selected_specialists* and capping each at
        :attr:`specialist_output_cap` characters.

        Unselected specialists are omitted entirely; selected specialists
        that returned empty strings are skipped silently.
        """
        parts: list[str] = []
        for key in selected_specialists:
            raw: str = state.get(f"{key}_output", "") or ""
            if not raw.strip():
                continue
            if len(raw) > self.specialist_output_cap:
                capped = (
                    raw[: self.specialist_output_cap]
                    + f"\n... [+{len(raw) - self.specialist_output_cap} chars truncated]"
                )
            else:
                capped = raw
            parts.append(f"---\n### {key.title()} Output\n{capped}")

        return "\n\n".join(parts) if parts else "(no specialist outputs)"

    # ------------------------------------------------------------------ #
    # 4. Response caching                                                  #
    # ------------------------------------------------------------------ #

    def make_cache_key(
        self,
        model_id: str,
        query: str,
        context_fingerprint: str = "",
    ) -> str:
        """
        Deterministic cache key from model identity, query, and optional
        context fingerprint (e.g. workdir path hash).
        """
        raw = f"{model_id}||{query.strip()}||{context_fingerprint}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def get_cached_response(self, cache_key: str) -> str | None:
        """Return a cached response if it exists and has not expired."""
        if not self.cache_file.exists():
            return None
        try:
            store: dict = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        entry = store.get(cache_key)
        if not entry:
            return None
        if time.time() - entry.get("ts", 0) > self.cache_ttl:
            return None
        return entry.get("response")

    def get_semantic_cached_response(
        self,
        model_id: str,
        query: str,
        context_fingerprint: str = "",
        min_similarity: float = 0.93,
    ) -> str | None:
        """
        Return a semantically similar cached response if available.

        Uses sparse TF cosine similarity over cached query embeddings. Only cache
        entries from the same model and context fingerprint are considered.
        """
        if not self.cache_file.exists():
            return None
        try:
            store: dict = json.loads(self.cache_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        now = time.time()
        query_vec = embed(query)
        best_sim = 0.0
        best_resp: str | None = None
        for entry in store.values():
            if not isinstance(entry, dict):
                continue
            if now - entry.get("ts", 0) > self.cache_ttl:
                continue
            if entry.get("model_id") != model_id:
                continue
            if entry.get("context_fingerprint", "") != context_fingerprint:
                continue
            emb = entry.get("query_embedding")
            if not isinstance(emb, dict):
                continue
            sim = cosine_sparse(query_vec, emb)
            if sim >= min_similarity and sim > best_sim:
                response = entry.get("response")
                if isinstance(response, str):
                    best_sim = sim
                    best_resp = response
        return best_resp

    def set_cached_response(self, cache_key: str, response: str) -> None:
        """Persist *response* to the cache under *cache_key*."""
        store: dict = {}
        if self.cache_file.exists():
            try:
                store = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                store = {}

        store[cache_key] = {"response": response, "ts": time.time()}

        # LRU eviction: keep only the most recent _CACHE_MAX_ENTRIES entries
        if len(store) > _CACHE_MAX_ENTRIES:
            sorted_keys = sorted(store, key=lambda k: store[k].get("ts", 0))
            for old_key in sorted_keys[: len(store) - _CACHE_MAX_ENTRIES]:
                del store[old_key]

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(store, indent=2), encoding="utf-8")

    def set_semantic_cached_response(
        self,
        cache_key: str,
        response: str,
        model_id: str,
        query: str,
        context_fingerprint: str = "",
    ) -> None:
        """
        Persist a cache entry with semantic metadata for approximate matching.
        """
        store: dict = {}
        if self.cache_file.exists():
            try:
                store = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                store = {}

        store[cache_key] = {
            "response": response,
            "ts": time.time(),
            "model_id": model_id,
            "context_fingerprint": context_fingerprint,
            "query": query,
            "query_embedding": embed(query),
        }

        if len(store) > _CACHE_MAX_ENTRIES:
            sorted_keys = sorted(store, key=lambda k: store[k].get("ts", 0))
            for old_key in sorted_keys[: len(store) - _CACHE_MAX_ENTRIES]:
                del store[old_key]

        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(store, indent=2), encoding="utf-8")

    def invalidate_cache(self) -> int:
        """Clear all cached responses. Returns the number of entries removed."""
        if not self.cache_file.exists():
            return 0
        try:
            store: dict = json.loads(self.cache_file.read_text(encoding="utf-8"))
            count = len(store)
        except (json.JSONDecodeError, OSError):
            return 0
        self.cache_file.write_text("{}", encoding="utf-8")
        return count

    # ------------------------------------------------------------------ #
    # 5. Context size estimation                                           #
    # ------------------------------------------------------------------ #

    def estimate_prompt_tokens(
        self,
        system_prompt: str,
        human_prompt: str,
    ) -> int:
        """Estimate total tokens for a single (system, human) prompt pair."""
        return estimate_tokens(system_prompt) + estimate_tokens(human_prompt)

    # ------------------------------------------------------------------ #
    # 6. Adaptive specialist stopping                                     #
    # ------------------------------------------------------------------ #

    def should_skip_specialists(
        self,
        query: str,
        complexity: QueryComplexity,
        planner_output: str,
    ) -> tuple[bool, float, str]:
        """
        Decide if specialists can be skipped safely for low-risk focused tasks.

        Returns:
            (skip, confidence_0_to_1, reason)
        """
        if complexity != QueryComplexity.FOCUSED:
            return False, 0.35, "non-focused query"

        lower = query.lower()
        high_risk_signals = (
            "security", "auth", "password", "token", "sql", "database",
            "migration", "deploy", "docker", "kubernetes", "payment",
            "encrypt", "compliance", "permission",
        )
        if any(sig in lower for sig in high_risk_signals):
            return False, 0.45, "high-risk domain keywords detected"

        has_structured_plan = "## " in planner_output or "1." in planner_output
        confidence = 0.9 if has_structured_plan else 0.82
        if len(planner_output.strip()) < 180:
            confidence -= 0.08

        skip = confidence >= 0.85
        reason = "focused low-risk task with high planner confidence" if skip else "planner confidence too low"
        return skip, max(0.0, min(1.0, confidence)), reason
