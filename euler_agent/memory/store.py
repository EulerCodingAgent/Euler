"""
Persistent long-term memory store with sparse TF-IDF retrieval.

Memory entries are stored per project directory in
~/.euler_agent/memory.json and survive across sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from euler_agent.analysis.tfidf import cosine_sparse, embed

MEMORY_DIR = Path.home() / ".euler_agent"
MEMORY_FILE = MEMORY_DIR / "memory.json"
MAX_ENTRIES = 500


@dataclass
class MemoryEntry:
    timestamp: str
    project: str
    goal: str
    result: str
    tags: list[str]
    embedding: dict[str, float]


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _load_raw() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    try:
        return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_raw(rows: list[dict]) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_memory(
    project: str,
    goal: str,
    result: str,
    tags: list[str] | None = None,
) -> None:
    """Persist a completed goal/result pair for future retrieval."""
    rows = _load_raw()
    memory_text = f"{goal.strip()}\n{result.strip()}\n{' '.join(tags or [])}"
    entry = MemoryEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        project=project,
        goal=goal.strip(),
        result=result.strip(),
        tags=tags or [],
        embedding=embed(memory_text),
    )
    rows.append(entry.__dict__)
    _save_raw(rows[-MAX_ENTRIES:])


def search_memory(project: str, query: str, limit: int = 3) -> list[MemoryEntry]:
    """
    Return the *limit* most semantically similar past entries for *project*.

    Handles legacy entries that stored dense float embeddings by re-embedding
    them on the fly using the current TF-IDF engine.
    """
    return [entry for _, entry in search_memory_scored(project, query, limit)]


def search_memory_scored(
    project: str,
    query: str,
    limit: int = 4,
) -> list[tuple[float, MemoryEntry]]:
    """
    Like :func:`search_memory` but returns ``(score, entry)`` pairs so callers
    can apply a relevance threshold before injecting memory into prompts.

    Args:
        project: Absolute path of the project directory (used as a namespace).
        query:   The current user goal to match against.
        limit:   Maximum results to return before threshold filtering.

    Returns:
        List of ``(cosine_similarity, MemoryEntry)`` sorted descending by score.
    """
    rows = _load_raw()
    if not rows:
        return []

    query_vec = embed(query)
    candidates: list[tuple[float, dict]] = []

    for row in rows:
        if row.get("project") != project:
            continue
        raw_emb = row.get("embedding")
        if not raw_emb:
            continue
        # Migrate old dense-float embeddings (list[float]) on the fly.
        if isinstance(raw_emb, list):
            raw_emb = embed(f"{row.get('goal', '')}\n{row.get('result', '')}")
            row["embedding"] = raw_emb
        score = cosine_sparse(query_vec, raw_emb)
        if score > 0:
            candidates.append((score, row))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [(score, MemoryEntry(**row)) for score, row in candidates[:limit]]
