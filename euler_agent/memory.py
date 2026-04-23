"""Persistent memory store with embedding-backed semantic retrieval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from math import sqrt
from pathlib import Path

from fastembed import TextEmbedding

MEMORY_DIR = Path.home() / ".euler_agent"
MEMORY_FILE = MEMORY_DIR / "memory.json"
_EMBEDDING_MODEL: TextEmbedding | None = None


def _get_embedding_model() -> TextEmbedding:
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        _EMBEDDING_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _EMBEDDING_MODEL


@dataclass
class MemoryEntry:
    timestamp: str
    project: str
    goal: str
    result: str
    tags: list[str]
    embedding: list[float]

def _embed_text(text: str) -> list[float]:
    model = _get_embedding_model()
    return list(next(model.embed([text])))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vec: list[float]) -> float:
    return sqrt(sum(v * v for v in vec))


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)


def _load_raw() -> list[dict]:
    if not MEMORY_FILE.exists():
        return []
    return json.loads(MEMORY_FILE.read_text(encoding="utf-8"))


def _save_raw(rows: list[dict]) -> None:
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    MEMORY_FILE.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def add_memory(project: str, goal: str, result: str, tags: list[str] | None = None) -> None:
    rows = _load_raw()
    memory_text = f"{goal.strip()}\n{result.strip()}\n{' '.join(tags or [])}"
    rows.append(
        MemoryEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            project=project,
            goal=goal.strip(),
            result=result.strip(),
            tags=tags or [],
            embedding=_embed_text(memory_text),
        ).__dict__
    )
    _save_raw(rows[-500:])  # keep a bounded long-term memory buffer


def search_memory(project: str, query: str, limit: int = 3) -> list[MemoryEntry]:
    rows = _load_raw()
    if not rows:
        return []
    query_embedding = _embed_text(query)
    candidates: list[tuple[float, dict]] = []
    for row in rows:
        if row.get("project") != project:
            continue
        stored_embedding = row.get("embedding")
        if not stored_embedding:
            continue
        score = _cosine_similarity(query_embedding, stored_embedding)
        if score > 0:
            candidates.append((score, row))
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    return [MemoryEntry(**row) for _, row in candidates[:limit]]
