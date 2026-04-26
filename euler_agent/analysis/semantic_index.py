"""
Repository semantic indexing and retrieval.

Uses a pure-Python sparse TF-IDF engine (no native dependencies).
Supports incremental rebuilds keyed on per-file SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from euler_agent.analysis.tfidf import cosine_sparse, embed

SUPPORTED_EXTENSIONS = {
    ".py", ".md", ".txt", ".json",
    ".yaml", ".yml", ".toml",
    ".js", ".ts", ".tsx", ".jsx", ".sql",
}

_SKIP_DIRS = {"venv", ".venv", "__pycache__", "node_modules", ".git", ".euler"}


@dataclass
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    content: str
    embedding: dict[str, float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _iter_repo_files(workdir: Path) -> list[Path]:
    files: list[Path] = []
    for path in workdir.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if any(part.startswith(".") for part in path.relative_to(workdir).parts):
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files.append(path)
    return files


def _chunk_file(
    path: Path,
    chunk_lines: int = 80,
    overlap: int = 20,
) -> list[tuple[int, int, str]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if not lines:
        return []
    chunks: list[tuple[int, int, str]] = []
    idx = 0
    while idx < len(lines):
        start = idx
        end = min(len(lines), idx + chunk_lines)
        content = "\n".join(lines[start:end]).strip()
        if content:
            chunks.append((start + 1, end, content))
        if end == len(lines):
            break
        idx = max(end - overlap, idx + 1)
    return chunks


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def index_path(
    workdir: str,
    output_path: str | None = None,
    incremental: bool = True,
) -> str:
    """
    Build (or update) the semantic index for *workdir*.

    Args:
        workdir: Root of the repository to index.
        output_path: Override location for the index file.
        incremental: When True, reuse chunks for files whose hash hasn't changed.

    Returns:
        Human-readable status string.
    """
    root = Path(workdir).resolve()
    index_file = (
        Path(output_path).resolve()
        if output_path
        else root / ".euler" / "semantic_index.json"
    )
    index_file.parent.mkdir(parents=True, exist_ok=True)

    old_payload: dict = {}
    if incremental and index_file.exists():
        try:
            old_payload = json.loads(index_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            old_payload = {}

    prior_chunks_by_file: dict[str, list[dict]] = {}
    for chunk in old_payload.get("chunks", []):
        prior_chunks_by_file.setdefault(chunk.get("path", ""), []).append(chunk)
    prior_manifest: dict[str, dict] = old_payload.get("manifest", {})

    rows: list[dict] = []
    manifest: dict[str, dict] = {}
    reused_files = 0
    rebuilt_files = 0

    for file_path in _iter_repo_files(root):
        rel = str(file_path.relative_to(root))
        file_hash = _hash_file(file_path)
        manifest[rel] = {"hash": file_hash, "mtime_ns": file_path.stat().st_mtime_ns}

        prior = prior_manifest.get(rel)
        if incremental and prior and prior.get("hash") == file_hash:
            existing = prior_chunks_by_file.get(rel, [])
            # Migrate old dense-float embeddings → sparse dicts on the fly.
            for chunk in existing:
                if isinstance(chunk.get("embedding"), list):
                    chunk["embedding"] = embed(
                        f"{chunk.get('path', '')}\n{chunk.get('content', '')}"
                    )
            rows.extend(existing)
            reused_files += 1
            continue

        rebuilt_files += 1
        for start_line, end_line, content in _chunk_file(file_path):
            rows.append(
                CodeChunk(
                    path=rel,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    embedding=embed(f"{rel}\n{content}"),
                ).__dict__
            )

    payload = {"root": str(root), "manifest": manifest, "chunks": rows}
    index_file.write_text(json.dumps(payload), encoding="utf-8")
    mode = "incremental" if incremental else "full"
    return (
        f"Indexed {len(rows)} chunks into {index_file} "
        f"({mode}; reused={reused_files}, rebuilt={rebuilt_files})"
    )


def search_index(workdir: str, query: str, limit: int = 5) -> list[dict]:
    """
    Return the top-*limit* code chunks most similar to *query*.

    Returns an empty list if no index has been built yet.
    """
    return [chunk for _, chunk in search_index_scored(workdir, query, limit)]


def search_index_scored(
    workdir: str,
    query: str,
    limit: int = 8,
) -> list[tuple[float, dict]]:
    """
    Like :func:`search_index` but returns ``(score, chunk)`` pairs so that
    callers can apply their own relevance threshold.

    Args:
        workdir: Root of the repository whose index should be searched.
        query:   Free-form search query (natural language or code snippet).
        limit:   Maximum number of results to return before threshold filtering.

    Returns:
        List of ``(cosine_similarity, chunk_dict)`` sorted descending by score.
        Empty list when no index exists.
    """
    root = Path(workdir).resolve()
    index_file = root / ".euler" / "semantic_index.json"
    if not index_file.exists():
        return []

    try:
        payload = json.loads(index_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []

    query_vec = embed(query)
    scored: list[tuple[float, dict]] = []

    for chunk in payload.get("chunks", []):
        raw_emb = chunk.get("embedding")
        if not raw_emb:
            continue
        # Normalise legacy dense embeddings (list[float]) to sparse on the fly.
        if isinstance(raw_emb, list):
            raw_emb = embed(f"{chunk.get('path', '')}\n{chunk.get('content', '')}")
        score = cosine_sparse(query_vec, raw_emb)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]
