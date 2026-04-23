"""Repository semantic indexing and retrieval."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from fastembed import TextEmbedding

SUPPORTED_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".sql",
}

_EMBED_MODEL: TextEmbedding | None = None


@dataclass
class CodeChunk:
    path: str
    start_line: int
    end_line: int
    content: str
    embedding: list[float]


def _get_model() -> TextEmbedding:
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _EMBED_MODEL


def _embed(text: str) -> list[float]:
    model = _get_model()
    return list(next(model.embed([text])))


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vec: list[float]) -> float:
    return sqrt(sum(v * v for v in vec))


def _cosine(left: list[float], right: list[float]) -> float:
    left_norm = _norm(left)
    right_norm = _norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return _dot(left, right) / (left_norm * right_norm)


def _iter_repo_files(workdir: Path) -> list[Path]:
    files: list[Path] = []
    for path in workdir.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts if part != "."):
            continue
        if "venv" in path.parts or ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        files.append(path)
    return files


def _chunk_file(path: Path, chunk_lines: int = 80, overlap: int = 20) -> list[tuple[int, int, str]]:
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
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def index_path(workdir: str, output_path: str | None = None, incremental: bool = True) -> str:
    root = Path(workdir).resolve()
    index_file = Path(output_path).resolve() if output_path else root / ".euler" / "semantic_index.json"
    index_file.parent.mkdir(parents=True, exist_ok=True)

    old_payload: dict = {}
    if incremental and index_file.exists():
        old_payload = json.loads(index_file.read_text(encoding="utf-8"))

    prior_chunks_by_file: dict[str, list[dict]] = {}
    prior_manifest: dict[str, dict] = {}
    for chunk in old_payload.get("chunks", []):
        prior_chunks_by_file.setdefault(chunk.get("path", ""), []).append(chunk)
    prior_manifest = old_payload.get("manifest", {})

    rows: list[dict] = []
    manifest: dict[str, dict] = {}
    reused_files = 0
    rebuilt_files = 0
    for file_path in _iter_repo_files(root):
        rel = str(file_path.relative_to(root))
        file_hash = _hash_file(file_path)
        mtime_ns = file_path.stat().st_mtime_ns
        manifest[rel] = {"hash": file_hash, "mtime_ns": mtime_ns}

        prior = prior_manifest.get(rel)
        if incremental and prior and prior.get("hash") == file_hash:
            rows.extend(prior_chunks_by_file.get(rel, []))
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
                    embedding=_embed(f"{rel}\n{content}"),
                ).__dict__
            )

    payload = {"root": str(root), "manifest": manifest, "chunks": rows}
    index_file.write_text(json.dumps(payload), encoding="utf-8")
    mode = "incremental" if incremental else "full"
    return (
        f"Indexed {len(rows)} chunks into {index_file} ({mode}; "
        f"reused_files={reused_files}, rebuilt_files={rebuilt_files})"
    )


def search_index(workdir: str, query: str, limit: int = 5) -> list[dict]:
    root = Path(workdir).resolve()
    index_file = root / ".euler" / "semantic_index.json"
    if not index_file.exists():
        return []
    payload = json.loads(index_file.read_text(encoding="utf-8"))
    query_embedding = _embed(query)
    scored: list[tuple[float, dict]] = []
    for chunk in payload.get("chunks", []):
        embedding = chunk.get("embedding")
        if not embedding:
            continue
        score = _cosine(query_embedding, embedding)
        if score <= 0:
            continue
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]
