"""Project instruction and repository context helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path


KNOWLEDGE_DIR_NAME = "Euler-Knowledge"
ROLE_MAP_FILE_NAME = "file_role_map.json"


def get_knowledge_dir(workdir: Path) -> Path:
    """Return the canonical project knowledge directory path."""
    return workdir / KNOWLEDGE_DIR_NAME


def get_role_map_path(workdir: Path) -> Path:
    """Return the role-mapping JSON path inside the knowledge folder."""
    return get_knowledge_dir(workdir) / ROLE_MAP_FILE_NAME


def list_knowledge_files(workdir: Path) -> list[Path]:
    """List non-hidden files under Euler-Knowledge excluding role map storage."""
    knowledge_dir = get_knowledge_dir(workdir)
    if not knowledge_dir.exists() or not knowledge_dir.is_dir():
        return []
    files: list[Path] = []
    for path in sorted(knowledge_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ROLE_MAP_FILE_NAME:
            continue
        rel_parts = path.relative_to(knowledge_dir).parts
        if any(part.startswith(".") for part in rel_parts):
            continue
        files.append(path)
    return files


def load_file_role_mapping(workdir: Path) -> dict[str, str]:
    """Load role mappings as {relative_knowledge_file_path: responsibility}."""
    role_map_path = get_role_map_path(workdir)
    if not role_map_path.exists() or not role_map_path.is_file():
        return {}
    try:
        payload = json.loads(role_map_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_items = payload.get("mappings", payload)
    if not isinstance(raw_items, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in raw_items.items():
        if isinstance(key, str) and isinstance(value, str):
            cleaned_key = key.strip().replace("\\", "/")
            cleaned_val = value.strip()
            if cleaned_key and cleaned_val:
                out[cleaned_key] = cleaned_val
    return out


def save_file_role_mapping(workdir: Path, relative_file_path: str, responsibility: str) -> None:
    """Upsert a single file->responsibility entry."""
    knowledge_dir = get_knowledge_dir(workdir)
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    role_map_path = get_role_map_path(workdir)
    mappings = load_file_role_mapping(workdir)
    mappings[relative_file_path.replace("\\", "/").strip()] = responsibility.strip()
    payload = {"mappings": dict(sorted(mappings.items(), key=lambda item: item[0].lower()))}
    role_map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def format_file_role_mapping(workdir: Path) -> str:
    """Human-readable mapping list for CLI rendering."""
    mappings = load_file_role_mapping(workdir)
    if not mappings:
        return "No file-role mappings saved."
    lines = ["File -> Responsibility"]
    for rel_path, role in sorted(mappings.items(), key=lambda item: item[0].lower()):
        lines.append(f"- {rel_path} -> {role}")
    return "\n".join(lines)


def _tokenize(value: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9_]+", value.lower()) if len(tok) > 1}


def _match_score(combined_query: str, rel_path: str, role: str) -> int:
    query_tokens = _tokenize(combined_query)
    candidate_tokens = _tokenize(rel_path.replace("/", " ") + " " + role)
    overlap = len(query_tokens & candidate_tokens)
    score = overlap * 3
    query_lc = combined_query.lower()
    role_lc = role.lower()
    if role_lc and role_lc in query_lc:
        score += 8
    path_stem = Path(rel_path).stem.lower().replace("_", " ")
    if path_stem and path_stem in query_lc:
        score += 5
    return score


def build_role_reference_context(
    workdir: Path,
    user_goal: str,
    plan: str,
    max_matches: int = 3,
    max_chars_per_file: int = 1400,
) -> str:
    """
    Match planner/goal text with file-role mappings and attach matching files.

    Returns a formatted context block or "None" if no relevant mapping matched.
    """
    mappings = load_file_role_mapping(workdir)
    if not mappings:
        return "None"
    combined_query = f"{user_goal}\n{plan}".strip()
    if not combined_query:
        return "None"

    scored: list[tuple[int, str, str]] = []
    for rel_path, role in mappings.items():
        score = _match_score(combined_query, rel_path, role)
        if score > 0:
            scored.append((score, rel_path, role))
    if not scored:
        return "None"

    scored.sort(key=lambda row: (-row[0], row[1].lower()))
    selected = scored[:max_matches]
    knowledge_dir = get_knowledge_dir(workdir)
    blocks: list[str] = []
    for score, rel_path, role in selected:
        candidate = (knowledge_dir / rel_path).resolve()
        try:
            if not candidate.exists() or not candidate.is_file():
                continue
            content = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if not content:
                continue
        except Exception:
            continue
        if len(content) > max_chars_per_file:
            content = content[:max_chars_per_file] + "\n... [truncated]"
        blocks.append(
            f"### {rel_path}\n"
            f"Role: {role}\n"
            f"Match Score: {score}\n"
            f"```text\n{content}\n```"
        )
    if not blocks:
        return "None"
    return "\n\n".join(blocks)


def load_euler_instruction_docs(workdir: Path) -> str:
    """
    Load instruction files from ./Euler-Knowledge/ as supporting project instructions.

    Supported:
      - Markdown (*.md)
      - JSON (*.json)
    """
    instruction_dir = get_knowledge_dir(workdir)
    if not instruction_dir.exists() or not instruction_dir.is_dir():
        return ""

    blocks: list[str] = []
    patterns = ("*.md", "*.json")
    files = sorted(
        f
        for pattern in patterns
        for f in instruction_dir.glob(pattern)
        if f.is_file()
    )
    for file_path in files:
        if file_path.name == ROLE_MAP_FILE_NAME:
            continue
        content = file_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        blocks.append(f"# {file_path.name}\n{content}")

    return "\n\n".join(blocks)
