"""
Named REPL sessions with JSONL checkpoints and git-based exit summaries.
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_new_session_name() -> str:
    """Local timestamp label for each fresh Euler launch (folder id stays unique)."""
    return datetime.now().strftime("chat-%Y-%m-%d-%H%M%S")


def sessions_root(workdir: Path) -> Path:
    return workdir.resolve() / ".euler" / "sessions"


def _slug(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name.strip()).strip("_")
    return s or "session"


@dataclass
class SessionMeta:
    session_id: str
    name: str
    workdir: str
    created_at: str
    updated_at: str
    git_head_start: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "workdir": self.workdir,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "git_head_start": self.git_head_start,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionMeta:
        return cls(
            session_id=str(data["session_id"]),
            name=str(data["name"]),
            workdir=str(data["workdir"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            git_head_start=data.get("git_head_start"),
        )


@dataclass
class SessionSummary:
    session_id: str
    name: str
    workdir: str
    created_at: str
    updated_at: str
    turn_count: int
    path: Path


@dataclass
class SessionHandle:
    """Active session: append-only turns + mutable meta on disk."""

    root: Path
    meta: SessionMeta
    turns_path: Path
    _meta_path: Path = field(repr=False)

    @property
    def session_id(self) -> str:
        return self.meta.session_id

    @property
    def name(self) -> str:
        return self.meta.name

    @property
    def workdir(self) -> Path:
        return Path(self.meta.workdir)

    def append_turn(
        self,
        role: str,
        text: str,
        *,
        max_chars: int = 500_000,
    ) -> None:
        if not text:
            return
        payload = text if len(text) <= max_chars else text[: max_chars - 80] + "\n… [truncated for checkpoint]\n"
        line = json.dumps(
            {"role": role, "text": payload, "ts": utc_now_iso()},
            ensure_ascii=False,
        )
        self.turns_path.parent.mkdir(parents=True, exist_ok=True)
        with self.turns_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        self.meta.updated_at = utc_now_iso()
        self._write_meta()

    def _write_meta(self) -> None:
        self._meta_path.write_text(
            json.dumps(self.meta.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def touch(self) -> None:
        """Refresh `updated_at` and flush meta (explicit checkpoint)."""
        self.meta.updated_at = utc_now_iso()
        self._write_meta()


def git_head(workdir: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(workdir.resolve()),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def git_diff_from_head(workdir: Path, start_head: str | None, *, max_bytes: int = 200_000) -> str:
    """
    Working tree + index diff versus start_head (commit when session began).
    Falls back to plain `git diff` if start_head is missing or invalid.
    """
    wd = workdir.resolve()
    try:
        if start_head:
            r = subprocess.run(
                ["git", "diff", start_head, "--"],
                cwd=str(wd),
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            if r.returncode == 0:
                return _cap_diff(r.stdout, max_bytes)
        r2 = subprocess.run(
            ["git", "diff", "--"],
            cwd=str(wd),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if r2.returncode == 0:
            return _cap_diff(r2.stdout, max_bytes)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(git diff failed: {exc})"
    return "(could not produce git diff)"


def _cap_diff(text: str, max_bytes: int) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text
    decoded = raw[:max_bytes].decode("utf-8", errors="replace")
    return decoded + "\n\n… [diff truncated for terminal]\n"


def git_diff_stats(workdir: Path, start_head: str | None) -> str:
    wd = workdir.resolve()
    try:
        if start_head:
            r = subprocess.run(
                ["git", "diff", "--stat", start_head, "--"],
                cwd=str(wd),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if r.returncode == 0 and r.stdout.strip():
                return r.stdout.strip()
        r2 = subprocess.run(
            ["git", "diff", "--stat", "--"],
            cwd=str(wd),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if r2.returncode == 0:
            return r2.stdout.strip() or "(no unstaged changes)"
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"(git diff --stat failed: {exc})"
    return "(could not produce git diff stats)"


def create_session(workdir: Path, name: str) -> SessionHandle:
    wd = workdir.resolve()
    root_dir = sessions_root(wd)
    root_dir.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex[:10]
    folder = f"{_slug(name)}_{token}"
    sdir = root_dir / folder
    if sdir.exists():
        raise FileExistsError(sdir)
    sdir.mkdir(parents=True, exist_ok=True)
    sid = folder
    now = utc_now_iso()
    head = git_head(wd)
    meta = SessionMeta(
        session_id=sid,
        name=name.strip(),
        workdir=str(wd),
        created_at=now,
        updated_at=now,
        git_head_start=head,
    )
    meta_path = sdir / "meta.json"
    turns_path = sdir / "turns.jsonl"
    meta_path.write_text(
        json.dumps(meta.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    turns_path.write_text("", encoding="utf-8")
    return SessionHandle(root=sdir, meta=meta, turns_path=turns_path, _meta_path=meta_path)


def load_session_handle(session_dir: Path) -> SessionHandle | None:
    meta_path = session_dir / "meta.json"
    turns_path = session_dir / "turns.jsonl"
    if not meta_path.is_file() or not turns_path.is_file():
        return None
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        meta = SessionMeta.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    return SessionHandle(root=session_dir, meta=meta, turns_path=turns_path, _meta_path=meta_path)


def iter_session_dirs(workdir: Path) -> Iterator[Path]:
    root = sessions_root(workdir)
    if not root.is_dir():
        return
    for child in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if child.is_dir() and (child / "meta.json").is_file():
            yield child


def list_sessions(workdir: Path) -> list[SessionSummary]:
    out: list[SessionSummary] = []
    wd = workdir.resolve()
    for sdir in iter_session_dirs(wd):
        h = load_session_handle(sdir)
        if h is None:
            continue
        n = _count_lines(h.turns_path)
        out.append(
            SessionSummary(
                session_id=h.session_id,
                name=h.name,
                workdir=h.meta.workdir,
                created_at=h.meta.created_at,
                updated_at=h.meta.updated_at,
                turn_count=n,
                path=h.root,
            )
        )
    return out


def _count_lines(path: Path) -> int:
    try:
        return sum(1 for _ in path.open("r", encoding="utf-8"))
    except OSError:
        return 0


def find_session_by_name_or_id(workdir: Path, token: str) -> Path | None:
    """
    Resolve session folder by exact session_id (folder name) or by display name
    (latest matching meta.name if multiple).
    """
    needle = token.strip()
    if not needle:
        return None
    wd = workdir.resolve()
    by_name: list[tuple[float, Path]] = []
    for sdir in iter_session_dirs(wd):
        h = load_session_handle(sdir)
        if h is None:
            continue
        if h.session_id == needle or sdir.name == needle:
            return sdir
        if h.name.lower() == needle.lower():
            try:
                mtime = sdir.stat().st_mtime
            except OSError:
                mtime = 0.0
            by_name.append((mtime, sdir))
    if by_name:
        by_name.sort(key=lambda x: x[0], reverse=True)
        return by_name[0][1]
    return None


def load_turns(session_dir: Path, *, tail: int | None = None) -> list[dict[str, Any]]:
    turns_path = session_dir / "turns.jsonl"
    if not turns_path.is_file():
        return []
    lines = turns_path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if tail is not None and tail > 0:
        return rows[-tail:]
    return rows


def format_turns_recap(turns: list[dict[str, Any]], *, max_lines: int = 40) -> str:
    parts: list[str] = []
    for row in turns[-max_lines:]:
        role = row.get("role", "?")
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        preview = text if len(text) <= 2000 else text[:1997] + "…"
        parts.append(f"[{role}]\n{preview}")
    return "\n\n---\n\n".join(parts) if parts else "(no turns recorded)"


def build_resume_context_prefix(turns: list[dict[str, Any]], *, max_user_snippets: int = 5) -> str:
    """Compact prefix for the next model call after resume (optional injection)."""
    user_snips: list[str] = []
    for row in turns:
        if row.get("role") != "user":
            continue
        t = str(row.get("text", "")).strip()
        if not t or t.startswith("/session"):
            continue
        one = t.replace("\n", " ").strip()
        if len(one) > 220:
            one = one[:217] + "…"
        user_snips.append(one)
    tail = user_snips[-max_user_snippets:]
    if not tail:
        return ""
    bullets = "\n".join(f"- {s}" for s in tail)
    return (
        "[Resumed Euler session — recent user messages for context]\n"
        f"{bullets}\n\n"
    )
