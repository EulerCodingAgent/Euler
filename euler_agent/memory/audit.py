"""Structured audit logging for autopilot runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def create_audit_run(workdir: str, goal: str, metadata: dict) -> tuple[str, Path]:
    run_id = uuid4().hex
    root = Path(workdir).resolve() / ".euler" / "audit"
    root.mkdir(parents=True, exist_ok=True)
    file_path = root / f"{run_id}.jsonl"
    started = {
        "type": "run_started",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "goal": goal,
        "metadata": metadata,
    }
    file_path.write_text(json.dumps(started) + "\n", encoding="utf-8")
    return run_id, file_path


def append_audit_event(file_path: Path, payload: dict) -> None:
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")
