"""Typed patch protocol models for REPL patch extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class PatchEdit(BaseModel):
    path: str
    operation: Literal["write", "delete"] = "write"
    content: str | None = None


class PatchEnvelope(BaseModel):
    edits: list[PatchEdit]


PatchTuple = tuple[Path, str | None, Literal["write", "delete"]]

