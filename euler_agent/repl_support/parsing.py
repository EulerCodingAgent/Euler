"""Typed parsing utilities for REPL slash commands and @references."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

from euler_agent.repl_support.constants import RANGED_FILE_REF_PATTERN


class SlashCommand(BaseModel):
    name: str
    payload: str = ""


class FileRefSpec(BaseModel):
    raw_ref: str
    path: str
    start_line: int | None = None
    end_line: int | None = None

    @property
    def is_ranged(self) -> bool:
        return self.start_line is not None and self.end_line is not None


class ConvertCodeSpec(BaseModel):
    source_lang: str
    target_lang: str
    arrow: Literal["->", "→"]


def parse_slash_command(user_input: str) -> SlashCommand | None:
    text = user_input.strip()
    if not text.startswith("/"):
        return None
    if " " not in text:
        return SlashCommand(name=text.lower(), payload="")
    head, tail = text.split(" ", 1)
    return SlashCommand(name=head.lower(), payload=tail.strip())


def parse_file_ref(raw_ref: str) -> FileRefSpec:
    cleaned = raw_ref.rstrip(".,;:!?)]}")
    match = RANGED_FILE_REF_PATTERN.match(cleaned)
    if not match:
        return FileRefSpec(raw_ref=raw_ref, path=cleaned)
    return FileRefSpec(
        raw_ref=raw_ref,
        path=match.group("path"),
        start_line=int(match.group("start")),
        end_line=int(match.group("end")),
    )


def parse_convert_code_spec(payload: str) -> ConvertCodeSpec | None:
    spec = payload.strip()
    if not spec:
        return None
    arrow = "→" if "→" in spec else "->" if "->" in spec else None
    if arrow is None:
        return None
    src, tgt = [part.strip() for part in re.split(r"→|->", spec, maxsplit=1)]
    if not src or not tgt:
        return None
    return ConvertCodeSpec(source_lang=src, target_lang=tgt, arrow=arrow)

