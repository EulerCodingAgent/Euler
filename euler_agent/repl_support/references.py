"""Reference expansion helpers for @file and URL context attachments."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from rich.markup import escape

from euler_agent.repl_support.constants import (
    CODE_EXTENSIONS,
    FILE_REF_PATTERN,
    FOLDER_FILE_LIMIT,
    SKIP_DIRS,
    URL_PATTERN,
    WEB_CONTENT_CHAR_LIMIT,
    WEB_FETCH_TIMEOUT_SEC,
)
from euler_agent.repl_support.parsing import parse_file_ref
from euler_agent.tools.ops import read_file


def collect_folder_files(folder: Path) -> list[Path]:
    files: list[Path] = []
    for candidate in sorted(folder.rglob("*")):
        if not candidate.is_file():
            continue
        if candidate.suffix.lower() not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in candidate.parts):
            continue
        files.append(candidate)
        if len(files) >= FOLDER_FILE_LIMIT:
            break
    return files


def attach_folder(folder: Path, ref: str, notes: list[str], ref_paths: set[Path]) -> str:
    files = collect_folder_files(folder)
    if not files:
        notes.append(f"[yellow]No code files found in @{escape(ref)}[/yellow]")
        return f"@{ref}"
    truncated = len(files) >= FOLDER_FILE_LIMIT
    blocks: list[str] = []
    attached: list[str] = []
    for file_path in files:
        try:
            content = read_file(str(file_path))
        except Exception as exc:
            notes.append(f"[yellow]Skipped {escape(file_path.name)}: {escape(str(exc))}[/yellow]")
            continue
        try:
            rel = file_path.relative_to(folder.parent)
        except ValueError:
            rel = file_path
        blocks.append(f"### {rel}\n[Attached file: {file_path}]\n```text\n{content}\n```")
        ref_paths.add(file_path)
        attached.append(file_path.name)
    suffix = f" (first {FOLDER_FILE_LIMIT})" if truncated else ""
    notes.append(f"[cyan]Attached @{escape(ref)}/ ({len(attached)} files{suffix})[/cyan]")
    header = f"@{ref}/\n[Attached folder: {folder} — {len(attached)} files{suffix}]"
    return header + "\n\n" + "\n\n".join(blocks)


def expand_file_references(user_input: str) -> tuple[str, list[str], set[Path]]:
    notes: list[str] = []
    ref_paths: set[Path] = set()

    def _replace_match(match: re.Match[str]) -> str:
        spec = parse_file_ref(match.group(1))
        raw_ref = spec.raw_ref
        ref = spec.path
        line_start = spec.start_line
        line_end = spec.end_line
        candidate = Path(ref)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / ref).resolve()
        if not candidate.exists():
            notes.append(f"[yellow]Could not resolve @{escape(ref)}[/yellow]")
            return match.group(0)
        if candidate.is_dir():
            if line_start is not None:
                notes.append(f"[yellow]Line ranges are not supported on folders (@{escape(raw_ref)})[/yellow]")
            return attach_folder(candidate, ref, notes, ref_paths)
        if not candidate.is_file():
            notes.append(f"[yellow]Could not resolve @{escape(ref)}[/yellow]")
            return match.group(0)
        try:
            content = read_file(str(candidate))
        except Exception as exc:
            notes.append(f"[red]Failed to read @{escape(ref)}: {escape(str(exc))}[/red]")
            return match.group(0)
        line_header = ""
        if line_start is not None and line_end is not None:
            if line_start <= 0 or line_end <= 0 or line_start > line_end:
                notes.append(f"[yellow]Invalid line range in @{escape(raw_ref)}; expected start ≤ end[/yellow]")
                return match.group(0)
            file_lines = content.splitlines()
            if line_start > len(file_lines):
                notes.append(
                    f"[yellow]Line {line_start} out of bounds in @{escape(raw_ref)}; file has {len(file_lines)} lines[/yellow]"
                )
                return match.group(0)
            clipped_end = min(line_end, len(file_lines))
            content = "\n".join(file_lines[line_start - 1 : clipped_end])
            line_header = f"[Attached line range: {line_start}-{clipped_end}]\n"
            notes.append(f"[cyan]Attached @{escape(raw_ref)}[/cyan]")
        else:
            notes.append(f"[cyan]Attached @{escape(ref)}[/cyan]")
        ref_paths.add(candidate)
        return f"@{raw_ref}\n[Attached file: {candidate}]\n{line_header}```text\n{content}\n```"

    resolved = FILE_REF_PATTERN.sub(_replace_match, user_input)
    return resolved, notes, ref_paths


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag in {"script", "style", "noscript"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self._skip_depth > 0:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def fetch_url_text(url: str) -> str:
    request = Request(
        url,
        headers={
            "User-Agent": "Euler-Agent/1.0 (+web-context)",
            "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.8",
        },
    )
    with urlopen(request, timeout=WEB_FETCH_TIMEOUT_SEC) as resp:
        content_type = (resp.headers.get("Content-Type", "") or "").lower()
        charset = resp.headers.get_content_charset() or "utf-8"
        raw = resp.read(200_000)
    body = raw.decode(charset, errors="replace")
    if "html" in content_type or "<html" in body.lower():
        parser = _HTMLTextExtractor()
        parser.feed(body)
        parser.close()
        text = parser.text()
    else:
        text = re.sub(r"\s+", " ", body).strip()
    if len(text) > WEB_CONTENT_CHAR_LIMIT:
        return text[:WEB_CONTENT_CHAR_LIMIT] + " ... [truncated]"
    return text


def expand_web_references(user_input: str) -> tuple[str, list[str]]:
    urls: list[str] = []
    seen: set[str] = set()
    for match in URL_PATTERN.finditer(user_input):
        raw = match.group(0).rstrip(".,;:!?)]}")
        if raw in seen:
            continue
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        seen.add(raw)
        urls.append(raw)
    if not urls:
        return user_input, []
    notes: list[str] = []
    blocks: list[str] = []
    for url in urls:
        try:
            text = fetch_url_text(url)
            if not text:
                notes.append(f"[yellow]Fetched URL but no readable content: {escape(url)}[/yellow]")
                continue
            notes.append(f"[cyan]Attached web context: {escape(url)}[/cyan]")
            blocks.append(f"### Web Source\n[Attached URL: {url}]\n```text\n{text}\n```")
        except HTTPError as exc:
            notes.append(f"[yellow]Failed to fetch {escape(url)} (HTTP {exc.code})[/yellow]")
        except URLError as exc:
            notes.append(f"[yellow]Failed to fetch {escape(url)} ({escape(str(exc.reason))})[/yellow]")
        except Exception as exc:
            notes.append(f"[yellow]Failed to fetch {escape(url)} ({escape(str(exc))})[/yellow]")
    if not blocks:
        return user_input, notes
    return user_input + "\n\n" + "\n\n".join(blocks), notes

