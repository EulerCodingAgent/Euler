"""Shared constants and regex patterns for REPL behavior."""

from __future__ import annotations

import re

FILE_REF_PATTERN = re.compile(r"@([^\s]+)")
URL_PATTERN = re.compile(r"\bhttps?://[^\s<>()\"']+")
RANGED_FILE_REF_PATTERN = re.compile(r"^(?P<path>.+):(?P<start>\d+)-(?P<end>\d+)$")
CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)
JSON_BLOCK_RE = re.compile(r"```json\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
FILE_PATH_COMMENT_RE = re.compile(
    r"^(?:#|//|/\*)\s*(?:file:\s*)?(?P<path>[^\s*]+\.\w+)"
)

CODE_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".go", ".rs", ".java", ".c", ".cpp", ".h", ".hpp",
    ".cs", ".rb", ".php", ".swift", ".kt", ".scala",
    ".r", ".sql", ".sh", ".bash", ".zsh", ".ps1",
    ".yaml", ".yml", ".toml", ".json", ".jsonc",
    ".md", ".txt", ".env", ".cfg", ".ini", ".conf",
    ".html", ".css", ".scss", ".sass", ".less",
    ".xml", ".proto", ".graphql",
})

SKIP_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".hg", ".svn",
    "node_modules", ".venv", "venv", ".env",
    "dist", "build", "out", "target", ".cache",
    ".idea", ".vscode", ".mypy_cache", ".pytest_cache",
    ".tox", "coverage", ".ruff_cache", "site-packages",
})

FOLDER_FILE_LIMIT = 40
WEB_FETCH_TIMEOUT_SEC = 10
WEB_CONTENT_CHAR_LIMIT = 8_000

DELETE_WORDS = frozenset({
    "delete", "remove", "rm", "erase", "wipe", "unlink",
    "get rid", "trash", "clean up", "cleanup",
})

ACTION_VERBS = frozenset({
    "fix", "refactor", "implement", "build", "write", "create",
    "update", "change", "add", "remove", "delete", "rename",
    "convert", "generate", "deploy", "patch", "rewrite", "optimize",
    "migrate", "scaffold", "test", "improve", "correct", "repair",
    "edit", "modify", "clean", "format", "lint", "upgrade", "extend",
    "complete", "finish", "solve", "debug",
})

NON_DELETE_ACTION_VERBS = frozenset(
    verb for verb in ACTION_VERBS if verb not in {"delete", "remove", "rm"}
)

QUESTION_FIRST_WORDS = frozenset({
    "explain", "what", "why", "how", "describe", "summarize",
    "tell", "show", "is", "are", "does", "can", "could",
    "should", "would", "hi", "hello", "hey",
})

