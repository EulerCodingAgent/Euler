"""
Cross-language conversion and migration engine.
"""

from __future__ import annotations

from euler_agent.prompts import SYSTEM_LANG_ANALYSER, SYSTEM_LANG_CONVERTER

SUPPORTED_LANGUAGES = {
    "python", "javascript", "typescript", "go", "rust", "java",
    "kotlin", "swift", "csharp", "cpp", "sql", "bash", "ruby",
}


def _normalise_lang(lang: str) -> str:
    mapping = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "rb": "ruby",
        "rs": "rust",
        "cs": "csharp",
        "c++": "cpp",
        "sh": "bash",
        "shell": "bash",
    }
    lower = lang.strip().lower()
    return mapping.get(lower, lower)


def analyse_migration(model, source_code: str, source_lang: str, target_lang: str) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n\n"
        f"Source code:\n```{source_lang}\n{source_code}\n```\n\n"
        "Produce the complete migration analysis."
    )
    response = model.invoke([
        SystemMessage(content=SYSTEM_LANG_ANALYSER),
        HumanMessage(content=prompt),
    ])
    return response.content if isinstance(response.content, str) else str(response.content)


def convert_code(
    model,
    source_code: str,
    source_lang: str,
    target_lang: str,
    analysis: str = "",
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    analysis_block = f"Migration analysis:\n{analysis}\n\n" if analysis else ""
    prompt = (
        f"Source language: {source_lang}\n"
        f"Target language: {target_lang}\n\n"
        f"{analysis_block}"
        f"Source code:\n```{source_lang}\n{source_code}\n```\n\n"
        "Produce complete, idiomatic, production-ready code in the target language."
    )
    response = model.invoke([
        SystemMessage(content=SYSTEM_LANG_CONVERTER),
        HumanMessage(content=prompt),
    ])
    return response.content if isinstance(response.content, str) else str(response.content)


def convert_file(
    model,
    file_path: str,
    target_lang: str,
) -> str:
    """
    Read a file, detect its source language from extension, and convert.
    Returns the converted code string.
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        return f"Error: file not found: {file_path}"

    ext_to_lang: dict[str, str] = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".kt": "kotlin",
        ".swift": "swift",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".rb": "ruby",
        ".sh": "bash",
        ".sql": "sql",
    }
    source_lang = ext_to_lang.get(path.suffix.lower(), "unknown")
    if source_lang == "unknown":
        return f"Error: cannot detect source language for extension {path.suffix}"

    target_lang_norm = _normalise_lang(target_lang)
    source_code = path.read_text(encoding="utf-8", errors="ignore")
    analysis = analyse_migration(model, source_code, source_lang, target_lang_norm)
    converted = convert_code(model, source_code, source_lang, target_lang_norm, analysis)
    return f"=== Migration Analysis ===\n{analysis}\n\n=== Converted Code ===\n{converted}"
