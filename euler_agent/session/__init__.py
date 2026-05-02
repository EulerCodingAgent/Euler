"""REPL session checkpoints and resume support."""

from euler_agent.session.checkpoint import (
    SessionHandle,
    SessionMeta,
    SessionSummary,
    build_resume_context_prefix,
    create_session,
    default_new_session_name,
    find_session_by_name_or_id,
    format_turns_recap,
    git_diff_from_head,
    git_diff_stats,
    git_head,
    load_session_handle,
    load_turns,
    list_sessions,
    sessions_root,
)

__all__ = [
    "SessionHandle",
    "SessionMeta",
    "SessionSummary",
    "build_resume_context_prefix",
    "create_session",
    "default_new_session_name",
    "find_session_by_name_or_id",
    "format_turns_recap",
    "git_diff_from_head",
    "git_diff_stats",
    "git_head",
    "load_session_handle",
    "load_turns",
    "list_sessions",
    "sessions_root",
]
