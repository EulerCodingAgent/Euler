"""Autonomous execution loop (plan -> act -> observe -> retry)."""

from __future__ import annotations

import json
import shutil
import subprocess
from difflib import unified_diff
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from rich.console import Console

from euler_agent.core.agent import EulerAgent
from euler_agent.memory.audit import append_audit_event, create_audit_run
from euler_agent.core.prompts import PRODUCTION_PREAMBLE  # noqa: F401 — imported for side-effect awareness
from euler_agent.guards.guardrails import (
    build_policy,
    ensure_inside_workdir,
    is_command_allowed,
    is_risky_command,
)
from euler_agent.guards.safety import guarded_replace_range, guarded_write
from euler_agent.tools.ops import (
    append_file,
    read_file,
    replace_in_files,
    run_terminal_command,
)

console = Console()


class ActionType(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    APPEND_FILE = "append_file"
    REPLACE_RANGE = "replace_range"
    REPLACE_IN_FILES = "replace_in_files"
    RUN_COMMAND = "run_command"
    DONE = "done"


class ActionPayload(BaseModel):
    type: ActionType
    path: str | None = None
    content: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    search: str | None = None
    replacement: str | None = None
    paths: list[str] | None = None
    command: str | None = None
    reason: str | None = None


class PlannerPayload(BaseModel):
    summary: str = ""
    actions: list[ActionPayload]


def _extract_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0]
    return json.loads(raw)


def _format_action_result(name: str, output: str) -> str:
    return f"[{name}] {output.strip()}"


def _extract_exit_code(command_output: str) -> int | None:
    for line in command_output.splitlines():
        if line.startswith("exit_code="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _is_risky_file_action(action_type: str, target: str) -> bool:
    lowered = target.replace("\\", "/").lower()
    risky_paths = (".github/workflows", "pyproject.toml", "package.json", "dockerfile")
    if action_type in {"write_file", "append_file", "replace_in_files"}:
        return any(marker in lowered for marker in risky_paths)
    return False


def _compact_text(text: str, max_chars: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars] + "... [truncated]"


def _observation_delta(previous: str, current: str, max_chars: int = 1500) -> str:
    if not previous.strip():
        return _compact_text(current, max_chars)
    diff = "".join(
        unified_diff(
            previous.splitlines(keepends=True),
            current.splitlines(keepends=True),
            fromfile="previous",
            tofile="current",
            n=1,
        )
    )
    if not diff.strip():
        return "No meaningful observation change."
    return _compact_text(diff, max_chars)


def _compact_history(history: list[str], max_items: int = 8, item_chars: int = 220) -> str:
    if not history:
        return "None"
    return "\n".join(_compact_text(item, item_chars) for item in history[-max_items:])


def _snapshot_file(snapshot_root: Path, workdir: Path, target_file: Path) -> None:
    rel = target_file.relative_to(workdir)
    destination = snapshot_root / rel
    destination.parent.mkdir(parents=True, exist_ok=True)
    if target_file.exists():
        shutil.copy2(target_file, destination)


def _restore_snapshot(snapshot_root: Path, workdir: Path, touched: set[Path]) -> list[str]:
    restored: list[str] = []
    for target in touched:
        rel = target.relative_to(workdir)
        snap_file = snapshot_root / rel
        if snap_file.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(snap_file, target)
            restored.append(str(target))
        elif target.exists():
            target.unlink()
            restored.append(str(target))
    return restored


def _is_inside_workdir(workdir: Path, rel_path: str) -> bool:
    candidate = (workdir / rel_path).resolve()
    return candidate == workdir or workdir in candidate.parents


def _validate_action_contract(action: ActionPayload, workdir: Path) -> tuple[bool, str]:
    if action.type == ActionType.DONE:
        return True, ""
    if action.type == ActionType.READ_FILE:
        if not action.path:
            return False, "read_file requires path"
        if Path(action.path).is_absolute() or not _is_inside_workdir(workdir, action.path):
            return False, "read_file path must be relative and inside workdir"
        return True, ""
    if action.type in {ActionType.WRITE_FILE, ActionType.APPEND_FILE}:
        if not action.path:
            return False, f"{action.type.value} requires path"
        if action.content is None:
            return False, f"{action.type.value} requires content"
        if Path(action.path).is_absolute() or not _is_inside_workdir(workdir, action.path):
            return False, f"{action.type.value} path must be relative and inside workdir"
        return True, ""
    if action.type == ActionType.REPLACE_RANGE:
        if not action.path:
            return False, "replace_range requires path"
        if action.content is None:
            return False, "replace_range requires content"
        if action.start_line is None or action.end_line is None:
            return False, "replace_range requires start_line and end_line"
        if action.start_line <= 0 or action.end_line <= 0 or action.start_line > action.end_line:
            return False, "replace_range line numbers are invalid"
        if Path(action.path).is_absolute() or not _is_inside_workdir(workdir, action.path):
            return False, "replace_range path must be relative and inside workdir"
        return True, ""
    if action.type == ActionType.REPLACE_IN_FILES:
        if not action.paths:
            return False, "replace_in_files requires paths"
        if action.search is None or action.replacement is None:
            return False, "replace_in_files requires search and replacement"
        for raw_path in action.paths:
            if Path(raw_path).is_absolute() or not _is_inside_workdir(workdir, raw_path):
                return False, "replace_in_files paths must be relative and inside workdir"
        return True, ""
    if action.type == ActionType.RUN_COMMAND:
        if not (action.command or "").strip():
            return False, "run_command requires command"
        return True, ""
    return False, f"unsupported action type: {action.type}"


def _git_status_paths(workdir: Path) -> set[str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(workdir),
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return set()
    paths: set[str] = set()
    for line in output.splitlines():
        if len(line) < 4:
            continue
        raw = line[3:].strip()
        if " -> " in raw:
            _, raw = raw.split(" -> ", 1)
        if raw:
            paths.add(raw.replace("\\", "/"))
    return paths


def _read_file_bytes(path: Path) -> bytes | None:
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except Exception:
        return None


def _snapshot_rel_paths(workdir: Path, rel_paths: set[str]) -> dict[str, bytes | None]:
    snap: dict[str, bytes | None] = {}
    for rel in rel_paths:
        path = (workdir / rel).resolve()
        if path == workdir or workdir not in path.parents:
            continue
        snap[rel] = _read_file_bytes(path)
    return snap


def _restore_bytes(path: Path, payload: bytes | None) -> None:
    if payload is None:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _restore_command_side_effects(
    workdir: Path,
    baseline_dirty_paths: set[str],
    baseline_snapshots: dict[str, bytes | None],
) -> list[str]:
    """
    Restore files changed by command/verification side effects in this round.

    Strategy:
      - revert pre-existing dirty paths to their round-start bytes snapshot
      - remove newly introduced dirty files
    """
    restored: list[str] = []
    now_dirty = _git_status_paths(workdir)
    introduced = now_dirty - baseline_dirty_paths

    # Restore any pre-existing dirty files that changed during this round.
    for rel, before in baseline_snapshots.items():
        current = _read_file_bytes((workdir / rel).resolve())
        if current != before:
            _restore_bytes((workdir / rel).resolve(), before)
            restored.append(str((workdir / rel).resolve()))

    # Best-effort cleanup for newly dirty files from command side effects.
    for rel in introduced:
        abs_path = (workdir / rel).resolve()
        if abs_path == workdir or workdir not in abs_path.parents:
            continue
        if abs_path.exists():
            try:
                abs_path.unlink()
                restored.append(str(abs_path))
            except Exception:
                # If we cannot unlink (tracked or permission issue), leave as-is.
                pass
    return restored


def run_autopilot(
    agent: EulerAgent,
    goal: str,
    workdir: str,
    max_rounds: int = 4,
    verify_command: str | None = None,
    max_file_mutations: int = 25,
    policy_profile: str = "normal",
    require_approval_for_risky: bool = True,
    auto_approve_risky: bool = False,
) -> str:
    """
    Iteratively ask the model for concrete actions and execute them.
    """
    wd = str(Path(workdir).resolve())
    observation = "No execution yet."
    previous_observation = ""
    history: list[str] = []
    mutation_count = 0
    policy = build_policy(
        profile=policy_profile,
        max_file_mutations=max_file_mutations,
        require_approval_for_risky=require_approval_for_risky,
    )
    snapshots_root = Path(wd) / ".euler" / "snapshots"
    snapshots_root.mkdir(parents=True, exist_ok=True)
    run_id, audit_file = create_audit_run(
        wd,
        goal,
        metadata={
            "policy_profile": policy.profile,
            "max_rounds": max_rounds,
            "max_file_mutations": policy.max_file_mutations,
            "require_approval_for_risky": policy.require_approval_for_risky,
            "auto_approve_risky": auto_approve_risky,
        },
    )
    append_audit_event(audit_file, {"type": "policy_resolved", "policy": policy.__dict__})

    for i in range(1, max_rounds + 1):
        console.print(f"[cyan]Autopilot round {i}/{max_rounds}[/cyan]")
        round_snapshot = snapshots_root / f"round_{i}"
        round_snapshot.mkdir(parents=True, exist_ok=True)
        round_workdir = Path(wd)
        baseline_dirty_paths = _git_status_paths(round_workdir)
        baseline_dirty_snapshot = _snapshot_rel_paths(round_workdir, baseline_dirty_paths)
        touched_files: set[Path] = set()
        planner_prompt = (
            "You are Euler Autopilot — an autonomous production-grade coding agent.\n\n"
            "## Code Quality Non-Negotiables\n"
            "Every file you write or modify MUST be:\n"
            "- Complete and immediately runnable — no stubs, no ellipsis, no TODOs.\n"
            "- Production-ready with error handling, logging, and type hints.\n"
            "- Idiomatic for the target language.\n"
            "- Free of hard-coded secrets; use environment variables via a config layer.\n"
            "- Using parameterised queries for any SQL.\n"
            "- Consistent in naming and style with the existing codebase.\n\n"
            "## Action Schema\n"
            "Return valid JSON ONLY (no markdown, no prose) in this exact schema:\n"
            '{"summary":"<one line>","actions":['
            '{"type":"read_file|write_file|append_file|replace_range|replace_in_files|run_command|done",'
            '"path":"<relative>","content":"<full file content when writing>",'
            '"start_line":0,"end_line":0,'
            '"search":"<exact text>","replacement":"<replacement text>",'
            '"paths":[],"command":"<shell command>","reason":"<why>"}'
            "]}\n\n"
            "## Strict Rules\n"
            "- All paths MUST be relative to workdir.\n"
            "- write_file content must be the COMPLETE file — never partial.\n"
            "- Use run_command only for: install deps, run tests, build.\n"
            "- If the goal is fully achieved, emit a single done action.\n"
            "- Prefer targeted replace_range over full rewrites when only a section changes.\n"
            "- Never emit more than the allowed actions per round.\n\n"
            f"- Max actions this round: {policy.max_actions_per_round}\n"
            f"- Remaining file mutation budget: {policy.max_file_mutations - mutation_count}\n\n"
            f"## Goal\n{goal}\n\n"
            f"## Workdir\n{wd}\n\n"
            f"## Observation delta from previous round\n"
            f"{_observation_delta(previous_observation, observation)}\n\n"
            f"## Compact action history (last 8)\n{_compact_history(history)}"
        )
        plan_text = agent.ask(planner_prompt, role="autonomous-production-builder")
        append_audit_event(
            audit_file,
            {
                "type": "planner_response",
                "run_id": run_id,
                "round": i,
                "preview": plan_text[:500],
            },
        )
        try:
            parsed = _extract_json(plan_text)
        except json.JSONDecodeError:
            observation = f"Planner returned non-JSON output:\n{plan_text}"
            history.append("[planner_error] non-json response")
            append_audit_event(
                audit_file,
                {
                    "type": "planner_error",
                    "run_id": run_id,
                    "round": i,
                    "error": "non-json response",
                },
            )
            continue

        try:
            planner_payload = PlannerPayload.model_validate(parsed)
        except ValidationError as exc:
            observation = f"Planner returned invalid action schema: {exc}"
            history.append("[planner_error] invalid action schema")
            append_audit_event(
                audit_file,
                {
                    "type": "planner_error",
                    "run_id": run_id,
                    "round": i,
                    "error": "invalid action schema",
                },
            )
            continue

        actions = planner_payload.actions
        if not actions:
            observation = "Planner returned no actions."
            history.append("[planner_error] empty actions")
            append_audit_event(
                audit_file,
                {
                    "type": "planner_error",
                    "run_id": run_id,
                    "round": i,
                    "error": "empty actions",
                },
            )
            continue
        if len(actions) > policy.max_actions_per_round:
            actions = actions[: policy.max_actions_per_round]
            history.append("[policy] Truncated actions to max_actions_per_round")

        step_outputs: list[str] = []
        for action in actions:
            action_type = action.type.value
            reason = action.reason or ""
            if reason:
                step_outputs.append(f"[reason] {reason}")
            valid, contract_error = _validate_action_contract(action, round_workdir)
            if not valid:
                step_outputs.append(f"[contract_rejected] {contract_error}")
                append_audit_event(
                    audit_file,
                    {
                        "type": "contract_rejected",
                        "run_id": run_id,
                        "round": i,
                        "action_type": action_type,
                        "error": contract_error,
                    },
                )
                continue
            if action_type == "done":
                final = planner_payload.summary or "Objective completed."
                history.append(_format_action_result("done", final))
                append_audit_event(
                    audit_file,
                    {"type": "run_completed", "run_id": run_id, "round": i, "summary": final},
                )
                append_audit_event(
                    audit_file,
                    {"type": "run_finished", "run_id": run_id, "status": "completed"},
                )
                return "\n".join(history + [f"[audit] run_id={run_id} file={audit_file}"])
            if action_type == "read_file":
                target = str(Path(wd) / (action.path or ""))
                if not ensure_inside_workdir(wd, target):
                    step_outputs.append("[guardrail] read_file blocked (outside workdir)")
                    continue
                output = read_file(target)
                step_outputs.append(_format_action_result("read_file", output[:2500]))
            elif action_type == "write_file":
                target_path = Path(wd) / (action.path or "")
                target = str(target_path)
                if (
                    _is_risky_file_action(action_type, target)
                    and policy.require_approval_for_risky
                    and not auto_approve_risky
                ):
                    step_outputs.append("[approval_required] risky write blocked (use --auto-approve-risky)")
                    append_audit_event(
                        audit_file,
                        {
                            "type": "approval_blocked",
                            "run_id": run_id,
                            "round": i,
                            "action_type": action_type,
                            "target": target,
                        },
                    )
                    continue
                if not ensure_inside_workdir(wd, target):
                    step_outputs.append("[guardrail] write_file blocked (outside workdir)")
                    continue
                if mutation_count >= policy.max_file_mutations:
                    step_outputs.append("[guardrail] write_file blocked (mutation limit reached)")
                    continue
                if target_path not in touched_files:
                    _snapshot_file(round_snapshot, Path(wd), target_path)
                    touched_files.add(target_path)
                output = guarded_write(target, action.content or "")
                step_outputs.append(_format_action_result("write_file", output))
                if output.startswith("Wrote"):
                    mutation_count += 1
            elif action_type == "append_file":
                target_path = Path(wd) / (action.path or "")
                target = str(target_path)
                if (
                    _is_risky_file_action(action_type, target)
                    and policy.require_approval_for_risky
                    and not auto_approve_risky
                ):
                    step_outputs.append("[approval_required] risky append blocked (use --auto-approve-risky)")
                    append_audit_event(
                        audit_file,
                        {
                            "type": "approval_blocked",
                            "run_id": run_id,
                            "round": i,
                            "action_type": action_type,
                            "target": target,
                        },
                    )
                    continue
                if not ensure_inside_workdir(wd, target):
                    step_outputs.append("[guardrail] append_file blocked (outside workdir)")
                    continue
                if mutation_count >= policy.max_file_mutations:
                    step_outputs.append("[guardrail] append_file blocked (mutation limit reached)")
                    continue
                if target_path not in touched_files:
                    _snapshot_file(round_snapshot, Path(wd), target_path)
                    touched_files.add(target_path)
                output = append_file(target, action.content or "")
                step_outputs.append(_format_action_result("append_file", output))
                mutation_count += 1
            elif action_type == "replace_range":
                target_path = Path(wd) / (action.path or "")
                target = str(target_path)
                if not ensure_inside_workdir(wd, target):
                    step_outputs.append("[guardrail] replace_range blocked (outside workdir)")
                    continue
                if mutation_count >= policy.max_file_mutations:
                    step_outputs.append("[guardrail] replace_range blocked (mutation limit reached)")
                    continue
                if target_path not in touched_files:
                    _snapshot_file(round_snapshot, Path(wd), target_path)
                    touched_files.add(target_path)
                output = guarded_replace_range(
                    target,
                    int(action.start_line or 1),
                    int(action.end_line or 1),
                    action.content or "",
                )
                step_outputs.append(_format_action_result("replace_range", output))
                if output.startswith("Replaced"):
                    mutation_count += 1
            elif action_type == "replace_in_files":
                path_objects = [Path(wd) / p for p in (action.paths or [])]
                paths = [str(p) for p in path_objects]
                if (
                    any(_is_risky_file_action(action_type, p) for p in paths)
                    and policy.require_approval_for_risky
                    and not auto_approve_risky
                ):
                    step_outputs.append(
                        "[approval_required] risky multi-file replace blocked (use --auto-approve-risky)"
                    )
                    append_audit_event(
                        audit_file,
                        {
                            "type": "approval_blocked",
                            "run_id": run_id,
                            "round": i,
                            "action_type": action_type,
                            "targets": paths[:10],
                        },
                    )
                    continue
                sandboxed_paths = [p for p in paths if ensure_inside_workdir(wd, p)]
                if mutation_count >= policy.max_file_mutations:
                    step_outputs.append("[guardrail] replace_in_files blocked (mutation limit reached)")
                    continue
                remaining = policy.max_file_mutations - mutation_count
                sandboxed_paths = sandboxed_paths[:remaining]
                for path in sandboxed_paths:
                    target_path = Path(path)
                    if target_path not in touched_files:
                        _snapshot_file(round_snapshot, Path(wd), target_path)
                        touched_files.add(target_path)
                output = replace_in_files(
                    paths=sandboxed_paths,
                    search=action.search or "",
                    replacement=action.replacement or "",
                )
                step_outputs.append(_format_action_result("replace_in_files", output))
                if output.startswith("Updated "):
                    first_part = output.split(" files", 1)[0]
                    try:
                        changed = int(first_part.replace("Updated ", "").strip())
                        mutation_count += changed
                    except ValueError:
                        mutation_count += 1
            elif action_type == "run_command":
                command = action.command or ""
                if is_risky_command(command) and policy.require_approval_for_risky and not auto_approve_risky:
                    step_outputs.append("[approval_required] risky command blocked (use --auto-approve-risky)")
                    append_audit_event(
                        audit_file,
                        {
                            "type": "approval_blocked",
                            "run_id": run_id,
                            "round": i,
                            "action_type": action_type,
                            "command": command,
                        },
                    )
                    continue
                allowed, reason_message = is_command_allowed(command, policy)
                if not allowed:
                    step_outputs.append(f"[guardrail] run_command blocked ({reason_message})")
                    append_audit_event(
                        audit_file,
                        {
                            "type": "guardrail_blocked",
                            "run_id": run_id,
                            "round": i,
                            "action_type": action_type,
                            "reason": reason_message,
                        },
                    )
                    continue
                try:
                    output = run_terminal_command(command, cwd=wd)
                except Exception as exc:
                    output = (
                        "exit_code=1\n"
                        "stdout:\n\n"
                        f"stderr:\nFailed to execute command safely: {exc}"
                    )
                step_outputs.append(_format_action_result("run_command", output[-3500:]))
                append_audit_event(
                    audit_file,
                    {
                        "type": "action_executed",
                        "run_id": run_id,
                        "round": i,
                        "action_type": action_type,
                        "command": command,
                        "exit_code": _extract_exit_code(output),
                    },
                )
                exit_code = _extract_exit_code(output)
                if exit_code not in (None, 0):
                    restored = _restore_snapshot(round_snapshot, Path(wd), touched_files) if touched_files else []
                    restored_side_effects = _restore_command_side_effects(
                        round_workdir,
                        baseline_dirty_paths,
                        baseline_dirty_snapshot,
                    )
                    restored.extend(restored_side_effects)
                    step_outputs.append(
                        _format_action_result(
                            "rollback",
                            (
                                f"Command failed (exit_code={exit_code}). "
                                f"Restored {len(restored)} files from round snapshot."
                            ),
                        )
                    )
                    append_audit_event(
                        audit_file,
                        {
                            "type": "rollback",
                            "run_id": run_id,
                            "round": i,
                            "reason": f"command failed exit_code={exit_code}",
                            "restored_files": restored,
                        },
                    )
                    break
            else:
                step_outputs.append(_format_action_result("unknown_action", action_type))

        if verify_command:
            allowed, reason_message = is_command_allowed(verify_command, policy)
            if allowed:
                try:
                    verification = run_terminal_command(verify_command, cwd=wd)
                except Exception as exc:
                    verification = (
                        "exit_code=1\n"
                        "stdout:\n\n"
                        f"stderr:\nFailed to execute verification command safely: {exc}"
                    )
                step_outputs.append(_format_action_result("verify_command", verification[-2500:]))
                append_audit_event(
                    audit_file,
                    {
                        "type": "verification_executed",
                        "run_id": run_id,
                        "round": i,
                        "command": verify_command,
                        "exit_code": _extract_exit_code(verification),
                    },
                )
                exit_code = _extract_exit_code(verification)
                if exit_code not in (None, 0):
                    restored = _restore_snapshot(round_snapshot, Path(wd), touched_files) if touched_files else []
                    restored_side_effects = _restore_command_side_effects(
                        round_workdir,
                        baseline_dirty_paths,
                        baseline_dirty_snapshot,
                    )
                    restored.extend(restored_side_effects)
                    step_outputs.append(
                        _format_action_result(
                            "rollback",
                            (
                                f"Verification failed (exit_code={exit_code}). "
                                f"Restored {len(restored)} files from round snapshot."
                            ),
                        )
                    )
                    append_audit_event(
                        audit_file,
                        {
                            "type": "rollback",
                            "run_id": run_id,
                            "round": i,
                            "reason": f"verification failed exit_code={exit_code}",
                            "restored_files": restored,
                        },
                    )
            else:
                step_outputs.append(f"[guardrail] verify_command blocked ({reason_message})")
                append_audit_event(
                    audit_file,
                    {
                        "type": "guardrail_blocked",
                        "run_id": run_id,
                        "round": i,
                        "action_type": "verify_command",
                        "reason": reason_message,
                    },
                )

        previous_observation = observation
        observation = _compact_text("\n".join(step_outputs), 3500)
        history.extend(step_outputs)
        append_audit_event(
            audit_file,
            {
                "type": "round_completed",
                "run_id": run_id,
                "round": i,
                "mutation_count": mutation_count,
                "output_preview": observation[:1000],
            },
        )

    append_audit_event(
        audit_file,
        {"type": "run_finished", "run_id": run_id, "status": "max_rounds_reached"},
    )
    return "\n".join(
        history
        + [
            "[status] Max rounds reached before done action.",
            f"[audit] run_id={run_id} file={audit_file}",
        ]
    )
