"""Autonomous execution loop (plan -> act -> observe -> retry)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from rich.console import Console

from euler_agent.agent import EulerAgent
from euler_agent.audit import append_audit_event, create_audit_run
from euler_agent.guardrails import (
    build_policy,
    ensure_inside_workdir,
    is_command_allowed,
    is_risky_command,
)
from euler_agent.safety import guarded_replace_range, guarded_write
from euler_agent.tools import (
    append_file,
    read_file,
    replace_in_files,
    run_terminal_command,
)

console = Console()


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
        touched_files: set[Path] = set()
        planner_prompt = (
            "Return valid JSON only in this schema:\n"
            '{'
            '"summary":"short summary",'
            '"actions":[{"type":"read_file|write_file|append_file|replace_range|replace_in_files|run_command|done",'
            '"path":"optional","content":"optional","start_line":0,"end_line":0,'
            '"search":"optional","replacement":"optional","paths":[],"command":"optional","reason":"optional"}]'
            "}\n\n"
            "Rules:\n"
            "- Use relative paths under the provided workdir.\n"
            "- Use run_command for tests/build.\n"
            "- If objective is complete, return one action with type=done.\n\n"
            f"- Max actions per round: {policy.max_actions_per_round}\n"
            f"- Remaining file mutations: {policy.max_file_mutations - mutation_count}\n"
            f"Goal:\n{goal}\n\n"
            f"Workdir:\n{wd}\n\n"
            f"Recent observation:\n{observation}\n\n"
            f"Action history:\n{chr(10).join(history[-8:]) or 'None'}"
        )
        plan_text = agent.ask(planner_prompt, role="autonomous-builder")
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

        actions = parsed.get("actions", [])
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
            action_type = action.get("type", "")
            reason = action.get("reason", "")
            if reason:
                step_outputs.append(f"[reason] {reason}")
            if action_type == "done":
                final = parsed.get("summary", "Objective completed.")
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
                target = str(Path(wd) / action["path"])
                if not ensure_inside_workdir(wd, target):
                    step_outputs.append("[guardrail] read_file blocked (outside workdir)")
                    continue
                output = read_file(target)
                step_outputs.append(_format_action_result("read_file", output[:2500]))
            elif action_type == "write_file":
                target_path = Path(wd) / action["path"]
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
                output = guarded_write(target, action.get("content", ""))
                step_outputs.append(_format_action_result("write_file", output))
                if output.startswith("Wrote"):
                    mutation_count += 1
            elif action_type == "append_file":
                target_path = Path(wd) / action["path"]
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
                output = append_file(target, action.get("content", ""))
                step_outputs.append(_format_action_result("append_file", output))
                mutation_count += 1
            elif action_type == "replace_range":
                target_path = Path(wd) / action["path"]
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
                    int(action.get("start_line", 1)),
                    int(action.get("end_line", 1)),
                    action.get("content", ""),
                )
                step_outputs.append(_format_action_result("replace_range", output))
                if output.startswith("Replaced"):
                    mutation_count += 1
            elif action_type == "replace_in_files":
                path_objects = [Path(wd) / p for p in action.get("paths", [])]
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
                    search=action.get("search", ""),
                    replacement=action.get("replacement", ""),
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
                command = action.get("command", "")
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
                output = run_terminal_command(command, cwd=wd)
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
                if exit_code not in (None, 0) and touched_files:
                    restored = _restore_snapshot(round_snapshot, Path(wd), touched_files)
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
                verification = run_terminal_command(verify_command, cwd=wd)
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
                if exit_code not in (None, 0) and touched_files:
                    restored = _restore_snapshot(round_snapshot, Path(wd), touched_files)
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

        observation = "\n".join(step_outputs)
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
