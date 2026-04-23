"""Autonomous execution loop (plan -> act -> observe -> retry)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console

from euler_agent.agent import EulerAgent
from euler_agent.guardrails import AutopilotPolicy, ensure_inside_workdir, is_command_allowed
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


def run_autopilot(
    agent: EulerAgent,
    goal: str,
    workdir: str,
    max_rounds: int = 4,
    verify_command: str | None = None,
    max_file_mutations: int = 25,
) -> str:
    """
    Iteratively ask the model for concrete actions and execute them.
    """
    wd = str(Path(workdir).resolve())
    observation = "No execution yet."
    history: list[str] = []
    mutation_count = 0
    policy = AutopilotPolicy(max_file_mutations=max_file_mutations)

    for i in range(1, max_rounds + 1):
        console.print(f"[cyan]Autopilot round {i}/{max_rounds}[/cyan]")
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
        try:
            parsed = _extract_json(plan_text)
        except json.JSONDecodeError:
            observation = f"Planner returned non-JSON output:\n{plan_text}"
            history.append("[planner_error] non-json response")
            continue

        actions = parsed.get("actions", [])
        if not actions:
            observation = "Planner returned no actions."
            history.append("[planner_error] empty actions")
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
                return "\n".join(history)
            if action_type == "read_file":
                target = str(Path(wd) / action["path"])
                if not ensure_inside_workdir(wd, target):
                    step_outputs.append("[guardrail] read_file blocked (outside workdir)")
                    continue
                output = read_file(target)
                step_outputs.append(_format_action_result("read_file", output[:2500]))
            elif action_type == "write_file":
                target = str(Path(wd) / action["path"])
                if not ensure_inside_workdir(wd, target):
                    step_outputs.append("[guardrail] write_file blocked (outside workdir)")
                    continue
                if mutation_count >= policy.max_file_mutations:
                    step_outputs.append("[guardrail] write_file blocked (mutation limit reached)")
                    continue
                output = guarded_write(target, action.get("content", ""))
                step_outputs.append(_format_action_result("write_file", output))
                if output.startswith("Wrote"):
                    mutation_count += 1
            elif action_type == "append_file":
                target = str(Path(wd) / action["path"])
                if not ensure_inside_workdir(wd, target):
                    step_outputs.append("[guardrail] append_file blocked (outside workdir)")
                    continue
                if mutation_count >= policy.max_file_mutations:
                    step_outputs.append("[guardrail] append_file blocked (mutation limit reached)")
                    continue
                output = append_file(target, action.get("content", ""))
                step_outputs.append(_format_action_result("append_file", output))
                mutation_count += 1
            elif action_type == "replace_range":
                target = str(Path(wd) / action["path"])
                if not ensure_inside_workdir(wd, target):
                    step_outputs.append("[guardrail] replace_range blocked (outside workdir)")
                    continue
                if mutation_count >= policy.max_file_mutations:
                    step_outputs.append("[guardrail] replace_range blocked (mutation limit reached)")
                    continue
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
                paths = [str(Path(wd) / p) for p in action.get("paths", [])]
                sandboxed_paths = [p for p in paths if ensure_inside_workdir(wd, p)]
                if mutation_count >= policy.max_file_mutations:
                    step_outputs.append("[guardrail] replace_in_files blocked (mutation limit reached)")
                    continue
                remaining = policy.max_file_mutations - mutation_count
                sandboxed_paths = sandboxed_paths[:remaining]
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
                allowed, reason_message = is_command_allowed(command, policy)
                if not allowed:
                    step_outputs.append(f"[guardrail] run_command blocked ({reason_message})")
                    continue
                output = run_terminal_command(command, cwd=wd)
                step_outputs.append(_format_action_result("run_command", output[-3500:]))
            else:
                step_outputs.append(_format_action_result("unknown_action", action_type))

        if verify_command:
            allowed, reason_message = is_command_allowed(verify_command, policy)
            if allowed:
                verification = run_terminal_command(verify_command, cwd=wd)
                step_outputs.append(_format_action_result("verify_command", verification[-2500:]))
            else:
                step_outputs.append(f"[guardrail] verify_command blocked ({reason_message})")

        observation = "\n".join(step_outputs)
        history.extend(step_outputs)

    return "\n".join(history + ["[status] Max rounds reached before done action."])
