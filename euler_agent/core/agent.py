"""
Multi-agent orchestration engine.

Graph:  planner
           │
    ┌──────┴──────────────────────────────────────┐
    │         parallel_specialists (8 agents)      │
    │  architect · coder · tester · security       │
    │  devops · db · documenter · refactor         │
    └──────────────────┬──────────────────────────┘
                       │
                  arbitrator
                       │
                   reviewer ──► END
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from euler_agent.config.context import load_euler_instruction_docs
from euler_agent.memory.store import add_memory, search_memory
from euler_agent.core.prompts import (
    SYSTEM_ARCHITECT,
    SYSTEM_ARBITRATOR,
    SYSTEM_CODER,
    SYSTEM_DB,
    SYSTEM_DEVOPS,
    SYSTEM_DOCUMENTER,
    SYSTEM_PLANNER,
    SYSTEM_REFACTOR,
    SYSTEM_REVIEWER,
    SYSTEM_SECURITY,
    SYSTEM_TESTER,
)
from euler_agent.core.providers import get_chat_model
from euler_agent.analysis.semantic_index import search_index


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    user_goal: str
    workdir: str
    instruction_docs: str
    memory_context: str
    semantic_context: str

    # planner
    plan: str

    # parallel specialists
    architect_output: str
    coder_output: str
    tester_output: str
    security_output: str
    devops_output: str
    db_output: str
    documenter_output: str
    refactor_output: str

    # consolidation
    arbitrated_output: str
    final_output: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _invoke(model, system_prompt: str, human_prompt: str) -> str:
    try:
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        return response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:
        msg = str(exc)
        if "400" in msg or "Bad Request" in msg:
            raise RuntimeError(
                "API returned 400 Bad Request.\n\n"
                "Most likely causes:\n"
                "  1) Missing/invalid API key (most common)\n"
                "  2) API key restrictions or network/proxy rewriting requests\n"
                "  3) Invalid model slug\n\n"
                "Quick checks:\n"
                "  - Run: Euler config show\n"
                "  - Re-enter your key: Euler config set --provider gemini --model gemini-2.5-flash\n"
                "  - Ensure the key is from Google AI Studio (usually starts with 'AIza')\n\n"
                "Common stable Gemini model slugs:\n"
                "  gemini-2.5-flash\n"
                "  gemini-2.5-pro\n"
                "  gemini-2.5-flash-lite\n"
                "  gemini-2.0-flash\n\n"
                f"Original error: {msg[:400]}"
            ) from exc
        raise


def _build_base_context(state: AgentState) -> str:
    return (
        f"## User Goal\n{state['user_goal']}\n\n"
        f"## Strategic Plan\n{state.get('plan', 'Not yet available.')}\n\n"
        f"## Project Memory (similar past goals)\n"
        f"{state.get('memory_context', 'None')}\n\n"
        f"## Semantically Relevant Code Hits\n"
        f"{state.get('semantic_context', 'None')}\n\n"
        f"## Project Instruction Docs (./Euler/*.md)\n"
        f"{state.get('instruction_docs', 'None')}"
    )


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class EulerAgent:
    def __init__(self, provider: str, model_name: str, api_key: str) -> None:
        self.model = get_chat_model(provider=provider, model=model_name, api_key=api_key)

    def ask(self, prompt: str, role: str = "assistant") -> str:
        from euler_agent.core.prompts import PRODUCTION_PREAMBLE
        system = f"{PRODUCTION_PREAMBLE}\n\nYou are Euler acting as {role}."
        return _invoke(self.model, system, prompt)

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _gather_context(self, state: AgentState) -> AgentState:
        workdir = state["workdir"]
        goal = state["user_goal"]

        memory_snippets = search_memory(project=workdir, query=goal, limit=4)
        memory_context = "\n\n".join(
            f"Past goal: {m.goal}\nOutcome: {m.result[:600]}"
            for m in memory_snippets
        ) or "None"

        semantic_hits = search_index(workdir=workdir, query=goal, limit=5)
        semantic_context = "\n\n".join(
            f"File: {h['path']} lines {h['start_line']}-{h['end_line']}\n"
            f"{h.get('content', '')[:600]}"
            for h in semantic_hits
        ) or "None"

        instruction_docs = load_euler_instruction_docs(Path(workdir)) or "None"

        return {
            "memory_context": memory_context,
            "semantic_context": semantic_context,
            "instruction_docs": instruction_docs,
        }

    def _planner(self, state: AgentState) -> AgentState:
        prompt = (
            f"## User Request\n{state['user_goal']}\n\n"
            f"## Project Memory (related past outcomes)\n"
            f"{state.get('memory_context', 'None')}\n\n"
            f"## Relevant Existing Code\n"
            f"{state.get('semantic_context', 'None')}\n\n"
            f"## Project Instructions\n"
            f"{state.get('instruction_docs', 'None')}\n\n"
            "Produce the full strategic plan in the format described in your role."
        )
        plan = _invoke(self.model, SYSTEM_PLANNER, prompt)
        return {"plan": plan}

    def _parallel_specialists(self, state: AgentState) -> AgentState:
        ctx = _build_base_context(state)

        specialists: dict[str, tuple[str, str]] = {
            "architect":  (SYSTEM_ARCHITECT,  f"Produce the full architecture blueprint.\n\n{ctx}"),
            "coder":      (SYSTEM_CODER,       f"Produce the complete implementation.\n\n{ctx}"),
            "tester":     (SYSTEM_TESTER,      f"Produce the complete test suite.\n\n{ctx}"),
            "security":   (SYSTEM_SECURITY,    f"Perform a full security review of the plan and implementation.\n\n{ctx}"),
            "devops":     (SYSTEM_DEVOPS,      f"Produce the full deployment and infra config.\n\n{ctx}"),
            "db":         (SYSTEM_DB,          f"Design and implement the data layer.\n\n{ctx}"),
            "documenter": (SYSTEM_DOCUMENTER,  f"Produce all documentation.\n\n{ctx}"),
            "refactor":   (SYSTEM_REFACTOR,    f"Identify and apply refactoring to any existing relevant code.\n\n{ctx}"),
        }

        results: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=8) as pool:
            future_to_key = {
                pool.submit(_invoke, self.model, sys_prompt, human_prompt): key
                for key, (sys_prompt, human_prompt) in specialists.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as exc:
                    results[key] = f"[specialist error: {exc}]"

        return {
            "architect_output":  results.get("architect", ""),
            "coder_output":      results.get("coder", ""),
            "tester_output":     results.get("tester", ""),
            "security_output":   results.get("security", ""),
            "devops_output":     results.get("devops", ""),
            "db_output":         results.get("db", ""),
            "documenter_output": results.get("documenter", ""),
            "refactor_output":   results.get("refactor", ""),
        }

    def _arbitrator(self, state: AgentState) -> AgentState:
        prompt = (
            f"## User Goal\n{state['user_goal']}\n\n"
            f"## Strategic Plan\n{state.get('plan', '')}\n\n"
            f"---\n### Architect Output\n{state.get('architect_output', '')}\n\n"
            f"---\n### Coder Output\n{state.get('coder_output', '')}\n\n"
            f"---\n### Tester Output\n{state.get('tester_output', '')}\n\n"
            f"---\n### Security Output\n{state.get('security_output', '')}\n\n"
            f"---\n### DevOps Output\n{state.get('devops_output', '')}\n\n"
            f"---\n### Database Output\n{state.get('db_output', '')}\n\n"
            f"---\n### Documenter Output\n{state.get('documenter_output', '')}\n\n"
            f"---\n### Refactor Output\n{state.get('refactor_output', '')}\n\n"
            "Arbitrate all specialist outputs into a single unified strategy."
        )
        merged = _invoke(self.model, SYSTEM_ARBITRATOR, prompt)
        return {"arbitrated_output": merged}

    def _reviewer(self, state: AgentState) -> AgentState:
        prompt = (
            f"## User Goal\n{state['user_goal']}\n\n"
            f"## Strategic Plan\n{state.get('plan', '')}\n\n"
            f"## Arbitrated Strategy\n{state.get('arbitrated_output', '')}\n\n"
            "Perform the final production review and deliver the complete, "
            "corrected, and deployment-ready answer."
        )
        final = _invoke(self.model, SYSTEM_REVIEWER, prompt)
        return {"final_output": final}

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("gather_context",      self._gather_context)
        graph.add_node("planner",             self._planner)
        graph.add_node("parallel_specialists", self._parallel_specialists)
        graph.add_node("arbitrator",          self._arbitrator)
        graph.add_node("reviewer",            self._reviewer)

        graph.add_edge(START,                "gather_context")
        graph.add_edge("gather_context",     "planner")
        graph.add_edge("planner",            "parallel_specialists")
        graph.add_edge("parallel_specialists", "arbitrator")
        graph.add_edge("arbitrator",         "reviewer")
        graph.add_edge("reviewer",           END)
        return graph.compile()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, user_goal: str, workdir: str | None = None) -> str:
        compiled = self._build_graph()
        resolved_workdir = str(Path(workdir or ".").resolve())
        initial: AgentState = {
            "user_goal": user_goal,
            "workdir": resolved_workdir,
        }
        try:
            result = compiled.invoke(initial)
        except RuntimeError:
            raise
        except Exception as exc:
            msg = str(exc)
            if "400" in msg or "Bad Request" in msg:
                raise RuntimeError(
                    "API returned 400 Bad Request.\n\n"
                    "Most likely causes:\n"
                    "  1) Missing/invalid API key (most common)\n"
                    "  2) API key restrictions or network/proxy rewriting requests\n"
                    "  3) Invalid model slug\n\n"
                    "Quick checks:\n"
                    "  - Run: Euler config show\n"
                    "  - Re-enter your key: Euler config set --provider gemini --model gemini-2.5-flash\n"
                    "  - Ensure the key is from Google AI Studio (usually starts with 'AIza')\n\n"
                    "Common stable Gemini model slugs:\n"
                    "  gemini-2.5-flash\n"
                    "  gemini-2.5-pro\n"
                    "  gemini-2.5-flash-lite\n"
                    "  gemini-2.0-flash\n\n"
                    f"Original error: {msg[:400]}"
                ) from exc
            raise
        final = result.get("final_output", "No output generated.")
        add_memory(
            project=resolved_workdir,
            goal=user_goal,
            result=final,
            tags=["run", "graph", "production"],
        )
        return final

    def generate_sql(self, requirement: str) -> str:
        from euler_agent.core.prompts import SYSTEM_DB
        prompt = (
            f"Generate production-ready SQL for this requirement.\n\n"
            f"Requirement:\n{requirement}\n\n"
            "Include: DDL if tables are implied, the query itself, "
            "parameterised form, indexes that improve performance, "
            "and a brief explanation."
        )
        return _invoke(self.model, SYSTEM_DB, prompt)

    def rewrite_selection(
        self,
        file_path: str,
        selected_text: str,
        instruction: str,
    ) -> str:
        from euler_agent.core.prompts import PRODUCTION_PREAMBLE
        system = (
            f"{PRODUCTION_PREAMBLE}\n\n"
            "You rewrite a selected code region according to an instruction. "
            "Return ONLY the replacement code — no markdown, no explanations. "
            "The replacement must be drop-in ready."
        )
        prompt = (
            f"File: {file_path}\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Selected code to replace:\n```\n{selected_text}\n```"
        )
        return _invoke(self.model, system, prompt)

    def convert_language(
        self,
        source_code: str,
        source_lang: str,
        target_lang: str,
    ) -> str:
        from euler_agent.tools.converter import convert_code, analyse_migration
        analysis = analyse_migration(self.model, source_code, source_lang, target_lang)
        return convert_code(self.model, source_code, source_lang, target_lang, analysis)

    def convert_file(self, file_path: str, target_lang: str) -> str:
        from euler_agent.tools.converter import convert_file
        return convert_file(self.model, file_path, target_lang)
