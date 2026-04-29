"""
Multi-agent orchestration engine — with token optimisation.

Graph:  classify_query
           │
        gather_context
           │
         planner
           │
    ┌──────┴──────────────────────────────────────┐
    │   parallel_specialists (selective subset)    │
    │  architect · coder · tester · security       │
    │  devops · db · documenter · refactor         │
    └──────────────────┬──────────────────────────┘
                       │
                  arbitrator
                       │
                   reviewer ──► END

Token optimisation applied at each stage:
  - classify_query   : decides complexity tier & which specialists to invoke
  - gather_context   : relevance-gated injection (cosine threshold)
  - parallel_specialists : only the selected subset is called
  - arbitrator       : specialist outputs are capped before aggregation
  - run()            : response cache checked before the graph runs
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from euler_agent.config.context import load_euler_instruction_docs
from euler_agent.memory.store import add_memory, search_memory_scored
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
from euler_agent.analysis.semantic_index import search_index_scored
from euler_agent.analysis.code_graph import load_code_graph, related_files_from_graph
from euler_agent.core.skills import render_skill_protocol
from euler_agent.optimization.token_optimizer import (
    QueryComplexity,
    TokenOptimizer,
    OptimizationResult,
)


# ---------------------------------------------------------------------------
# Module-level optimiser instance (shared across all EulerAgent instances)
# ---------------------------------------------------------------------------

_OPTIMIZER = TokenOptimizer()
_SPECIALIST_ORDER: list[str] = [
    "architect",
    "coder",
    "tester",
    "security",
    "devops",
    "db",
    "documenter",
    "refactor",
]


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    user_goal: str
    workdir: str
    instruction_docs: str
    memory_context: str
    semantic_context: str

    # optimisation metadata (set by _classify_query node)
    complexity: str                   # QueryComplexity.value
    selected_specialists: list[str]   # specialist keys to invoke
    planner_confidence: float
    skip_specialists: bool
    skip_reason: str
    stage_tokens: dict[str, int]

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


def _compress_text_block(text: str, cap: int = 1800) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= cap:
        return compact
    return compact[:cap] + "... [truncated]"


def _workdir_fingerprint(workdir: str) -> str:
    """Cheap hash of the workdir path used in cache keys."""
    return hashlib.md5(workdir.encode()).hexdigest()[:8]


def _repo_state_fingerprint(workdir: str) -> str:
    """
    Hash repository state so cache invalidates when code changes.

    Uses:
      - git HEAD
      - git status --porcelain
    Falls back to workdir fingerprint for non-git folders.
    """
    cwd = str(Path(workdir).resolve())
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        raw = f"{cwd}|{head}|{status}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return _workdir_fingerprint(cwd)


def _ordered_specialists(keys: list[str]) -> list[str]:
    rank = {name: i for i, name in enumerate(_SPECIALIST_ORDER)}
    return sorted(keys, key=lambda k: rank.get(k, 10_000))


def _extract_file_mentions(text: str) -> set[str]:
    # Heuristic: parse common path-like mentions in specialist output.
    candidates = re.findall(r"(?:^|[\s`\"'])((?:[\w.-]+/)+[\w.-]+\.\w+)", text or "")
    return {c.strip() for c in candidates if c.strip()}


def _build_conflict_report(state: AgentState, selected: list[str]) -> str:
    file_to_specialists: dict[str, list[str]] = {}
    for specialist in selected:
        output = state.get(f"{specialist}_output", "") or ""
        for path in _extract_file_mentions(output):
            file_to_specialists.setdefault(path, []).append(specialist)
    conflicts = {f: names for f, names in file_to_specialists.items() if len(names) > 1}
    if not conflicts:
        return "No explicit file-level conflicts detected from specialist outputs."
    lines = ["Detected potential file-level conflicts:"]
    for path in sorted(conflicts):
        ordered = _ordered_specialists(conflicts[path])
        lines.append(f"- {path}: {', '.join(ordered)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class EulerAgent:
    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: str,
        base_url: str = "",
    ) -> None:
        self._provider = provider
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url
        self.model = get_chat_model(provider=provider, model=model_name, api_key=api_key, base_url=base_url)
        self._model_id = f"{provider}/{model_name}"
        self._last_run_stats: dict[str, Any] = {}
        self._stats_seq = 0
        self._max_specialist_workers = 3

    def _new_model_client(self):
        return get_chat_model(
            provider=self._provider,
            model=self._model_name,
            api_key=self._api_key,
            base_url=self._base_url,
        )

    def _record_stage_tokens(
        self,
        state: AgentState,
        stage: str,
        system_prompt: str,
        human_prompt: str,
    ) -> None:
        total = _OPTIMIZER.estimate_prompt_tokens(system_prompt, human_prompt)
        stage_tokens = dict(state.get("stage_tokens", {}))
        stage_tokens[stage] = stage_tokens.get(stage, 0) + total
        state["stage_tokens"] = stage_tokens

    def get_last_run_stats(self) -> dict[str, Any]:
        return dict(self._last_run_stats)

    def ask(self, prompt: str, role: str = "assistant") -> str:
        from euler_agent.core.prompts import PRODUCTION_PREAMBLE
        system = (
            f"{PRODUCTION_PREAMBLE}\n\n"
            f"You are Euler acting as {role}.\n\n"
            f"{render_skill_protocol(role, 'solve the user request with production-ready precision')}"
        )
        self._last_run_stats = {
            "seq": self._stats_seq + 1,
            "kind": "ask",
            "cache_hit": False,
            "specialists_used": 0,
            "specialists_total": 8,
            "stage_tokens": {
                "ask": _OPTIMIZER.estimate_prompt_tokens(system, prompt),
            },
        }
        self._stats_seq += 1
        return _invoke(self.model, system, prompt)

    # ------------------------------------------------------------------
    # Optimisation node
    # ------------------------------------------------------------------

    def _classify_query(self, state: AgentState) -> AgentState:
        """
        Classify the user goal to determine pipeline complexity and the
        minimal specialist set required.  Adds ``complexity`` and
        ``selected_specialists`` to state for downstream nodes.
        """
        result: OptimizationResult = _OPTIMIZER.classify_query(state["user_goal"])
        return {
            "complexity": result.complexity.value,
            "selected_specialists": result.selected_specialists,
        }

    # ------------------------------------------------------------------
    # Graph nodes
    # ------------------------------------------------------------------

    def _gather_context(self, state: AgentState) -> AgentState:
        workdir = state["workdir"]
        goal = state["user_goal"]
        complexity_str = state.get("complexity", QueryComplexity.FULL.value)
        try:
            complexity = QueryComplexity(complexity_str)
        except ValueError:
            complexity = QueryComplexity.FULL

        # --- memory (relevance-gated) ---
        memory_scored = search_memory_scored(project=workdir, query=goal, limit=6)
        filtered_memory = _OPTIMIZER.filter_memory_hits(memory_scored, complexity)
        memory_context = "\n\n".join(
            f"Past goal: {m.goal}\nLesson: {_compress_text_block(getattr(m, 'lesson_card', m.result), 550)}"
            for m in filtered_memory
        ) or "None"

        # --- semantic index (relevance-gated) ---
        semantic_scored = search_index_scored(workdir=workdir, query=goal, limit=30)
        filtered_hits = _OPTIMIZER.filter_semantic_hits(semantic_scored, complexity)
        top_seed_files = [h.get("path", "") for h in filtered_hits if h.get("path")]

        # Optional graph-aware expansion for tighter, high-precision context.
        graph_payload = load_code_graph(workdir)
        extra_related_files: list[str] = []
        if graph_payload and top_seed_files:
            extra_related_files = related_files_from_graph(graph_payload, top_seed_files, limit=3)

        existing_paths = {h.get("path", "") for h in filtered_hits}
        graph_augmented_hits = filtered_hits[:]
        for score, hit in semantic_scored:
            if score < _OPTIMIZER.relevance_threshold:
                continue
            path = hit.get("path", "")
            if path in extra_related_files and path not in existing_paths:
                graph_augmented_hits.append(hit)
                existing_paths.add(path)
            if len(graph_augmented_hits) >= (len(filtered_hits) + 3):
                break

        semantic_context = "\n\n".join(
            f"File: {h['path']} lines {h['start_line']}-{h['end_line']}\n"
            f"{h.get('content', '')[:500]}"
            for h in graph_augmented_hits
        ) or "None"

        # --- instruction docs ---
        # For focused queries, truncate heavy instruction docs to save tokens.
        raw_docs = load_euler_instruction_docs(Path(workdir)) or "None"
        if complexity == QueryComplexity.FOCUSED and len(raw_docs) > 1_200:
            instruction_docs = raw_docs[:1_200] + "\n... [truncated]"
        else:
            instruction_docs = raw_docs

        return {
            "memory_context": memory_context,
            "semantic_context": semantic_context,
            "instruction_docs": instruction_docs,
        }

    def _planner(self, state: AgentState) -> AgentState:
        skill_protocol = render_skill_protocol(
            "strategic planner",
            "build an execution plan that downstream specialists can implement directly",
        )
        prompt = (
            f"## User Request\n{state['user_goal']}\n\n"
            f"## Project Memory (related past outcomes)\n"
            f"{state.get('memory_context', 'None')}\n\n"
            f"## Relevant Existing Code\n"
            f"{state.get('semantic_context', 'None')}\n\n"
            f"## Project Instructions\n"
            f"{state.get('instruction_docs', 'None')}\n\n"
            f"## Skill Workflow\n{skill_protocol}\n\n"
            "Produce the full strategic plan in the format described in your role."
        )
        self._record_stage_tokens(state, "planner", SYSTEM_PLANNER, prompt)
        plan = _invoke(self.model, SYSTEM_PLANNER, prompt)
        complexity_str = state.get("complexity", QueryComplexity.FULL.value)
        try:
            complexity = QueryComplexity(complexity_str)
        except ValueError:
            complexity = QueryComplexity.FULL
        skip, confidence, reason = _OPTIMIZER.should_skip_specialists(
            state["user_goal"],
            complexity,
            plan,
        )
        return {
            "plan": plan,
            "skip_specialists": skip,
            "planner_confidence": confidence,
            "skip_reason": reason,
        }

    def _route_after_planner(self, state: AgentState) -> str:
        if state.get("skip_specialists", False):
            return "reviewer"
        return "parallel_specialists"

    def _parallel_specialists(self, state: AgentState) -> AgentState:
        ctx = _build_base_context(state)
        def _ctx_with_skill(role: str, objective: str, base_instruction: str) -> str:
            return (
                f"{base_instruction}\n\n"
                f"## Skill Workflow\n{render_skill_protocol(role, objective)}\n\n"
                f"{ctx}"
            )

        # Full catalogue of available specialists
        all_specialists: dict[str, tuple[str, str]] = {
            "architect": (
                SYSTEM_ARCHITECT,
                _ctx_with_skill("systems architect", "define module contracts and system boundaries", "Produce the full architecture blueprint."),
            ),
            "coder": (
                SYSTEM_CODER,
                _ctx_with_skill("senior engineer", "deliver the full implementation with complete code", "Produce the complete implementation."),
            ),
            "tester": (
                SYSTEM_TESTER,
                _ctx_with_skill("qa engineer", "deliver complete test coverage for implemented behavior", "Produce the complete test suite."),
            ),
            "security": (
                SYSTEM_SECURITY,
                _ctx_with_skill("security engineer", "identify vulnerabilities and provide concrete remediations", "Perform a full security review of the plan and implementation."),
            ),
            "devops": (
                SYSTEM_DEVOPS,
                _ctx_with_skill("devops engineer", "produce deployable infra and runtime operations setup", "Produce the full deployment and infra config."),
            ),
            "db": (
                SYSTEM_DB,
                _ctx_with_skill("database engineer", "design schema, migrations, and safe query patterns", "Design and implement the data layer."),
            ),
            "documenter": (
                SYSTEM_DOCUMENTER,
                _ctx_with_skill("technical writer", "produce implementation-aligned developer documentation", "Produce all documentation."),
            ),
            "refactor": (
                SYSTEM_REFACTOR,
                _ctx_with_skill("refactor engineer", "improve maintainability without behavior changes", "Identify and apply refactoring to any existing relevant code."),
            ),
        }

        # Restrict to the specialists selected by _classify_query
        selected_keys_raw: list[str] = state.get(
            "selected_specialists",
            list(all_specialists.keys()),  # fallback: all
        )
        selected_keys = _ordered_specialists(selected_keys_raw)
        specialists = {k: v for k, v in all_specialists.items() if k in selected_keys}
        for key, (sys_prompt, human_prompt) in specialists.items():
            self._record_stage_tokens(state, f"specialist:{key}", sys_prompt, human_prompt)

        results: dict[str, str] = {}
        workers = max(1, min(len(specialists), self._max_specialist_workers))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_key = {
                pool.submit(_invoke, self._new_model_client(), sys_prompt, human_prompt): key
                for key, (sys_prompt, human_prompt) in specialists.items()
            }
            for future in as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    results[key] = future.result()
                except Exception as exc:
                    results[key] = f"[specialist error: {exc}]"

        # Populate all output keys (unselected specialists get empty string)
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
        selected = state.get("selected_specialists", [
            "architect", "coder", "tester", "security",
            "devops", "db", "documenter", "refactor",
        ])
        selected = _ordered_specialists(selected)

        # Use the optimiser to compress specialist outputs before aggregation
        compressed_outputs = _OPTIMIZER.compress_specialist_outputs(state, selected)
        conflict_report = _build_conflict_report(state, selected)
        merge_policy = (
            "## Deterministic Merge Policy\n"
            "- Preserve user goal and planner acceptance criteria as highest priority.\n"
            "- Specialist precedence for direct conflicts:\n"
            f"  {', '.join(_SPECIALIST_ORDER)}\n"
            "- Security constraints override conflicting implementation details.\n"
            "- Prefer minimally invasive edits when two options are equivalent.\n"
            "- Output one conflict-resolution decision per conflicting file.\n"
        )

        skill_protocol = render_skill_protocol(
            "lead engineer arbitrator",
            "merge specialist outputs into one coherent production plan",
        )
        prompt = (
            f"## User Goal\n{state['user_goal']}\n\n"
            f"## Strategic Plan\n{state.get('plan', '')}\n\n"
            f"{merge_policy}\n"
            f"## Conflict Detection\n{conflict_report}\n\n"
            f"{compressed_outputs}\n\n"
            f"## Skill Workflow\n{skill_protocol}\n\n"
            "Arbitrate all specialist outputs into a single unified strategy."
        )
        self._record_stage_tokens(state, "arbitrator", SYSTEM_ARBITRATOR, prompt)
        merged = _invoke(self.model, SYSTEM_ARBITRATOR, prompt)
        return {"arbitrated_output": merged}

    def _reviewer(self, state: AgentState) -> AgentState:
        skip_note = ""
        if state.get("skip_specialists", False):
            skip_note = (
                "## Specialist Stage\n"
                "Skipped by adaptive early-exit optimization.\n"
                f"Reason: {state.get('skip_reason', '')}\n"
                f"Planner confidence: {state.get('planner_confidence', 0.0):.2f}\n\n"
            )
        skill_protocol = render_skill_protocol(
            "principal reviewer",
            "produce the final production-ready output with correctness guarantees",
        )
        prompt = (
            f"## User Goal\n{state['user_goal']}\n\n"
            f"## Strategic Plan\n{state.get('plan', '')}\n\n"
            f"{skip_note}"
            f"## Arbitrated Strategy\n{state.get('arbitrated_output', '')}\n\n"
            f"## Skill Workflow\n{skill_protocol}\n\n"
            "Perform the final production review and deliver the complete, "
            "corrected, and deployment-ready answer."
        )
        self._record_stage_tokens(state, "reviewer", SYSTEM_REVIEWER, prompt)
        final = _invoke(self.model, SYSTEM_REVIEWER, prompt)
        return {"final_output": final}

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("classify_query",       self._classify_query)
        graph.add_node("gather_context",       self._gather_context)
        graph.add_node("planner",              self._planner)
        graph.add_node("parallel_specialists", self._parallel_specialists)
        graph.add_node("arbitrator",           self._arbitrator)
        graph.add_node("reviewer",             self._reviewer)

        graph.add_edge(START,                  "classify_query")
        graph.add_edge("classify_query",       "gather_context")
        graph.add_edge("gather_context",       "planner")
        graph.add_conditional_edges(
            "planner",
            self._route_after_planner,
            {
                "parallel_specialists": "parallel_specialists",
                "reviewer": "reviewer",
            },
        )
        graph.add_edge("parallel_specialists", "arbitrator")
        graph.add_edge("arbitrator",           "reviewer")
        graph.add_edge("reviewer",             END)
        return graph.compile()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def run(self, user_goal: str, workdir: str | None = None) -> str:
        resolved_workdir = str(Path(workdir or ".").resolve())
        stage_tokens: dict[str, int] = {}
        cache_hit = False

        # --- response cache check (fast path) ---
        repo_fingerprint = _repo_state_fingerprint(resolved_workdir)
        cache_key = _OPTIMIZER.make_cache_key(
            self._model_id,
            user_goal,
            repo_fingerprint,
        )
        cached = _OPTIMIZER.get_cached_response(cache_key)
        if cached is not None:
            cache_hit = True
            self._last_run_stats = {
                "seq": self._stats_seq + 1,
                "kind": "run",
                "cache_hit": True,
                "specialists_used": 0,
                "specialists_total": 8,
                "stage_tokens": {},
            }
            self._stats_seq += 1
            return cached
        semantic_cached = _OPTIMIZER.get_semantic_cached_response(
            self._model_id,
            user_goal,
            repo_fingerprint,
        )
        if semantic_cached is not None:
            cache_hit = True
            self._last_run_stats = {
                "seq": self._stats_seq + 1,
                "kind": "run",
                "cache_hit": True,
                "specialists_used": 0,
                "specialists_total": 8,
                "stage_tokens": {},
            }
            self._stats_seq += 1
            return semantic_cached

        # --- full pipeline ---
        compiled = self._build_graph()
        initial: AgentState = {
            "user_goal": user_goal,
            "workdir": resolved_workdir,
            "stage_tokens": {},
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
        stage_tokens = result.get("stage_tokens", {}) or {}
        selected = result.get("selected_specialists", []) or []
        self._last_run_stats = {
            "seq": self._stats_seq + 1,
            "kind": "run",
            "cache_hit": cache_hit,
            "specialists_used": len(selected),
            "specialists_total": 8,
            "stage_tokens": stage_tokens,
        }
        self._stats_seq += 1

        # Persist to response cache and long-term memory
        _OPTIMIZER.set_semantic_cached_response(
            cache_key=cache_key,
            response=final,
            model_id=self._model_id,
            query=user_goal,
            context_fingerprint=repo_fingerprint,
        )
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
