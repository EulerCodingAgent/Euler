"""Multi-agent orchestration with LangGraph."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from euler_agent.context import load_euler_instruction_docs
from euler_agent.memory import add_memory, search_memory
from euler_agent.providers import get_chat_model
from euler_agent.semantic_index import search_index


class AgentState(TypedDict, total=False):
    user_goal: str
    plan: str
    architect_output: str
    coder_output: str
    tester_output: str
    arbitrated_output: str
    review_output: str
    final_output: str
    workdir: str


def _safe_invoke(model, system_prompt: str, human_prompt: str) -> str:
    response = model.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ]
    )
    return response.content if isinstance(response.content, str) else str(response.content)


class EulerAgent:
    def __init__(self, provider: str, model_name: str, api_key: str) -> None:
        self.model = get_chat_model(provider=provider, model=model_name, api_key=api_key)

    def ask(self, prompt: str, role: str = "assistant") -> str:
        return _safe_invoke(self.model, f"You are Euler acting as {role}.", prompt)

    def _planner(self, state: AgentState) -> AgentState:
        memory_snippets = search_memory(
            project=state["workdir"],
            query=state["user_goal"],
            limit=3,
        )
        memory_context = "\n\n".join(
            f"Past goal: {m.goal}\nPast result: {m.result[:800]}" for m in memory_snippets
        ) or "None"
        semantic_hits = search_index(
            workdir=state["workdir"],
            query=state["user_goal"],
            limit=3,
        )
        semantic_context = "\n\n".join(
            (
                f"File: {hit.get('path')} lines {hit.get('start_line')}-{hit.get('end_line')}\n"
                f"{hit.get('content', '')[:700]}"
            )
            for hit in semantic_hits
        ) or "None"
        prompt = (
            "Break the request into clear implementation subtasks and constraints.\n\n"
            f"User request:\n{state['user_goal']}\n\n"
            f"Relevant long-term project memory:\n{memory_context}\n\n"
            f"Relevant semantic code hits:\n{semantic_context}"
        )
        plan = _safe_invoke(
            self.model,
            "You are an architect agent. Produce concise actionable plans.",
            prompt,
        )
        return {"plan": plan}

    def _parallel_specialists(self, state: AgentState) -> AgentState:
        workdir = Path(state["workdir"])
        instruction_docs = load_euler_instruction_docs(workdir)
        base_context = (
            f"Plan:\n{state['plan']}\n\n"
            f"User goal:\n{state['user_goal']}\n\n"
            f"Supporting instruction docs from ./Euler/*.md:\n{instruction_docs or 'None'}"
        )

        def run_architect() -> str:
            return _safe_invoke(
                self.model,
                (
                    "You are an architecture specialist. Define module boundaries, "
                    "execution order, risk points, and integration constraints."
                ),
                base_context,
            )

        def run_coder() -> str:
            return _safe_invoke(
                self.model,
                "You are a coding agent. Produce implementation-ready output and concrete code actions.",
                base_context,
            )

        def run_tester() -> str:
            return _safe_invoke(
                self.model,
                (
                    "You are a test specialist. Design unit/integration/regression checks, "
                    "failure scenarios, and acceptance criteria."
                ),
                base_context,
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            architect_future = pool.submit(run_architect)
            code_future = pool.submit(run_coder)
            tester_future = pool.submit(run_tester)
            architect_output = architect_future.result()
            code_output = code_future.result()
            tester_output = tester_future.result()

        return {
            "architect_output": architect_output,
            "coder_output": code_output,
            "tester_output": tester_output,
        }

    def _arbitrator(self, state: AgentState) -> AgentState:
        prompt = (
            f"Goal:\n{state['user_goal']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Architect proposal:\n{state['architect_output']}\n\n"
            f"Coder proposal:\n{state['coder_output']}\n\n"
            f"Tester proposal:\n{state['tester_output']}\n\n"
            "Resolve conflicts and produce one unified implementation strategy with clear steps."
        )
        merged = _safe_invoke(
            self.model,
            "You are an arbitration agent. Resolve specialist conflicts and pick best combined path.",
            prompt,
        )
        return {"arbitrated_output": merged}

    def _reviewer(self, state: AgentState) -> AgentState:
        prompt = (
            f"Goal:\n{state['user_goal']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Arbitrated strategy:\n{state['arbitrated_output']}\n\n"
            "Return the best final answer with self-corrections, implementation steps, and test checklist."
        )
        review = _safe_invoke(
            self.model,
            "You are a reviewer agent. Merge outputs and fix mistakes.",
            prompt,
        )
        return {"review_output": review, "final_output": review}

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("planner", self._planner)
        graph.add_node("parallel_specialists", self._parallel_specialists)
        graph.add_node("arbitrator", self._arbitrator)
        graph.add_node("reviewer", self._reviewer)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "parallel_specialists")
        graph.add_edge("parallel_specialists", "arbitrator")
        graph.add_edge("arbitrator", "reviewer")
        graph.add_edge("reviewer", END)
        return graph.compile()

    def run(self, user_goal: str, workdir: str | None = None) -> str:
        compiled = self._build_graph()
        resolved_workdir = str(Path(workdir or ".").resolve())
        state: AgentState = {
            "user_goal": user_goal,
            "workdir": resolved_workdir,
        }
        result = compiled.invoke(state)
        final = result.get("final_output", "No output generated.")
        add_memory(
            project=resolved_workdir,
            goal=user_goal,
            result=final,
            tags=["run", "graph"],
        )
        return final

    def generate_sql(self, requirement: str) -> str:
        prompt = f"Create a robust SQL query for this requirement:\n{requirement}"
        return _safe_invoke(
            self.model,
            "You are a senior SQL engineer. Return SQL first, then a short explanation.",
            prompt,
        )

    def rewrite_selection(
        self,
        file_path: str,
        selected_text: str,
        instruction: str,
    ) -> str:
        prompt = (
            f"Rewrite only the selected code according to instruction.\n\n"
            f"Instruction:\n{instruction}\n\n"
            f"Selected code from {file_path}:\n{selected_text}"
        )
        return _safe_invoke(
            self.model,
            "You rewrite selected code only. Return only the replacement code.",
            prompt,
        )
