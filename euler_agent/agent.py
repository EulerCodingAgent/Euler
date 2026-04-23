"""Multi-agent orchestration with LangGraph."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from euler_agent.context import load_euler_instruction_docs
from euler_agent.providers import get_chat_model


class AgentState(TypedDict, total=False):
    user_goal: str
    plan: str
    research_output: str
    code_output: str
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

    def _planner(self, state: AgentState) -> AgentState:
        prompt = (
            "Break the request into clear implementation subtasks and constraints.\n\n"
            f"User request:\n{state['user_goal']}"
        )
        plan = _safe_invoke(
            self.model,
            "You are an architect agent. Produce concise actionable plans.",
            prompt,
        )
        return {"plan": plan}

    def _parallel_workers(self, state: AgentState) -> AgentState:
        workdir = Path(state["workdir"])
        instruction_docs = load_euler_instruction_docs(workdir)
        base_context = (
            f"Plan:\n{state['plan']}\n\n"
            f"User goal:\n{state['user_goal']}\n\n"
            f"Supporting instruction docs from ./Euler/*.md:\n{instruction_docs or 'None'}"
        )

        def run_research() -> str:
            return _safe_invoke(
                self.model,
                "You are a research agent. Identify risks, dependencies, and execution order.",
                base_context,
            )

        def run_coder() -> str:
            return _safe_invoke(
                self.model,
                "You are a coding agent. Produce implementation-ready output and concrete code actions.",
                base_context,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            research_future = pool.submit(run_research)
            code_future = pool.submit(run_coder)
            research_output = research_future.result()
            code_output = code_future.result()

        return {"research_output": research_output, "code_output": code_output}

    def _reviewer(self, state: AgentState) -> AgentState:
        prompt = (
            f"Goal:\n{state['user_goal']}\n\n"
            f"Plan:\n{state['plan']}\n\n"
            f"Research:\n{state['research_output']}\n\n"
            f"Code:\n{state['code_output']}\n\n"
            "Return the best merged answer, with self-corrections where needed."
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
        graph.add_node("parallel_workers", self._parallel_workers)
        graph.add_node("reviewer", self._reviewer)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "parallel_workers")
        graph.add_edge("parallel_workers", "reviewer")
        graph.add_edge("reviewer", END)
        return graph.compile()

    def run(self, user_goal: str, workdir: str | None = None) -> str:
        compiled = self._build_graph()
        state: AgentState = {
            "user_goal": user_goal,
            "workdir": str(Path(workdir or ".").resolve()),
        }
        result = compiled.invoke(state)
        return result.get("final_output", "No output generated.")

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
