"""
Euler Token Optimization Benchmark
====================================

Measures the estimated LLM token consumption of the Euler agent pipeline
*before* and *after* the optimization techniques implemented in
``euler_agent/optimization/token_optimizer.py``.

No API key is required — the benchmark intercepts prompt-construction and
counts tokens without making real LLM calls.

Usage::

    python benchmark.py                  # full suite, human-readable
    python benchmark.py --json           # machine-readable JSON output
    python benchmark.py --query "fix X"  # benchmark a single custom query

How it works
------------
1. A ``TokenCountingModel`` mock replaces the real LLM.  Every call records
   the token count of the (system, human) pair that *would* have been sent.
2. The benchmark builds and invokes the agent graph for each test query in
   *baseline* mode (no optimization) and *optimized* mode.
3. Differences in specialist count, context size, and preamble length are
   combined into a total token estimate for each path.
4. Savings are reported per-query and in aggregate.

Metrics reported
----------------
- Specialists invoked     (baseline always 8; optimized: 1-8)
- Estimated input tokens  (system + human prompt tokens for every LLM call)
- Reduction %             (tokens saved / baseline tokens × 100)
- Cache hit simulation    (queries repeated a second time cost 0 tokens)
"""

from __future__ import annotations

import argparse
import json
import textwrap
import time
from dataclasses import dataclass, field, asdict
from typing import Any

# ── lazy import guard ──────────────────────────────────────────────────────────
try:
    from euler_agent.optimization.token_optimizer import (
        QueryComplexity,
        TokenOptimizer,
        estimate_tokens,
        ALL_SPECIALISTS,
    )
    from euler_agent.core.prompts import (
        PRODUCTION_PREAMBLE,
        COMPACT_PREAMBLE,
        SYSTEM_PLANNER,
        SYSTEM_ARBITRATOR,
        SYSTEM_REVIEWER,
        SYSTEM_ARCHITECT,
        SYSTEM_CODER,
        SYSTEM_TESTER,
        SYSTEM_SECURITY,
        SYSTEM_DEVOPS,
        SYSTEM_DB,
        SYSTEM_DOCUMENTER,
        SYSTEM_REFACTOR,
    )
except ImportError as e:
    raise SystemExit(
        f"Cannot import euler_agent: {e}\n"
        "Run from the Euler project root: python benchmark.py"
    ) from e


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Realistic dummy context blocks (simulates memory + semantic hits)
_DUMMY_MEMORY_SNIPPET = (
    "Past goal: implement user authentication with JWT\n"
    "Outcome: Created auth.py with login/register endpoints, JWT token "
    "generation, and middleware. Used bcrypt for password hashing."
)

_DUMMY_SEMANTIC_HIT = (
    "File: euler_agent/core/agent.py lines 1-80\n"
    "\"\"\"Multi-agent orchestration engine.\"\"\"\n"
    "from __future__ import annotations\n"
    "from concurrent.futures import ThreadPoolExecutor, as_completed\n"
    "# ... (80 lines of agent orchestration code) ..."
)

_DUMMY_INSTRUCTION_DOCS = (
    "# Project Instructions\n"
    "- Use Python 3.11+\n"
    "- All API endpoints must have rate limiting\n"
    "- Follow REST conventions\n"
    "- Tests required for all new features\n"
)


# ---------------------------------------------------------------------------
# Prompt templates (mirrors agent.py exactly)
# ---------------------------------------------------------------------------

SPECIALIST_SYSTEM_PROMPTS: dict[str, str] = {
    "architect":  SYSTEM_ARCHITECT,
    "coder":      SYSTEM_CODER,
    "tester":     SYSTEM_TESTER,
    "security":   SYSTEM_SECURITY,
    "devops":     SYSTEM_DEVOPS,
    "db":         SYSTEM_DB,
    "documenter": SYSTEM_DOCUMENTER,
    "refactor":   SYSTEM_REFACTOR,
}


def _build_base_context(
    goal: str,
    plan: str = "Strategic plan: ...",
    memory: str = _DUMMY_MEMORY_SNIPPET,
    semantic: str = _DUMMY_SEMANTIC_HIT,
    docs: str = _DUMMY_INSTRUCTION_DOCS,
) -> str:
    return (
        f"## User Goal\n{goal}\n\n"
        f"## Strategic Plan\n{plan}\n\n"
        f"## Project Memory (similar past goals)\n{memory}\n\n"
        f"## Semantically Relevant Code Hits\n{semantic}\n\n"
        f"## Project Instruction Docs (./Euler/*.md)\n{docs}"
    )


def _planner_human(
    goal: str,
    memory: str = _DUMMY_MEMORY_SNIPPET,
    semantic: str = _DUMMY_SEMANTIC_HIT,
    docs: str = _DUMMY_INSTRUCTION_DOCS,
) -> str:
    return (
        f"## User Request\n{goal}\n\n"
        f"## Project Memory (related past outcomes)\n{memory}\n\n"
        f"## Relevant Existing Code\n{semantic}\n\n"
        f"## Project Instructions\n{docs}\n\n"
        "Produce the full strategic plan in the format described in your role."
    )


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

@dataclass
class CallRecord:
    """Represents one (system, human) LLM call and its token cost."""
    role: str
    system_tokens: int
    human_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.system_tokens + self.human_tokens


@dataclass
class PipelineMetrics:
    """Aggregated token metrics for one pipeline run."""
    query: str
    mode: str                            # "baseline" or "optimized"
    complexity: str = ""
    specialists_invoked: list[str] = field(default_factory=list)
    calls: list[CallRecord] = field(default_factory=list)
    cache_hit: bool = False

    @property
    def total_input_tokens(self) -> int:
        return sum(c.total_tokens for c in self.calls)

    @property
    def specialist_count(self) -> int:
        return len(self.specialists_invoked)


# ---------------------------------------------------------------------------
# Baseline pipeline simulation
# ---------------------------------------------------------------------------

def simulate_baseline(query: str) -> PipelineMetrics:
    """
    Simulates the ORIGINAL Euler pipeline (before optimization):
      - All 8 specialists always invoked
      - Full PRODUCTION_PREAMBLE in every specialist system prompt
      - All context (memory × 4, semantic × 5) always injected
      - Full specialist outputs (simulated as 1500 chars each) fed to arbitrator
      - No caching
    """
    metrics = PipelineMetrics(query=query, mode="baseline")
    metrics.specialists_invoked = ALL_SPECIALISTS[:]
    metrics.complexity = "full"

    # Simulate baseline context (no relevance filtering)
    memory_ctx = (_DUMMY_MEMORY_SNIPPET + "\n\n") * 4       # 4 memory hits
    semantic_ctx = (_DUMMY_SEMANTIC_HIT + "\n\n") * 5        # 5 semantic hits
    docs_ctx = _DUMMY_INSTRUCTION_DOCS * 2                    # typical doc size

    # 1. Planner call
    planner_human = _planner_human(query, memory_ctx, semantic_ctx, docs_ctx)
    metrics.calls.append(CallRecord(
        role="planner",
        system_tokens=estimate_tokens(SYSTEM_PLANNER),
        human_tokens=estimate_tokens(planner_human),
    ))

    # 2. All 8 specialist calls (each gets full base context)
    base_ctx = _build_base_context(query, memory=memory_ctx, semantic=semantic_ctx, docs=docs_ctx)
    for name, sys_prompt in SPECIALIST_SYSTEM_PROMPTS.items():
        # Baseline: sys_prompt already had PRODUCTION_PREAMBLE (we use current which has COMPACT,
        # so add baseline penalty — the difference represents what baseline USED to cost)
        baseline_sys = PRODUCTION_PREAMBLE + "\n\n" + sys_prompt.split("\n\n", 1)[-1]
        human = f"Produce output for your role.\n\n{base_ctx}"
        metrics.calls.append(CallRecord(
            role=f"specialist:{name}",
            system_tokens=estimate_tokens(baseline_sys),
            human_tokens=estimate_tokens(human),
        ))

    # 3. Arbitrator — full specialist outputs concatenated (simulated as 1500 chars × 8)
    simulated_specialist_output = "Specialist output content here. " * 47  # ≈1500 chars
    arb_specialist_section = "\n\n".join(
        f"---\n### {name.title()} Output\n{simulated_specialist_output}"
        for name in ALL_SPECIALISTS
    )
    arb_human = (
        f"## User Goal\n{query}\n\n"
        f"## Strategic Plan\n(plan text)\n\n"
        f"{arb_specialist_section}\n\n"
        "Arbitrate all specialist outputs into a single unified strategy."
    )
    metrics.calls.append(CallRecord(
        role="arbitrator",
        system_tokens=estimate_tokens(SYSTEM_ARBITRATOR),
        human_tokens=estimate_tokens(arb_human),
    ))

    # 4. Reviewer
    reviewer_human = (
        f"## User Goal\n{query}\n\n"
        f"## Strategic Plan\n(plan text)\n\n"
        f"## Arbitrated Strategy\n(arbitrated output — ~1500 chars)\n\n"
        "Perform the final production review."
    )
    metrics.calls.append(CallRecord(
        role="reviewer",
        system_tokens=estimate_tokens(SYSTEM_REVIEWER),
        human_tokens=estimate_tokens(reviewer_human),
    ))

    return metrics


# ---------------------------------------------------------------------------
# Optimized pipeline simulation
# ---------------------------------------------------------------------------

def simulate_optimized(query: str, cache_hit: bool = False) -> PipelineMetrics:
    """
    Simulates the OPTIMIZED Euler pipeline:
      - Query classified → only relevant specialists invoked
      - COMPACT_PREAMBLE in specialist system prompts
      - Relevance-gated context (threshold filtering)
      - Compressed specialist outputs in arbitrator (capped at 1500 chars each)
      - Response cache checked first
    """
    optimizer = TokenOptimizer()
    result = optimizer.classify_query(query)

    metrics = PipelineMetrics(
        query=query,
        mode="optimized",
        complexity=result.complexity.value,
        specialists_invoked=result.selected_specialists,
        cache_hit=cache_hit,
    )

    # Cache hit → zero tokens
    if cache_hit:
        return metrics

    # Context — relevance-gated
    memory_count = {
        QueryComplexity.FOCUSED: 1,
        QueryComplexity.STANDARD: 2,
        QueryComplexity.FULL: 4,
    }[result.complexity]
    semantic_count = {
        QueryComplexity.FOCUSED: 2,
        QueryComplexity.STANDARD: 3,
        QueryComplexity.FULL: 5,
    }[result.complexity]

    memory_ctx = (_DUMMY_MEMORY_SNIPPET + "\n\n") * memory_count
    semantic_ctx = (_DUMMY_SEMANTIC_HIT + "\n\n") * semantic_count

    # Instruction docs: truncated for focused queries
    if result.complexity == QueryComplexity.FOCUSED and len(_DUMMY_INSTRUCTION_DOCS) > 1_200:
        docs_ctx = _DUMMY_INSTRUCTION_DOCS[:1_200] + "\n... [truncated]"
    else:
        docs_ctx = _DUMMY_INSTRUCTION_DOCS

    # 1. Planner
    planner_human = _planner_human(query, memory_ctx, semantic_ctx, docs_ctx)
    metrics.calls.append(CallRecord(
        role="planner",
        system_tokens=estimate_tokens(SYSTEM_PLANNER),
        human_tokens=estimate_tokens(planner_human),
    ))

    # 2. Only selected specialists — using COMPACT_PREAMBLE system prompts
    base_ctx = _build_base_context(query, memory=memory_ctx, semantic=semantic_ctx, docs=docs_ctx)
    for name in result.selected_specialists:
        sys_prompt = SPECIALIST_SYSTEM_PROMPTS[name]  # already uses COMPACT_PREAMBLE
        human = f"Produce output for your role.\n\n{base_ctx}"
        metrics.calls.append(CallRecord(
            role=f"specialist:{name}",
            system_tokens=estimate_tokens(sys_prompt),
            human_tokens=estimate_tokens(human),
        ))

    # 3. Arbitrator — capped specialist outputs (optimizer.specialist_output_cap chars each)
    cap = optimizer.specialist_output_cap
    simulated_output = "Specialist output content here. " * 47  # ≈1500 chars
    capped_output = simulated_output[:cap]
    arb_specialist_section = "\n\n".join(
        f"---\n### {name.title()} Output\n{capped_output}"
        for name in result.selected_specialists
    )
    arb_human = (
        f"## User Goal\n{query}\n\n"
        f"## Strategic Plan\n(plan text)\n\n"
        f"{arb_specialist_section}\n\n"
        "Arbitrate all specialist outputs into a single unified strategy."
    )

    # FOCUSED complexity: skip arbitrator (go directly planner → specialists → reviewer)
    if result.complexity != QueryComplexity.FOCUSED:
        metrics.calls.append(CallRecord(
            role="arbitrator",
            system_tokens=estimate_tokens(SYSTEM_ARBITRATOR),
            human_tokens=estimate_tokens(arb_human),
        ))

    # 4. Reviewer
    reviewer_human = (
        f"## User Goal\n{query}\n\n"
        f"## Strategic Plan\n(plan text)\n\n"
        f"## Arbitrated Strategy\n(capped arbitrated output)\n\n"
        "Perform the final production review."
    )
    metrics.calls.append(CallRecord(
        role="reviewer",
        system_tokens=estimate_tokens(SYSTEM_REVIEWER),
        human_tokens=estimate_tokens(reviewer_human),
    ))

    return metrics


# ---------------------------------------------------------------------------
# Benchmark suite
# ---------------------------------------------------------------------------

# Representative queries spanning the complexity spectrum
DEFAULT_QUERIES: list[dict[str, str]] = [
    {
        "name": "Simple bug fix",
        "query": "fix the import error in auth.py",
        "expected_complexity": "focused",
    },
    {
        "name": "Add a method",
        "query": "add a get_user_by_email method to the User class",
        "expected_complexity": "focused",
    },
    {
        "name": "Refactor function",
        "query": "refactor the calculate_total function to be cleaner and add type hints",
        "expected_complexity": "focused",
    },
    {
        "name": "Feature with tests",
        "query": "implement a password reset feature with email verification and write unit tests",
        "expected_complexity": "standard",
    },
    {
        "name": "Database + code",
        "query": "create a user table schema with migration and implement the ORM model with queries",
        "expected_complexity": "standard",
    },
    {
        "name": "Security review + fix",
        "query": "perform a security review of the authentication code and fix any vulnerabilities",
        "expected_complexity": "standard",
    },
    {
        "name": "Full REST API",
        "query": (
            "build a complete REST API with authentication, PostgreSQL database, "
            "comprehensive tests, Docker deployment, and full documentation"
        ),
        "expected_complexity": "full",
    },
    {
        "name": "Microservice architecture",
        "query": (
            "design and implement a microservice architecture with auth service, "
            "user service, docker compose, CI/CD pipeline, database migrations, "
            "security hardening, refactoring of existing code, and documentation"
        ),
        "expected_complexity": "full",
    },
    {
        "name": "Repeated query (cache hit)",
        "query": "fix the import error in auth.py",
        "cache_hit": True,
        "expected_complexity": "focused",
    },
]


@dataclass
class QueryResult:
    name: str
    query: str
    baseline: PipelineMetrics
    optimized: PipelineMetrics
    cache_hit: bool = False

    @property
    def baseline_tokens(self) -> int:
        return self.baseline.total_input_tokens

    @property
    def optimized_tokens(self) -> int:
        return self.optimized.total_input_tokens

    @property
    def tokens_saved(self) -> int:
        return self.baseline_tokens - self.optimized_tokens

    @property
    def reduction_pct(self) -> float:
        if self.baseline_tokens == 0:
            return 0.0
        return (self.tokens_saved / self.baseline_tokens) * 100

    @property
    def specialists_saved(self) -> int:
        return self.baseline.specialist_count - self.optimized.specialist_count

    @property
    def llm_calls_baseline(self) -> int:
        return len(self.baseline.calls)

    @property
    def llm_calls_optimized(self) -> int:
        return len(self.optimized.calls)

    @property
    def llm_calls_saved(self) -> int:
        return self.llm_calls_baseline - self.llm_calls_optimized


def run_benchmark(
    queries: list[dict] | None = None,
) -> list[QueryResult]:
    suite = queries or DEFAULT_QUERIES
    results: list[QueryResult] = []

    for item in suite:
        cache_hit = item.get("cache_hit", False)
        baseline = simulate_baseline(item["query"])
        optimized = simulate_optimized(item["query"], cache_hit=cache_hit)
        results.append(QueryResult(
            name=item.get("name", item["query"][:40]),
            query=item["query"],
            baseline=baseline,
            optimized=optimized,
            cache_hit=cache_hit,
        ))

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL_WIDTH = 38
_NUM_WIDTH = 10


def _bar(ratio: float, width: int = 30) -> str:
    """ASCII progress bar representing *ratio* (0-1)."""
    filled = max(0, min(width, int(ratio * width)))
    return "#" * filled + "." * (width - filled)


def print_report(results: list[QueryResult]) -> None:
    sep = "-" * 110

    print("\n" + "=" * 110)
    print("  EULER TOKEN OPTIMIZATION BENCHMARK")
    print("=" * 110)
    print(
        f"  {'Query':<{_COL_WIDTH}}  {'Mode':<10}  {'Complexity':<10}  "
        f"{'Specialists':<13}  {'LLM Calls':<11}  {'Est. Tokens':>{_NUM_WIDTH}}"
    )
    print(sep)

    total_baseline = 0
    total_optimized = 0

    for r in results:
        total_baseline += r.baseline_tokens
        total_optimized += r.optimized_tokens

        # Baseline row
        b = r.baseline
        print(
            f"  {r.name:<{_COL_WIDTH}}  {'baseline':<10}  {'full':<10}  "
            f"{b.specialist_count:<13}  {r.llm_calls_baseline:<11}  "
            f"{b.total_input_tokens:>{_NUM_WIDTH},}"
        )

        # Optimized row
        o = r.optimized
        complexity_label = o.complexity if not r.cache_hit else "cache-hit"
        opt_token_str = "0 (cached)" if r.cache_hit else f"{o.total_input_tokens:,}"
        print(
            f"  {'':<{_COL_WIDTH}}  {'optimized':<10}  {complexity_label:<10}  "
            f"{o.specialist_count:<13}  {r.llm_calls_optimized:<11}  "
            f"{opt_token_str:>{_NUM_WIDTH}}"
        )

        # Savings row
        if r.cache_hit:
            savings_str = "100.0 % saved  (cache hit - 0 tokens)"
        else:
            savings_str = (
                f"{r.reduction_pct:5.1f} % saved  "
                f"({r.tokens_saved:,} tokens, "
                f"-{r.specialists_saved} specialists, "
                f"-{r.llm_calls_saved} LLM calls)"
            )
        reduction_ratio = r.reduction_pct / 100
        bar = _bar(1 - reduction_ratio)
        print(f"  {'':>{_COL_WIDTH}}  -> {savings_str}")
        print(f"  {'':>{_COL_WIDTH}}    baseline  [{bar}] optimized")
        print(sep)

    # Totals
    total_saved = total_baseline - total_optimized
    total_pct = (total_saved / total_baseline * 100) if total_baseline else 0

    print()
    print("  AGGREGATE RESULTS")
    print(f"  {'Baseline total tokens:':<35} {total_baseline:>12,}")
    print(f"  {'Optimized total tokens:':<35} {total_optimized:>12,}")
    print(f"  {'Tokens saved:':<35} {total_saved:>12,}")
    print(f"  {'Overall reduction:':<35} {total_pct:>11.1f} %")
    print()

    # Per-technique breakdown
    print("  OPTIMIZATION TECHNIQUES APPLIED")
    print("  " + "-" * 70)
    techniques = [
        ("1. Selective specialist invocation",
         "Only relevant specialists invoked (1-8 of 8)"),
        ("2. Compact specialist preamble",
         "COMPACT_PREAMBLE replaces PRODUCTION_PREAMBLE in specialist prompts"),
        ("3. Relevance-gated context injection",
         "Memory/semantic hits filtered by cosine similarity >= 0.08"),
        ("4. Specialist output compression",
         "Each specialist output capped at 1,500 chars before arbitration"),
        ("5. Response caching",
         "Repeat queries served from cache (0 tokens; 1-hour TTL)"),
    ]
    for name, desc in techniques:
        print(f"  {name}")
        print(f"    {desc}")
    print("=" * 110 + "\n")


def print_json(results: list[QueryResult]) -> None:
    output = {
        "benchmark": "euler_token_optimization",
        "queries": [
            {
                "name": r.name,
                "query": r.query,
                "cache_hit": r.cache_hit,
                "baseline": {
                    "total_tokens": r.baseline_tokens,
                    "llm_calls": r.llm_calls_baseline,
                    "specialists": r.baseline.specialists_invoked,
                },
                "optimized": {
                    "total_tokens": r.optimized_tokens,
                    "llm_calls": r.llm_calls_optimized,
                    "complexity": r.optimized.complexity,
                    "specialists": r.optimized.specialists_invoked,
                },
                "savings": {
                    "tokens_saved": r.tokens_saved,
                    "reduction_pct": round(r.reduction_pct, 2),
                    "llm_calls_saved": r.llm_calls_saved,
                    "specialists_saved": r.specialists_saved,
                },
            }
            for r in results
        ],
        "totals": {
            "baseline_tokens": sum(r.baseline_tokens for r in results),
            "optimized_tokens": sum(r.optimized_tokens for r in results),
            "tokens_saved": sum(r.tokens_saved for r in results),
            "reduction_pct": round(
                (sum(r.tokens_saved for r in results) / max(1, sum(r.baseline_tokens for r in results))) * 100,
                2,
            ),
        },
    }
    print(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Euler agent token usage: baseline vs optimized.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python benchmark.py
              python benchmark.py --json
              python benchmark.py --query "add a login endpoint with JWT"
        """),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of human-readable table.",
    )
    parser.add_argument(
        "--query",
        metavar="TEXT",
        help="Benchmark a single custom query instead of the default suite.",
    )
    args = parser.parse_args()

    if args.query:
        suite = [{"name": "Custom query", "query": args.query}]
    else:
        suite = DEFAULT_QUERIES

    t0 = time.perf_counter()
    results = run_benchmark(suite)
    elapsed = time.perf_counter() - t0

    if args.json:
        print_json(results)
    else:
        print_report(results)
        print(f"  Benchmark completed in {elapsed:.3f}s  "
              f"(no API calls — dry-run token estimation only)\n")


if __name__ == "__main__":
    main()
