"""Dependency and symbol graph extraction."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GraphNode:
    id: str
    kind: str
    file: str


@dataclass
class GraphEdge:
    source: str
    target: str
    relation: str


def _parse_python(path: Path, rel: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    module_node = GraphNode(id=f"module:{rel}", kind="module", file=rel)
    nodes.append(module_node)

    source = path.read_text(encoding="utf-8", errors="ignore")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return nodes, edges

    for item in ast.walk(tree):
        if isinstance(item, ast.FunctionDef):
            fn_id = f"function:{rel}:{item.name}"
            nodes.append(GraphNode(id=fn_id, kind="function", file=rel))
            edges.append(GraphEdge(source=module_node.id, target=fn_id, relation="defines"))
        elif isinstance(item, ast.ClassDef):
            cls_id = f"class:{rel}:{item.name}"
            nodes.append(GraphNode(id=cls_id, kind="class", file=rel))
            edges.append(GraphEdge(source=module_node.id, target=cls_id, relation="defines"))
        elif isinstance(item, ast.Import):
            for alias in item.names:
                dep = alias.name
                dep_id = f"import:{dep}"
                nodes.append(GraphNode(id=dep_id, kind="import", file=rel))
                edges.append(GraphEdge(source=module_node.id, target=dep_id, relation="imports"))
        elif isinstance(item, ast.ImportFrom):
            dep = item.module or ""
            if dep:
                dep_id = f"import:{dep}"
                nodes.append(GraphNode(id=dep_id, kind="import", file=rel))
                edges.append(GraphEdge(source=module_node.id, target=dep_id, relation="imports"))
    return nodes, edges


def build_code_graph(workdir: str, output_path: str | None = None) -> str:
    root = Path(workdir).resolve()
    graph_file = Path(output_path).resolve() if output_path else root / ".euler" / "code_graph.json"
    graph_file.parent.mkdir(parents=True, exist_ok=True)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for path in root.rglob("*.py"):
        if any(part.startswith(".") for part in path.parts if part != "."):
            continue
        if "venv" in path.parts or ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(root))
        file_nodes, file_edges = _parse_python(path, rel)
        nodes.extend(file_nodes)
        edges.extend(file_edges)

    unique_nodes = {node.id: node for node in nodes}
    unique_edges = {(edge.source, edge.target, edge.relation): edge for edge in edges}
    payload = {
        "root": str(root),
        "nodes": [node.__dict__ for node in unique_nodes.values()],
        "edges": [edge.__dict__ for edge in unique_edges.values()],
    }
    graph_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return f"Graph saved to {graph_file} ({len(unique_nodes)} nodes, {len(unique_edges)} edges)"
