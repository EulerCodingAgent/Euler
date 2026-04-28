"""Dependency and symbol graph extraction."""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
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


def _parse_jsts(path: Path, rel: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    module_node = GraphNode(id=f"module:{rel}", kind="module", file=rel)
    nodes.append(module_node)
    source = path.read_text(encoding="utf-8", errors="ignore")

    function_names = re.findall(r"\bfunction\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", source)
    function_names += re.findall(r"\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\(", source)
    class_names = re.findall(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b", source)
    imports = re.findall(r"""import\s+.*?\s+from\s+["']([^"']+)["']""", source)
    imports += re.findall(r"""require\(["']([^"']+)["']\)""", source)

    for fn in sorted(set(function_names)):
        fn_id = f"function:{rel}:{fn}"
        nodes.append(GraphNode(id=fn_id, kind="function", file=rel))
        edges.append(GraphEdge(source=module_node.id, target=fn_id, relation="defines"))
    for cls in sorted(set(class_names)):
        cls_id = f"class:{rel}:{cls}"
        nodes.append(GraphNode(id=cls_id, kind="class", file=rel))
        edges.append(GraphEdge(source=module_node.id, target=cls_id, relation="defines"))
    for imp in sorted(set(imports)):
        imp_id = f"import:{imp}"
        nodes.append(GraphNode(id=imp_id, kind="import", file=rel))
        edges.append(GraphEdge(source=module_node.id, target=imp_id, relation="imports"))

    return nodes, edges


def _parse_sql(path: Path, rel: str) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    module_node = GraphNode(id=f"module:{rel}", kind="module", file=rel)
    nodes.append(module_node)
    source = path.read_text(encoding="utf-8", errors="ignore")
    lowered = source.lower()

    create_tables = re.findall(r"\bcreate\s+table\s+([a-zA-Z_][a-zA-Z0-9_\.]*)", lowered)
    select_tables = re.findall(r"\bfrom\s+([a-zA-Z_][a-zA-Z0-9_\.]*)", lowered)
    join_tables = re.findall(r"\bjoin\s+([a-zA-Z_][a-zA-Z0-9_\.]*)", lowered)

    for table in sorted(set(create_tables)):
        table_id = f"table:{table}"
        nodes.append(GraphNode(id=table_id, kind="table", file=rel))
        edges.append(GraphEdge(source=module_node.id, target=table_id, relation="defines_table"))

    for table in sorted(set(select_tables + join_tables)):
        table_id = f"table:{table}"
        nodes.append(GraphNode(id=table_id, kind="table", file=rel))
        edges.append(GraphEdge(source=module_node.id, target=table_id, relation="references_table"))

    return nodes, edges


def build_code_graph(workdir: str, output_path: str | None = None) -> str:
    root = Path(workdir).resolve()
    graph_file = Path(output_path).resolve() if output_path else root / ".euler" / "code_graph.json"
    graph_file.parent.mkdir(parents=True, exist_ok=True)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.parts if part != "."):
            continue
        if "venv" in path.parts or ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        rel = str(path.relative_to(root))
        if path.suffix == ".py":
            file_nodes, file_edges = _parse_python(path, rel)
        elif path.suffix.lower() in {".js", ".jsx", ".ts", ".tsx"}:
            file_nodes, file_edges = _parse_jsts(path, rel)
        elif path.suffix.lower() == ".sql":
            file_nodes, file_edges = _parse_sql(path, rel)
        else:
            continue
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


def load_code_graph(workdir: str) -> dict | None:
    """
    Load an existing graph payload from .euler/code_graph.json.
    """
    root = Path(workdir).resolve()
    graph_file = root / ".euler" / "code_graph.json"
    if not graph_file.exists():
        return None
    try:
        return json.loads(graph_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def related_files_from_graph(
    graph_payload: dict,
    seed_files: list[str],
    limit: int = 4,
) -> list[str]:
    """
    Return files related to *seed_files* by shared graph targets.

    Heuristic: files are related when they connect to the same imported module,
    symbol, or table nodes.
    """
    nodes = graph_payload.get("nodes", [])
    edges = graph_payload.get("edges", [])
    if not nodes or not edges or not seed_files:
        return []

    node_to_file: dict[str, str] = {}
    for node in nodes:
        node_id = str(node.get("id", ""))
        node_file = str(node.get("file", ""))
        if node_id and node_file:
            node_to_file[node_id] = node_file

    file_targets: dict[str, set[str]] = defaultdict(set)
    target_to_files: dict[str, set[str]] = defaultdict(set)
    for edge in edges:
        source = str(edge.get("source", ""))
        target = str(edge.get("target", ""))
        source_file = node_to_file.get(source)
        if not source_file or not target:
            continue
        file_targets[source_file].add(target)
        target_to_files[target].add(source_file)

    seed_set = set(seed_files)
    seed_targets: set[str] = set()
    for seed in seed_set:
        seed_targets.update(file_targets.get(seed, set()))

    scores: dict[str, int] = defaultdict(int)
    for target in seed_targets:
        for fpath in target_to_files.get(target, set()):
            if fpath not in seed_set:
                scores[fpath] += 1

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [path for path, _ in ranked[:limit]]
