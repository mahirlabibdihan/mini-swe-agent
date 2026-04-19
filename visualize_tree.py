#!/usr/bin/env python3
"""Render a search tree JSON into a Graphviz tree with score labels.

Supports two common formats:
1) Nested tree (children are full child objects).
2) Flat node list (children are child ids).

Examples:
  python visualize_tree.py debug_tree.json
  python visualize_tree.py debug_nodes.json --format png --output tree.png
    python visualize_tree.py debug_tree.json --score-field value
    python visualize_tree.py debug_tree.json --show-unseen
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass
class Edge:
    parent: str
    child: str


def _escape_dot_label(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _short_id(node_id: str) -> str:
    return node_id[:8] if node_id else "unknown"


def _score_to_color(score: float | None) -> str:
    # Red -> yellow -> green gradient for quick visual scan.
    if score is None or not isinstance(score, (int, float)) or math.isnan(score):
        return "#eeeeee"
    s = max(0.0, min(1.0, float(score)))
    if s < 0.5:
        t = s / 0.5
        r, g, b = 231, int(76 + (201 - 76) * t), 60
    else:
        t = (s - 0.5) / 0.5
        r, g, b = int(241 + (46 - 241) * t), int(196 + (204 - 196) * t), 15
    return f"#{r:02x}{g:02x}{b:02x}"


def _state_key(node: Dict[str, Any]) -> str:
    # branch = node.get("branch") or "<none>"
    commit = node.get("commit") or "<none>"
    return f"{commit}"


def _build_state_color_map(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    # Deterministic one-to-one mapping for states present in this tree.
    palette = [
        "#e6194b",  # red
        "#3cb44b",  # green
        "#ffe119",  # yellow
        "#4363d8",  # blue
        "#f58231",  # orange
        "#911eb4",  # purple
        "#46f0f0",  # cyan
        "#f032e6",  # magenta
        "#bcf60c",  # lime
        "#fabebe",  # light pink
    ]
    states = sorted({_state_key(n) for n in nodes.values() if not _none_commit(n)})
    return {state: palette[i % len(palette)] for i, state in enumerate(states)}


def _none_commit(node: Dict[str, Any]) -> bool:
    return node.get("commit") is None


def _format_score(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def parse_nested_tree(root: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Edge], str]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Edge] = []

    def dfs(node: Dict[str, Any]) -> None:
        node_id = node.get("id")
        if not node_id:
            return
        nodes[node_id] = node
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                child_id = child.get("id")
                if child_id:
                    edges.append(Edge(parent=node_id, child=child_id))
                dfs(child)

    dfs(root)
    root_id = root.get("id")
    if not root_id:
        raise ValueError("Root node does not contain an 'id' field.")
    return nodes, edges, root_id


def parse_flat_nodes(flat_nodes: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], List[Edge], str]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Edge] = []
    children_seen: set[str] = set()

    for node in flat_nodes:
        node_id = node.get("id")
        if node_id:
            nodes[node_id] = node

    for node in flat_nodes:
        parent_id = node.get("id")
        if not parent_id:
            continue
        for child in node.get("children", []) or []:
            child_id = child if isinstance(child, str) else child.get("id")
            if child_id and child_id in nodes:
                edges.append(Edge(parent=parent_id, child=child_id))
                children_seen.add(child_id)

    roots = [n_id for n_id in nodes if n_id not in children_seen]
    if not roots:
        raise ValueError("Could not infer a root node from flat list.")

    roots.sort(key=lambda n_id: nodes[n_id].get("level", 10**9))
    return nodes, edges, roots[0]


def load_tree(path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[Edge], str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return parse_nested_tree(data)
    if isinstance(data, list):
        return parse_flat_nodes(data)
    raise ValueError("JSON root must be either an object (nested tree) or a list (flat nodes).")


def filter_visible(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Edge],
    hide_unseen: bool,
) -> Tuple[Dict[str, Dict[str, Any]], List[Edge]]:
    if not hide_unseen:
        return nodes, edges

    kept = {node_id: n for node_id, n in nodes.items() if n.get("visible", True)}
    kept_edges = [e for e in edges if e.parent in kept and e.child in kept]
    return kept, kept_edges


def build_dot(
    nodes: Dict[str, Dict[str, Any]],
    edges: Iterable[Edge],
    root_id: str,
    score_field: str,
    include_command: bool,
    color_by: str,
) -> str:
    lines: List[str] = []
    lines.append("digraph SearchTree {")
    lines.append("  rankdir=TB;")
    lines.append("  splines=true;")
    lines.append("  overlap=false;")
    lines.append('  node [shape=circle style="filled" fontname="Helvetica" fontsize=10 width=0.65 fixedsize=true];')
    lines.append('  edge [color="#999999"];')

    state_colors = _build_state_color_map(nodes) if color_by == "state" else {}

    for node_id, node in nodes.items():
        score_val = node.get(score_field)
        if _none_commit(node):
            color = "#bdbdbd"
        elif color_by == "state":
            color = state_colors[_state_key(node)]
        else:
            color = _score_to_color(score_val if isinstance(score_val, (int, float)) else None)
        shape = "doublecircle" if (node_id == root_id or node.get("is_submission", False)) else "circle"
        label = _escape_dot_label(_format_score(score_val))
        lines.append(
            f'  "{node_id}" [label="{label}" fillcolor="{color}" shape="{shape}" penwidth="1.2"];'
        )

    for edge in edges:
        if edge.parent in nodes and edge.child in nodes:
            child_node = nodes[edge.child]
            if child_node.get("modifies_code", False):
                lines.append(
                    f'  "{edge.parent}" -> "{edge.child}" [color="#d62728" penwidth=2.0];'
                )
            else:
                lines.append(f'  "{edge.parent}" -> "{edge.child}" [color="#999999"];')

    lines.append("}")
    return "\n".join(lines)


def maybe_render(dot_path: Path, out_path: Path, fmt: str) -> None:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        print("Graphviz 'dot' not found. Wrote DOT only:", dot_path)
        return
    subprocess.run([dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out_path)], check=True)
    print(f"Rendered: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize search-tree JSON with score labels.")
    parser.add_argument("json_file", type=Path, help="Path to debug tree JSON file")
    parser.add_argument(
        "--score-field",
        default="merged_value",
        choices=["merged_value", "value"],
        help="Node field used for score labels",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path for rendered image (png/svg/pdf)",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "svg", "pdf"],
        help="Render format if Graphviz is installed",
    )
    parser.add_argument(
        "--dot",
        type=Path,
        default=None,
        help="Optional DOT output path (default: <json_stem>.dot)",
    )
    parser.add_argument(
        "--hide-unseen",
        action="store_true",
        default=True,
        help="Hide nodes where visible=false (enabled by default)",
    )
    parser.add_argument(
        "--show-unseen",
        action="store_true",
        help="Include nodes where visible=false",
    )
    parser.add_argument(
        "--include-command",
        action="store_true",
        help="Include compact command text in node labels",
    )
    parser.add_argument(
        "--color-by",
        default="state",
        choices=["state", "score"],
        help="Color nodes by commit state (default) or by score",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    json_path: Path = args.json_file
    if not json_path.exists():
        print(f"Input file does not exist: {json_path}", file=sys.stderr)
        return 1

    try:
        nodes, edges, root_id = load_tree(json_path)
        hide_unseen = args.hide_unseen and not args.show_unseen
        nodes, edges = filter_visible(nodes, edges, hide_unseen)
        if root_id not in nodes:
            # If root got filtered out (e.g., invisible root), keep it for consistency.
            nodes_all, _, _ = load_tree(json_path)
            nodes[root_id] = nodes_all[root_id]

        dot_text = build_dot(
            nodes=nodes,
            edges=edges,
            root_id=root_id,
            score_field=args.score_field,
            include_command=args.include_command,
            color_by=args.color_by,
        )
    except Exception as exc:
        print(f"Failed to process JSON: {exc}", file=sys.stderr)
        return 2

    dot_path = args.dot or json_path.with_suffix(".dot")
    dot_path.write_text(dot_text, encoding="utf-8")
    print(f"DOT written: {dot_path}")

    if args.output is not None:
        try:
            maybe_render(dot_path, args.output, args.format)
        except subprocess.CalledProcessError as exc:
            print(f"Graphviz rendering failed: {exc}", file=sys.stderr)
            return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
