#!/usr/bin/env python3
"""Render a search tree JSON into a Graphviz tree with score labels or interactive HTML.

Supports two common formats:
1) Nested tree (children are full child objects).
2) Flat node list (children are child ids).

Output formats:
- PNG/SVG/PDF: Static images via Graphviz
- HTML: Interactive Cytoscape.js visualization (no dependencies required to view)

Examples:
  python visualize_tree.py debug_tree.json
  python visualize_tree.py debug_tree.json --format png --output tree.png
  python visualize_tree.py debug_tree.json --format html --output tree.html
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


def _with_alpha(hex_color: str, alpha: float) -> str:
    """Return #RRGGBBAA for Graphviz-compatible color alpha."""
    if not isinstance(hex_color, str) or not hex_color.startswith("#") or len(hex_color) < 7:
        return hex_color
    a = max(0.0, min(1.0, float(alpha)))
    return f"{hex_color[:7]}{int(round(a * 255)):02x}"


def _state_key(node: Dict[str, Any]) -> str:
    # branch = node.get("branch") or "<none>"
    commit = node.get("state_hash") or node.get("commit") or "<none>"
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
        "#008080",  # teal
        "#e6beff",  # lavender
        "#9a6324",  # brown
        "#fffac8",  # light yellow
        "#800000",  # maroon
        "#aaffc3",  # mint
        "#808000",  # olive
    ]
    states = sorted({_state_key(n) for n in nodes.values()})
    return {state: palette[i % len(palette)] for i, state in enumerate(states)}


def _none_commit(node: Dict[str, Any]) -> bool:
    return node.get("commit") is None


def _format_score(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def parse_nested_tree(root: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[Edge], str]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Edge] = []
    visited: set[str] = set()

    def dfs(node: Dict[str, Any]) -> None:
        node_id = node.get("id")
        if not node_id:
            return
        
        # If we've already processed this node, don't traverse its children again
        # (handles DAG case where multiple parents can point to same child)
        if node_id in visited:
            return
        
        if node.get("value") is not None:
            visited.add(node_id)
            
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
        return parse_nested_tree(data["children"][0])
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

    kept = {node_id: n for node_id, n in nodes.items() if n.get("visible", False)}
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
    lines.append('  node [shape=circle style="filled" fontname="Helvetica" fontsize=14 width=0.65 fixedsize=true];')
    lines.append('  edge [color="#999999"];')

    state_colors = _build_state_color_map(nodes) if color_by == "state" else {}

    
    for node_id, node in nodes.items():
        score_val = node.get("merged_value") if node.get("is_terminating", False) else node.get(score_field)
        order_val = node.get("order")
        is_terminating = bool(node.get("is_terminating", False))
        pass_val = node.get("pass")
        is_cached = node.get("cache_hit") is not None
        is_executed = node.get("executed", False)
        # if not node.get("executed", False):
        #     color = "#bdbdbd"
        if color_by == "state":
            color = state_colors[_state_key(node)]
        else:
            color = _score_to_color(score_val if isinstance(score_val, (int, float)) else None)
        
        # Lighten color for non-executed nodes
        if not is_executed:
            # Convert hex to RGB, make lighter, convert back
            rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            lighter = tuple(int(c + (255 - c) * 0.0) for c in rgb)
            color = f"#{lighter[0]:02x}{lighter[1]:02x}{lighter[2]:02x}"

        # Highlight terminating node validity when pass is set.
        penwidth = "1.2"
        border_color = "#333333"
        border_style = "solid"
        if is_terminating and pass_val is True:
            border_color = "#1b8f3a"
            penwidth = "3.0"
        elif is_terminating and pass_val is False:
            border_color = "#c62828"
            penwidth = "3.0"

        # Fade borders for non-executed nodes to match execution emphasis.
        if not is_executed:
            border_color = _with_alpha(border_color, 0.5)

        # Fade label text for non-executed nodes.
        font_color = _with_alpha("#111111", 0.55) if not is_executed else "#111111"
        
        # Make cached nodes visually distinct with dashed and thicker border
        if is_cached:
            border_style = "filled,dashed"
            # Graphviz does not expose a direct dash-length control for node borders,
            # so a thicker pen makes the dash and gap pattern much more noticeable.
            penwidth = str(max(float(penwidth), 2.0))
        else:
            border_style = "filled"

        shape = "doublecircle" if (node_id == root_id or node.get("is_submission", False)) else "circle"
        score_str = _format_score(score_val)
        # Include order in label if available
        term_prefix = ""
        # if is_terminating and pass_val is True:
        #     term_prefix = "VALID\\n"
        # elif is_terminating and pass_val is False:
        #     term_prefix = "INVALID\\n"

        if order_val != 0 and order_val is not None:
            label = _escape_dot_label(f"{term_prefix}#{order_val}\n{score_str}")
        else:
            label = _escape_dot_label(f"{term_prefix}{score_str}")
        lines.append(
            f'  "{node_id}" [label="{label}" fillcolor="{color}" shape="{shape}" color="{border_color}" penwidth="{penwidth}" style="{border_style}" fontcolor="{font_color}"];'
        )

    for edge in edges:
        if edge.parent in nodes and edge.child in nodes:
            child_node = nodes[edge.child]
            real_child = (child_node.get("parent") is None or child_node.get("parent") == nodes[edge.parent]["id"])
            is_child_executed = child_node.get("executed", False)
            
            # Determine edge color based on child node type
            if child_node.get("modifies_code", False):
                color = "#d62728"
            elif child_node.get("is_terminating", False):
                color = "#2ca02c"
            else:
                color = "#000000"
            
            # Lighten color for non-executed edges
            if not is_child_executed:
                rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
                lighter = tuple(int(c + (255 - c) * 0.0) for c in rgb)
                color = f"#{lighter[0]:02x}{lighter[1]:02x}{lighter[2]:02x}"
            
            if real_child:
                lines.append(
                    f'  "{edge.parent}" -> "{edge.child}" [color="{color}" penwidth=2.0];'
                )
            else: # dashed
                lines.append(
                    f'  "{edge.parent}" -> "{edge.child}" [color="{color}" penwidth=2.0 style=dashed];'
                )

    lines.append("}")
    return "\n".join(lines)


def maybe_render(dot_path: Path, out_path: Path, fmt: str) -> None:
    dot_bin = shutil.which("dot")
    if not dot_bin:
        print("Graphviz 'dot' not found. Wrote DOT only:", dot_path)
        return
    subprocess.run([dot_bin, f"-T{fmt}", str(dot_path), "-o", str(out_path)], check=True)
    print(f"Rendered: {out_path}")


def build_cytoscape_html(
    nodes: Dict[str, Dict[str, Any]],
    edges: List[Edge],
    root_id: str,
    score_field: str,
    color_by: str,
) -> str:
    """Generate interactive HTML with Cytoscape.js visualization."""
    
    state_colors = _build_state_color_map(nodes) if color_by == "state" else {}
    
    # Build node elements
    cy_nodes = []
    for node_id, node in nodes.items():
        score_val = node.get("merged_value") if node.get("is_terminating", False) else node.get(score_field)
        order_val = node.get("order")
        is_terminating = bool(node.get("is_terminating", False))
        pass_val = node.get("pass")
        # if not node.get("executed", False):
        #     color = "#bdbdbd"
        if color_by == "state":
            color = state_colors[_state_key(node)]
        else:
            color = _score_to_color(score_val if isinstance(score_val, (int, float)) else None)
        
        score_str = _format_score(score_val)
        # Include order in label if available
        label = f"{score_str}\n#{order_val}" if order_val is not None else score_str
        if is_terminating and pass_val is True:
            label = f"VALID\n{label}"
        elif is_terminating and pass_val is False:
            label = f"INVALID\n{label}"
        is_root = node_id == root_id or node.get("is_submission", False)

        border_color = "#333"
        border_width = "2" if is_root else "1"
        if is_terminating and pass_val is True:
            border_color = "#1b8f3a"
            border_width = "3"
        elif is_terminating and pass_val is False:
            border_color = "#c62828"
            border_width = "3"
        
        # Use dashed + thicker border for cached nodes so they are easy to spot
        is_cached = node.get("cache_hit") is not None
        is_executed = node.get("executed", False)
        cached_border_width = "4" if is_cached else border_width
        node_opacity = "0.5" if not is_executed else "1.0"
        border_opacity = "0.5" if not is_executed else "1.0"
        label_opacity = "0.55" if not is_executed else "1.0"
        
        cy_nodes.append({
            "data": {
                "id": node_id,
                "label": label,
                "commit": node.get("commit"),
                "is_root": is_root,
                "score": score_val,
                "order": order_val,
                "observation": node.get("observation", ""),
                "last_action": node.get("last_action", {}),
                "test_status": node.get("test_status", []),
                "modifies_code": node.get("modifies_code", False),
                "modified_files": node.get("modified_files", []),
                "read_files": node.get("read_files", []),
                "is_terminating": node.get("is_terminating", False),
                "pass": pass_val,
                "executed": node.get("executed", False),
                "visible": node.get("visible", True),
                "visits": node.get("visits", 0),
                "background-color": color,
                "border-width": cached_border_width,
                "border-color": border_color,
                "border-style": "solid",
                "border-opacity": border_opacity,
                "opacity": node_opacity,
                "label-opacity": label_opacity,
                "is_cached": is_cached,
            },
        })
    
    # Build edge elements
    cy_edges = []
    for edge in edges:
        if edge.parent in nodes and edge.child in nodes:
            child_node = nodes[edge.child]
            is_child_executed = child_node.get("executed", False)
            edge_opacity = "1.0" if is_child_executed else "0.5"
            
            # Determine edge color and width based on child node type
            if child_node.get("modifies_code", False):
                line_color = "#d62728"
                width = "2.5"
            elif child_node.get("is_terminating", False):
                line_color = "#2ca02c"
                width = "2.5"
            else:
                line_color = "#000000"
                width = "1.5"
            
            cy_edges.append({
                "data": {
                    "id": f"{edge.parent}-{edge.child}",
                    "source": edge.parent,
                    "target": edge.child,
                },
                "style": {
                    "line-color": line_color,
                    "width": width,
                    "opacity": edge_opacity,
                }
            })
    
    elements = cy_nodes + cy_edges
    elements_json = json.dumps(elements)
    
    # Build HTML with proper escaping
    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Search Tree Visualization</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.24.0/cytoscape.min.js"></script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f5f5f5;
        }}
        
        .container {{
            display: flex;
            width: 100%;
            height: 100%;
        }}
        
        #cy {{
            flex: 1;
            background: white;
        }}
        
        #info {{
            width: 300px;
            padding: 15px;
            box-sizing: border-box;
            background: #f9f9f9;
            border-left: 1px solid #ddd;
            overflow-y: auto;
            font-size: 13px;
        }}
        
        .info-section {{
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }}
        
        .info-section:last-child {{
            border-bottom: none;
        }}
        
        .info-label {{
            font-weight: bold;
            color: #333;
            display: block;
            margin-bottom: 5px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #666;
        }}
        
        .info-value {{
            color: #333;
            margin-left: 0;
            word-break: break-word;
            white-space: pre-wrap;
            line-height: 1.4;
        }}
        
        .info-value code {{
            background: #e8e8e8;
            padding: 2px 4px;
            border-radius: 2px;
            font-family: 'Courier New', monospace;
            font-size: 11px;
        }}
        
        .info-title {{
            font-size: 14px;
            font-weight: bold;
            color: #007bff;
            margin-bottom: 10px;
            text-transform: capitalize;
        }}
        
        .controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: white;
            padding: 10px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            z-index: 10;
        }}
        
        button {{
            padding: 6px 12px;
            margin-right: 5px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 3px;
            cursor: pointer;
            font-size: 12px;
        }}
        
        button:hover {{
            background: #0056b3;
        }}
        
        .empty-state {{
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100%;
            color: #999;
            text-align: center;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="controls">
        <button onclick="fitView()">Fit View</button>
        <button onclick="resetView()">Clear</button>
    </div>
    <div class="container">
        <div id="cy"></div>
        <div id="info">
            <div class="empty-state">
                <div>Click on a node or edge to see details</div>
            </div>
        </div>
    </div>

    <script>
        const elements = {elements_json};
        
        function parseObservation(obsStr) {{
            if (!obsStr) return {{ returncode: 'N/A', output: 'N/A', warning: '' }};
            
            const returncodeMatch = obsStr.match(/<returncode>(.*?)<\/returncode>/s);
            const outputMatch = obsStr.match(/<output>([\s\S]*?)<\/output>/);
            const warningMatch = obsStr.match(/<warning>([\s\S]*?)<\/warning>/);
            
            return {{
                returncode: returncodeMatch ? returncodeMatch[1] : 'N/A',
                output: outputMatch ? outputMatch[1] : obsStr,
                warning: warningMatch ? warningMatch[1] : ''
            }};
        }}
        
        function formatCommand(cmd) {{
            if (!cmd) return 'N/A';
            if (typeof cmd === 'object') return JSON.stringify(cmd);
            return cmd.substring(0, 200) + (cmd.length > 200 ? '...' : '');
        }}
        
        function formatThought(thought) {{
            if (!thought) return '';
            // Extract just the THOUGHT line, not the full block
            const lines = thought.split('\\n');
            const thoughtLine = lines.find(l => l.startsWith('THOUGHT:'));
            if (thoughtLine) {{
                return thoughtLine.substring(8).trim().substring(0, 300) + (thoughtLine.length > 308 ? '...' : '');
            }}
            return thought.substring(0, 300) + (thought.length > 300 ? '...' : '');
        }}
        
        const cy = cytoscape({{
            container: document.getElementById('cy'),
            elements: elements,
            style: [
                {{
                    selector: 'node',
                    style: {{
                        'label': 'data(label)',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'width': '40px',
                        'height': '40px',
                        'font-size': '10px',
                        'background-color': 'data(background-color)',
                        'border-width': 'data(border-width)',
                        'border-color': 'data(border-color)',
                        'border-style': 'data(border-style)',
                        'border-opacity': 'data(border-opacity)',
                        'opacity': 'data(opacity)',
                        'text-opacity': 'data(label-opacity)',
                    }}
                }},
                {{
                    selector: 'node[is_cached]',
                    style: {{
                        'border-style': 'dashed',
                        'border-dash-pattern': [14, 10],
                        'border-width': '4px',
                    }}
                }},
                {{
                    selector: 'node:selected',
                    style: {{
                        'border-width': '3px',
                        'border-color': '#ff6600',
                    }}
                }},
                {{
                    selector: 'edge',
                    style: {{
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'line-color': 'data(line-color)',
                        'target-arrow-color': 'data(line-color)',
                        'width': 'data(width)',
                        'arrow-scale': 1.2,
                    }}
                }},
                {{
                    selector: 'edge:selected',
                    style: {{
                        'line-color': '#ff6600',
                        'target-arrow-color': '#ff6600',
                        'width': '4px',
                    }}
                }}
            ],
            layout: {{
                name: 'breadthfirst',
                directed: true,
                roots: '[id = "{root_id}"]',
                spacingFactor: 1.5,
                avoidOverlap: true,
                avoidOverlapPadding: 20
            }}
        }});
        
        cy.on('tap', 'node', function(event) {{
            cy.elements().unselect();
            const node = event.target;
            node.select();
            const data = node.data();
            const info = document.getElementById('info');
            
            const obs = parseObservation(data.observation);
            const lastAction = data.last_action || {{}};
            const testStatus = data.test_status || [];
            const testStr = testStatus.map(t => t.status === 'PASSED' ? '✓ ' + t.name : '✗ ' + t.name).join('\\n');
            
            let html = '<div class="info-title">Node</div>';
            html += '<div class="info-section"><span class="info-label">ID</span><span class="info-value"><code>' + data.id.substring(0, 12) + '</code></span></div>';
            html += '<div class="info-section"><span class="info-label">Score</span><span class="info-value">' + (data.score !== null ? data.score.toFixed(3) : 'N/A') + '</span></div>';
            if (data.order !== undefined && data.order !== null) {{
                html += '<div class="info-section"><span class="info-label">Expansion Order</span><span class="info-value">#' + data.order + '</span></div>';
            }}
            html += '<div class="info-section"><span class="info-label">Status</span><span class="info-value">' + (data.executed ? 'Executed' : 'Not executed') + ' | Visits: ' + data.visits + '</span></div>';
            if (data.is_terminating && data.pass !== null && data.pass !== undefined) {{
                html += '<div class="info-section"><span class="info-label">Terminating Validity</span><span class="info-value">' + (data.pass ? 'VALID' : 'INVALID') + '</span></div>';
            }}
            
            if (data.modifies_code) {{
                html += '<div class="info-section"><span class="info-label">Modified Files</span><span class="info-value"><code>' + (data.modified_files.join(', ') || 'N/A') + '</code></span></div>';
            }}
            
            if (data.read_files.length > 0) {{
                html += '<div class="info-section"><span class="info-label">Read Files</span><span class="info-value"><code>' + data.read_files.join(', ') + '</code></span></div>';
            }}
            
            html += '<div class="info-section"><span class="info-label">Command</span><span class="info-value"><code>' + formatCommand(lastAction.command) + '</code></span></div>';
            
            if (lastAction.thought) {{
                html += '<div class="info-section"><span class="info-label">Thought</span><span class="info-value"><code>' + formatThought(lastAction.thought) + '</code></span></div>';
            }}
            
            const obsOutput = obs.output.substring(0, 400) + (obs.output.length > 400 ? '...' : '');
            html += '<div class="info-section"><span class="info-label">Observation</span><span class="info-value">Return Code: <code>' + obs.returncode + '</code><br/><code>' + obsOutput + '</code>';
            if (obs.warning) {{
                html += '<br/>Warning: <code>' + obs.warning.substring(0, 200) + '</code>';
            }}
            html += '</span></div>';
            
            if (testStr) {{
                html += '<div class="info-section"><span class="info-label">Tests</span><span class="info-value"><code>' + testStr + '</code></span></div>';
            }}
            
            info.innerHTML = html;
        }});
        
        cy.on('tap', 'edge', function(event) {{
            cy.elements().unselect();
            const edge = event.target;
            edge.select();
            const edgeData = edge.data();
            const sourceNode = cy.getElementById(edgeData.source);
            const sourceData = sourceNode.data();
            const info = document.getElementById('info');
            
            const lastAction = sourceData.last_action || {{}};
            let actionType = 'Unknown';
            if (lastAction.COMMAND_TYPE) {{
                actionType = lastAction.COMMAND_TYPE;
            }} else if (sourceData.modifies_code) {{
                actionType = 'Code Modification';
            }} else if (sourceData.is_terminating) {{
                actionType = 'Terminating Action';
            }}
            
            const obs = parseObservation(sourceData.observation);
            
            let html = '<div class="info-title">Edge / Action</div>';
            html += '<div class="info-section"><span class="info-label">From Node</span><span class="info-value"><code>' + edgeData.source.substring(0, 12) + '</code></span></div>';
            html += '<div class="info-section"><span class="info-label">To Node</span><span class="info-value"><code>' + edgeData.target.substring(0, 12) + '</code></span></div>';
            html += '<div class="info-section"><span class="info-label">Action Type</span><span class="info-value">' + actionType + '</span></div>';
            html += '<div class="info-section"><span class="info-label">Command</span><span class="info-value"><code>' + formatCommand(lastAction.command) + '</code></span></div>';
            
            if (lastAction.thought) {{
                html += '<div class="info-section"><span class="info-label">Reasoning</span><span class="info-value"><code>' + formatThought(lastAction.thought) + '</code></span></div>';
            }}
            
            const resultOutput = obs.output.substring(0, 300) + (obs.output.length > 300 ? '...' : '');
            html += '<div class="info-section"><span class="info-label">Result</span><span class="info-value">Return Code: <code>' + obs.returncode + '</code><br/>Output: <code>' + resultOutput + '</code></span></div>';
            
            if (sourceData.modifies_code) {{
                html += '<div class="info-section"><span class="info-label">Code Changes</span><span class="info-value"><code>' + sourceData.modified_files.join(', ') + '</code></span></div>';
            }}
            
            info.innerHTML = html;
        }});
        
        cy.on('tap', function(event) {{
            if (event.target === cy) {{
                cy.elements().unselect();
                document.getElementById('info').innerHTML = `
                    <div class="empty-state">
                        <div>Click on a node or edge to see details</div>
                    </div>
                `;
            }}
        }});
        
        function fitView() {{
            cy.fit();
        }}
        
        function resetView() {{
            cy.elements().unselect();
            document.getElementById('info').innerHTML = `
                <div class="empty-state">
                    <div>Click on a node or edge to see details</div>
                </div>
            `;
        }}
        
        // Auto-fit on load
        cy.ready(function() {{
            cy.fit();
        }});
    </script>
</body>
</html>"""
    
    return html


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize search-tree JSON with score labels.")
    parser.add_argument("json_file", type=Path, help="Path to debug tree JSON file or directory containing tree JSONs")
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
        help="Output file path for rendered image (png/svg/pdf) or HTML (ignored if input is directory)",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "svg", "pdf", "html"],
        help="Render format: png/svg/pdf (requires Graphviz) or html (interactive)",
    )
    parser.add_argument(
        "--dot",
        type=Path,
        default=None,
        help="Optional DOT output path (default: <json_stem>.dot, ignored if input is directory)",
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
    parser.add_argument(
        "--redo",
        action="store_true",
        help="Regenerate directory outputs even when the target already exists",
    )
    return parser.parse_args()


def _process_single_tree(
    json_path: Path,
    score_field: str,
    format: str,
    dot_path: Path | None,
    hide_unseen: bool,
    include_command: bool,
    color_by: str,
    output_path: Path | None = None,
) -> bool:
    """Process a single tree JSON file and render to output.
    
    Returns True on success, False on failure.
    """
    try:
        nodes, edges, root_id = load_tree(json_path)
        nodes, edges = filter_visible(nodes, edges, hide_unseen)
        if root_id not in nodes:
            nodes_all, _, _ = load_tree(json_path)
            nodes[root_id] = nodes_all[root_id]

        # Handle HTML output separately
        if format == "html":
            html_text = build_cytoscape_html(
                nodes=nodes,
                edges=edges,
                root_id=root_id,
                score_field=score_field,
                color_by=color_by,
            )
            html_path = output_path or json_path.with_suffix(".html")
            html_path.write_text(html_text, encoding="utf-8")
            print(f"  → {html_path}")
            return True

        # Handle traditional DOT/PNG/SVG/PDF output
        dot_text = build_dot(
            nodes=nodes,
            edges=edges,
            root_id=root_id,
            score_field=score_field,
            include_command=include_command,
            color_by=color_by,
        )
    except Exception as exc:
        print(f"  ✗ Failed to process: {exc}", file=sys.stderr)
        return False

    dot_out = dot_path or json_path.with_suffix(".dot")
    dot_out.write_text(dot_text, encoding="utf-8")

    if output_path is not None:
        try:
            maybe_render(dot_out, output_path, format)
        except subprocess.CalledProcessError as exc:
            print(f"  ✗ Graphviz rendering failed: {exc}", file=sys.stderr)
            return False
    else:
        print(f"  → {dot_out}")

    return True


def main() -> int:
    args = parse_args()
    input_path: Path = args.json_file
    
    if not input_path.exists():
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        return 1

    hide_unseen = args.hide_unseen and not args.show_unseen
    
    # Handle directory input: find all *.tree.json files recursively
    if input_path.is_dir():
        tree_files = sorted(input_path.rglob("*.tree.json"))
        if not tree_files:
            print(f"No .tree.json files found in {input_path}", file=sys.stderr)
            return 1
        
        print(f"Found {len(tree_files)} tree files. Processing...")
        success_count = 0
        
        for tree_file in tree_files:
            instance_dir = tree_file.parent
            # Generate output in the same directory as the tree file
            output_path = instance_dir / f"tree.{args.format}"
            
            rel_path = tree_file.relative_to(input_path)
            if output_path.exists() and not args.redo:
                print(f"  {rel_path} (skipping existing {output_path.name}; use --redo to overwrite)")
                continue

            print(f"  {rel_path}")
            if _process_single_tree(
                json_path=tree_file,
                score_field=args.score_field,
                format=args.format,
                dot_path=None,  # Auto-generate DOT path in same directory
                hide_unseen=hide_unseen,
                include_command=args.include_command,
                color_by=args.color_by,
                output_path=output_path,  # Always render to this path
            ):
                success_count += 1
        
        print(f"\nProcessed {success_count}/{len(tree_files)} tree files successfully.")
        return 0 if success_count > 0 else 1
    
    # Handle single file input
    else:
        if not _process_single_tree(
            json_path=input_path,
            score_field=args.score_field,
            format=args.format,
            dot_path=args.dot,
            hide_unseen=hide_unseen,
            include_command=args.include_command,
            color_by=args.color_by,
            output_path=args.output,
        ):
            return 2


if __name__ == "__main__":
    raise SystemExit(main())
