import json
import ast
from pathlib import PurePosixPath


def parse_python_content(file_content: str):
    """
    Equivalent to parse_python_file(...), but works from in-memory content.
    """
    try:
        parsed_data = ast.parse(file_content)
    except Exception:
        return [], [], file_content.splitlines()

    class_info = []
    function_names = []
    class_methods = set()
    lines = file_content.splitlines()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": lines[n.lineno - 1 : n.end_lineno],
                        }
                    )
                    class_methods.add(n.name)

            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": lines[node.lineno - 1 : node.end_lineno],
                    "methods": methods,
                }
            )

        elif isinstance(node, ast.FunctionDef):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": lines[node.lineno - 1 : node.end_lineno],
                    }
                )

    return class_info, function_names, lines


def result_to_structure(result: str):
    """
    Convert env.execute JSONL output into the same structure as create_structure(),
    except WITHOUT an artificial repo root.
    """
    structure = {}

    for obj in result:
        path = PurePosixPath(obj["id"])

        # Remove the leading "path\\n"
        _, _, file_text = obj["content"].partition("\n")

        classes, functions, lines = parse_python_content(file_text)

        curr = structure

        # Build directory tree
        for part in path.parts[:-1]:
            curr = curr.setdefault(part, {})

        # Insert Python file payload
        curr[path.name] = {
            "classes": classes,
            "functions": functions,
            "text": lines,
        }

    return structure

def remove_redundancy(node: dict):
    """
    Remove redundant information from the structure to save space.
    For example, "text" under a filename includes function/class codes, also "functions", "classes" include the same code snippets, so we can truncate "text" to remove redundancy.
    """
    # Folder detection: dict keys are subnodes and are themselves dicts
    if all(isinstance(v, dict) for v in node.values()):
        # Folder: recurse into children
        for child in node.values():
            remove_redundancy(child)
        return
    
    # Class detection: has 'methods' key
    if "methods" in node:
        # Replace method bodies with placeholders
        lines = node["text"]
        class_offset = node["start_line"]
        for method_node in node.get("methods", []):
            start, end = method_node["start_line"], method_node["end_line"]
            lines[start - class_offset:end - class_offset + 1] = ["...."] * (end - (start - 1))
            lines[start - class_offset] = f"....<method:{method_node['name']}[{start}:{end}]>...."
        node["text"] = [line for line in lines if line != "...."] # Remove redundant lines which are fully replaced by placeholders
         
    # File detection: has 'classes' or 'functions' keys
    if "classes" in node or "functions" in node:
        lines = node["text"]

        # Replace class code with placeholders
        for class_node in node.get("classes", []):
            start, end = class_node["start_line"], class_node["end_line"]
            lines[start - 1:end] = ["...."] * (end - (start - 1))
            lines[start - 1] = f"....<class:{class_node['name']}[{start}:{end}]>...."

        # Replace function code with placeholders
        for func_node in node.get("functions", []):
            start, end = func_node["start_line"], func_node["end_line"]
            if lines[start - 1].strip().startswith("...."):
                continue
            lines[start - 1:end] = ["...."] * (end - (start - 1))
            lines[start - 1] = f"....<function:{func_node['name']}[{start}:{end}]>...."

        node["text"] = [line for line in lines if line != "...."] # Remove redundant lines which are fully replaced by placeholders

        # Recurse into classes/functions
        for class_node in node.get("classes", []):
            remove_redundancy(class_node)
        for func_node in node.get("functions", []):
            remove_redundancy(func_node)
        return

    # Leaf node (method/function/module-level code) → nothing to do
    return


class RepoNode:
    def __init__(
        self,
        name: str,
        node_type: str,
        text=None,
        start_line=None,
        end_line=None,
        parent=None,
    ):
        self.name = name
        self.type = node_type  # "folder", "file", "class", "function", "method"
        self.text = text or []  # list of lines
        self.start_line = start_line
        self.end_line = end_line

        self.parent = parent
        self.children = []

        self.self_score = 0.0
        self.score = 0.0  # propagated score

    # -------------------------
    # Tree utilities
    # -------------------------

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def is_leaf(self):
        return len(self.children) == 0

    def full_path(self):
        if self.parent is None or self.parent.name is None:
            return self.name
        return self.parent.full_path() + "/" + self.name
    
    def qualified_name(self):
        if self.type in ("folder", "file"):
            return self.full_path()

        # Walk upward to find file
        node = self
        parts = []
        file_path = None

        while node is not None:
            if node.type == "file":
                file_path = node.full_path()
                break
            parts.append(node.name)
            node = node.parent

        parts.reverse()

        if file_path:
            if parts:
                return file_path + "::" + ".".join(parts)
            return file_path

        return self.full_path()

    def __repr__(self):
        return f"<Node {self.type}:{self.full_path()} score={self.score}>"
    
    # -----------------------------
    # DFS tree print method
    # -----------------------------
    def print(self, indent=0, show_score=False):
        """
        Print the tree rooted at this node in DFS order.
        """
        prefix = "  " * indent
        score_str = f" [score={self.score:.2f}]" if show_score else ""
        print(f"{prefix}{self.type}: {self.name}{score_str}")

        for child in self.children:
            child.print(indent + 1, show_score=show_score)


def dict_to_tree(name, obj, parent=None):
    """
    Convert structure_opt dict into Node tree.
    """

    # -------------------------
    # 1️⃣ Folder detection
    # -------------------------
    if isinstance(obj, dict) and all(isinstance(v, dict) for v in obj.values()):
        node = RepoNode(name=name, node_type="folder", parent=parent)

        for child_name, child_obj in obj.items():
            child_node = dict_to_tree(child_name, child_obj, node)
            node.add_child(child_node)

        return node

    # -------------------------
    # 2️⃣ File detection
    # -------------------------
    if isinstance(obj, dict) and (
        "classes" in obj or "functions" in obj
    ):
        node = RepoNode(
            name=name,
            node_type="file",
            text=obj.get("text", []),
            parent=parent,
        )

        # Classes
        for cls in obj.get("classes", []):
            child = dict_to_tree(cls["name"], cls, node)
            node.add_child(child)

        # Functions
        for fn in obj.get("functions", []):
            child = dict_to_tree(fn["name"], fn, node)
            node.add_child(child)

        return node

    # -------------------------
    # 3️⃣ Class detection
    # -------------------------
    if isinstance(obj, dict) and "methods" in obj:
        node = RepoNode(
            name=name,
            node_type="class",
            text=obj.get("text", []),
            start_line=obj.get("start_line"),
            end_line=obj.get("end_line"),
            parent=parent,
        )

        for method in obj.get("methods", []):
            child = dict_to_tree(method["name"], method, node)
            node.add_child(child)

        return node

    # -------------------------
    # 4️⃣ Function / Method
    # -------------------------
    if isinstance(obj, dict):
        # If parent is class → this is a method
        if parent and parent.type == "class":
            node_type = "method"
        else:
            node_type = "function"

        return RepoNode(
            name=name,
            node_type=node_type,
            text=obj.get("text", []),
            start_line=obj.get("start_line"),
            end_line=obj.get("end_line"),
            parent=parent,
        )

    raise ValueError(f"Unknown structure for {name}")

def collect_rankable_nodes(root):
    """
    Return a flat list of all nodes with text for BM25 ranking.
    """
    nodes = []

    def dfs(node):
        if node.text:
            nodes.append(node)
        for child in node.children:
            dfs(child)

    dfs(root)
    return nodes

def collect_all_nodes(root):
    nodes = []

    def dfs(node):
        nodes.append(node)
        for child in node.children:
            dfs(child)

    dfs(root)
    return nodes

def propagate_scores(node):
    """
    Set node.score = max(self_score, max(children scores))
    """
    for child in node.children:
        propagate_scores(child)

    if node.children:
        node.score = max([node.self_score] + [c.score for c in node.children])
        # all_scores = [node.self_score] + [c.score for c in node.children]
        # node.score = sum(all_scores) / len(all_scores)
    else:
        node.score = node.self_score