import json
import sys

def dfs_print_last_action_code(node, depth=0):
    if not node.get("visible", True):
        return

    indent = "  " * depth
    last_action = node.get("last_action", {})
    code = "<None>"
    if last_action:
        command = last_action.get("command")
        if command is not None:
            code = command[:50].replace("\n", "\\n") + ("..." if len(command) > 50 else "")

    # Safely handle None values for value and merged_value
    value_str = f"{node['value']:.4f}" if node.get('value') is not None else "None"
    merged_value_str = f"{node['merged_value']:.4f}" if node.get('merged_value') is not None else "None"

    if node.get('modifies_code', False):
        print(f"{indent}{code} *-> {value_str} ({merged_value_str})")
    else:
        print(f"{indent}{code} -> {value_str} ({merged_value_str})")

    # Traverse children
    for child in node.get("children", []):
        dfs_print_last_action_code(child, depth + 1)
        
def print_commit_tree(node, parent=None, depth=0):
    if node["branch"] is None:
        return
    indent = "  " * depth
    if parent is None or parent['commit'] != node['commit'] or parent['branch'] != node['branch']:
        print(f"{indent}[{node['branch']}:{node['commit'][:7]}]")
        depth += 1
    for child in node.get("children", []):
        print_commit_tree(child, node, depth)

# Load the single root node JSON (not a list)
# with open('tree.json', 'r', encoding="utf-8") as f:
#     root_node = json.load(f)

file_path = sys.argv[1] if len(sys.argv) > 1 else "debug_tree.json"

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"Loaded {len(content)} characters from {file_path}")

        if not content.strip():
            raise ValueError("File is empty!")

        root_node = json.loads(content)  # <-- this is the step that usually fails

    print("JSON parsed successfully!")
    
except json.JSONDecodeError as e:
    print(f"❌ JSONDecodeError at line {e.lineno}, column {e.colno}: {e.msg}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

# Start DFS from the root
dfs_print_last_action_code(root_node)

print_commit_tree(root_node)