import json

def dfs_print_last_action_code(node, depth=0):
    if node["visible"] == False:
        return
    indent = "  " * depth
    last_action = node.get("last_action", {})
    if last_action is not None:
        code = last_action.get("command")
        if code is not None:
            code = code[0:50].replace("\n", "\\n") + ("..." if len(code) > 50 else "")
        else:
            code = "<None>"
        if code is not None:
            if node['modifies_code']:
                print(f"{indent}{code} *-> {node['value']:.4f} ({node['merged_value']:.4f})")
            else:
                print(f"{indent}{code} -> {node['value']:.4f} ({node['merged_value']:.4f})")
            
        depth = depth + 1
    
    for child in node.get("children", []):
        dfs_print_last_action_code(child, depth)

# Load the single root node JSON (not a list)
# with open('tree.json', 'r', encoding="utf-8") as f:
#     root_node = json.load(f)

file_path = "debug_tree.json"

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
