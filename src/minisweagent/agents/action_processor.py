from minisweagent.agents.tree_search_node import TreeSearchNode
from typing import List, Any, Optional
from tqdm import tqdm
import random
            
def merge_nodes(node_list: List[tuple[float, TreeSearchNode]], merge_strategy = "sum") -> List[TreeSearchNode]:
    if merge_strategy == "none":
        return node_list
    
    node_dict = {}
    for node in node_list:
        score = node.value
        command = node.last_action["command"]
        if command not in node_dict:
            node_dict[command] = (score, 1, node)
        else:
            existing_score, c,  existing_node = node_dict[command]
            if existing_score >= score:
                best_node = existing_node
                node.prune()
            else:
                best_node = node
                existing_node.prune()
                
            if merge_strategy == "sum":
                new_score = existing_score + score
                c += 1
            elif merge_strategy == "max":
                new_score = max(existing_score, score)
            node_dict[command] = (new_score, c, best_node)

    merged_node_list = list(node_dict.values())
    
    tree_nodes = []
    for score, c, node in merged_node_list:
        node.merged_value = score
        # Adjust value based on number of merges
        if merge_strategy == "sum":
            new_value = score * (2 / (1 + c))
            print(f">> Merged {c} nodes for action '{node.last_action['command']}'. Value adjusted: {score:.4f} -> {new_value:.4f}")
            node.merged_value = new_value
            
        tree_nodes.append(node)
        
    return tree_nodes
        