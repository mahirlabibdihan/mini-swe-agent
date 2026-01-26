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
            node_dict[command] = (score, node)
        else:
            existing_score, existing_node = node_dict[command]
            if existing_score >= score:
                best_node = existing_node
                node.prune()
            else:
                best_node = node
                existing_node.prune()
                
            if merge_strategy == "sum":
                new_score = 0.7 * existing_score + 0.7 * score
            elif merge_strategy == "max":
                new_score = max(existing_score, score)
            node_dict[command] = (new_score, best_node)

    merged_node_list = list(node_dict.values())
    
    tree_nodes = []
    for score, node in merged_node_list:
        node.merged_value = score
        tree_nodes.append(node)
        
    return tree_nodes
        