from minisweagent.agents.tree_search_node import TreeSearchNode
from typing import List, Any, Optional
from tqdm import tqdm
import random

def evaluate_nodes(node_list, goal):
    for new_node in tqdm(node_list, desc="Evaluating nodes"):
        if new_node.value is None:
            # A random number from 0 to 1 for now; replace with proper evaluation later
            new_node.value = random.random() 
            
def merge_nodes(node_list: List[tuple[float, TreeSearchNode]], merge_strategy = "sum") -> List[tuple[float, TreeSearchNode]]:
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
                new_score = existing_score + score
            elif merge_strategy == "max":
                new_score = max(existing_score, score)
            node_dict[command] = (new_score, best_node)
            
    merged_node_list = list(node_dict.values())
    return merged_node_list
        