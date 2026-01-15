from minisweagent.agents.tree_search_node import TreeSearchNode
from typing import List, Any, Optional
from tqdm import tqdm
import random

class ActionProcessor:
    merge_strategy = "sum" # or "max"
    
    @classmethod
    def configure(cls, merge_strategy):
        cls.merge_strategy = merge_strategy
        
    @classmethod
    def convert_action_to_nodes(cls, actions: List[str], curr_node):
        """Get unique nodes from the current observation and actions."""
        if len(curr_node.children) > 0 and len(actions) == 0:
            # checkpoint
            pass
        else:
            # Generate children nodes
            for formatted_action in actions:
                new_node = TreeSearchNode(
                    last_action=formatted_action,
                )
                curr_node.add_child(new_node)
        
        tree_nodes = []
        for node in curr_node.children:
            tree_nodes.append(node)
                
        return tree_nodes
    
    @classmethod
    def evaluate_nodes(cls, node_list, goal):
        for new_node in tqdm(node_list, desc="Evaluating nodes"):
            if new_node.value is None:
                # A random number from 0 to 1 for now; replace with proper evaluation later
                new_node.value = random.random() 
    
    @classmethod
    def merge_nodes(cls, node_list: List[tuple[float, TreeSearchNode]]) -> List[tuple[float, TreeSearchNode]]:
        if cls.merge_strategy == "none":
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
                    
                if cls.merge_strategy == "sum":
                    new_score = existing_score + score
                elif cls.merge_strategy == "max":
                    new_score = max(existing_score, score)
                node_dict[command] = (new_score, best_node)
                
        merged_node_list = list(node_dict.values())
        return merged_node_list
            