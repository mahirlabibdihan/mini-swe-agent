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
    def evaluate_actions(cls, action_nodes, goal):
        action_list = []  # Array of (score, Node) pair
        for new_node in tqdm(action_nodes, desc="Evaluating nodes"):
            if new_node.value is None:
                # A random number from 0 to 1 for now; replace with proper evaluation later
                new_node.value = random.random() 
            action_list.append((new_node.value, new_node))
        return action_list
    
    @classmethod
    def merge_actions(cls, action_list: List[tuple[float, TreeSearchNode]]) -> List[tuple[float, TreeSearchNode]]:
        if cls.merge_strategy == "none":
            return action_list
        
        action_dict = {}
        for score, node in action_list:
            command = node.last_action["command"]
            if command not in action_dict:
                action_dict[command] = (score, node)
            else:
                existing_score, existing_node = action_dict[command]
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
                action_dict[command] = (new_score, best_node)
                
        merged_action_list = list(action_dict.values())
        return merged_action_list
            