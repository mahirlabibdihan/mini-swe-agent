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
            node_dict[command] = [node]
        else:
            node_dict[command].append(node)
            
    merged_node_list = list(node_dict.values())
    tree_nodes = []
    for nodes in merged_node_list:
        if len(nodes) > 1:
            # print(f">> Merging {len(nodes)} nodes for action '{nodes[0].last_action['command']}'")
            nodes.sort(key=lambda x: x.value, reverse=True)
            
            if merge_strategy == "sum":
                merged_score = nodes[0].value + (0.7 / len(nodes)) * sum([n.value for n in nodes[1:]])
                nodes[0].merged_value = merged_score
            elif merge_strategy == "avg":
                merged_score = sum([n.value for n in nodes]) / len(nodes)
                nodes[0].merged_value = merged_score 
            else:
                nodes[0].merged_value = nodes[0].value 
            
            for n in nodes[1:]:
                n.prune()
        else:
            nodes[0].merged_value = nodes[0].value
        
        tree_nodes.append(nodes[0])
    
    return tree_nodes
        