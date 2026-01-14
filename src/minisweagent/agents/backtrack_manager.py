from minisweagent.agents.tree_search_node import TreeSearchNode
from minisweagent import Environment

class BacktrackManager:
    @classmethod
    def backtrack(cls, start_node: TreeSearchNode, end_node: TreeSearchNode, env: Environment):
        if start_node.commit == end_node.commit:
            return
        print(f">> Backtracking from {start_node.branch}:{start_node.commit} to {end_node.branch}:{end_node.commit}")
        # env.execute(f"git checkout {end_node.branch}")
        env.execute(f"git checkout {end_node.commit}")