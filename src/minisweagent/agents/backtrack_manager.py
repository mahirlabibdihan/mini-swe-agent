from minisweagent.agents.tree_search_node import TreeSearchNode
from minisweagent import Environment

class BacktrackManager:
    @classmethod
    def backtrack(cls, start_node: TreeSearchNode, end_node: TreeSearchNode, env: Environment):
        print(f">> Backtracking from {start_node.branch}:{start_node.commit} to {end_node.parent.branch}:{end_node.parent.commit}")
        # env.execute(f"git checkout {end_node.parent.branch}")
        env.execute(f"git checkout {end_node.parent.commit}")