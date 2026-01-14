from minisweagent.agents.action_queue_manager import ActionQueueManager
from minisweagent.agents.tree_search_node import TreeSearchNode


class ActionSelector:
    frontier_budget: int = 4
    max_depth: int = 20
    max_steps:int = 20
    
    def __init__(self):
        self.action_queue_manager = ActionQueueManager(frontier_budget=self.frontier_budget)
        self.n_prune = 0
        
    def reset(self):
        self.action_queue_manager.reset()
        self.n_prune = 0
        
    @classmethod
    def configure(cls, frontier_budget=4, max_depth=20):
        cls.frontier_budget = frontier_budget
        cls.max_depth = max_depth
        
    def add_actions(self, curr_node, actions, check_budget=True):
        if check_budget and self.action_queue_manager.is_out_of_budget():
            self.action_queue_manager.minimize()
            self.n_prune += 1
            print(f"Queue size {self.action_queue_manager.length()}. Tree pruned.")
        else:
            print(f"Queue size {self.action_queue_manager.length()}. Adding new actions...")
            
        for score, new_node in actions:
            if new_node.level >= self.max_depth:
                print(f"Non-terminating Node {new_node.last_action['code']} exceeded max depth {self.max_depth}, skipping...")
                new_node.prune()
                continue
            self.action_queue_manager.push(-score, new_node)
    
    def _make_terminating_action(self, curr_node):
        node = TreeSearchNode(
            last_action={
                "command": f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached",
                "thought": "THOUGHT: MAX STEPS REACHED ```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```",
                "extra": None
            },
        )
        node.value = 0.0
        curr_node.add_child(node)
        return node
    
    def _handle_max_steps(self, n_expanded, curr_node):
        if n_expanded < self.max_steps:
            return None
        return self._make_terminating_action(curr_node)
    
    def select_action(self, curr_node, n_expanded):
        # 1. Handle max-step pruning
        node = self._handle_max_steps(n_expanded, curr_node)
        if node: return node
        
        if self.action_queue_manager.length() > 0:
            neg_score, best_node = self.action_queue_manager.pop()
        else:
            best_node = self._make_terminating_action(curr_node)
            print("Action queue empty. Forcing terminating action.")
        return best_node