import uuid
from minisweagent.utils.log import instance_logger

class TreeSearchNode:
    def __init__(self, last_action):
        self.id = str(uuid.uuid4())
        self.parent = None
        self.children = []
        self.value = None
        self.merged_value = None
        self.commit = None
        self.branch = None
        self.executed = False
        self.observation = None
        self.epsilon = None
        self.last_action = last_action
        self.visible = True
        self.level = 0
        self.order = 0
        self.itr = 0
        self.system_generated = False
        self.modifies_code = False
        self.modified_files = []
        self.merged = False
        self.read_files = []
        self.is_terminating = False
        self.is_submission = False
        self.invalid_termination = False
        self.visits = 0
        self.is_system_response = False
        self.raw_observation = None
        self.fails_tests = False
        self.n_history = 0
        self.changes = []
        self.diff_size = 0
        self.history_summary = None
        self.test_status = []
        self.raw_value = None
        # Detailed breakdown of how the node's score/value was calculated.
        # Filled by the agent's reward computation for debugging and analysis.
        self.score_calculation = None
        self._pass = None
        
    def __lt__(self, other):
        # based on frequency
        if self.last_action is None or self.last_action.get("command") is None:
            return True
        if other.last_action is None or other.last_action.get("command") is None:
            return False
        return self.visits < other.visits
        
    def get_path_value(self, discount_factor=1.0, max_steps=None):
        score = 0.0
        current = self
        f = 1.0
        norm = 0.0
        steps = 0

        while (
            current is not None
            and current.value is not None
            and (max_steps is None or steps < max_steps) # NEW: Limit the number of steps, so that deep nodes don't get too much advantage
        ):
            score += f * current.merged_value
            norm += f

            current = current.parent
            f *= discount_factor
            steps += 1

        return score / norm if norm > 0 else 0.0
    
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self
        if child_node.last_action is not None:
            child_node.level = self.level + 1
        
    def prune(self):
        self.visible = False
        
    def is_expanded(self):
        for child in self.children:
            if child.branch:
                return True
        return False
    
    def to_json(self):
        return [{
            "id": self.id,
            "raw_value": self.raw_value,
            "value": self.value,
            "merged_value": self.merged_value,
            "level": self.level,
            "commit": self.commit,
            "branch": self.branch,
            "executed": self.executed,
            "visible": self.visible,
            "visits": self.visits,
            "epsilon": self.epsilon,
            "modified_files": self.modified_files,
            "modifies_code": self.modifies_code,
            "diff_size": self.diff_size,
            "read_files": self.read_files,
            "last_action": {
                "command": self.last_action["command"],
                "thought": self.last_action["thought"],
            } if self.last_action else None,
            "test_status": self.test_status,
            "observation": self.observation,
            "children": [child.id for child in self.children],
        }, *[node for child in self.children for node in child.to_json()]]
    
    def from_tree(self, tree_data):
        self.id = tree_data["id"]
        self.value = tree_data["value"]
        self.merged_value = tree_data["merged_value"]
        self.raw_value = tree_data.get("raw_value", None)
        self.level = tree_data["level"]
        self.commit = tree_data["commit"]
        self.branch = tree_data["branch"]
        self.executed = tree_data["executed"]
        self.visible = tree_data["visible"]
        self.itr = tree_data["itr"]
        self.order = tree_data["order"]
        self.merged = tree_data["merged"]
        self.system_generated = tree_data["system_generated"]
        self.is_terminating = tree_data["is_terminating"]
        self.is_submission = tree_data["is_submission"]
        self.visits = tree_data["visits"]
        self.epsilon = tree_data.get("epsilon", None)
        self.modified_files = tree_data.get("modified_files", [])
        self.modifies_code = tree_data.get("modifies_code", False)
        self.diff_size = tree_data.get("diff_size", 0)
        self.read_files = tree_data.get("read_files", [])
        self.last_action = tree_data.get("last_action", None)
        self.test_status = tree_data.get("test_status", [])
        self.observation = tree_data.get("observation", None)
        self.history_summary = tree_data.get("history_summary", None)
        self.score_calculation = tree_data.get("score_calculation", None)
        self._pass = tree_data.get("pass", None)
        # Children will be linked in a separate step after all nodes are created.
        for child_data in tree_data.get("children", []):
            if child_data.get("executed") is None:
                child_node = TreeSearchNode(last_action=None)
                child_node.id = child_data.get("id")
            else:
                child_node = TreeSearchNode(last_action=child_data.get("last_action", None))
                child_node.from_tree(child_data)
            self.add_child(child_node)
            if child_data.get("executed") is None:
                child_node.parent = None
                child_node.merged = True
            
    def to_tree(self):
        try:
            response = {
                "id": self.id,
                "raw_value": self.raw_value,
                "value": self.value,
                "merged_value": self.merged_value,
                "level": self.level,
                "commit": self.commit,
                "branch": self.branch,
                "executed": self.executed,
                "visible": self.visible,
                "visits": self.visits,
                "order": self.order,
                "itr": self.itr,
                "merged": self.merged,
                "system_generated": self.system_generated,
                "is_terminating": self.is_terminating,
                "is_submission": self.is_submission,
                "epsilon": self.epsilon,
                "modified_files": self.modified_files,
                "modifies_code": self.modifies_code,
                "parent": self.parent.id if self.parent else None,
                "diff_size": self.diff_size,
                "read_files": self.read_files,
                "last_action": {
                    "command": self.last_action["command"],
                    "thought": self.last_action["thought"],
                    "type": self.last_action.get("type", "unknown"),
                } if self.last_action else None,
                "test_status": self.test_status,
                "history_summary": self.history_summary,
                "score_calculation": self.score_calculation,
                "children": [
                    child.to_tree() if not child.merged or (child.parent and child.parent.id == self.id) else {"id": child.id}
                    for child in self.children
                ],
                "observation": self.observation,
            }
            if self.is_terminating:
                response["pass"] = self._pass
        except Exception as e:
            instance_logger.debug(f">> Failed to convert node {self.id} to tree: {e}")
            response = None
        
        return response