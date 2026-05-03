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
        
    def __lt__(self, other):
        # based on frequency
        if self.last_action is None or self.last_action.get("command") is None:
            return True
        if other.last_action is None or other.last_action.get("command") is None:
            return False
        return self.visits < other.visits
        
    def get_path_value(self, discount_factor=1.0, max_steps=5):
        score = 0.0
        current = self
        f = 1.0
        norm = 0.0
        steps = 0

        while (
            current is not None
            and current.value is not None
            # and steps < max_steps # NEW: Limit the number of steps, so that deep nodes don't get too much advantage
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
    
    def to_tree(self):
        try:
            response = {
                "id": self.id,
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
                "children": [child.to_tree() for child in self.children],
                "observation": self.observation,
            }
        except Exception as e:
            instance_logger.debug(f">> Failed to convert node {self.id} to tree: {e}")
            response = None
        
        return response