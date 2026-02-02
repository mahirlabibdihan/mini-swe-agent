import uuid

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
        self.modifies_code = False
        self.modified_files = []
        self.read_files = []
        self.is_terminating = False
        self.invalid_termination = False
        self.visits = 0
        self.is_system_response = False
        self.return_code = None
        self.n_history = 0
        
    def __lt__(self, other):
        # based on frequency
        if self.last_action is None or self.last_action.get("command") is None:
            return True
        if other.last_action is None or other.last_action.get("command") is None:
            return False
        return self.visits < other.visits
        
    def get_path_value(self, discount_factor=1.0):
        score = 0
        current = self
        f = 1.0
        norm = 0.0

        while current.value is not None:
            score += f * current.merged_value
            norm += f
            current = current.parent
            f *= discount_factor
        
        return score / norm if norm > 0 else 0
    
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self
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
            "read_files": self.read_files,
            "last_action": {
                "command": self.last_action["command"],
                "thought": self.last_action["thought"],
            } if self.last_action else None,
            "observation": self.observation,
            "children": [child.id for child in self.children],
        }, *[node for child in self.children for node in child.to_json()]]
    
    def to_tree(self):
        return {
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
            "read_files": self.read_files,
            "last_action": {
                "command": self.last_action["command"],
                "thought": self.last_action["thought"],
            } if self.last_action else None,
            "children": [child.to_tree() for child in self.children],
            "observation": self.observation,
        }