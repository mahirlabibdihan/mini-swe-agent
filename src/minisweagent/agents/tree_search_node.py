import uuid

class TreeSearchNode:
    def __init__(self, last_action):
        self.id = str(uuid.uuid4())
        self.parent = None
        self.children = []
        self.value = None
        self.commit = None
        self.branch = None
        self.observation = None
        self.last_action = last_action
        self.visible = True
        self.level = 0
        self.has_write_child = False
        
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