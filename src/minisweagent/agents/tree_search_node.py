class TreeSearchNode:
    def __init__(self, last_action):
        self.parent = None
        self.children = []
        self.value = None
        self.commit = None
        self.branch = None
        self.observation = None
        self.last_action = last_action
        self.visible = True
        self.level = 0
        
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self
        child_node.level = self.level + 1
        
    def prune(self):
        self.visible = False