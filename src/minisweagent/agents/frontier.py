import heapq

class Frontier:
    def __init__(self, budget=None):
        self.initial_budget = budget
        self.budget = budget
        self.queue = []

    def reset(self):
        self.budget = self.initial_budget
        self.queue = []
            
    def clear(self):
        self.queue = []
        
    def reduce_budget(self):
        if self.budget is not None:
            self.budget = max(1, self.budget - 1)
        else:
            raise ValueError("Budget is not set. Cannot reduce budget.")
        
    def is_out_of_budget(self):
        return self.budget is not None and len(self.queue) > self.budget

    def push(self, priority, node):
        heapq.heappush(self.queue, (priority, node))
    
    def pop(self):
        return heapq.heappop(self.queue)
    
    def empty(self):
        return len(self.queue) == 0
    
    def length(self):
        return len(self.queue)
    
    def minimize(self):
        # path_score = []
                
        # for neg_score, node in self.queue:
        #     path_score.append((node.get_path_value(), neg_score, node))
        
        # top_paths = heapq.nlargest(self.budget, path_score, key=lambda x: x[0])
        
        # # Rebuild queue
        # self.queue = []
        # for path_score, neg_score, node in top_paths:
        #     self.push(neg_score, node)
        
        # Keep only the top 'budget' nodes
        self.queue = heapq.nsmallest(self.budget, self.queue)
        heapq.heapify(self.queue)