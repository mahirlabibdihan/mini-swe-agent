import heapq

class Frontier:
    def __init__(self, budget):
        self.initial_budget = budget
        self.budget = budget
        self.queue = []

    def reset(self):
        self.budget = self.initial_budget
        self.queue = []
            
    def clear(self):
        self.queue = []
        
    def reduce_budget(self):
        self.budget = max(1, self.budget - 1)
        
    def is_out_of_budget(self):
        return len(self.queue) > self.budget

    def push(self, priority, node):
        heapq.heappush(self.queue, (priority, node))
    
    def pop(self):
        return heapq.heappop(self.queue)
    
    def empty(self):
        return len(self.queue) == 0
    
    def length(self):
        return len(self.queue)
    
    def minimize(self):
        pass