import heapq

class ActionQueueManager:
    def __init__(self, frontier_budget):
        self.initial_frontier_budget = frontier_budget
        self.frontier_budget = frontier_budget
        self.queue = []

    def reset(self):
        self.frontier_budget = self.initial_frontier_budget
        self.queue = []
            
    def clear(self):
        self.queue = []
        
    def reduce_budget(self):
        self.frontier_budget = max(1, self.frontier_budget - 1)
        
    def is_out_of_budget(self):
        return len(self.queue) > self.frontier_budget

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