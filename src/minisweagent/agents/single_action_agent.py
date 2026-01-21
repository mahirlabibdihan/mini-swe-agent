from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, TerminatingException, Submitted, ExecutionTimeoutError
from minisweagent.agents.tree_search_node import TreeSearchNode   
from minisweagent.agents.frontier import Frontier
from minisweagent.agents.action_analyzer import is_terminating
from typing import List, Any, Optional
from tabulate import tabulate
import time
import json

class NoActionFound(Exception):
    """Raised when the agent has reached its cost or step limit."""
    
class SingleActionAgentConfig(AgentConfig):
    depth_limit: int = 20
    """The maximum depth allowed for any node."""

class SingleActionAgent(DefaultAgent):
    def __init__(self, 
                 *args,
                 config_class=SingleActionAgentConfig, 
                 **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.tree_root = self.tree_node = None
        self.n_actions = 0
        self.n_explored = 0
        self.n_expanded = 0
        self.n_submissions = 0
        self.frontier = Frontier(budget=1)
        self.node_map = {}
    
    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        
        self._reset()
        
        while True:
            try:
                self.step()
            except NoActionFound as e:
                self.add_message("system", str(e))
            except NonTerminatingException as e:
                self.add_message("user", str(e))
                self.tree_node.observation = str(e)
            except TerminatingException as e:
                self.add_message("user", str(e))
                self.tree_node.observation = str(e)
                return type(e).__name__, str(e)
            
    def _handle_max_steps(self):
        if self.n_expanded < self.config.step_limit:
            return None
        return self._make_terminating_action(self.tree_node)
    
    def _make_terminating_action(self, curr_node):
        node = self._create_node(
            last_action={
                "command": f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached",
                "thought": "THOUGHT: MAX STEPS REACHED\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```",
                "extra": None
            },
        )
        node.value = 0.0
        node.is_terminating = True
        curr_node.add_child(node)
        return node
    
    def _add_actions_to_frontier(self, actions: List[TreeSearchNode]):
        for new_node in actions:
            if is_terminating(new_node.last_action):
                self.n_submissions += 1
            if new_node.level >= self.config.depth_limit:
                print(f"Non-terminating Node {new_node.last_action['command']} exceeded max depth {self.config.depth_limit}, skipping...")
                new_node.prune()
                continue
            self.frontier.push(-new_node.merged_value, new_node)
            
    def _select_action(self):
        # 1. Handle max-step pruning
        node = self._handle_max_steps()
        if node: return node
        
        if self.frontier.length() > 0:
            neg_score, best_node = self.frontier.pop()
        else:
            best_node = self._make_terminating_action(self.tree_node)
            print("Action queue empty. Forcing terminating action.")
        
        return best_node
    
    def _create_node(self, last_action: dict = None) -> TreeSearchNode:
        node = TreeSearchNode(
            last_action=last_action,
        )
        self.node_map[node.id] = node
        return node
    def _reset(self):
        self.frontier.reset()
        self.tree_root = self.tree_node = self._create_node()        
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
        self.tree_node.observation = self.render_template(self.config.instance_template)
       
    def _generate_new_nodes(self, n_actions) -> List[TreeSearchNode]:
        nodes = []
        # flag = True
        for i in range(n_actions):
            # Execute action to get observation
            try:
                response = self.query()
                action = self.parse_action(response)
                print(f"Generated action #{i+1}: {action['action']}")
                # Convert action to node
                new_node = self._create_node(
                    last_action={
                        "command": action["action"],
                        "thought": action["content"],
                        "extra": action["extra"]
                    },
                )
            except NonTerminatingException as e:
                observation = str(e)
                # Convert action to node
                new_node = self._create_node(
                    last_action={
                        "command": None,
                        "thought": response["content"],
                        "extra": response["extra"]
                    },
                )
                new_node.observation = observation
                print(f">> Invalid Response: {response["content"]}")
                
            time.sleep(2)  # To avoid rate limiting
            
            nodes.append(new_node)
        
        return nodes
              
    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        tree_nodes = self._generate_new_nodes(1)
        tree_nodes = self._update_tree(tree_nodes)
        self._update_frontier(tree_nodes)
        best_node = self._select_action()
        self.tree_node = best_node
        
        self.frontier.reset()
         
        if self.tree_node.last_action["extra"]:
            self.add_message("assistant", **{"content": self.tree_node.last_action["thought"], "extra": self.tree_node.last_action.get("extra", {})})
        else: # Action generated by System
            self.add_message("system", self.tree_node.last_action["thought"])
            
        print(f">> Executing selected action: {self.tree_node.last_action['command']}")
        if self.tree_node.last_action["command"] is None and self.tree_node.observation is not None: # For invalid action, no need to re-execute
            observation = self.tree_node.observation
        else:
            output = self.get_observation(
                {
                    "action": self.tree_node.last_action["command"]
                }
            )
            observation = self.render_template(self.config.action_observation_template, output=output)
        self.n_expanded += 1
        
        self.add_message("user", observation)
        self.tree_node.observation = observation
        self.tree_node.executed = True
        return self.tree_node.observation
    
    def get_observation(self, action: dict) -> dict:
        """Execute the action and return the observation."""
        output = self.execute_action(action)
        return output
    
    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.n_expanded or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        
        messages = []
        curr = self.tree_node
        while curr is not None:
            if curr.observation is not None:
                messages.append(
                    {
                        "role": "user", 
                        "content": curr.observation, 
                    }
                )
            if curr.last_action is not None:
                messages.append(
                    {
                        "role": "assistant",
                        "content": curr.last_action["thought"],
                    }
                )   
            curr = curr.parent
        
        messages.append({
            "role": "system",
            "content": self.render_template(self.config.system_template),
        })
        messages.reverse()
        
        # save to file for debugging
        with open("debug_messages.json", "w") as f:
            json.dump(messages, f, indent=4)
            
        response = self.model.query(messages)
        return response
    
    def _process_nodes(self, tree_nodes: List[TreeSearchNode]) -> List[TreeSearchNode]:
        self.n_actions += len(self.tree_node.children)
        return tree_nodes
    
    def _update_frontier(self, tree_nodes: List[TreeSearchNode]):  
        if len(tree_nodes) == 0:
            return               
        self._add_actions_to_frontier([tree_nodes[0]])
    
    def _update_tree(self, tree_nodes):
        if len(tree_nodes) > 0:
            for node in tree_nodes:
                self.tree_node.add_child(node)
            tree_nodes = self._process_nodes(tree_nodes)

        return tree_nodes
    
