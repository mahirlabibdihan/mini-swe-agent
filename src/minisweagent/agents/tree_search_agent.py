from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, Submitted, ExecutionTimeoutError
from minisweagent.agents.tree_search_node import TreeSearchNode   
from minisweagent.agents.action_selector import ActionSelector
from minisweagent.agents.action_processor import ActionProcessor
from minisweagent.agents.backtrack_manager import BacktrackManager
from typing import List, Any, Optional
from tabulate import tabulate
import time
import subprocess
import datetime

class TreeSearchAgentConfig(AgentConfig):
    search_depth: int = 5
    """The depth of the tree search."""
    breadth_limit: int = 3
    """The maximum number of branches to explore at each node."""

class TreeSearchAgent(DefaultAgent):
    def __init__(self, 
                 *args,  
                 action_selector: ActionSelector = None,
                 backtrack_manager: BacktrackManager = None,
                 config_class=TreeSearchAgentConfig, 
                 **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.tree_root = self.tree_node = TreeSearchNode(
            last_action=None,
        )
        self.n_actions = 0
        self.n_explored = 0
        self.n_expanded = 0
        self.n_backtracks = 0
        self.action_selector = action_selector
        self.backtrack_manager = backtrack_manager
        if self.action_selector is not None:
            self.action_selector.reset()
    
    def generate_new_actions(self):
        actions = []
        for _ in range(self.config.breadth_limit):
            action = self.parse_action(self.query())
            actions.append({
                "command": action["action"],
                "thought": action["content"],
                "extra": action["extra"]
            })
            print(f"Generated action: {action['action']}")
            time.sleep(2)  # To avoid rate limiting
        return actions
    
    def execute_action(self, action: dict) -> dict:
        try:
            output = self.env.execute(action)
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
            raise ExecutionTimeoutError(
                self.render_template(self.config.timeout_template, action=action, output=output)
            )
        self.has_finished(output)
        return output | {"action": action}
    
    def create_unique_branch(self, base_name="auto"):
        """Create a new branch with a unique name"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"{base_name}-{timestamp}"
        self.env.execute(f"git checkout -b {branch_name}")
        return branch_name

    def repo_has_changes(self):
        """Check if there are any unstaged or uncommitted changes"""
        observation = self.env.execute("git status --porcelain")
        return bool(observation["output"])
    
    def commit_changes(self, message="Automated commit"):
        """Stage all changes and commit"""
        print(">> Committing changes to the repository...")
        self.env.execute("git add .")
        self.env.execute(f'git commit -m "{message}"')
        return self.env.execute("git rev-parse HEAD")["output"].strip()

    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        actions = self.generate_new_actions()

        flag = True
        while True:
            best_node = self.adjust_tree(actions, add_actions=flag)
            if best_node.parent == self.tree_node:
                self.tree_node = best_node
                break
            elif self.backtrack_manager is not None:
                BacktrackManager.backtrack(self.tree_node, best_node, self.env)
                self.tree_node = best_node
                self.n_backtracks += 1
                break
            
            print("Best node is not a child of the current node, re-adjusting the tree...")
            flag = False
            
        if best_node.parent is None or best_node.parent.branch is None or best_node.parent.is_expanded():
            best_node.branch = self.create_unique_branch(base_name="ts-agent")
            print(f">> Switching to branch: {best_node.branch}\n{self.env.execute('git branch')['output'].strip()}")
        else:
            best_node.branch = best_node.parent.branch
            print(f">> Staying on branch: {best_node.branch}")
        
        self.add_message("assistant", **{"content": best_node.last_action["thought"], "extra": best_node.last_action.get("extra", {})})
        self.get_observation(best_node.last_action["command"])
        
        if self.repo_has_changes():
            best_node.commit = self.commit_changes(f"Commit after: {best_node.last_action['command']}")
        else:
            best_node.commit = best_node.parent.commit
            
        return best_node.observation
    
        
    def get_observation(self, action: dict) -> dict:
        """Execute the action and return the observation."""
        self.n_expanded += 1
        output = self.execute_action(action)
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output
    
    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        response = self.model.query(self.messages)
        return response
    
    def process_actions(self, actions: List[str]) -> List[dict]:
        tree_nodes = ActionProcessor.convert_action_to_nodes(actions, self.tree_node)
        self.n_actions += len(self.tree_node.children)
        
        print(f"# {len(tree_nodes)} new actions generated at level {self.tree_node.level}:")
        for node in tree_nodes:
            print(f"- {node.last_action['command']}")
            
        action_list = ActionProcessor.evaluate_actions(tree_nodes, self.extra_template_vars["task"])
        final_action_list = ActionProcessor.merge_actions(action_list)

        reward_data = []
        for score, new_node in final_action_list:
            self.n_explored += 1
            reward_data.append(
                [
                    (
                        (new_node.last_action["command"][:100] + "...")
                        if len(new_node.last_action["command"]) > 100
                        else new_node.last_action["command"]
                    ),
                    f"{new_node.value:.6f}",
                    f"{score:.6f}",
                ]
            )
        
        if len(reward_data) > 0:
            print(
                tabulate(
                    reward_data,
                    headers=["Action", "Reward", "Merged"],
                    tablefmt="grid",
                    colalign=("left", "center", "center"),
                )
            )
            
        return final_action_list
    
    def adjust_tree(self, actions, add_actions=True):
        if add_actions:
            actions = self.process_actions(actions)
            if self.action_selector is not None:
                self.action_selector.add_actions(self.tree_node, actions)
        else:
            actions = []

        # Choose best action based on selection strategy
        if self.action_selector is not None:
            best_node = self.action_selector.select_action(
                self.tree_node, self.n_expanded
            )
            return best_node
        else:
            # Default: choose the first action
            if len(actions) == 0:
                raise RuntimeError("No actions to select from.")
            return actions[0][1]
    
