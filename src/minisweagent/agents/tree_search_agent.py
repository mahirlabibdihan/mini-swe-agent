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
        
        self.tree_node.branch = self.env.execute("git branch --show-current")["output"].strip()
        self.tree_node.commit = self.get_commit_hash()
        
    def generate_new_nodes(self) -> List[TreeSearchNode]:
        nodes = []
        # flag = True
        for _ in range(self.config.breadth_limit):
            action = self.parse_action(self.query())
            print(f"Generated action: {action['action']}")
            
            # Convert action to node
            new_node = TreeSearchNode(
                last_action={
                    "command": action["action"],
                    "thought": action["content"],
                    "extra": action["extra"]
                },
            )
            nodes.append(new_node)
            
            # Execute action to get observation
            output = self.env.execute(action["action"])
            observation = self.render_template(self.config.action_observation_template, output=output)
            new_node.observation = observation
            if self.repo_has_changes():
                new_node.modifies_code = True
                # Rollback changes
                print(">> Write-task detected.")
                # flag = False
                self.env.execute("git restore .")
                self.tree_node.has_write_child = True
            
            time.sleep(2)  # To avoid rate limiting
            # if flag:
            #     print("No write-task detected, stopping further action generation.")
            #     break
        return nodes
    
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
        self.add_message("system", f'THOUGHT: Need to create a new branch before committing changes. ```bash\ngit checkout -b {branch_name}\n```')
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
        output = self.env.execute("git rev-parse HEAD")
        self.add_message("system", f'THOUGHT: Commit changes of the last command.\n\n```bash\ngit add . >/dev/null 2>&1 && git commit -m "{message}" >/dev/null 2>&1 && git rev-parse HEAD\n```')
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        return output["output"].strip()
        
    def get_commit_hash(self):
        """Get the current commit hash"""
        return self.env.execute("git rev-parse HEAD")["output"].strip()

    def is_detached_head(self):
        """Check if the current HEAD is detached"""
        status = self.env.execute("git status")
        return "HEAD detached at" in status["output"]
    
    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        tree_nodes = self.generate_new_nodes()

        flag = True
        while True:
            best_node = self.adjust_tree(tree_nodes, add_nodes=flag)
            if best_node.parent == self.tree_node:
                self.tree_node = best_node
                break
            elif self.backtrack_manager is not None:
                if best_node.parent.commit != self.tree_node.commit:
                    print(f">> Backtracking to [{best_node.parent.branch} {best_node.parent.commit[:7]}]")
                    # env.execute(f"git checkout {best_node.parent.branch}")
                    self.env.execute(f"git checkout {best_node.parent.commit}")
                    self.add_message("system", f"THOUGHT: Backtrack needed to execute the highest-rewarded action.\n\n```bash\ngit checkout {best_node.parent.commit}\n```")
                print(">> Backtrack needed to execute the highest-rewarded action.")
                    
                self.tree_node = best_node
                self.n_backtracks += 1
                break
            
            print("Best node is not a child of the current node, re-adjusting the tree...")
            flag = False
                    
        if best_node.last_action["extra"]:
            self.add_message("assistant", **{"content": best_node.last_action["thought"], "extra": best_node.last_action.get("extra", {})})
        else:
            self.add_message("system", best_node.last_action["thought"])
        
        # If this is a terminating action,
        # self.env.execute(f"git checkout {self.tree_root.branch}")
        # self.env.execute(f"git diff {self.tree_root.branch}..{best_node.parent.branch} | git apply")
        
        # Undo
        # git restore . 
        # git checkout -
        
        potential_termination = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in best_node.last_action["command"]
        
        if potential_termination:
            print(">> Potentially terminating action detected, preparing final output...")
            self.env.execute(f"git checkout {self.tree_root.branch}")
            self.env.execute(f"git diff {self.tree_root.branch}..{best_node.parent.branch} | git apply")
            self.add_message("system", "THOUGHT: Preparing final output before submission.\n\n```bash\ngit checkout {self.tree_root.branch} && git diff {self.tree_root.branch}..{best_node.parent.branch} | git apply\n```")
            
        output = self.get_observation(best_node.last_action["command"])
        
        if potential_termination:
            print(">> Wasn't terminating after all, reverting to previous state.")
            self.env.execute("git restore .")
            self.env.execute(f"git checkout -")
            self.add_message("system", "THOUGHT: Reverting changes as the submission failed.\n\n```bash\ngit restore . && git checkout -\n```")
            
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        best_node.observation = observation
        
        if best_node.modifies_code:
            if best_node.parent.branch == self.tree_root.branch or self.is_detached_head():
                best_node.branch = self.create_unique_branch(base_name="ts-agent")
                print(f">> Switching to branch: {best_node.branch}\n{self.env.execute('git branch')['output'].strip()}")
            else:
                best_node.branch = best_node.parent.branch
                print(f">> Staying on branch: {best_node.branch}")
            
            best_node.commit = self.commit_changes(f"Commit after: {best_node.last_action['command']}")
            print(f">> New commit created: {best_node.commit}")
        else:
            best_node.commit = self.get_commit_hash()
            best_node.branch = best_node.parent.branch
            print(f">> No changes detected, staying on commit: {best_node.commit}")
            
        return best_node.observation
    
    def get_observation(self, action: dict) -> dict:
        """Execute the action and return the observation."""
        self.n_expanded += 1
        output = self.execute_action(action)
        return output
    
    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.step_limit <= self.model.n_calls or 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        
        messages = []
        curr = self.tree_node
        while curr.last_action is not None:
            self.messages.append({
                "role": "user", 
                "content": curr.observation, 
                "timestamp": time.time(),
            })
            messages.append(
                {
                    "role": "assistant",
                    "content": curr.last_action["thought"],
                    "extra": curr.last_action.get("extra", {}),
                    "timestamp": time.time(),
                }
            )   
            curr = curr.parent
        
        messages.append({
            "role": "user",
            "content": self.render_template(self.config.instance_template),
            "timestamp": time.time(),
        })
        messages.append({
            "role": "system",
            "content": self.render_template(self.config.system_template),
            "timestamp": time.time(),
        })
        messages.reverse()
        response = self.model.query(messages)
        return response
    
    def process_nodes(self, tree_nodes: List[str]) -> List[dict]:
        self.n_actions += len(self.tree_node.children)
        print(f"# {len(tree_nodes)} new nodes generated at level {self.tree_node.level}:")
        for node in tree_nodes:
            print(f"- {node.last_action['command']}")
            
        ActionProcessor.evaluate_nodes(tree_nodes, self.extra_template_vars["task"])
        score_node_list = ActionProcessor.merge_nodes(tree_nodes)

        reward_data = []
        for score, new_node in score_node_list:
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
            
        return score_node_list
    
    def adjust_tree(self, tree_nodes, add_nodes=True):
        if add_nodes or len(tree_nodes) > 0:
            tree_nodes = self.process_nodes(tree_nodes)
            # Add the node with the highest score as a child
            best_score, best_node = max(tree_nodes, key=lambda x: x[0])
            if best_node.modifies_code:
                for score, node in tree_nodes:
                    self.tree_node.add_child(node)
            else:
                self.tree_node.add_child(best_node)
                tree_nodes = [(best_score, best_node)]  
                
            if self.action_selector is not None:
                self.action_selector.add_actions(self.tree_node, tree_nodes)
        else:
            tree_nodes = []
        # Choose best node based on selection strategy
        if self.action_selector is not None:
            best_node = self.action_selector.select_action(
                self.tree_node, self.n_expanded
            )
            return best_node
        else:
            # Default: choose the first action
            if len(tree_nodes) == 0:
                raise RuntimeError("No actions to select from.")
            return tree_nodes[0][1]
    
