from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, FormatError, TerminatingException, Submitted, ExecutionTimeoutError
from minisweagent.agents.tree_search_node import TreeSearchNode   
import minisweagent.agents.action_processor as action_processor
from minisweagent.agents.frontier import Frontier
from minisweagent.agents.action_analyzer import is_terminating
from typing import List, Any, Optional
from tabulate import tabulate
import time
import subprocess
import datetime
import json
from minisweagent import Model, Environment
from minisweagent.agents.reward_model import RewardModel
from tqdm import tqdm
from minisweagent.agents.single_action_agent import SingleActionAgentConfig, SingleActionAgent

class RewardGuidedAgentConfig(SingleActionAgentConfig):
    branching_factor: int = 3
    """The maximum number of branches to explore at each node."""

class RewardGuidedAgent(SingleActionAgent):
    def __init__(self, 
                 model: Model, env: Environment,
                 reward_model: RewardModel, 
                 *,
                 config_class=RewardGuidedAgentConfig, 
                 **kwargs):
        super().__init__(model, env, config_class=config_class, **kwargs)
        self.frontier = Frontier(budget=self.config.branching_factor)
        self.reward_model = reward_model
            
    def _get_commit_hash(self):
        """Get the current commit hash"""
        return self.env.execute("git rev-parse HEAD")["output"].strip()
    
    def _create_pseudo_root(self):
        if self._repo_has_changes():
            self.env.execute(f"git checkout -b ts-agent-root")
            self.env.execute(f"git add .")
            self.env.execute(f"git commit -m 'Committing changes before starting tree search'")
            action = "git checkout -b ts-agent-root >/dev/null 2>&1 && git add . >/dev/null 2>&1 && git commit -m 'Committing changes before starting tree search' >/dev/null 2>&1 && git rev-parse HEAD"
            self.add_message("system", f"THOUGHT: Need to commit changes before starting tree search.\n\n```bash\n{action}\n```")
        else:
            self.env.execute(f"git checkout -b ts-agent-root")
            action = "git checkout -b ts-agent-root >/dev/null 2>&1 && git rev-parse HEAD"
            self.add_message("system", f"THOUGHT: Switching to new branch before starting tree search.\n\n```bash\n{action}\n```")
            
        output = self.env.execute("git rev-parse HEAD")
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        
        new_node = self._create_node()
        self.tree_node.add_child(
            new_node
        )
        new_node.branch = f"ts-agent-root"
        new_node.commit = self._get_commit_hash() 
        self.tree_node.executed = True
        self.tree_node = new_node
        self.tree_node.executed = True
        
    def _commit_changes(self, message="Automated commit"):
        """Stage all changes and commit"""
        print(">> Committing changes to the repository...")
        output = self.env.execute("git add .")
        output = self.env.execute(f'git commit -m "{message}"')
        
        output = self.env.execute("git rev-parse HEAD")
        self.add_message("system", f'THOUGHT: Commit changes of the last command.\n\n```bash\ngit add . >/dev/null 2>&1 && git commit -m "{message}" >/dev/null 2>&1 && git rev-parse HEAD\n```')
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        if self._repo_has_changes():
            raise Exception(">> Warning: Changes still detected after commit.")
        return output["output"].strip()
    
    def _reset(self):
        super()._reset()
        self.tree_root.branch = self.env.execute("git branch --show-current")["output"].strip()
        self.tree_root.commit = self._get_commit_hash()
        self._create_pseudo_root()
       
    def _repo_has_changes(self):
        """Check if there are any unstaged or uncommitted changes"""
        observation = self.env.execute("git status --porcelain")
        if bool(observation["output"]):
            print(">> Repository has unstaged or uncommitted changes.")
            print(observation["output"])
        return bool(observation["output"])
    
    def _get_modified_files(self):
        """Get the list of modified files in the repo"""
        observation = self.env.execute("git diff --name-only")
        return observation["output"].splitlines()
       
    def _generate_action(self):
        """
        Generate an action from the model and parse it
        
        Returns:
            response (dict): The raw response from the model
            action (dict): The parsed action
            error (str | None): The error message if parsing failed
        """
        
        response = self.query()
        try:
            action = self.parse_action(response)
            return response, action, None
        except FormatError as e:
            return response, None, str(e)
        
    def _generate_new_nodes(self, n_actions) -> List[TreeSearchNode]:
        nodes = []
        # flag = True
        for i in range(n_actions):
            # Execute action to get observation
            potential_termination = False
            response, action, error = self._generate_action()
            if error is None:
                print(f"Generated action #{i+1}: {action['action']}")
                new_node = self._create_node(
                    last_action={
                        "command": action["action"],
                        "thought": action["content"],
                        "extra": action["extra"]
                    },
                )
            else:
                new_node = self._create_node(
                    last_action={
                        "command": None,
                        "thought": response["content"],
                        "extra": response["extra"]
                    },
                )
            
            if error is None:
                try:
                    # Be-aware of potential terminating actions
                    potential_termination = is_terminating(action['action'])
                    if potential_termination:
                        self.env.execute(f"git checkout {self.tree_root.branch}")
                        self.env.execute(f"git diff {self.tree_root.branch}..{self.tree_node.branch} | git apply")
                    output = self.env.execute(action["action"])
                    if potential_termination:
                        self.env.execute("git restore .")
                        self.env.execute("git checkout -")
                    observation = self.render_template(self.config.action_observation_template, output=output) 
                except (TimeoutError, subprocess.TimeoutExpired) as e:
                    output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
                    observation = self.render_template(self.config.timeout_template, action=action["action"], output=output)

                # Check for terminating action
                lines = output.get("output", "").lstrip().splitlines(keepends=True)
                if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
                    print(">> Terminating action detected.")
                    new_node.is_terminating = True   
                # Check for code modifications
                elif self._repo_has_changes():
                    new_node.modifies_code = True
                    new_node.modified_files = self._get_modified_files()
                    # Rollback changes
                    print(">> Write-task detected.")
                    self.env.execute("git restore .")

                if new_node.is_terminating != potential_termination:
                    print(">> Warning: Invalid terminating action detected. Skipping this action...")
                    time.sleep(2)  # To avoid rate limiting
                    continue    
            else:
                print(f"Generated action #{i+1}: <<Invalid Action>>")
                observation = error
            
            new_node.observation = observation
            nodes.append(new_node)

            time.sleep(2)  # To avoid rate limiting
        return nodes
    
    def _stage_to_main_branch(self):
        self.env.execute(f"git checkout {self.tree_root.branch}")
        self.env.execute(f"git diff {self.tree_root.branch}..{self.tree_node.parent.branch} | git apply")
        self.env.execute(f"git branch | grep '^  ts-agent' | sed 's/^  //' | xargs -r git branch -D") # Clean up temp branches
        self.add_message("system", f"THOUGHT: Preparing final output before submission.\n\n```bash\ngit checkout {self.tree_root.branch} && git diff {self.tree_root.branch}..{self.tree_node.parent.branch} | git apply && git branch | grep '^  ts-agent' | sed 's/^  //' | xargs -r git branch -D\n```")
            
    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        if self.tree_node.is_terminating:
            self._create_pseudo_root()
            
        tree_nodes = self._generate_new_nodes(self.config.branching_factor)
        tree_nodes = self._update_tree(tree_nodes)
        self._update_frontier(tree_nodes)
        best_node = self._select_action()
        self.tree_node = best_node
        
        self.frontier.reset()
        
        if self.tree_node.is_terminating:
            self._stage_to_main_branch()

        if self.tree_node.last_action["extra"]:
            self.add_message("assistant", **{"content": self.tree_node.last_action["thought"], "extra": self.tree_node.last_action.get("extra", {})})
        else: # Action generated by System
            self.add_message("system", self.tree_node.last_action["thought"])
            
        print(f">> Executing selected action: {self.tree_node.last_action['command']}")
        if self.tree_node.last_action["command"] is None or (not self.tree_node.is_terminating and not self.tree_node.modifies_code): # For read-only action, no need to re-execute
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
        self.tree_node.branch = self.tree_node.parent.branch
        if self.tree_node.modifies_code:
            self.tree_node.commit = self._commit_changes()
            print(f">> New commit created: {self.tree_node.commit}")
        else:
            self.tree_node.commit = self._get_commit_hash()
            print(f">> No changes detected, staying on commit: {self.tree_node.commit}")

        return self.tree_node.observation
    
    def _evaluate_nodes(self, node_list):
        for new_node in tqdm(node_list, desc="Evaluating nodes"):
            if new_node.value is None:
                new_node.value = self.reward_model.compute_reward(new_node, self.extra_template_vars["task"])
                if new_node.last_action["command"] is None:
                    # Penalize invalid actions
                    new_node.value = 0.5 * new_node.value
            
    def _process_nodes(self, tree_nodes: List[str]) -> List[TreeSearchNode]:
        self.n_actions += len(self.tree_node.children)
        print(f"# {len(tree_nodes)} new nodes generated at level {self.tree_node.level}:")
        for node in tree_nodes:
            print(f"- {node.last_action['command']}")
            
        self._evaluate_nodes(tree_nodes)
        tree_nodes = action_processor.merge_nodes(tree_nodes)

        reward_data = []
        for new_node in tree_nodes:
            self.n_explored += 1
            reward_data.append(
                [
                    (
                        (new_node.last_action["command"][:100] + "...")
                        if new_node.last_action["command"] is not None and len(new_node.last_action["command"]) > 100
                        else new_node.last_action["command"]
                    ),
                    f"{new_node.value:.6f}",
                    f"{new_node.merged_value:.6f}",
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
            
        return tree_nodes
    
    def _update_frontier(self, tree_nodes: List[TreeSearchNode]):  
        if len(tree_nodes) == 0:
            return                    
        self._add_actions_to_frontier(tree_nodes)