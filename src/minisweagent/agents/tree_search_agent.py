from pathlib import Path
from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, TerminatingException, Submitted, ExecutionTimeoutError
from minisweagent.agents.tree_search_node import TreeSearchNode   
from minisweagent.agents.reward_guided_agent import RewardGuidedAgentConfig, RewardGuidedAgent
from minisweagent.agents.single_action_agent import NoActionFound
import minisweagent.agents.action_processor as action_processor
from minisweagent.agents.frontier import Frontier
from minisweagent.agents.action_analyzer import is_terminating
from minisweagent.agents.reward_model import RewardModel
from typing import List, Any, Optional
from tabulate import tabulate
import time
import subprocess
import datetime
import json
import heapq
import math
from rank_bm25 import BM25Okapi
import pickle
import os
import numpy as np
from minisweagent.utils.log import instance_logger

class TreeSearchAgentConfig(RewardGuidedAgentConfig):
    frontier_budget: int = 4
    """The maximum number of nodes allowed in the action queue."""
    epsilon: float = 0.3
    selection_scope: str = "local"
    """Scope for action selection: 'local' or 'global'."""
    max_expansion: int = 5
    """Maximum number of nodes to expand per step."""

import json

class TreeSearchAgent(RewardGuidedAgent):
    def __init__(self, 
                 *args,  
                 config_class=TreeSearchAgentConfig, 
                 **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.frontier = Frontier(budget=self.config.frontier_budget)
        self.n_backtracks = 0
        self.n_prune = 0
        self.curr_epsilon = self.config.epsilon
        self.phase = 1
        
    def _reset(self):
        super()._reset()
        self.curr_epsilon = self.config.epsilon
        issue_tokens = self.task.split()
        scores = self.bm25.get_scores(issue_tokens)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        self.relevance_dict = dict(zip(self.file_ids, scores))
        
    def _backtrack(self, target_node):
        instance_logger.debug(f">> Backtracking from [{self.tree_node.id}] to [{target_node.id}]")
        
        if target_node.commit != self.tree_node.commit:
            instance_logger.debug(f">> Backtracking from [{self.tree_node.commit[:7]}] to [{target_node.commit[:7]}]")

            # if self._get_branch_head(target_node.branch) != target_node.commit:
            self.env.execute(f"git checkout {target_node.commit}")
            self.add_message("system", f"THOUGHT: Backtracking to node:{target_node.id}.\n\n```bash\ngit checkout {target_node.commit}\n```")
            # else:
            #     self.env.execute(f"git checkout {target_node.branch}")
            #     self.add_message("system", f"THOUGHT: Backtracking to node:{target_node.id}.\n\n```bash\ngit checkout {target_node.branch}\n```")
        
        else:
            self.add_message("system", f"THOUGHT: Backtracking to node:{target_node.id}.")
    
    def is_promising(self, node: TreeSearchNode) -> bool:
        """Check if a node is promising based on epsilon threshold."""
        if self.curr_epsilon is None or node.parent.value is None:
            return True
        
        # A node is promising if it increases value within epsilon threshold or has high absolute value
        return node.value >= node.parent.value - self.curr_epsilon or node.value > 0.8
        # return True # TEMPORARY: Disable pruning to see how it affects performance. Will re-enable after testing.
    
    def _calculate_path_write_reward(self, node: TreeSearchNode) -> float:
        """Calculate average reward of write actions along the path to the node."""
        total_reward = 0.0
        write_actions = 0
        current = node
        while current.parent is not None:
            if current.modifies_code and current.merged_value is not None:
                total_reward += current.merged_value
                write_actions += 1
            current = current.parent
        if write_actions == 0:
            return 0.0
        return total_reward / write_actions
    
    def _handle_max_steps(self):
        # instance_logger.debug(f">> Checking for max steps... {self.n_expanded} / {self.config.step_limit}")
        if self.n_expanded + 1 < self.config.step_limit:
            return None
        
        instance_logger.debug(">> Max steps reached, selecting best terminating action...")
        
        # Find the best terminating action in the tree
        terminating_nodes = [
            n for n in self.node_map.values()
            if n.is_terminating and n.merged_value is not None
        ]
        best_node = max(
            terminating_nodes,
            key=lambda x: x.merged_value,
            default=None
        )
        if best_node is None:
            # Find the best path (Whose average reward on write actions is highest)
            candidates = [
                n for n in self.node_map.values()
                if n.merged_value is not None
                and n.visible
                and not n.executed
                and not n.is_terminating
            ]
            
            # For each candidate, compute average write reward along path
            best_node = None
            best_value = float("-inf")

            for node in candidates:
                value = self._calculate_path_write_reward(node)
                if value > best_value:
                    best_value = value
                    best_node = node
            
            if best_value == 0:
                best_node = max(candidates, key=lambda x: x.merged_value, default=None) # Fallback to max reward node
                print(f">> No write actions found, fallback to node with highest merged value [{best_node.id}] with merged value {best_node.merged_value}")
            else:
                print(f">> Best node based on path write reward is [{best_node.id}] with average write reward {best_value}")
                
            # Next action is the best_node and terminate after that -> Has some issue with "local" scope. TODO: Fix that
            best_node.visits += 1
            term_node = self._make_terminating_action(best_node)
            print(f">> Adding terminating node [{term_node.id}] under best node [{best_node.id}] with path write reward {best_value}")
            # best_node.add_child(term_node)
        
        return best_node

    # def go_to_best_expandable_node(self):
    #     best_node = best_node = max(
    #         (n for n in self.node_map.values()
    #         if n.merged_value is not None
    #         and not n.is_terminating
    #         and n.visible
    #         and self.is_promising(n)
    #         and len(n.children) < self.config.max_expansion),
    #         key=lambda x: x.merged_value,
    #         default=None
    #     )
    #     self._backtrack(best_node)
    #     if best_node is not None:
    #         self.tree_node = best_node
    #     else:
    #         self.tree_node = self.tree_root.children[0] # Fallback to first child of root
    #         instance_logger.debug(">> No expandable nodes found, reverting to root.")
            
            
    def compute_alpha(self, progress: float, alpha_min: float = 0.4, alpha_max: float = 0.8) -> float:
        """
        Compute alpha based on fraction of executed nodes.

        Args:
            progress: float in [0,1], fraction of executed nodes
            alpha_min: minimum alpha (late in search)
            alpha_max: maximum alpha (early in search)
        Returns:
            alpha: float in [alpha_min, alpha_max]
        """
        progress = min(max(progress, 0.0), 1.0)  # clamp
        return alpha_max - (alpha_max - alpha_min) * progress

    def _go_to_best_executable_node(self, k: int = 1) -> List[TreeSearchNode]:
        # TODO: Find best on a subtree basis. Root should be provided.
        def node_priority(n, gamma=0.9, max_depth=50):
            path_value = n.get_path_value(gamma)
            depth_score = math.log1p(n.level) / math.log1p(max_depth)
            alpha = self.compute_alpha(progress=self.n_expanded / self.config.step_limit)
            return alpha * path_value + (1 - alpha) * depth_score

        # Materialize candidates (important)
        while True:
            candidates = [
                n for n in self.node_map.values()
                if n.merged_value is not None
                and n.visible
                and not n.executed
                and n.id != self.tree_root.id
                and (self.phase > 1 or not n.modifies_code)
                and (self.phase > 2 or not n.is_terminating)
            ]
        
            if not candidates:
                if self.phase == 1:
                    best_node = self._switch_to_phase_2()
                    if best_node is not None:
                        return best_node
                elif self.phase == 2:
                    instance_logger.debug(":: Switching to phase 3: Allowing terminating actions.")
                    self.phase = 3
                    continue
                     
                raise NoActionFound("No executable nodes found in the tree.")
            break
            
                    
        # Find max depth among all nodes
        max_depth = max(n.level for n in candidates)
        
        # Step 1: top-k by priority
        top_k = heapq.nlargest(
            min(k, len(candidates)),
            candidates,
            key=lambda x: node_priority(x, max_depth=max_depth)
        )
        
        # Step 2: lowest-valued node among top-k
        last_node = min(top_k, key=lambda x: x.value)
            
        if self.curr_epsilon is not None:
            self.curr_epsilon = max(self.curr_epsilon, last_node.parent.value - last_node.value) # Increase epsilon to be less strict
        
        self._update_frontier(top_k)
        best_node = self._select_action()
        
        if best_node.parent != self.tree_node:
            self._backtrack(best_node.parent)          
            self.n_backtracks += 1   
        
        return best_node
        
    def _find_best_write_leaf(self) -> TreeSearchNode:
        write_leaves = [
            n for n in self.node_map.values()
            if n.visible
            and not n.executed
            and n.modifies_code
            and not n.is_terminating
            and n.merged_value is not None
        ]
        if not write_leaves:
            return None
        best_leaf = max(write_leaves, key=lambda x: x.get_path_value(0.85))
        return best_leaf

    def _find_best_read_leaf(self) -> TreeSearchNode:
        read_leaves = [
            n for n in self.node_map.values()
            if n.visible
            and not n.executed
            and not n.is_terminating
            and n.merged_value is not None
        ]
        if not read_leaves:
            return None
        best_leaf = max(read_leaves, key=lambda x: x.get_path_value(0.85))
        return best_leaf
        
    def _switch_to_phase_2(self):
        best_leaf = self._find_best_write_leaf()
        if best_leaf is not None:
            instance_logger.debug(":: Switching to phase 2: Prioritizing write actions.")
            self.phase = 2  # Switch to phase 2 after 30% of steps
            self.node_map = {best_leaf.id: best_leaf} # Clearing all other nodes as they are now stale when we switch to phase 2 with a promising write action. This is a design choice to focus the search on the promising write action and its subtree, but it can be adjusted to keep some of the tree if desired.
            self.frontier.clear()  # Clear frontier when switching phases
            self._update_frontier([best_leaf])
            return best_leaf
        else:
            best_leaf = self._find_best_read_leaf()
            if best_leaf is not None:
                instance_logger.debug(":: Switching to phase 2: No promising write actions, converging on best read action for phase 2.")
                self.phase = 2
                self.node_map = {best_leaf.id: best_leaf}
                self.frontier.clear()  # Clear frontier when switching phases
                self._update_frontier([best_leaf])
                self.frontier.reduce_budget() # Reduce budget to focus on this promising read action
                return best_leaf
        return None
    
    def _switch_to_phase_3(self):
        instance_logger.debug(":: Switching to phase 3: Allowing terminating actions.")
        self.phase = 3  # Switch to phase 3
        self.frontier.clear()  # Clear frontier when switching 
        
        
    def step(self) -> dict:
        """Query the LM, execute the action, return the observation."""
        if self.tree_node.is_terminating:
            self._create_pseudo_root()
            
        if self.tree_node.visits == 0:
            tree_nodes = self._generate_new_nodes(min(self.config.branching_factor, self.config.max_expansion - len(self.tree_node.children)))
            if self.phase == 1:
                for n in tree_nodes:
                    if n.invalid_termination:
                        n.prune()
                        # Prune terminating actions in phase 1. Because there's no modifications yet.
                        
                        
            self._update_tree(tree_nodes)

        self.tree_node.visits += 1
        
        self.tree_node.epsilon = self.curr_epsilon
        
        best_node = None       
        while best_node is None:
            if self.config.selection_scope == "local":
                self.frontier.clear() # Local frontier only
            
            if self.config.selection_scope == "global" and (self.n_submissions >= 2) and self.phase == 2:
                instance_logger.debug(":: Switching to phase 3: Allowing terminating actions.")
                self.phase = 3  # Switch to phase 3 after 50% of steps
                candidates = [
                    n for n in self.node_map.values()
                    if n.merged_value is not None
                    and n.visible
                    and not n.executed
                    and n.id != self.tree_root.id
                    and n.is_terminating
                    and self.is_promising(n)
                    and n.parent.id != self.tree_node.id
                ]
                # Adding left out terminating nodes from previous phases
                self._update_frontier(candidates)
                self.add_message("system", "Switching to phase 3: Allowing terminating actions.")
                # TODO: May do more sophisticated checks for choosing the best terminating node
                
            if self.config.selection_scope == "local" or self.tree_node.visits == 1: # Local or First visit
                unexecuted = [
                    c for c in self.tree_node.children 
                    if not c.executed 
                    and c.visible 
                    and self.is_promising(c) 
                    and (self.phase > 1 or not c.modifies_code or self.config.selection_scope == "local") 
                    and (self.phase > 2 or not c.is_terminating or self.config.selection_scope == "local")
                ] # Unexecuted + Promising
                self._update_frontier(unexecuted)
                
                if self.tree_node.value is not None and self.config.epsilon is not None:
                    if len(unexecuted) > 0:
                        self.curr_epsilon = max(self.curr_epsilon - .03*self.config.epsilon, .7*self.config.epsilon)  # Decrease epsilon to be more strict, but not too much
                    else:
                        self.curr_epsilon = self.curr_epsilon + .03*self.config.epsilon # Increase epsilon to be less strict
            
            if self.config.selection_scope == "global" and self.phase == 1 and (
                (self.n_modifications >= 2 and (self.n_expanded >= min(self.config.step_limit/3, 10) or self.frontier.empty()))
                or
                (self.n_modifications >= 1 and  self.n_expanded >= min(self.config.step_limit/2, 25))
                or
                (self.n_expanded >= 2*self.config.step_limit/3)
            ):
                self._switch_to_phase_2()
                 
            
            if not self.frontier.empty():
                best_node = self._select_action()
                if best_node.parent != self.tree_node:
                    self._backtrack(best_node.parent)          
                    self.n_backtracks += 1   
                    instance_logger.debug(">> Backtrack needed to execute the highest-rewarded action.")
                    
            elif self.config.selection_scope == "local":
                # Backtrack to parent
                if self.tree_node.last_action is not None:
                    self._backtrack(self.tree_node.parent)
                    self.tree_node = self.tree_node.parent
                    self.n_backtracks += 1
                    self.tree_node.visits += 1
                    instance_logger.debug(">> No promising actions locally, backtracking to parent node.")
                else:
                    # Go to the highest-rewarded node globally
                    best_node = self._go_to_best_executable_node()
                    self.add_message("system", "No promising actions found locally, backtracking to best action.")
                
            else:
                # Go to the highest-rewarded node globally
                instance_logger.debug(">> Frontier is empty, searching globally for best executable node.")
                best_node = self._go_to_best_executable_node()
                self.add_message("system", "No promising actions found globally, backtracking to best action.")
                
        self.tree_node = best_node
        self.tree_node.parent.visits += 1
       
        if self.tree_node.is_terminating:
            self._stage_to_main_branch()
            self.tree_node.is_submission = True
            self.frontier.reset()
 
        if self.tree_node.last_action["extra"]:
            self.add_message("assistant", **{"content": self.tree_node.last_action["thought"], "extra": self.tree_node.last_action.get("extra", {})})
        else: # Action generated by System
            self.add_message("system", self.tree_node.last_action["thought"])
            
        instance_logger.debug(f">> Executing selected action #{self.n_expanded + 1}: {self.tree_node.last_action['command']}")
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
        
        if self.tree_node.modifies_code:
            # if self._is_detached_head():
            #     self.tree_node.branch = self._create_unique_branch(base_name="ts-agent")
            #     instance_logger.debug(f">> Switching to branch: {self.tree_node.branch}\n{self.env.execute('git branch')['output'].strip()}")
            # else:
            #     self.tree_node.branch = self.tree_node.parent.branch
            #     instance_logger.debug(f">> Staying on branch: {self.tree_node.branch}")
            
            self.tree_node.commit, is_submodule_commit = self._commit_changes()
            instance_logger.debug(f">> New commit created: {self.tree_node.commit}")
            
            if is_submodule_commit:
                instance_logger.debug(">> Detected submodule commit, marking this step as point-of-no-return.")
                self.frontier.clear()  # Clear frontier when switching phases
                self.node_map = {self.tree_node.id: self.tree_node} # Prune all other nodes as they are now stale   
        else:
            self.tree_node.commit = self._get_commit_hash()
            # self.tree_node.branch = self.tree_node.parent.branch
            instance_logger.debug(f">> No changes detected, staying on commit: {self.tree_node.commit}")
            
        try:
            with open("debug_tree.json", "w", encoding="utf-8") as f:
                json.dump(self.tree_root.to_tree(), f, indent=4, ensure_ascii=False)

            with open("debug_nodes.json", "w", encoding="utf-8") as f:
                json.dump(self.tree_root.to_json(), f, indent=4, ensure_ascii=False)
        except Exception as e:
            instance_logger.debug(f">> Failed to dump tree for debugging: {e}")
            # traceback
            import traceback
            instance_logger.debug(traceback.format_exc())
            # instance.logger()
            
            
        return self.tree_node.observation
    
    def _create_unique_branch(self, base_name="auto"):
        """Create a new branch with a unique name"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        branch_name = f"{base_name}-{timestamp}"
        self.env.execute(f"git checkout -b {branch_name}")
        self.add_message("system", f'THOUGHT: Need to create a new branch before committing changes. ```bash\ngit checkout -b {branch_name}\n```')
        return branch_name
        
    def _get_branch_head(self, branch_name):
        """Get the commit hash of the head of a branch"""
        return self.env.execute(f"git rev-parse {branch_name}")["output"].strip()

    def _is_detached_head(self):
        """Check if the current HEAD is detached"""
        status = self.env.execute("git status")
        return "HEAD detached at" in status["output"]
    
    def _update_frontier(self, tree_nodes: List[TreeSearchNode]):
        if len(tree_nodes) == 0:
            return
        if self.frontier.is_out_of_budget():
            self.frontier.minimize()
            self.n_prune += 1
            instance_logger.debug(f"Queue size {self.frontier.length()}. Tree pruned.")
        else:
            instance_logger.debug(f"Queue size {self.frontier.length()}. Adding new actions...")
            
        # PRUNE READ Action
        # best_node = max(tree_nodes, key=lambda x: x.merged_value)
        # if not best_node.modifies_code:
        #     tree_nodes = [best_node]  
            
        self._add_actions_to_frontier(tree_nodes)
    