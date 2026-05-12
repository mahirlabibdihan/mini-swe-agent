from pathlib import Path
from platform import node
import uuid

from matplotlib.hatch import get_path
from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, FormatError, TerminatingException, Submitted, ExecutionTimeoutError
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class TreeSearchAgentConfig(RewardGuidedAgentConfig):
    frontier_budget: int = None
    """The maximum number of nodes allowed in the action queue."""
    epsilon: float = 0.3
    selection_scope: str = "local"
    """Scope for action selection: 'local' or 'global'."""
    max_expansion: int = 5
    """Maximum number of nodes to expand per step."""
    sub_thres: int = 3
    """Number of submissions after which to switch to phase 3."""
    u_sub_thres: int = 1
    """Number of unique submissions after which to switch to phase 3."""
    itr_limit: int = 4
    top_k_tree_pruning: int = 4
    

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
        self.itr = 1
        # create an empty list of size self.config.itr_limit+1
        self.node_map_itr = [{} for _ in range(self.config.itr_limit+2)] # NEW:
        self.terminating_nodes = {}

    def _reset(self):
        super()._reset()
        self.curr_epsilon = self.config.epsilon
        # issue_tokens = self.task.split()
        # scores = self.bm25.get_scores(issue_tokens)
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        # self.relevance_dict = dict(zip(self.file_ids, scores))
        
    def _backtrack(self, target_node):
        instance_logger.debug(f">> Backtracking from [{self.tree_node.id}] to [{target_node.id}]")
        n_commit = self._estimate_commit(target_node)
        if n_commit != self.tree_node.commit:
            instance_logger.debug(f">> Backtracking from [{self.tree_node.commit[:7]}] to [{n_commit[:7]}]")

            # if self._get_branch_head(target_node.branch) != target_node.commit:
            out = self.env.execute(f"git checkout {n_commit}")
            if out["returncode"] != 0:
                instance_logger.error(f"Git checkout failed: {out['stderr']}")
                raise Exception(f"Git checkout failed: {out['stderr']}")
            self.add_message("system", f"THOUGHT: Backtracking to node:{target_node.id}.\n\n```bash\ngit checkout {n_commit}\n```")
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
    
    def _get_solution_patch(self, t_node: TreeSearchNode) -> str:
        if not t_node.is_terminating:
            return None
        
        if t_node.parent.commit is None:
            return None
        
        out = self.env.execute(f"git diff {t_node.parent.commit} {self._get_root_commit()}")
        return out["output"].strip(), out
    
    def _make_terminating_action(self, curr_node):
        node = self._action_to_node(None,
            {
                "action": f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached",
                "content": "THOUGHT: Time to submit final output\nCOMMAND_TYPE: [SUBMIT]\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```",
                "extra": None,
            } , None, curr_node)
        
        curr_node.add_child(node)
        node.system_generated = True
        return node
    
    def _generate_terminating_nodes(self):
        # if self.n_submissions >= self.config.sub_thres and len(self.terminating_nodes) >= self.config.u_sub_thres:
        #     return # NEW: 
        
        old_active_node = self.tree_node
        
        edit_nodes = self._get_all_unique_edit_paths()
        if len(edit_nodes) > 0:
            instance_logger.debug(f">> Generating terminating nodes for {len(edit_nodes)} unique edit paths.")
            
        count = 0
        for node in edit_nodes:
            if not node.executed:
                if not node.modifies_code:
                    node.commit = self._estimate_commit(node)
                    if node.commit in self.terminating_nodes:
                        continue # We already have a terminating node for this commit, so skip
                    node.executed = True
                    node.order = self.n_expanded + 1
                    self._backtrack(node)
                    self.tree_node = node
                    term_node = self._make_terminating_action(node)
                    term_node.value = term_node.merged_value = self._evaluate_node(term_node)
                    if self.terminating_nodes.get(node.commit) is None:
                        self.terminating_nodes[node.commit] = []
                    self.terminating_nodes[node.commit].append(term_node)
                else:
                    # Modified action not executed yet, so execute it to get the observation and value
                    # Need to backtrack to the parent node to execute this node
                    self._backtrack(node.parent)
                    self.tree_node = node
                    try:
                        self.env.execute(node.last_action["command"])
                    except (TimeoutError, subprocess.TimeoutExpired) as e:
                        instance_logger.warning(f"Execution of node [{node.id}] command timed out during terminating node generation: {e}")
                                
                    node.commit, _ = self._commit_changes()
                    node.order = self.n_expanded + 1
                    instance_logger.debug(f">> New commit created: {self.tree_node.commit}")
                    node.executed = True
                    term_node = self._make_terminating_action(node)
                    term_node.value = term_node.merged_value = self._evaluate_node(term_node)
                    if self.terminating_nodes.get(node.commit) is None:
                        self.terminating_nodes[node.commit] = []
                    self.terminating_nodes[node.commit].append(term_node)
            else:
                if node.commit in self.terminating_nodes:
                    continue # We already have a terminating node for this commit, so skip
                self._backtrack(node)
                self.tree_node = node
                term_node = self._make_terminating_action(node)
                term_node.value = term_node.merged_value = self._evaluate_node(term_node)
                if self.terminating_nodes.get(node.commit) is None:
                    self.terminating_nodes[node.commit] = []
                self.terminating_nodes[node.commit].append(term_node)
            
            count += 1
            # if count >= self.config.top_k_tree_pruning:
            #     break # Only generate terminating nodes for top-k unique edit paths to limit combinatorial explosion
                
        self._backtrack(old_active_node)
        self.tree_node = old_active_node
        
                    
    def _get_best_terminating_node(self) -> Optional[TreeSearchNode]:
        self._generate_terminating_nodes() # Ensure we have generated terminating nodes for all current edit paths
        
        terminating_nodes = [
            n
            for n in self.all_node_map.values() # OLD
            if n.is_terminating and n.merged_value is not None
        ]
        
        if not terminating_nodes:
            return None
        
        best_node = max(
            terminating_nodes,
            key=lambda x: (
                # x.merged_value, # OLD
                0.8 * x.merged_value + (1 - (x.parent.order / self.config.step_limit)) * 0.1 + 0.1 * (not x.system_generated), # NEW:  Should give priority to early discovered solutions
                # NEW: Give priority to AI generated nodes
                x.get_path_value(0.85) # NEW: In case of tie
            ),
            default=None
        )
        
        for commit, t_nodes in self.terminating_nodes.items():
            sorted_terms = sorted(
                t_nodes,
                key=lambda x: (
                    # OLD: merged_value
                    0.8 * x.merged_value + (1 - (x.parent.order / self.config.step_limit)) * 0.1 + 0.1 * (not x.system_generated), # NEW:  Should give priority to early discovered solutions
                    x.get_path_value(0.85) # NEW: In case of tie
                ),
                reverse=True
            )
            for t in sorted_terms[1:]:
                t.visible = False # Hide suboptimal terminating nodes for the same commit to reduce clutter in the tree visualization
            
        return best_node
    
    
    # OLD:
    # def _handle_max_steps(self):
    #     # instance_logger.debug(f">> Checking for max steps... {self.n_expanded} / {self.config.step_limit}")
    #     if self.n_expanded + 1 < self.config.step_limit:
    #         return None
        
    #     instance_logger.debug(">> Max steps reached, selecting best terminating action...")
        
    #     # Find the best terminating action in the tree
    #     terminating_nodes = [
    #         n for n in self.node_map.values()
    #         if n.is_terminating and n.merged_value is not None
    #     ]
    #     best_node = max(
    #         terminating_nodes,
    #         key=lambda x: x.merged_value,
    #         default=None
    #     )
    #     if best_node is None:
    #         # Find the best path (Whose average reward on write actions is highest)
    #         candidates = [
    #             n for n in self.node_map.values()
    #             if n.merged_value is not None
    #             and n.visible
    #             and not n.executed
    #             and not n.is_terminating
    #         ]
            
    #         # For each candidate, compute average write reward along path
    #         best_node = None
    #         best_value = float("-inf")

    #         for node in candidates:
    #             value = self._calculate_path_write_reward(node)
    #             if value > best_value:
    #                 best_value = value
    #                 best_node = node
            
    #         if best_value == 0:
    #             best_node = max(candidates, key=lambda x: x.merged_value, default=None) # Fallback to max reward node
    #             print(f">> No write actions found, fallback to node with highest merged value [{best_node.id}] with merged value {best_node.merged_value}")
    #         else:
    #             print(f">> Best node based on path write reward is [{best_node.id}] with average write reward {best_value}")
                
    #         # Next action is the best_node and terminate after that -> Has some issue with "local" scope. TODO: Fix that
    #         best_node.visits += 1
    #         term_node = self._make_terminating_action(best_node)
    #         print(f">> Adding terminating node [{term_node.id}] under best node [{best_node.id}] with path write reward {best_value}")
    #         # best_node.add_child(term_node)
        
    #     return best_node
    
    # NEW:
    def _handle_max_steps(self):
        # instance_logger.debug(f">> Checking for max steps... {self.n_expanded} / {self.config.step_limit}")
        if self.n_expanded + 1 < self.config.step_limit:
            return None
        
        self.node_map_itr[self.itr] = self.node_map
        
        instance_logger.debug(">> Max steps reached, selecting best terminating action...")
        
        best_node = self._get_best_terminating_node()
        
        if best_node is None:
            candidates = self._get_topk_edit_paths(k=self.config.top_k_tree_pruning, to_execute=False) # <----
                    
            # NEW: We can generate a terminating node under all the paths with at least one edit action, evaluate those terminating nodes, and pick the best one among them.
            if len(candidates) > 0:
                futures = []
                max_workers = 4
                term_nodes = []
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for node in tqdm(candidates, desc="Generating nodes"):
                        node.visits += 1
                        term_node = self._make_terminating_action(node)
                        term_nodes.append(term_node)
                        instance_logger.debug(f">> Adding terminating node [{term_node.id}] under node [{node.id}]")
                        futures.append((term_node, executor.submit(self._evaluate_node, term_node)))
                    if futures:
                        for node, future in tqdm(futures, total=len(futures), desc="Waiting for node scores"):
                            node.merged_value = node.value = future.result()
                
                # Now select the best terminating node among the generated ones
                best_node = self._get_best_terminating_node()
                best_node = best_node.parent # We want to execute the parent node which is the original candidate node. The terminating node will be executed implicitly after that.
            else:
                raise NoActionFound("Can't make any modifications within the step limit.")
                candidates = [
                    n for n in self.node_map.values()
                    if not n.executed
                    and not n.is_terminating
                    and n.visible
                    and n.merged_value is not None
                ]
                best_node = max(candidates, key=lambda x: x.merged_value, default=None) # Fallback to max reward node
                print(f">> No write actions found, fallback to node with highest merged value [{best_node.id}] with merged value {best_node.merged_value}")
                term_node = self._make_terminating_action(best_node)
                best_node.visits += 1
                print(f">> Adding terminating node [{term_node.id}] under node [{best_node.id}]")
        
        return best_node

    def go_to_best_expandable_node(self):
        best_node = best_node = max(
            (n for n in self.node_map.values()
            if n.merged_value is not None
            and not n.is_terminating
            and n.visible
            and self.is_promising(n)
            and len(n.children) < self.config.max_expansion),
            key=lambda x: x.merged_value,
            default=None
        )
        self._backtrack(best_node)
        if best_node is not None:
            self.tree_node = best_node
        else:
            self.tree_node = self.tree_root.children[0] # Fallback to first child of root
            instance_logger.debug(">> No expandable nodes found, reverting to root.")
            
    def _go_to_best_executable_node(self, k: int = 1) -> List[TreeSearchNode]:
        # TODO: Find best on a subtree basis. Root should be provided.

        # Materialize candidates (important)
        while True:
            candidates = [
                n for n in self.node_map.values()
                if n.merged_value is not None
                and n.visible
                and not n.executed
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
            key=lambda x: self.node_priority(x, max_depth=max_depth)
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
    
    def _find_all_write_leaves(self) -> TreeSearchNode:
        write_leaves = [
            n for n in self.node_map.values()
            if n.visible
            and not n.executed
            and n.modifies_code
            and not n.is_terminating
            and n.merged_value is not None
        ]
        if not write_leaves:
            return []
        # convert to a sorted list of all write leaves based on path value
        sorted_leaves = sorted(write_leaves, key=lambda x: x.get_path_value(0.85), reverse=True)
        return sorted_leaves

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
    
    def _find_all_read_leaves(self) -> TreeSearchNode:
        read_leaves = [
            n for n in self.node_map.values()
            if n.visible
            and not n.executed
            and not n.is_terminating
            and n.merged_value is not None
        ]
        if not read_leaves:
            return []
        # convert to a sorted list of all read leaves based on path value
        sorted_leaves = sorted(read_leaves, key=lambda x: x.get_path_value(0.85), reverse=True)
        return sorted_leaves
        
    def _switch_to_phase_2(self):
        # TODO: Keep top-k nodes instead of just the best node
        # best_leaf = self._find_best_write_leaf()
        sorted_writes = self._find_all_write_leaves()
        sorted_reads = self._find_all_read_leaves()
        sorted_leaves = sorted_writes + sorted_reads
        k = 1
        if len(sorted_leaves) > 0:
            self.phase = 2  # Switch to phase 2 after 30% of steps
            # Keep top k, say k = 3 for example
            top_k = sorted_leaves[:k]
            self.node_map = {leaf.id: leaf for leaf in top_k}
            
            # Clear frontier when switching phases
            self.frontier.clear()
            best_leaf = top_k[0]
            self._update_frontier([best_leaf])
            
            if len(sorted_writes) > 0:
                instance_logger.debug(":: Switching to phase 2: Prioritizing write actions.")
            else:
                instance_logger.debug(":: Switching to phase 2: No promising write actions, converging on best read action for phase 2.")
                self.frontier.reduce_budget() # Reduce budget to focus on this promising read action
            # Start with the best leaf
            return best_leaf

        return None
    
    def _switch_to_phase_3(self):
        instance_logger.debug(":: Switching to phase 3: Allowing terminating actions.")
        self.phase = 3  # Switch to phase 3
        self.frontier.clear()  # Clear frontier when switching 
        
    def _has_write_parent(self, node: TreeSearchNode) -> bool:
        current = node.parent
        while current is not None:
            if current.modifies_code:
                return True
            current = current.parent
        return False
    
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

    def node_priority(self, n, gamma=0.9, max_depth=50):
        path_value = n.get_path_value(gamma)
        return path_value # NEW
        depth_score = math.log1p(n.level) / math.log1p(max_depth)
        alpha = self.compute_alpha(progress=self.n_expanded / self.config.step_limit)
        return alpha * path_value + (1 - alpha) * depth_score
    

    # def _best_terminating_node(self):
    #     terminating_nodes = [
    #         n for n in self.node_map.values()
    #         if n.is_terminating and n.merged_value is not None
    #     ]
    #     if not terminating_nodes:
    #         return None
        
    #     # Find the modified filenames from diff
    #     for t in terminating_nodes:
    #         patch = t.observation
    #         modified_files = self._extract_changed_files(patch)
    #         # Now calculate average reward of modified files based on relevance dict
    #         if len(modified_files) > 0:
    #             file_rewards = [
    #                 self.relevance_dict[f]
    #                 for f in modified_files
    #                 if f in self.relevance_dict
    #             ]
                
    #             if len(file_rewards) > 0:
    #                 avg_file_reward = sum(file_rewards) / len(file_rewards)
    #             else:
    #                 avg_file_reward = 0  # or handle this case however you prefer
            
        
    #     return best_node
    def _get_path(self, node):
        path = []
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]
    
    def _count_shared_path_length(self, node_A, node_B):
        path_A = self._get_path(node_A)
        path_B = self._get_path(node_B)
        
        i = 0
        while i < min(len(path_A), len(path_B)) and path_A[i] == path_B[i]:
            i += 1
        
        return i
    
    def _get_max_divergence_path_length(self, node_A, node_B):
        path_A = self._get_path(node_A)
        path_B = self._get_path(node_B)
        
        i = 0 # shared path length
        while i < min(len(path_A), len(path_B)) and path_A[i] == path_B[i]:
            i += 1
            
        return max(len(path_A) - i, len(path_B) - i)
        
    def query_two_nodes(self, node_A, node_B) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        
        messages = self.get_messages_two_nodes(node_A, node_B)
        # save to file for debugging
        with open("debug_messages.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
        response = self.model.query(messages)
        if "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in response["content"]:
            response["content"] = response["content"].replace("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached")
        return response
    
    def _generate_merge_action(self, node_A, node_B):
        """
        Generate an action from the model and parse it
        
        Returns:
            response (dict): The raw response from the model
            action (dict): The parsed action
            error (str | None): The error message if parsing failed
        """
        
        response = self.query_two_nodes(node_A, node_B)
        try:
            action = self.parse_action(response)
            return response, action, None
        except FormatError as e:
            return response, None, str(e)
        
    def _merge_nodes(self, node_A, node_B):            
        self.SYSTEM_PROMPT = self.candidates[0]["SYSTEM_PROMPT"]
        self.USER_PROMPT = self.candidates[0]["USER_PROMPT"]
        node_A.commit = node_A.parent.commit
        node_B.commit = node_B.parent.commit
        node_A.executed = node_B.executed = True
        node_A.itr = node_B.itr = self.itr + 1
        node_A.order = node_B.order = self.n_expanded + 1
        # self.n_expanded += 1
        response, action, error = self._generate_merge_action(node_A, node_B)
        instance_logger.debug(f">> Merging nodes [{node_A.id}] and [{node_B.id}] with shared path length {self._count_shared_path_length(node_A, node_B)} and max divergence path length {self._get_max_divergence_path_length(node_A, node_B)}")
        
        merged_node = self._action_to_node(response, action, error, node_A)
        
        return  merged_node # A is parent, since it has higher value
            
    def get_messages_two_nodes(self, node_A, node_B) -> List[dict]:
        if node_A.commit != node_B.commit:
            raise ValueError("Nodes must be on the same commit to merge their paths for messaging.")

        def linearize_path(path):
            messages = []
            for node in path:
                if node.last_action is None:
                    continue
                messages.append({"role": "user", "content": node.observation})
                messages.append({"role": "assistant", "content": node.last_action["thought"]})
            return messages
        
        def format_suffix(path):
            lines = []
            for node in path:
                if node.last_action is None:
                    continue
                lines.append("Action: " + node.last_action["thought"])
                lines.append("Observation: " + node.observation + "\n")
            return "\n".join(lines)
    
        path_a = self._get_path(node_A)
        path_b = self._get_path(node_B)
        
        # find LCA
        i = 0
        while i < min(len(path_a), len(path_b)) and path_a[i] == path_b[i]:
            i += 1
            
        common = path_a[:i]
        suffix_a = path_a[i:]
        suffix_b = path_b[i:]
        
        messages = []
        
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT
        })
        messages.append({
            "role": "user",
            "content": self.USER_PROMPT
        })
        
        # shared prefix (unchanged)
        messages.extend(linearize_path(common))

        # divergence marker (VERY important)
        messages.append({
            "role": "user",
            "content": f"""
From this point, two alternative action sequences were explored. 
Your task is to reason across both and decide the best next action.

=== Path A ===
{format_suffix(suffix_b)}

=== Path B ===
{format_suffix(suffix_a)}

Given both trajectories, what is the best next action to take from this point?
"""   
        })
            
        return messages
    
    
    def _make_pairs_elite(self, bucket):
        n = len(bucket)
        used = set()
        pairs = []

        for i in range(n):
            if i in used:
                continue

            best_j = None
            best_score = float("inf")

            for j in range(i+1, n):
                if j in used:
                    continue

                # bucket items are (node, priority) tuples: extract node objects
                ni = bucket[i][0]
                nj = bucket[j][0]

                score = self._get_max_divergence_path_length(ni, nj)

                if score < best_score:
                    best_score = score
                    best_j = j

            if best_j is not None:
                pairs.append((bucket[i], bucket[best_j]))
                used.add(i)
                used.add(best_j)
            else:
                pairs.append((bucket[i], None))
                used.add(i)

        return pairs
    
    def _make_pairs_hungarian(self, bucket):
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        import math
        
        n = len(bucket)

        # If empty or single node
        if n <= 1:
            return [(bucket[0], None)] if n == 1 else []

        # Make even-sized by padding dummy nodes
        even_n = n if n % 2 == 0 else n + 1

        # Keep original bucket items (may be tuples). For scoring we extract node objects.
        nodes = bucket + [None] * (even_n - n)

        cost = np.zeros((even_n, even_n))

        INF = 1e9

        for i in range(even_n):
            for j in range(even_n):
                if i == j:
                    cost[i][j] = INF
                else:
                    ni = nodes[i]
                    nj = nodes[j]

                    if ni is None or nj is None:
                        cost[i][j] = 0  # allow dummy pairing
                    else:
                        ni_obj = ni[0]
                        nj_obj = nj[0]
                        cost[i][j] = self._get_max_divergence_path_length(ni_obj, nj_obj)
                        
        row_ind, col_ind = linear_sum_assignment(cost)
        
        used = set()
        pairs = []

        for i, j in zip(row_ind, col_ind):
            if i in used or j in used:
                continue

            used.add(i)
            used.add(j)

            ni = nodes[i]
            nj = nodes[j]

            if ni is None and nj is None:
                continue
            elif ni is None:
                continue
            elif nj is None:
                pairs.append((ni, None))
            else:
                pairs.append((ni, nj))

        return pairs


    # Global Greedy
    def _make_pairs(self, bucket):
        import heapq

        n = len(bucket)

        if n == 0:
            return []
        if n == 1:
            return [(bucket[0], None)]

        heap = []
        for i in range(n):
            for j in range(i + 1, n):
                cost = self._get_max_divergence_path_length(bucket[i][0], bucket[j][0])
                heapq.heappush(heap, (cost, i, j))  # Store cost, priorities, and indices

        active = set(range(n))
        pairs = []

        while heap and len(active) > 1:
            cost, i, j = heapq.heappop(heap)

            if cost > 5:
                break

            if i not in active or j not in active:
                continue

            # store indices along with nodes
            pairs.append((i, j))

            active.remove(i)
            active.remove(j)

        # remaining singles
        for i in active:
            pairs.append((i, None))

        # 🔑 sort by earliest appearance in bucket
        pairs.sort(key=lambda x: x[0])

        # convert indices → actual nodes
        result = []
        for i, j in pairs:
            if j is None:
                result.append((bucket[i], None))
            else:
                result.append((bucket[i], bucket[j]))

        return result

    def _coalesce_dual_nodes(self, nodes, k):
        buckets = {}
        bucket_count = 0
        
        chunk_size = 2
        priority = 1
        for node in nodes:
            if node.modifies_code: 
                if bucket_count < k:
                    buckets[str(uuid.uuid4())] = [(node, priority)]
                    bucket_count += 1
                    priority += 1
            elif not buckets.get(node.parent.commit):
                if bucket_count < k:
                    buckets[node.parent.commit] = [(node, priority)]
                    bucket_count += 1
                    priority += 1
            # else: # NEW
            elif len(buckets[node.parent.commit]) % chunk_size != 0: # OLD
                buckets[node.parent.commit].append((node, priority))
                priority += 1
            # OLD
            elif bucket_count < k:
                buckets[node.parent.commit].append((node, priority))
                bucket_count += 1
                priority += 1
               
               
        old_tree_node = self.tree_node

         
        # Merge
        merged_nodes = []
        for commit, bucket in buckets.items():
            if len(bucket) == 1:
                merged_nodes.append(bucket[0])
            else:
                # pairs = self._make_pairs(bucket) # NEW
                pairs = self._make_pairs_elite(bucket) # OLD
                # TODO: We need to make pair of nodes, so that size of prefix is maximized and suffix is minimized.
                for p in pairs:
                    if p[1] is None:
                        merged_nodes.append(p[0])
                    else:
                        self._backtrack(p[0][0])
                        self.tree_node = p[0][0]
                        merged_node = self._merge_nodes(p[0][0], p[1][0])
                        p[0][0].add_child(merged_node)
                        p[1][0].add_child(merged_node)
                        merged_node.merged = True
                        merged_node.parent = p[0][0]
                        # if merged_node.is_terminating: # NEW
                        merged_node.value = merged_node.merged_value = self._evaluate_node(merged_node)
                        # else:
                            # merged_node.value = merged_node.merged_value = (0.8 * p[0][0].merged_value + 0.2 * p[1][0].merged_value) # We can also experiment with other ways of aggregating values, like max or min, or even giving more weight to the node with higher value. This is a hyperparameter that can be tuned based on the task and the size of the tree.
                        merged_nodes.append((merged_node, p[0][1])) # Keep the highest priority among merged nodes
        
        self._backtrack(old_tree_node)
        self.tree_node = old_tree_node
        
        # Return only nodes (drop priority) but preserve ordering by priority
        return [x[0] for x in sorted(merged_nodes, key=lambda x: x[1])]
    
    
    def _coalesce_nodes(self, nodes, k):
        buckets = {}
        bucket_count = 0
        
        chunk_size = 2
        priority = 1
        for node in nodes:
            if node.modifies_code: 
                if bucket_count < k:
                    buckets[self._estimate_commit(node)] = [(node, priority)]
                    bucket_count += 1
                    priority += 1
            elif not buckets.get(node.parent.commit):
                if bucket_count < k:
                    buckets[node.parent.commit] = [(node, priority)]
                    bucket_count += 1
                    priority += 1
            elif len(buckets[node.parent.commit]) < chunk_size * k: # NEW --- we are discarding bad nodes. So, we may not find pairs for some commits, but that's ok. We will just keep the good nodes and discard the bad ones.
                buckets[node.parent.commit].append((node, priority))
                priority += 1
               
        old_tree_node = self.tree_node

         
        # Merge
        merged_nodes = []
        new_buckets = {}
        while len(merged_nodes) < k and len(buckets) > 0:
            for commit, bucket in buckets.items():
                if len(bucket) == 1:
                    merged_nodes.append(bucket[0])
                elif len(bucket) > 1:
                    best_j = None
                    best_score = float("inf")

                    for node in bucket[1:]:
                        ni = bucket[0][0]
                        nj = node[0]

                        score = self._get_max_divergence_path_length(ni, nj)

                        if score < best_score and score <= 5:
                            best_score = score
                            best_j = node

                    if best_j is not None:
                        a_node = bucket[0][0]
                        b_node = best_j[0]
                        self._backtrack(a_node)
                        self.tree_node = a_node
                        merged_node = self._merge_nodes(a_node, b_node)
                        a_node.add_child(merged_node)
                        b_node.add_child(merged_node)
                        merged_node.merged = True
                        merged_node.parent = a_node
                        if merged_node.is_terminating:
                            merged_node.value = merged_node.merged_value = self._evaluate_node(merged_node)
                        else:
                            merged_node.value = merged_node.merged_value = (0.8 * a_node.merged_value + 0.2 * b_node.merged_value)
                        merged_nodes.append((merged_node, bucket[0][1])) # Keep the highest priority among merged nodes
                        if merged_node.modifies_code:
                            new_buckets[self._estimate_commit(merged_node)] = [item for item in bucket if item[0] not in [a_node, b_node]] # Add remaining nodes in the bucket to the new bucket based on the commit of the merged node
                        else:
                            new_buckets[commit] = [item for item in bucket if item[0] not in [a_node, b_node]] # Add remaining nodes in the bucket to the new bucket based on the original commit since merged node doesn't modify code
                    else:
                        merged_nodes.append(bucket[0])
                        new_buckets[commit] = bucket[1:] # Keep the remaining nodes in the bucket for future merging
                
                
            buckets = new_buckets
            new_buckets = {}
            
        self._backtrack(old_tree_node)
        self.tree_node = old_tree_node
        
        # Return only nodes (drop priority) but preserve ordering by priority
        return [x[0] for x in sorted(merged_nodes, key=lambda x: x[1])]
    
    def _estimate_commit(self, node):
        curr = node
        while curr is not None:
            if curr.commit is not None:
                return curr.commit
            elif curr.modifies_code: # Scenario: When a merged node modifies code. And we try to merge the merged node with another node.
                return curr.id
            curr = curr.parent
        return None
        
    def _coalesce_nodes_aggressively(self, nodes, k):
        buckets = {}
        bucket_count = 0
        
        # NEW: Merge more aggressively until we have at most k nodes. Preserve a priority for ordering.
        node_count = 0
        priority = 1
        for node in nodes:
            n_commit = self._estimate_commit(node)
            if not buckets.get(n_commit):
                if bucket_count < k:
                    buckets[n_commit] = [(node, priority)]
                    bucket_count += 1
                    node_count += 1
                    priority += 1
            else:
                buckets[n_commit].append((node, priority))
                node_count += 1
                priority += 1
                
        
        new_buckets = {
            # inititialize empty array for each commit in buckets
            commit: [] for commit in buckets.keys()
        }
        while node_count > k:
            no_progress = True
            for commit, bucket in buckets.items():
                if len(bucket) > 1:
                    pairs = self._make_pairs(bucket)
                    for p in pairs:
                        # p elements are taken from bucket and may be (node, priority) tuples
                        if p[1] is None:
                            # keep the tuple as-is
                            new_buckets[commit].append(p[0])
                        else:
                            a_item = p[0]
                            b_item = p[1]
                            # a_item and b_item are (node, priority) tuples
                            a_node = a_item[0]
                            b_node = b_item[0]
                            merged_node = self._merge_nodes(a_node, b_node)
                            a_node.add_child(merged_node)
                            b_node.add_child(merged_node)
                            merged_node.merged = True
                            merged_node.parent = a_node
                            if merged_node.is_terminating:
                                merged_node.value = merged_node.merged_value = self._evaluate_node(merged_node)
                            else:
                                merged_node.value = merged_node.merged_value = (0.8 * a_node.merged_value + 0.2 * b_node.merged_value)
                            node_count -= 1
                            # compute new priority: prefer earlier appearance
                            a_pr = a_item[1]
                            b_pr = b_item[1]
                            new_pr = min(a_pr, b_pr)
                            if merged_node.modifies_code:
                                new_buckets[self._estimate_commit(merged_node)] = [(merged_node, new_pr)]
                            else:
                                new_buckets[commit].append((merged_node, new_pr))
                            no_progress = False
                        # else: # If we have already merged enough nodes to be within the limit, we can just add the remaining nodes without merging to preserve information and encourage exploration of different paths.
                        #     new_buckets[commit].append(p[0])
                        #     new_buckets[commit].append(p[1])
                            
                else:    
                    new_buckets[commit] = bucket
                          
            buckets = new_buckets
            new_buckets = {
                # inititialize empty array for each commit in buckets
                commit: [] for commit in buckets.keys()
            }
            
            if no_progress: # If we went through all buckets and couldn't merge any nodes, we should break to avoid infinite loop.
                break
            
        merged_nodes = []
        for commit, bucket in buckets.items():
            merged_nodes.extend(bucket)

        # merged_nodes are tuples (node, priority); return plain nodes preserving order
        return [x[0] for x in merged_nodes]
        
    def _slice_topk(self, nodes: List[TreeSearchNode], k: int) -> List[TreeSearchNode]:
        # Future Work
        # Instead of discarding nodes beyond top-k, we can gather information from them and merge nodes with same commit.
        # Merging nodes means, let's say 3 nodes in the array have the same commit, then we will give the diverge path trajectory of all of them to agent, and ask it to consider all information and generate next step based on that. This way we can keep more information from the tree and also encourage exploration of different paths while still keeping the search focused on promising trajectories. This can be especially helpful in cases where the reward signal is sparse and we want to gather as much information as possible to guide the search.
        # Since the array is sorted, we will traverse it from start to end, and keep adding nodes to the top-k array until we have k unique commits.
        # We don't want to merge more than 2 nodes for now, so let's say there are 3 same commit at the start of an array, we will group the first 2, and treat 3rd as separate commit to encourage exploration of different paths. This is a hyperparameter that can be tuned based on the task and the size of the tree.
        
        merged_nodes = self._coalesce_dual_nodes(nodes, k) 
        # merged_nodes = self._coalesce_nodes_aggressively(nodes, k)
                 
        # return nodes[:k] # OLD
        return merged_nodes[:k] # NEW:
    
    def _get_all_unique_edit_paths(self, to_execute=False):
        # For each commit, find the best node
        sorted_nodes = sorted(
            (
                n
                for n in self.all_node_map.values()
                if not n.is_terminating
                # and not n.executed # NEW
                and n.visible
                and n.merged_value is not None
                and (not to_execute or n.level < self.config.depth_limit)
                and (n.parent.commit != self._get_root_commit() or n.modifies_code)
                # and (n.commit is not None and n.commit not in self.terminating_nodes)
                # and (n.commit is None and self._estimate_commit(n) not in self.terminating_nodes)
            ),
            key=lambda n: n.get_path_value(0.85),
            reverse=True
        )
        
        # Now pick the best node for each unique commit, and return those nodes as unique edit paths
        unique_paths = {}
        for node in sorted_nodes:
            commit = self._estimate_commit(node)
            if commit not in unique_paths:
                unique_paths[commit] = node
                
        return list(unique_paths.values())
        
    def _prune_priority(self, node, max_depth):
        # Early explored nodes should get some advantage
        return self.node_priority(
                        node, 
                        gamma=0.85,
                        max_depth=max_depth
                    ) * 0.95 + (1 - (node.parent.order / self.config.step_limit)) * 0.05,
        
                    
    def _get_topk_edit_paths(self, k=None, to_execute=True):
        if k is None:
            k = self.config.top_k_tree_pruning
        curr_i = self.itr 
        sorted_leaves = []
        
        while len(sorted_leaves) < k and curr_i > 0:
            max_depth = max(n.level for n in self.node_map_itr[curr_i].values())
            candidates = sorted(
                (
                    n
                    for n in self.node_map_itr[curr_i].values()
                    if not n.executed
                    and not n.is_terminating
                    and n.visible
                    and n.merged_value is not None
                    and (not to_execute or n.level < self.config.depth_limit)
                    and  (n.parent.commit != self._get_root_commit() or n.modifies_code)
                ),
                key=lambda n: self._prune_priority(n, max_depth=max_depth),
                reverse=True
            )
            sorted_leaves.extend(candidates)
            curr_i -= 1
        
        # OLD2
        # max_depth = max(n.level for n in self.all_node_map.values())
        # sorted_leaves = sorted(
        #     (
        #         n
        #         for n in self.all_node_map.values()
        #         if not n.executed
        #         and not n.is_terminating
        #         and n.visible
        #         and n.merged_value is not None
        #         and (not to_execute or n.level < self.config.depth_limit)
        #         and  (n.parent.commit != self._get_root_commit() or n.modifies_code)
        #     ),
        #     key=lambda n: (
        #         n.parent.itr,  # NEW: recency bias (higher = newer)
        #         # n.parent.itr == self.itr,  # OLD: Prioritize nodes generated in the current iteration to encourage exploitation of new promising paths
        #         self.node_priority(
        #             n, 
        #             gamma=0.85,
        #             max_depth=max_depth
        #         ),
        #     ),
        #     reverse=True
        # )
        
        # OLD:
        # candidates = [
        #     n for n in self.node_map.values()
        #     if not n.executed
        #     and not n.is_terminating
        #     and n.visible
        #     and n.merged_value is not None
        #     and n.level < self.config.depth_limit
        #     and n.commit != self._get_root_commit()
        # ]
        
        # sorted_leaves = sorted(
        #     candidates,
        #     key=lambda x: self.node_priority(x, gamma=0.85, max_depth=max(n.level for n in candidates)),
        #     reverse=True
        # )
        
        # if len(sorted_leaves) < k:
        #     candidates = [
        #         n 
        #         for n in self.all_node_map.values() # OLD
        #         if not n.executed
        #         and not n.is_terminating
        #         and n.visible
        #         and n.merged_value is not None
        #         and n.level < self.config.depth_limit
        #         and n.commit != self._get_root_commit()
        #     ]
            
        #     sorted_leaves = sorted(
        #         candidates,
        #         key=lambda x: self.node_priority(x, gamma=0.85, max_depth=max(n.level for n in candidates)),
        #         reverse=True
        #     )
            
        top_k = self._slice_topk(sorted_leaves, k)
        instance_logger.debug(f">> Found {len(top_k)} edit paths")
        return top_k 
    
              
    def _update_iteration(self):
        self.node_map_itr[self.itr] = self.node_map # NEW:
        
        if self.n_submissions >= self.config.sub_thres and len(self.terminating_nodes) >= self.config.u_sub_thres:
        # if len(self.terminating_nodes) >= self.config.sub_thres: # Too harsh
            # TODO: Should we just terminate or consider terminating actions from here?
            # We are done exploring. Now check the tree if there is any terminating action. If multiple, choose the one with highest path value/reward. If none, choose the one with highest path value among all nodes and run sequentially from there until we reach a terminating node.   
            best_node = self._get_best_terminating_node()
            instance_logger.debug(">> Discovered enough solutions. Best terminating node: [{}] with merged value {}".format(best_node.id, best_node.merged_value))
            self.frontier.clear()
            self.node_map = {best_node.id: best_node}
            self._update_frontier([best_node]) # Update frontier with the best node to encourage exploitation of the best solution path
        elif self.itr == self.config.itr_limit:
            if self.mode == "evaluation":
                top_k = self._get_topk_edit_paths(k=1)
                if len(top_k) > 0:
                    best_node = top_k[0]
                else:
                    # If no nodes with edits are found, fallback to any promising node regardless of edits to at least have some path to follow.
                    max_depth = max(n.level for n in self.node_map.values())
                    best_node = max(
                        (
                            n for n in self.node_map.values() 
                            if n.merged_value is not None 
                            and not n.is_terminating
                            and n.visible 
                            and not n.executed
                            and n.level < self.config.depth_limit
                        ),
                        key=lambda n: self._prune_priority(n, max_depth=max_depth),
                        default=None
                    )
                    
                    if best_node is None: # --- All available nodes are terminating
                        best_node = self._get_best_terminating_node()
                        instance_logger.debug(">> No non-terminating actions available for execution. Best terminating node: [{}] with merged value {}".format(best_node.id, best_node.merged_value if best_node else "N/A"))
                        
                    else: 
                        instance_logger.debug(">> Iteration limit reached. Selecting best node based on merged value: [{}] with merged value {}".format(best_node.id, best_node.merged_value if best_node else "N/A"))
                self.frontier.clear()
                self.node_map = {best_node.id: best_node} # Prune the rest of the tree by keeping only the best node in the node map
                self._update_frontier([best_node])
        else: 
            if self.mode == "evaluation":
                self.frontier.clear()
                # Update frontier with top-k non-terminating leaves [EXPLOITATION]
                # Find max depth among all nodes
                max_depth = max(n.level for n in self.node_map.values())
                if self.itr + 1 < self.config.itr_limit:
                    sorted_leaves = sorted(
                        (
                            n
                            for n in self.node_map.values()
                            if not n.executed
                            and not n.is_terminating
                            and n.visible
                            and n.merged_value is not None
                            and n.level < self.config.depth_limit
                        ),
                        key=lambda n: self._prune_priority(n, max_depth=max_depth),
                        reverse=True,
                    )
                    top_k = self._slice_topk(sorted_leaves, k=self.config.top_k_tree_pruning) # Keep top-k
                else: # On the last iteration, prioritize nodes with edits regardless of score to encourage exploitation of promising edit paths. If not enough edit paths are found, go for read paths.
                    instance_logger.debug(">> Iteration {} reached. Prioritizing nodes with edits for exploitation.".format(self.itr + 1))
                    sorted_writes = self._get_topk_edit_paths(k=self.config.top_k_tree_pruning)
                    top_k = sorted_writes
                    # If not enough edit paths are found, go for read paths.
                    # TODO: If len(sorted_writes) > 0, we shouldn't gather reads. Just keep the edit paths even if they are less than k, since we want to encourage exploitation of promising edit paths when we are close to iteration limit. We can experiment with different values of k for edit paths and read paths to find the best balance between exploitation and exploration.
                    if len(sorted_writes) < self.config.top_k_tree_pruning:
                        sorted_reads = sorted(
                            (
                                # n for n in self.all_node_map.values() # OLD
                                n for n in self.node_map.values() 
                                if not n.executed
                                and not n.is_terminating
                                and n.visible
                                and n.merged_value is not None
                                and n.level < self.config.depth_limit
                                and (n.parent.commit == self._get_root_commit() and not n.modifies_code)
                            ),
                            key=lambda n: self._prune_priority(n, max_depth=max_depth),
                            reverse=True,
                        )
                        top_k.extend(self._slice_topk(sorted_reads, k=self.config.top_k_tree_pruning - len(sorted_writes)))
                    
                # Don't put terminating nodes in frontier.
                self._update_frontier([
                    n for n in top_k if not n.is_terminating
                ]) # -> It will sort based on merged_value
                for n in top_k:
                    if n.is_terminating:
                        self.n_submissions += 1
                        self.terminating_nodes[n.parent.commit] = self.terminating_nodes.get(n.parent.commit, []) + [n]
                        
                # Keep top-k active nodes
                self.node_map = {n.id: n for n in top_k}
                instance_logger.debug(f">> Iteration {self.itr}: Updating frontier with top {len(top_k)} non-terminating leaves based on path value.")
        self.itr += 1

    def _expand(self):
        if self.mode == "simulation":
            tree_nodes = [
                n for n in self.tree_node.children if not n.system_generated and n.merged_value is not None
            ]
            instance_logger.debug(f"# {len(tree_nodes)} new nodes generated at level {self.tree_node.level}:")
            reward_data = []
            for new_node in tree_nodes:
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
                instance_logger.debug(
                    tabulate(
                        reward_data,
                        headers=["Action", "Reward", "Merged"],
                        tablefmt="grid",
                        colalign=("left", "center", "center"),
                    )
                )
            return 
        
        if self.tree_node.visits == 0:
            tree_nodes = self._generate_new_nodes(self.config.branching_factor)
            tree_nodes = self._update_tree(tree_nodes)
            for node in tree_nodes:
                if node.is_terminating:
                    if self.terminating_nodes.get(node.parent.commit) is None:
                        self.terminating_nodes[node.parent.commit] = []
                    self.terminating_nodes[node.parent.commit].append(node)
                    
    def _has_reached_finish_line(self):
        # Find the max node.order among all nodes in the tree (from self.tree_root). If self.n_expanded has reached that, then we have reached the finish line.

        def _find_max_order(node):
            max_order = node.order
            for child in node.children:
                max_order = max(max_order, _find_max_order(child))
            return max_order

        max_order = _find_max_order(self.tree_root)
        # print(f">> Max order in the tree: {max_order}, Current expanded nodes: {self.n_expanded}")
        return self.n_expanded == max_order or self.n_expanded == max_order - 1 # TODO: max_order varies. Needs to fix.

    def _get_best_terminating_node_from_checkpoint(self):
        terminating_nodes = []
        def _collect_terminating_nodes(node):
            if node.is_terminating and node.visible:
                terminating_nodes.append(node)
            for child in node.children:
                _collect_terminating_nodes(child)
            return terminating_nodes
        
        terminating_nodes = _collect_terminating_nodes(self.tree_root)
        if len(terminating_nodes) == 0:
            return None
        
        for n in terminating_nodes:
            # TEMP: Re-evaluate terminating nodes to fix value bug. We can remove this once the bug is fixed.
            # If n has a sibling terminating node, evaluate both and merge
            if n.merged_value > n.value:
                # Find sibling terminating nodes
                sibling_terminating_nodes = [
                    s for s in n.parent.children 
                    if s.is_terminating
                ]
                for s in sibling_terminating_nodes:
                    s.value = self._evaluate_node(s)
                action_processor.merge_nodes(sibling_terminating_nodes)  
            else:
                n.value = n.merged_value = self._evaluate_node(n)
            n.is_submission = False

        best_node = max(
            terminating_nodes,
            key=lambda x: (
                # x.merged_value, # OLD
                0.8 * x.merged_value + (1 - (x.parent.order / self.config.step_limit)) * 0.1 + 0.1 * (not x.system_generated), # NEW:  Should give priority to early discovered solutions
                # NEW: Give priority to AI generated nodes
                x.get_path_value(0.85) # NEW: In case of tie
            ),
            default=None
        )

        return best_node
        

        
    def _get_best_node_from_checkpoint(self):
        # traverse the tree from self.tree_root and find the node with node.order == self.n_expanded + 1
        # instance_logger.debug(f">> Finding best node with order {self.n_expanded + 1} in the tree for iteration {self.itr}")
        def _search_best(node):
            if node.itr == self.itr and node.order == self.n_expanded + 1:
                return node
            
            elif not node.children:
                return None
            
            for child in node.children:
                # TEMP
                if child.value is None and child.merged_value is not None:
                    child.value = child.merged_value
                    instance_logger.debug(">> Fixing Value BUG")

                best_node = _search_best(child)
                if best_node:
                    return best_node
            return None
        
        best_node = _search_best(self.tree_root)
        return best_node
        
    def _select(self):
        best_node = None 
        while best_node is None:
            if self.mode == "evaluation" and self.tree_node.visits == 1:
                # Only update frontier when node first expanded.
                candidates = []
                # if self.itr > self.config.itr_limit:
                #     # prioritize terminating nodes when iteration limit is reached to encourage exploitation and avoid over-exploration which can lead to noise and long backtracking
                candidates = [
                    c for c in self.tree_node.children 
                    if not c.executed 
                    and c.visible 
                    and not c.is_terminating # Don't need to execute terminating actions. Generating them is enough.
                    and (c.level < self.itr * 5 or (self.itr > self.config.itr_limit and c.level < self.config.depth_limit)) # Don't expand too deep to avoid noise and long backtracking
                ]
                
                if self.itr > self.config.itr_limit:
                    self.frontier.clear() # Clear frontier when iteration limit is reached to focus on exploitation of promising node
                    if len(candidates) == 0: # Only terminating actions are left
                        # TODO: Expand other write paths if n_submissions < config.sub_thres. Since we have some budget left.
                        self.node_map_itr[self.itr] = self.node_map # NEW:
                        candidates = [self._get_best_terminating_node()] # Get the best terminating node among all nodes in the current iteration. Generating new nodes won't help at this point, so we directly go for the best terminating node if there are no non-terminating nodes left to execute.
                         
                self._update_frontier(candidates)
                
            if ((self.itr == 1 and self.n_expanded >= 10) or self.n_expanded >= 10 + (self.itr-1) * 10) and self.itr <= self.config.itr_limit:
                self._update_iteration()

            
            while best_node is None:
                if self.mode == "simulation":
                    best_node = self._get_best_node_from_checkpoint()
                    if best_node:
                        break

                if not self.frontier.empty():
                    best_node = self._select_action()
                    if best_node.parent != self.tree_node:
                        self._backtrack(best_node.parent)    
                        best_node.parent.visits += 1
                        self.n_backtracks += 1   
                        instance_logger.debug(">> Backtrack needed to execute the highest-rewarded action.")
                elif self.mode == "simulation" and self._has_reached_finish_line():
                    # Find best terminating node
                    instance_logger.debug(">> Finish line reached. Searching for best terminating node in the tree.")
                    best_node = self._get_best_terminating_node_from_checkpoint()
                else:
                    instance_logger.debug(f">> No actions in frontier. Updating iteration to expand more nodes. Current iteration: {self.itr}")
                    # NEW:
                    self._update_iteration()
                    
                    # OLD:
                    # max_depth = max(n.level for n in self.node_map.values())
                    # scored_leaves = [
                    #     (n, self.node_priority(n, gamma=0.85, max_depth=max_depth))
                    #     for n in self.node_map.values()
                    #     if not n.executed
                    #     and not n.is_terminating
                    #     and n.visible
                    #     and n.merged_value is not None
                    #     and n.level < self.config.depth_limit
                    # ]
                    # # sort by score
                    # scored_leaves.sort(key=lambda x: x[1], reverse=True)
                    # # unpack if needed
                    # sorted_leaves = [n for n, _ in scored_leaves]
                    # top_k = sorted_leaves[:3] # Keep top 3
                    # self._update_frontier(top_k) # -> It will sort based on merged_value
                    # # Keep top-k active nodes
                    # self.node_map = {n.id: n for n in top_k}
                    # self.itr += 1
                    
        return best_node
    
    def _act(self):
        if self.tree_node.is_terminating:
            self._stage_to_main_branch()
            self.tree_node.is_submission = True
            self.frontier.reset()
            
        if not self.tree_node.system_generated:
            self.add_message("assistant", **{"content": self.tree_node.last_action["thought"], "extra": self.tree_node.last_action.get("extra", {})})
        else: # Action generated by System
            self.add_message("system", self.tree_node.last_action["thought"])
        instance_logger.debug(f">> Executing selected action #{self.n_expanded + 1}: {self.tree_node.last_action['command']}")
        self.tree_node.executed = True

    def _observe(self):
        if self.mode == "simulation" or self.tree_node.last_action["command"] is None or (not self.tree_node.is_terminating and not self.tree_node.modifies_code): # For read-only action, no need to re-execute
            observation = self.tree_node.observation
            if self.tree_node.is_terminating:
                raise Submitted("".join(self.tree_node.observation))
        else: # For write action, need to change the environment and get new (which will be basically same as before) observation
            output = self.get_observation(
                {
                    "action": self.tree_node.last_action["command"]
                }
            )
            observation = self.render_template(self.config.action_observation_template, output=output)
            
        instance_logger.debug(f">> Observation: {observation[:200]}...") # Log only the beginning of the observation to avoid cluttering the logs
        self.add_message("user", observation)
        self.tree_node.observation = observation

        if self.mode == "evaluation":
            if self.tree_node.modifies_code:            
                self.tree_node.commit, _ = self._commit_changes()
                instance_logger.debug(f">> New commit created: {self.tree_node.commit}")
                
            else:
                self.tree_node.commit = self._get_commit_hash()
                instance_logger.debug(f">> No changes detected, staying on commit: {self.tree_node.commit}")
                
    def step(self) -> dict:
        if self.tree_node.is_terminating:
            self._create_pseudo_root()
            
        self._expand()
        
        if self.mode == "evaluation":
            self.tree_node.visits += 1
            self.tree_node.itr = self.itr
            self.tree_node.order = self.n_expanded
            
        self.tree_node = self._select()
        self._act()
        self._observe()
        self.n_expanded += 1
        
        with open("debug_tree.json", "w", encoding="utf-8") as f:
            json.dump(self.tree_root.to_tree(), f, indent=4, ensure_ascii=False)

        with open("debug_nodes.json", "w", encoding="utf-8") as f:
            json.dump(self.tree_root.to_json(), f, indent=4, ensure_ascii=False)
                
        return self.tree_node.observation
        
    def step_v2(self) -> dict:
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
            
            if self.config.selection_scope == "global" and (self.n_submissions >= self.config.sub_thres) and self.phase == 2:
                instance_logger.debug(":: Switching to phase 3: Allowing terminating actions.")
                self.phase = 3  # Switch to phase 3 after 50% of steps
                candidates = [
                    n for n in self.node_map.values()
                    if n.merged_value is not None
                    and n.visible
                    and not n.executed
                    and n.is_terminating
                    and self.is_promising(n)
                    and n.parent.id != self.tree_node.id
                ]
                # Adding left out terminating nodes from previous phases
                self._update_frontier(candidates)
                self.add_message("system", "Switching to phase 3: Allowing terminating actions.")
                # TODO: May do more sophisticated checks for choosing the best terminating node
                # Compare test status?
                
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
                
            if self.config.selection_scope == "global" and self.phase == 2 and self.n_modifications == 0 and self.n_expanded >= 4*self.config.step_limit/5 and self.frontier.budget > 2:
                self.frontier.reduce_budget() # Try hard to find some modification actions if we haven't found any yet in phase 2
                 
            
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
            instance_logger.debug(f"Queue size {self.frontier.length()}. Adding new {len(tree_nodes)} actions...")
            
        # PRUNE READ Action
        # best_node = max(tree_nodes, key=lambda x: x.merged_value)
        # if not best_node.modifies_code:
        #     tree_nodes = [best_node]  
            
        self._add_actions_to_frontier(tree_nodes)
    