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
    itr_limit: int = 4
    

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
        issue_tokens = self.task.split()
        scores = self.bm25.get_scores(issue_tokens)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        self.relevance_dict = dict(zip(self.file_ids, scores))
        
    def _backtrack(self, target_node):
        instance_logger.debug(f">> Backtracking from [{self.tree_node.id}] to [{target_node.id}]")
        n_commit = self._estimate_commit(target_node)
        if n_commit != self.tree_node.commit:
            instance_logger.debug(f">> Backtracking from [{self.tree_node.commit[:7]}] to [{n_commit[:7]}]")

            # if self._get_branch_head(target_node.branch) != target_node.commit:
            self.env.execute(f"git checkout {n_commit}")
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
    
    def _get_best_terminating_node(self) -> Optional[TreeSearchNode]:
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
                0.9 * x.merged_value + (1 - (x.parent.order / self.config.step_limit)) * 0.1, # NEW:  Should give priority to early discovered solutions
                x.get_path_value(0.85) # NEW: In case of tie
            ),
            default=None
        )
        
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
            candidates = self._get_topk_edit_paths(k=3, to_execute=False) # <----
                    
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
                        print(f">> Adding terminating node [{term_node.id}] under node [{node.id}]")
                        futures.append((term_node, executor.submit(self._evaluate_node, term_node)))
                    if futures:
                        for term_node, future in tqdm(futures, total=len(futures), desc="Waiting for node scores"):
                            term_node.merged_value = term_node.value = future.result()
                
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
        return self._action_to_node(response, action, error, node_A) # A is parent, since it has higher value
            
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

                score = self._get_max_divergence_path_length(bucket[i], bucket[j])

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
                        cost[i][j] = self._get_max_divergence_path_length(ni, nj)
                        
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
                cost = self._get_max_divergence_path_length(bucket[i], bucket[j])
                heapq.heappush(heap, (cost, i, j))

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
        
        # OLD: Just merge 2 and ignore the rest
        chunk_size = 2
        for node in nodes:
            if node.modifies_code: 
                if bucket_count < k:
                    buckets[str(uuid.uuid4())] = [node]
                    bucket_count += 1
            elif not buckets.get(node.parent.commit):
                if bucket_count < k:
                    buckets[node.parent.commit] = [node]
                    bucket_count += 1
            elif len(buckets[node.parent.commit]) % chunk_size != 0:
                buckets[node.parent.commit].append(node)
            elif bucket_count < k:
                buckets[node.parent.commit].append(node)
                bucket_count += 1
                
        # Merge
        merged_nodes = []
        for commit, bucket in buckets.items():
            if len(bucket) == 1:
                merged_nodes.append(bucket[0])
            else:
                pairs = self._make_pairs(bucket)
                # TODO: We need to make pair of nodes, so that size of prefix is maximized and suffix is minimized.
                for p in pairs:
                    if p[1] is None:
                        merged_nodes.append(p[0])
                    else:
                        merged_node = self._merge_nodes(p[0], p[1])
                        p[0].add_child(merged_node)
                        p[1].add_child(merged_node)
                        merged_node.merged = True
                        merged_node.parent = p[0]
                        # merged_node.value = merged_node.merged_value = self._evaluate_node(merged_node) 
                        merged_node.value = merged_node.merged_value = (0.8 * p[0].merged_value + 0.2 * p[1].merged_value) # We can also experiment with other ways of aggregating values, like max or min, or even giving more weight to the node with higher value. This is a hyperparameter that can be tuned based on the task and the size of the tree.
                        merged_nodes.append(merged_node)
        
        return merged_nodes
    
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
        
        # NEW: Merge more aggressively until we have at most k nodes. So, that no path gets just discarded, but gets merged with other paths and we can still gather information from it.
        node_count = 0
        for node in nodes:
            n_commit = self._estimate_commit(node)
            if not buckets.get(n_commit):
                if bucket_count < k:
                    buckets[n_commit] = [node]
                    bucket_count += 1
                    node_count += 1
            else:
            # elif len(buckets[node.parent.commit]) < 2 * k: # NEW
                buckets[n_commit].append(node)
                node_count += 1
                
        
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
                        if p[1] is None:
                            new_buckets[commit].append(p[0])
                        else:
                        # elif node_count > k:
                            merged_node = self._merge_nodes(p[0], p[1])
                            p[0].add_child(merged_node)
                            p[1].add_child(merged_node)
                            merged_node.merged = True
                            merged_node.parent = p[0]
                            merged_node.value = merged_node.merged_value = (0.8 * p[0].merged_value + 0.2 * p[1].merged_value)
                            node_count -= 1
                            if merged_node.modifies_code:
                                new_buckets[self._estimate_commit(merged_node)] = [merged_node]
                            else:
                                new_buckets[commit].append(merged_node)
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
            
        return merged_nodes
        
    def _slice_topk(self, nodes: List[TreeSearchNode], k: int) -> List[TreeSearchNode]:
        # Future Work
        # Instead of discarding nodes beyond top-k, we can gather information from them and merge nodes with same commit.
        # Merging nodes means, let's say 3 nodes in the array have the same commit, then we will give the diverge path trajectory of all of them to agent, and ask it to consider all information and generate next step based on that. This way we can keep more information from the tree and also encourage exploration of different paths while still keeping the search focused on promising trajectories. This can be especially helpful in cases where the reward signal is sparse and we want to gather as much information as possible to guide the search.
        # Since the array is sorted, we will traverse it from start to end, and keep adding nodes to the top-k array until we have k unique commits.
        # We don't want to merge more than 2 nodes for now, so let's say there are 3 same commit at the start of an array, we will group the first 2, and treat 3rd as separate commit to encourage exploration of different paths. This is a hyperparameter that can be tuned based on the task and the size of the tree.
        
        # merged_nodes = self._coalesce_dual_nodes(nodes, k)
        merged_nodes = self._coalesce_nodes_aggressively(nodes, k)
                 
        # return nodes[:k] # OLD
        return merged_nodes[:k] # NEW:
    
        
    def _get_topk_edit_paths(self, k=3, to_execute=True):
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
                key=lambda n: (
                    self.node_priority(
                        n, 
                        gamma=0.85,
                        max_depth=max_depth
                    ),
                ),
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
        print(f">> Found {len(top_k)} edit paths")
        return top_k 
    
              
    def _update_iteration(self):
        self.node_map_itr[self.itr] = self.node_map # NEW:
        
        if self.n_submissions >= self.config.sub_thres:
        # if len(self.terminating_nodes) >= self.config.sub_thres: # Too harsh
            # TODO: Should we just terminate or consider terminating actions from here?
            
            # We are done exploring. Now check the tree if there is any terminating action. If multiple, choose the one with highest path value/reward. If none, choose the one with highest path value among all nodes and run sequentially from there until we reach a terminating node.   
            best_node = self._get_best_terminating_node()
            instance_logger.debug(">> Discovered enough solutions. Best terminating node: [{}] with merged value {}".format(best_node.id, best_node.merged_value))
            self.frontier.clear()
            self.itr += 1
            self.node_map = {best_node.id: best_node}
          
            return best_node
            
        elif self.itr == self.config.itr_limit:
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
                    key=lambda x: self.node_priority(x, gamma=0.85, max_depth=max_depth),
                    default=None
                )
                
                if best_node is None: # --- All available nodes are terminating
                    best_node = self._get_best_terminating_node()
                    instance_logger.debug(">> No non-terminating actions available for execution. Best terminating node: [{}] with merged value {}".format(best_node.id, best_node.merged_value if best_node else "N/A"))
                    
                else: 
                    instance_logger.debug(">> Iteration limit reached. Selecting best node based on merged value: [{}] with merged value {}".format(best_node.id, best_node.merged_value if best_node else "N/A"))

            self.frontier.clear()
            self.itr += 1
            self.node_map = {best_node.id: best_node} # Prune the rest of the tree by keeping only the best node in the node map
            
            return best_node

        else: 
            
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
                    key=lambda n: self.node_priority(n, gamma=0.85, max_depth=max_depth),
                    reverse=True,
                )
                top_k = self._slice_topk(sorted_leaves, k=3) # Keep top 3
            else: # On the last iteration, prioritize nodes with edits regardless of score to encourage exploitation of promising edit paths. If not enough edit paths are found, go for read paths.
                instance_logger.debug(">> Iteration {} reached. Prioritizing nodes with edits for exploitation.".format(self.itr + 1))
                sorted_writes = self._get_topk_edit_paths(k=3)
                top_k = sorted_writes
                # If not enough edit paths are found, go for read paths.
                if len(sorted_writes) < 3:
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
                        key=lambda n: (
                            # n.parent.itr,  # OLD: recency bias (higher = newer)
                            self.node_priority(n, gamma=0.85, max_depth=max_depth)
                        ),
                        reverse=True,
                    )
                    top_k.extend(self._slice_topk(sorted_reads, k=3 - len(sorted_writes)))
                
            self._update_frontier(top_k) # -> It will sort based on merged_value

            # Keep top-k active nodes
            self.node_map = {n.id: n for n in top_k}
            self.itr += 1
            instance_logger.debug(f">> Iteration {self.itr}: Updating frontier with top {len(top_k)} non-terminating leaves based on path value.")
        return None

    def step(self) -> dict:
        if self.tree_node.is_terminating:
            self._create_pseudo_root()
            
        if self.tree_node.visits == 0:
            tree_nodes = self._generate_new_nodes(self.config.branching_factor)
            tree_nodes = self._update_tree(tree_nodes)
            
            for node in tree_nodes:
                if node.is_terminating:
                    if self.terminating_nodes.get(node.parent.commit) is None:
                        self.terminating_nodes[node.parent.commit] = []
                    self.terminating_nodes[node.parent.commit].append(node)
        
        self.tree_node.visits += 1
        self.tree_node.itr = self.itr
        self.tree_node.order = self.n_expanded
        
        best_node = None 
        while best_node is None:
            if self.tree_node.visits == 1:
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
                        self.node_map_itr[self.itr] = self.node_map # NEW:
                        candidates = [self._get_best_terminating_node()] # Get the best terminating node among all nodes in the current iteration. Generating new nodes won't help at this point, so we directly go for the best terminating node if there are no non-terminating nodes left to execute.
                         
                self._update_frontier(candidates)
                
            if self.n_expanded >= self.itr * 10 and self.itr <= self.config.itr_limit:
                best_node = self._update_iteration()

            if best_node is None:
                if not self.frontier.empty():
                    best_node = self._select_action()
                    if best_node.parent != self.tree_node:
                        self._backtrack(best_node.parent)    
                        best_node.parent.visits += 1
                        self.n_backtracks += 1   
                        instance_logger.debug(">> Backtrack needed to execute the highest-rewarded action.")
                else:
                    instance_logger.debug(f">> No actions in frontier. Updating iteration to expand more nodes. Current iteration: {self.itr}")
                    # NEW:
                    best_node = self._update_iteration()
                    
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
        
        self.tree_node = best_node
                
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
        else: # For write action, need to change the environment and get new (which will be basically same as before) observation
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
            self.tree_node.commit, _ = self._commit_changes()
            instance_logger.debug(f">> New commit created: {self.tree_node.commit}")
            
        else:
            self.tree_node.commit = self._get_commit_hash()
            instance_logger.debug(f">> No changes detected, staying on commit: {self.tree_node.commit}")
            
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
    