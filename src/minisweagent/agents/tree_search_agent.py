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
import litellm

class TreeSearchAgentConfig(RewardGuidedAgentConfig):
    """Maximum number of nodes to expand per step."""
    sub_thres: int = 3
    """Number of submissions after which to terminate."""
    u_sub_thres: int = 1
    """Number of unique submissions after which to terminate."""
    itr_limit: int = 4
    top_k_tree_pruning: int = 4
    

import json

class TreeSearchAgent(RewardGuidedAgent):
    def __init__(self, 
                 *args,  
                 config_class=TreeSearchAgentConfig, 
                 **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.frontier = Frontier()
        self.n_backtracks = 0
        self.itr = 1
        # create an empty list of size self.config.itr_limit+1
        self.node_map_itr = [{} for _ in range(self.config.itr_limit+2)] # NEW:
        self.terminating_nodes = {}
        self.rtv = []

        
    def _backtrack(self, target_node):
        n_commit = self._estimate_commit(target_node)
        if n_commit != self.tree_node.commit:
            _type = "Backtracking" if self.tree_node.id != target_node.parent.id else "Forwarding"
            instance_logger.debug(f">> {_type} from [{self.tree_node.id}] to [{target_node.id}]")
            instance_logger.debug(f">> {_type} from [{self.tree_node.commit[:7]}] to [{n_commit[:7]}]")
            out = self.env.execute(f"git checkout {n_commit}")
            if out["returncode"] != 0:
                instance_logger.error(f"Git checkout failed: {out['stderr']}")
                raise Exception(f"Git checkout failed: {out['stderr']}")
            self.add_message("system", f"THOUGHT: {_type} to node:{target_node.id}.\n\n```bash\ngit checkout {n_commit}\n```")
        elif self.tree_node.id != target_node.parent.id:
            instance_logger.debug(f">> Backtracking from [{self.tree_node.id}] to [{target_node.id}]")
            self.add_message("system", f"THOUGHT: Backtracking to node:{target_node.id}.")
        else:
            instance_logger.debug(f">> Forwarding from [{self.tree_node.id}] to [{target_node.id}]")

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
        
        edit_nodes = self._get_all_unique_edit_paths() # TODO: Should consider top 16 unique edit paths to limit combinatorial explosion. We can also experiment with different ways of selecting which paths to consider, like giving priority to paths with higher value or more recent paths. This is a hyperparameter that can be tuned based on the task and the size of the tree.
        if len(edit_nodes) > 0:
            instance_logger.debug(f">> Generating terminating nodes for {len(edit_nodes)} unique edit paths.")
            
        count = 0
        unevaluated_terms = []
        for node in edit_nodes:
            if not node.executed:
                if not node.modifies_code:
                    if node.commit in self.terminating_nodes:
                        continue # We already have a terminating node for this commit, so skip
                    node.executed = True
                    node.order = self.n_expanded + 1
                    self._backtrack(node)
                    self.tree_node = node
                    term_node = self._make_terminating_action(node)
                    if term_node.invalid_termination:
                        instance_logger.debug(f">> Invalid terminating node [{term_node.id}] under node [{node.id}], skipping.")
                        continue
                    # term_node.value = term_node.merged_value = self._evaluate_node(term_node)
                    unevaluated_terms.append(term_node)
                    if self.terminating_nodes.get(node.commit) is None:
                        self.terminating_nodes[node.commit] = []
                    self.terminating_nodes[node.commit].append(term_node)
                else:
                    # Modified action not executed yet, so execute it to get the observation and value
                    # Need to backtrack to the parent node to execute this node
                    if node.commit in self.terminating_nodes:
                        continue
                    node.executed = True
                    node.order = self.n_expanded + 1
                    self._backtrack(node)
                    self.tree_node = node
                    
                    # try:
                    #     self.env.execute(node.last_action["command"])
                    # except (TimeoutError, subprocess.TimeoutExpired) as e:
                    #     instance_logger.warning(f"Execution of node [{node.id}] command timed out during terminating node generation: {e}") 
                    # node.commit, _ = self._commit_changes()
                    
                    # instance_logger.debug(f">> New commit created: {self.tree_node.commit}")
                    
                    term_node = self._make_terminating_action(node)
                    if term_node.invalid_termination:
                        instance_logger.debug(f">> Invalid terminating node [{term_node.id}] under node [{node.id}], skipping.")
                        continue
                    # term_node.value = term_node.merged_value = self._evaluate_node(term_node)
                    unevaluated_terms.append(term_node)
                    if self.terminating_nodes.get(node.commit) is None:
                        self.terminating_nodes[node.commit] = []
                    self.terminating_nodes[node.commit].append(term_node)
            else:
                if node.commit in self.terminating_nodes:
                    continue # We already have a terminating node for this commit, so skip
                self._backtrack(node)
                self.tree_node = node
                term_node = self._make_terminating_action(node)
                if term_node.invalid_termination:
                    instance_logger.debug(f">> Invalid terminating node [{term_node.id}] under node [{node.id}], skipping.")
                    continue
                # term_node.value = term_node.merged_value = self._evaluate_node(term_node)
                unevaluated_terms.append(term_node)
                if self.terminating_nodes.get(node.commit) is None:
                    self.terminating_nodes[node.commit] = []
                self.terminating_nodes[node.commit].append(term_node)
            
            count += 1
            # if count >= self.config.top_k_tree_pruning:
            #     break # Only generate terminating nodes for top-k unique edit paths to limit combinatorial explosion
                
        # Evaluate all terminating nodes in parallel
        if len(unevaluated_terms) > 0:
            instance_logger.debug(f">> Evaluating {len(unevaluated_terms)} terminating nodes in parallel...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_node = {executor.submit(self._evaluate_node, node): node for node in unevaluated_terms}
                for future in tqdm(as_completed(future_to_node), total=len(unevaluated_terms), desc="Evaluating terminating nodes"):
                    node = future_to_node[future]
                    try:
                        score = future.result()
                        node.value = node.merged_value = score
                    except Exception as e:
                        instance_logger.error(f"Error evaluating terminating node [{node.id}]: {e}")
                        node.value = node.merged_value = float("-inf") # If evaluation fails, set value to -inf to avoid selecting this node
                        
        self._backtrack(old_active_node)
        self.tree_node = old_active_node
        
                    
    def _get_best_terminating_node(self) -> Optional[TreeSearchNode]:
        self._generate_terminating_nodes() # Ensure we have generated terminating nodes for all current edit paths
    
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
        
        terminating_nodes = [
            n
            for n in self.all_node_map.values() # OLD
            if n.is_terminating and n.merged_value is not None and n.visible
        ]
        
        if not terminating_nodes:
            return None
        
        return self._recursive_tournament_voting(terminating_nodes) # NEW
    
    # NEW:
    def _handle_max_steps(self):
        # instance_logger.debug(f">> Checking for max steps... {self.n_expanded} / {self.config.step_limit}")
        # if self.n_expanded + 1 < self.config.step_limit: # OLD
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
    
    
    def _generate_merge_action(self, node_A, node_B):
        """
        Generate an action from the model and parse it
        
        Returns:
            response (dict): The raw response from the model
            action (dict): The parsed action
            error (str | None): The error message if parsing failed
        """
        
        messages = self.get_messages_two_nodes(node_A, node_B)
        
        max_retries = 3
        for i in range(max_retries):  # Retry mechanism in case of parsing errors
            response = self.query(messages)
            try:
                action = self.parse_action(response)
                return response, action, None, i
            except FormatError as e:
                messages.append({
                    "role": "assistant",
                    "content": response["content"]
                })
                messages.append({
                    "role": "user",
                    "content": str(e)
                })
                if i == max_retries - 1:
                    instance_logger.debug(f">> Failed to parse action after {max_retries} attempts. Returning error.")
                    return response, None, str(e), i
                else:
                    instance_logger.debug(f">> Error parsing action. Retrying #{i+1}...")
                    
    def _merge_nodes(self, node_A, node_B):            
        self.SYSTEM_PROMPT = self.candidates[0]["SYSTEM_PROMPT"]
        self.USER_PROMPT = self.candidates[0]["USER_PROMPT"]
        node_A.commit = node_A.parent.commit
        node_B.commit = node_B.parent.commit
        node_A.executed = node_B.executed = True
        node_A.itr = node_B.itr = self.itr + 1
        node_A.order = node_B.order = self.n_expanded + 1
        # self.n_expanded += 1
        response, action, error, retries = self._generate_merge_action(node_A, node_B)
        instance_logger.debug(f">> Merging nodes [{node_A.id}] and [{node_B.id}] with shared path length {self._count_shared_path_length(node_A, node_B)} and max divergence path length {self._get_max_divergence_path_length(node_A, node_B)}")
        
        merged_node = self._action_to_node(response, action, error, node_A, retries)
        return  merged_node # A is parent, since it has higher value

    def linearize_path(self, path):
        messages = []
        for node in path:
            if node.last_action is None:
                continue
            messages.append({"role": "user", "content": node.observation})
            messages.append({"role": "assistant", "content": node.last_action["thought"]})
        return messages
    
    def format_suffix(self, path):
        lines = []
        for node in path:
            if node.last_action is None:
                continue
            lines.append("Action: " + node.last_action["thought"])
            lines.append("Observation: " + node.observation + "\n")
        return "\n".join(lines)
    
    def _get_trajectory(self, node):
        curr = node
        trajectory = []
        while curr.parent is not None:
            if curr.last_action is not None:
                trajectory.append({
                    "action": curr.last_action["thought"],
                    "observation": curr.observation                  
                })
            curr = curr.parent
            
        trajectory = trajectory[::-1]  # reverse to get correct order from root to node
        return trajectory
    
    def _stringify_trajectory(self, trajectory):
        return "\n\n".join(
            [f"Action:\n{step['action']}\n\nObservation:\n{step['observation']}" for step in trajectory]
        )
    
    def _summarize_solution(self, node):
        # Cache summary to avoid repeated LLM calls
        if node.solution_summary:
            return node.solution_summary

        trajectory = self._get_trajectory(node)
        summary = None
        for i in range(len(trajectory)):
            messages = [
                {
                "role": "system",
                "content": """
You are summarizing a candidate SWE-bench solution trajectory.

Your goal is to produce a concise technical summary that helps another model compare competing solutions.

Focus on:
- Root cause identified
- Files/modules modified
- Main fix strategy
- Important implementation details
- Edge cases handled
- Potential risks or weaknesses
- Final outcome

Be concise but information dense.
Do NOT speculate beyond the trajectory.
"""
                },
                {
                    "role": "user",
                    "content": f"""
<pr_description>
{self.task}
</pr_description>

<candidate_solution>
{"(prior steps are truncated)\n\n" if i>0 else "" + self._stringify_trajectory(trajectory[i:])}
</candidate_solution>
"""
                }
            ]

            try:
                response = self.model.query(messages)
                summary = response["content"].strip() + "\n\nFinal Patch:\n" + self.reward_model.format_patch(node.observation, max_chars=3000)
                instance_logger.debug(f">> Generated summary for node [{node.id}] with trajectory length {len(trajectory)-i}")
                # instance_logger.debug(f"Summary:\n{summary[:500]}...")
                break
            except (litellm.exceptions.ContextWindowExceededError, litellm.exceptions.BadRequestError) as e:
                instance_logger.debug(f">> Failed to summarize solution: {e}. Retrying...")
                
        if summary is None:
            raise Exception("Failed to generate summary for solution due to context window limitations.")
        
        node.solution_summary = summary
        return summary
    
    def _get_voting_messages(self, node_A, node_B):
        solution_A = self._summarize_solution(node_A)
        solution_B = self._summarize_solution(node_B)

        messages = []

        messages.append({
        "role": "system",
        "content": """
You are an expert software engineer evaluating candidate fixes for a SWE-bench task.

Your task is to determine which proposed solution is MORE LIKELY to correctly solve the issue described in the PR/task description.

Evaluation criteria:
- Correctness
- Completeness
- Edge case handling
- Regression risk
- Alignment with the intended fix
- Whether the patch addresses the root cause

You must choose exactly ONE solution:
- 1
- 2
"""
    })

        messages.append({
            "role": "user",
            "content": f"""
<pr_description>
{self.task}
</pr_description>

<solution_1>
{solution_A}
</solution_1>

<solution_2>
{solution_B}
</solution_2>

Analyze both solutions carefully and explain your reasoning.

At the very end of your response, output the verdict in EXACTLY this format:

```json
{{
    "verdict": 1
}}
```

or

```json
{{
    "verdict": 2
}}
```
"""
        })

        return messages
    
    def _parse_voting_response(self, response):
        """
        Extract JSON block from triple backticks and parse verdict.
        Expected format:

        ```json
        {
            "verdict": 1
        }
        ```
        """
        import json
        import re

        # Extract last fenced code block
        matches = re.findall(
            r"```(?:json)?\s*(.*?)\s*```",
            response,
            re.DOTALL,
        )
    
        if not matches:
            raise ValueError(
                f"No fenced JSON block found in response:\n{response}"
            )

        json_block = matches[-1]

        try:
            data = json.loads(json_block)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in fenced block: {e}\n\n{json_block}"
            )

        verdict = data.get("verdict")

        if verdict not in [1, 2]:
            raise ValueError(f"Invalid verdict: {verdict}")

        return verdict

    def _get_best_solution_by_voting(self, node_A: tuple, node_B: tuple) -> tuple:
        messages = self._get_voting_messages(node_A[0], node_B[0])
        max_retries = 3
        for i in range(max_retries):
            response = self.model.query(messages)
            try:
                verdict = self._parse_voting_response(response["content"])
                break
            except ValueError as e:
                messages.append({
                    "role": "assistant",
                    "content": response["content"]
                })
                messages.append({
                    "role": "user",
                    "content": str(e)
                })
                if i == max_retries - 1:
                    instance_logger.debug(f">> Failed to parse action after {max_retries} attempts. Returning error.")
                    return node_A if node_A[1] <= node_B[1] else node_B
                else:
                    instance_logger.debug(f">> Error parsing action. Retrying #{i+1}...")
        # write message and response to file debug_vote.txt
        with open("debug_vote.txt", "w") as f:
            f.write("Messages:\n")
            for m in messages:
                f.write(f"{m['role']}:\n{m['content']}\n\n")
            f.write(f"Response:\n{response['content']}\n\n")
            f.write(f"Final verdict: {verdict}\n")
            f.write("="*50 + "\n\n")
        
        return node_A if verdict == 1 else node_B

    def _recursive_tournament_voting(self, terminating_nodes):
        if len(terminating_nodes) == 0:
            return None
        
        sorted_nodes = sorted(
            terminating_nodes,
            key=lambda x: (
                # x.merged_value, # OLD
                0.8 * x.merged_value + (1 - (x.parent.order / self.config.step_limit)) * 0.1 + 0.1 * (not x.system_generated), # NEW:  Should give priority to early discovered solutions
                # NEW: Give priority to AI generated nodes
                x.get_path_value(0.85) # NEW: In case of tie
            ),
            reverse=True
        )
        
        candidates = sorted_nodes[:2*self.config.top_k_tree_pruning]  # 4*
        candidates = [(n, i) for i, n in enumerate(candidates)]
        # Generate summary in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_node = {executor.submit(self._summarize_solution, n[0]): n for n in candidates}
            for future in tqdm(as_completed(future_to_node), total=len(candidates), desc="Generating summaries for tournament voting"):
                node = future_to_node[future]
                try:
                    future.result()  # We just want to ensure the summary is generated and cached
                except Exception as e:
                    instance_logger.error(f"Error generating summary for node [{node[0].id}]: {e}")
                    
        # recursive tournament voting
        round = 1
        while len(candidates) > 1:
            new_candidates = []
            # if len(candidates) % 2 == 1:
            #     new_candidates.append(candidates[0]) # If odd number of candidates, give a bye to the best candidate (which is the first one in the sorted list)
            #     pairs = self._make_pairs_elite(candidates[1:])
            # else:
            pairs = self._make_pairs_elite(candidates)

            vote_pairs = [p for p in pairs if p[1] is not None]
            with ThreadPoolExecutor(max_workers=min(4, max(1, len(vote_pairs)))) as executor:
                future_to_pair = {
                    executor.submit(self._get_best_solution_by_voting, p[0], p[1]): p
                    for p in vote_pairs
                }

                for future in tqdm(
                    as_completed(future_to_pair),
                    total=len(future_to_pair),
                    desc=f"Voting round {round}",
                ):
                    best = future.result()
                    new_candidates.append(best)

            for p in pairs:
                if p[1] is None:
                    # raise Exception("This should not happen since we are giving bye to the best candidate when there is an odd number of candidates.")
                    new_candidates.append(p[0])
                # else:
                #     best = self._get_best_solution_by_voting(p[0], p[1])
                #     new_candidates.append(best)
            
            # sort by index to preserve original order as much as possible
            self.rtv.append({
                "round": round,
                "candidates": [c[0].id for c in candidates],
                "pairs": [(p[0][0].id, p[1][0].id if p[1] is not None else None) for p in pairs],
                "winners": [c[0].id for c in new_candidates]
            })
            
            instance_logger.debug(f">> Tournament Round {round}: {len(candidates)} candidates, {len(new_candidates)} winners")
            new_candidates.sort(key=lambda x: x[1])
            candidates = new_candidates
            round += 1
                
        return candidates[0][0] if len(candidates) > 0 else None
    
    def get_messages_two_nodes(self, node_A, node_B) -> List[dict]:
        if node_A.commit != node_B.commit:
            raise ValueError("Nodes must be on the same commit to merge their paths for messaging.")

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
        messages.extend(self.linearize_path(common))

        # divergence marker (VERY important)
        messages.append({
            "role": "user",
            "content": f"""
From this point, two alternative action sequences were explored. 
Your task is to reason across both and decide the best next action.

=== Path A ===
{self.format_suffix(suffix_b)}

=== Path B ===
{self.format_suffix(suffix_a)}

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
    
    def _get_merge_node_score_2(self, node_A, node_B, merged_node):
        merged_node.parent = node_B
        score_2 = self._evaluate_node(merged_node)
        merged_node.raw_value = None # _evaluate_node ignores evaluating if raw_value is set
        merged_node.parent = node_A
        score_1 = self._evaluate_node(merged_node)
        weight_1 = node_A.get_path_value(0.85)
        weight_2 = node_B.get_path_value(0.85)
        total_weight = weight_1 + weight_2
        weight_1 = weight_1 / total_weight
        weight_2 = weight_2 / total_weight
        # if score_1 >= score_2:
        #     merged_node.parent = p[0][0]
        # else:
        #     merged_node.parent = p[1][0]
        return (weight_1 * score_1 + weight_2 * score_2)
    
    def _get_merge_node_score_1(self, node_A, node_B, merged_node):
        merged_node.parent = node_A
        if merged_node.is_terminating: # NEW
            return self._evaluate_node(merged_node)
        else:
            return (0.8 * node_A.merged_value + 0.2 * node_B.merged_value) # We can also experiment with other ways of aggregating values, like max or min, or even giving more weight to the node with higher value. This is a hyperparameter that can be tuned based on the task and the size of the tree.

    def _get_merge_node_parent_1(self, node_A, node_B, merged_node):
        return node_A
    
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
                        merged_node.value = merged_node.merged_value = self._get_merge_node_score_2(p[0][0], p[1][0], merged_node) 
                        merged_node.parent = self._get_merge_node_parent_1(p[0][0], p[1][0], merged_node)
                        merged_nodes.append((merged_node, p[0][1])) # Keep the highest priority among merged nodes
        
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
    
    def _slice_topk(self, nodes: List[TreeSearchNode], k: int) -> List[TreeSearchNode]:
        # Future Work
        # Instead of discarding nodes beyond top-k, we can gather information from them and merge nodes with same commit.
        # Merging nodes means, let's say 3 nodes in the array have the same commit, then we will give the diverge path trajectory of all of them to agent, and ask it to consider all information and generate next step based on that. This way we can keep more information from the tree and also encourage exploration of different paths while still keeping the search focused on promising trajectories. This can be especially helpful in cases where the reward signal is sparse and we want to gather as much information as possible to guide the search.
        # Since the array is sorted, we will traverse it from start to end, and keep adding nodes to the top-k array until we have k unique commits.
        # We don't want to merge more than 2 nodes for now, so let's say there are 3 same commit at the start of an array, we will group the first 2, and treat 3rd as separate commit to encourage exploration of different paths. This is a hyperparameter that can be tuned based on the task and the size of the tree.
        
        merged_nodes = self._coalesce_dual_nodes(nodes, k) 
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
        return node.get_path_value(0.85) * 0.95 + (1 - (node.parent.order / self.config.step_limit)) * 0.05,
        
    # TODO: Try a different variant, where we only keep paths which don't have a terminating node yet for the corresponding commit. 
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
                        key=lambda n: self._prune_priority(n, max_depth=max_depth), # TODO: On N-2 iteration, we may give 0.1 weight for being edit action
                        reverse=True,
                    )
                    top_k = self._slice_topk(sorted_leaves, k=self.config.top_k_tree_pruning) # Keep top-k
                else: # On the last iteration, prioritize nodes with edits regardless of score to encourage exploitation of promising edit paths. If not enough edit paths are found, go for read paths.
                    instance_logger.debug(">> Iteration {} reached. Prioritizing nodes with edits for exploitation.".format(self.itr + 1))
                    sorted_writes = self._get_topk_edit_paths(k=self.config.top_k_tree_pruning)
                    top_k = sorted_writes
                    # If not enough edit paths are found, go for read paths. Because, we have one more iteration left, instead of settling for low-quality edit paths, try more.
                    # TODO: [BAD IDEA] If len(sorted_writes) > 0, we shouldn't gather reads. Just keep the edit paths even if they are less than k, since we want to encourage exploitation of promising edit paths when we are close to iteration limit. We can experiment with different values of k for edit paths and read paths to find the best balance between exploitation and exploration.
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
            n.is_submission = False
            n.executed = False
            
        return self._recursive_tournament_voting(terminating_nodes)
        
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
                    if len(candidates) == 0 or self.n_expanded + 1 >= self.config.step_limit: # Only terminating actions are left
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
                    self._backtrack(best_node)  
                    if best_node.parent != self.tree_node:
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
        # if self.mode == "simulation" or self.tree_node.last_action["command"] is None or (not self.tree_node.is_terminating and not self.tree_node.modifies_code): # For read-only action, no need to re-execute
        observation = self.tree_node.observation
        if self.tree_node.is_terminating:
            raise Submitted("".join(self.tree_node.observation))
        # else: # For write action, need to change the environment and get new (which will be basically same as before) observation
        #     output = self.get_observation(
        #         {
        #             "action": self.tree_node.last_action["command"]
        #         }
        #     )
        #     observation = self.render_template(self.config.action_observation_template, output=output)
            
        instance_logger.debug(f">> Observation: {observation[:200]}...") # Log only the beginning of the observation to avoid cluttering the logs
        self.add_message("user", observation)
        self.tree_node.observation = observation

        # if self.mode == "evaluation":
        #     if self.tree_node.modifies_code:            
        #         self.tree_node.commit, _ = self._commit_changes()
        #         instance_logger.debug(f">> New commit created: {self.tree_node.commit}")
                
        #     else:
        #         self.tree_node.commit = self._get_commit_hash()
        #         instance_logger.debug(f">> No changes detected, staying on commit: {self.tree_node.commit}")
    
        
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

    def _update_frontier(self, tree_nodes: List[TreeSearchNode]):
        if len(tree_nodes) == 0:
            instance_logger.debug(f">> Frontier size {self.frontier.length()}. No new nodes to add.")
            return
        
        instance_logger.debug(f"Frontier size {self.frontier.length()}. Adding new {len(tree_nodes)} actions...")    
        self._add_actions_to_frontier(tree_nodes)
    