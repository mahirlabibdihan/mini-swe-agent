from minisweagent.agents.default import FormatError, Submitted
from minisweagent.agents.tree_search_node import TreeSearchNode   
from minisweagent.agents.reward_guided_agent import RewardGuidedAgentConfig, RewardGuidedAgent
import minisweagent.agents.action_processor as action_processor
from minisweagent.agents.frontier import Frontier
from typing import List, Any, Optional
import json
from minisweagent.utils.log import instance_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import litellm
import json
class TreeSearchAgentConfig(RewardGuidedAgentConfig):
    """Maximum number of nodes to expand per step."""
    sub_thres: int = 3
    """Number of submissions after which to terminate."""
    u_sub_thres: int = 1
    """Number of unique submissions after which to terminate."""
    itr_limit: int = 4
    top_k_tree_pruning: int = 4
    reconcile: bool = True
    augment_solutions: bool = True
    defer_termination: bool = True
    force_convergence: bool = True
    solution_selector: str = "rtv"
    summarize_system_template: str = ""
    summarize_user_template: str = ""
    voting_system_template: str = ""
    voting_user_template: str = ""
    
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
        n_commit = target_node.commit
        if n_commit != self.tree_node.commit:
            _type = "Backtracking" if self.tree_node.id != target_node.parent.id else "Forwarding"
            instance_logger.debug(f">> {_type} from [{self.tree_node.id}] to [{target_node.id}]")
            instance_logger.debug(f">> {_type} from [{self.tree_node.commit[:7]}] to [{n_commit[:7]}]")
            command = (
                "git reset --hard HEAD && git clean -fd && "
                f"git checkout --detach {n_commit}"
            )
            out = self.env.execute(command)
            if out["returncode"] != 0:
                error = out.get("stderr") or out.get("output") or "Unknown error"
                instance_logger.error(f"Git checkout failed: {error}")
                raise Exception(f"Git checkout failed: {error}")
            self.add_message(
                "system",
                f"THOUGHT: {_type} to node:{target_node.id}.\n\n"
                f"```bash\n{command}\n```",
            )
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
        # if not node.cache_hit:
        #     self.action_cache[f"{curr_node.state_hash}_{node.last_action['command']}"] = node.id
        return node
    
    def _generate_terminating_nodes(self):
        if not self.config.augment_solutions:
            return
        old_active_node = self.tree_node
        
        edit_nodes = self._get_all_unique_edit_paths() # [:4*self.config.top_k_tree_pruning] # TODO: Should consider top 16 unique edit paths to limit combinatorial explosion. We can also experiment with different ways of selecting which paths to consider, like giving priority to paths with higher value or more recent paths. This is a hyperparameter that can be tuned based on the task and the size of the tree.
        if len(edit_nodes) > 0:
            instance_logger.debug(f">> Generating terminating nodes for {len(edit_nodes)} unique edit paths.")
            
        unevaluated_terms = []
        for node in edit_nodes:
            if self._get_term_key(node) in self.terminating_nodes:
                continue # We already have a terminating node for this commit, so skip
               
            self._backtrack(node)
            self.tree_node = node
            term_node = self._make_terminating_action(node)
            if term_node.invalid_termination:
                instance_logger.debug(f">> Invalid terminating node [{term_node.id}] under node [{node.id}], skipping.")
                continue
            
            if self.terminating_nodes.get(self._get_term_key(term_node)) is None:
                self.terminating_nodes[self._get_term_key(term_node)] = []
            
            if not node.executed and len(unevaluated_terms) < 4*self.config.top_k_tree_pruning:
                unevaluated_terms.append(term_node)
            else:
                term_node.value = term_node.merged_value = 0.0

            if not node.executed:
                node.executed = True
                node.order = self.n_expanded + 1

            self.terminating_nodes[self._get_term_key(term_node)].append(term_node)
            
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
        
    def _get_state_key(self, node):
        return node.commit
        # return node.state_hash
        
    def _get_term_key(self, node):
        return node.state_hash
        
    def _get_best_terminating_node(self) -> Optional[TreeSearchNode]:
        self._generate_terminating_nodes() # Ensure we have generated terminating nodes for all current edit paths
    
        for state_hash, t_nodes in self.terminating_nodes.items():
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
            if n.is_terminating and n.raw_value is not None and n.merged_value is not None and n.visible
        ]
        
        if not terminating_nodes:
            return None
        
        return self._recursive_tournament_voting(terminating_nodes) # NEW
    
    # NEW:
    def _handle_max_steps(self):
        return None


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

        node_A.executed = node_B.executed = True
        node_A.itr = node_B.itr = self.itr + 1
        node_A.order = node_B.order = self.n_expanded + 1
        # self.n_expanded += 1
        response, action, error, retries = self._generate_merge_action(node_A, node_B)
        instance_logger.debug(f">> Merging nodes [{node_A.id}] and [{node_B.id}] with shared path length {self._count_shared_path_length(node_A, node_B)} and max divergence path length {self._get_max_divergence_path_length(node_A, node_B)}")
        
        merged_node = self._action_to_node(response, action, error, node_A, retries)
        # if not merged_node.cache_hit:
        #     self.action_cache[f"{merged_node.state_hash}_{merged_node.last_action['command']}"] = merged_node.id
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
            [f"Action:\n{step['action']}\n\nObservation:\n{self.reward_model.format_observation(step['observation'])}" for step in trajectory]
        )
    
    def _summarize_solution(self, node):
        # Cache summary to avoid repeated LLM calls
        if node.solution_summary:
            return node.solution_summary

        trajectory = self._get_trajectory(node)
        summary = None
        for i in range(len(trajectory)):
            # Build candidate solution block
            candidate_block = ("(prior steps are truncated)\n\n" if i > 0 else "") + self._stringify_trajectory(trajectory[i:])

            system_content = self.render_template(self.config.summarize_system_template)
            user_content = self.render_template(
                self.config.summarize_user_template,
                candidate_solution=candidate_block,
            )

            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ]

            try:
                response = self.model.query(messages)
                summary = response["content"].strip() + "\n\nFinal Patch:\n" + self.reward_model.format_patch(node.observation, max_chars=3000)
                instance_logger.debug(f">> Generated summary for node [{node.id}] with trajectory length {len(trajectory)-i}")
                # instance_logger.debug(f"Summary:\n{summary[:500]}...")
                break
            except (litellm.exceptions.ContextWindowExceededError, litellm.exceptions.BadRequestError) as e:
                instance_logger.debug(f">> #{i} Failed to summarize solution: {e}. Retrying...")
                instance_logger.debug(f">> Candidate block length: {len(candidate_block)}, Trajectory length: {len(trajectory)-i}")
                
        if summary is None:
            raise Exception("Failed to generate summary for solution due to context window limitations.")
        
        node.solution_summary = summary
        return summary
    
    def _get_voting_messages(self, node_A, node_B):
        solution_A = self._summarize_solution(node_A)
        solution_B = self._summarize_solution(node_B)

        system_content = self.render_template(self.config.voting_system_template)
        user_content = self.render_template(
            self.config.voting_user_template,
            solution_1=solution_A,
            solution_2=solution_B,
        )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

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
        
        if self.config.solution_selector != "rtv":
            return sorted_nodes[0] 
        
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
                    new_candidates.append(p[0])

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
    
    
    def _recursive_tournament_voting_2(self, terminating_nodes):
        if len(terminating_nodes) == 0:
            return None
        
        sorted_nodes = sorted(
            terminating_nodes,
            key=lambda x: (
                # -x.parent.order, # TEMP
                # x.merged_value, # OLD
                0.8 * x.merged_value + (1 - (x.parent.order / self.config.step_limit)) * 0.1 + 0.1 * (not x.system_generated), # NEW:  Should give priority to early discovered solutions
                # NEW: Give priority to AI generated nodes
                x.get_path_value(0.85) # NEW: In case of tie
            ),
            reverse=True
        )
        # return sorted_nodes[0] # First discovered solution
        
        candidates = sorted_nodes[:2*self.config.top_k_tree_pruning]  # 4*
        candidates = [(n, i) for i, n in enumerate(candidates)]
        # candidates.reverse()
        # recursive tournament voting
        round = 1
        while len(candidates) > 1:
            new_candidates = []
            pairs = self._make_pairs_elite(candidates)
            # pairs = []
            # for i in range(0, len(candidates), 2):
            #     first = candidates[i]
            #     second = candidates[i + 1] if i + 1 < len(candidates) else None
            #     pairs.append((first, second))
            
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
                    new_candidates.append(p[0])
                # else:
                #     new_candidates.append(self._get_best_solution_by_voting(p[0], p[1]))

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
        if self._get_state_key(node_A) != self._get_state_key(node_B):
            raise ValueError("Nodes must be on the same state to merge their paths for messaging.")

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

<path_1>
{self.format_suffix(suffix_b)}
</path_1>

<path_2>
{self.format_suffix(suffix_a)}
</path_2>

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
            if not buckets.get(self._get_state_key(node)):
                if bucket_count < k:
                    buckets[self._get_state_key(node)] = [(node, priority)]
                    bucket_count += 1
                    priority += 1
            # else: # NEW
            elif len(buckets[self._get_state_key(node)]) % chunk_size != 0: # OLD
                buckets[self._get_state_key(node)].append((node, priority))
                priority += 1
            # OLD
            elif bucket_count < k:
                buckets[self._get_state_key(node)].append((node, priority))
                bucket_count += 1
                priority += 1
                  
        old_tree_node = self.tree_node

        # Merge
        merged_nodes = []
        unevaluated_merged_nodes = []
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
                        # merged_node.value = merged_node.merged_value = self._get_merge_node_score_2(p[0][0], p[1][0], merged_node) 
                        merged_node.parent = self._get_merge_node_parent_1(p[0][0], p[1][0], merged_node)
                        merged_nodes.append((merged_node, p[0][1])) # Keep the highest priority among merged nodes
                        unevaluated_merged_nodes.append((p[0][0], p[1][0], merged_node))
        
        if len(unevaluated_merged_nodes) > 0:
            instance_logger.debug(f">> Evaluating {len(unevaluated_merged_nodes)} merged nodes in parallel...")
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_node = {executor.submit(self._get_merge_node_score_2, p_1, p_2, node): node for p_1, p_2, node in unevaluated_merged_nodes}
                for future in tqdm(as_completed(future_to_node), total=len(unevaluated_merged_nodes), desc="Evaluating merged nodes"):
                    node = future_to_node[future]
                    try:
                        score = future.result()
                        node.value = node.merged_value = score
                    except Exception as e:
                        instance_logger.error(f"Error evaluating merged node [{node.id}]: {e}")
                        node.value = node.merged_value = float("-inf") # If evaluation fails, set value to -inf to avoid selecting this node

        self._backtrack(old_tree_node)
        self.tree_node = old_tree_node
        
        # Return only nodes (drop priority) but preserve ordering by priority
        return [x[0] for x in sorted(merged_nodes, key=lambda x: x[1])]
    
    def _slice_topk(self, nodes: List[TreeSearchNode], k: int) -> List[TreeSearchNode]:
        if self.config.reconcile:
            merged_nodes = self._coalesce_dual_nodes(nodes, k) 
            return merged_nodes[:k] # NEW:
        else:
            return nodes[:k]
    
    def _get_all_unique_edit_paths(self, to_execute=False):
        # For each commit, find the best node
        sorted_nodes = sorted(
            (
                n
                for n in self.all_node_map.values()
                if not n.is_terminating
                and not n.executed # NEW
                and n.visible
                and n.merged_value is not None
                and (not to_execute or n.level < self.config.depth_limit)
                and (n.commit != self._get_root_commit())
            ),
            key=lambda n: n.get_path_value(0.85),
            reverse=True
        )
        
        # Now pick the best node for each unique commit, and return those nodes as unique edit paths
        unique_paths = {}
        for node in sorted_nodes:
            # OLD
            if self._get_term_key(node) not in unique_paths:
                unique_paths[self._get_term_key(node)] = node
        
        sorted_nodes = sorted(
            (
                n
                for n in self.all_node_map.values()
                if not n.is_terminating
                # and not n.executed # NEW
                and n.visible
                and n.merged_value is not None
                and (not to_execute or n.level < self.config.depth_limit)
                and (n.commit != self._get_root_commit())
            ),
            key=lambda n: n.get_path_value(0.85),
            reverse=True
        )
        
        for node in sorted_nodes:
            # OLD
            if self._get_term_key(node) not in unique_paths:
                unique_paths[self._get_term_key(node)] = node
                
        return list(unique_paths.values())
        
    def _prune_priority(self, node):
        # Early explored nodes should get some advantage
        return node.get_path_value(0.85) * 0.95 + (1 - (node.parent.order / self.config.step_limit)) * 0.05,
        
    # TODO: Try a different variant, where we only keep paths which don't have a terminating node yet for the corresponding commit. 
    def _get_topk_edit_paths(self, k=None, to_execute=True):
        if k is None:
            k = self.config.top_k_tree_pruning
        curr_i = self.itr 
        sorted_leaves = []
        
        while len(sorted_leaves) < k and curr_i > 0:
            candidates = sorted(
                (
                    n
                    for n in self.node_map_itr[curr_i].values()
                    if not n.executed
                    and not n.is_terminating
                    and n.visible
                    and n.merged_value is not None
                    and (not to_execute or n.level < self.config.depth_limit)
                    and  (n.commit != self._get_root_commit())
                    and n not in sorted_leaves
                ),
                key=lambda n: self._prune_priority(n),
                reverse=True
            )
            sorted_leaves.extend(candidates)
            curr_i -= 1
            
        # NEW: 62
        # sorted_leaves = sorted(
        #      (
        #             n
        #             for n in self.all_node_map.values()
        #             if not n.executed
        #             and not n.is_terminating
        #             and n.visible
        #             and n.merged_value is not None
        #             and (not to_execute or n.level < self.config.depth_limit)
        #             and  (n.commit != self._get_root_commit())
        #         ),
        #         key=lambda n: self._prune_priority(n),
        #         reverse=True
        # )

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
        
        elif self.itr > self.config.itr_limit:
            if self.mode == "evaluation":
                best_node = self._get_best_terminating_node()
                if best_node is None:
                    raise Exception("ERROR: No solution found.")
                    instance_logger.debug(">> Fallback: Generating empty submission.")
                    best_parent = max(
                        (
                            n for n in self.node_map.values() 
                            if n.merged_value is not None 
                            and not n.is_terminating
                            and n.visible 
                            and not n.executed
                            and n.level < self.config.depth_limit
                        ),
                        key=lambda n: self._prune_priority(n),
                        default=None
                    )
                    best_parent.executed = True
                    best_parent.order = self.n_expanded + 1
                    self._backtrack(best_parent)
                    self.tree_node = best_parent
                    best_node = self._make_terminating_action(best_parent)
                    best_node.value = best_node.merged_value = 0.0
                    
                instance_logger.debug(">> Iteration limit exceeded. Best terminating node: [{}] with merged value {}".format(best_node.id, best_node.merged_value))
                self.frontier.clear()
                self.node_map = {best_node.id: best_node} # Prune the rest of
                self._update_frontier([best_node]) # Update frontier with the best node to encourage exploitation of the best solution path 
        elif self.itr == self.config.itr_limit and self.config.force_convergence:
            if self.mode == "evaluation":
                top_k = self._get_topk_edit_paths(k=1)
                if len(top_k) > 0:
                    best_node = top_k[0]
                else:
                    # If no nodes with edits are found, fallback to any promising node regardless of edits to at least have some path to follow.
                    # max_depth = max(n.level for n in self.node_map.values())
                    best_node = max(
                        (
                            n for n in self.node_map.values() 
                            if n.merged_value is not None 
                            and not n.is_terminating
                            and n.visible 
                            and not n.executed
                            and n.level < self.config.depth_limit
                        ),
                        key=lambda n: self._prune_priority(n),
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
                if self.itr + 1 < self.config.itr_limit or not self.config.force_convergence:
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
                        key=lambda n: self._prune_priority(n), # TODO: On N-2 iteration, we may give 0.1 weight for being edit action
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
                                and (n.commit == self._get_root_commit())
                            ),
                            key=lambda n: self._prune_priority(n),
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
                        self.terminating_nodes[self._get_term_key(n)] = self.terminating_nodes.get(self._get_term_key(n), []) + [n]
                        
                # Keep top-k active nodes
                self.node_map = {n.id: n for n in top_k}
                instance_logger.debug(f">> Iteration {self.itr + 1}: Updating frontier with top {len(top_k)} non-terminating leaves based on path value.")
        self.itr += 1

    def _expand(self):
        if self.mode == "simulation":
            tree_nodes = [
                n for n in self.tree_node.children if not n.system_generated and n.merged_value is not None
            ]
            instance_logger.debug(f"# {len(tree_nodes)} new nodes generated at level {self.tree_node.level}:")
            self._print_candidates(tree_nodes)
            return 
        
        if self.tree_node.visits == 0:
            tree_nodes = self._generate_new_nodes(self.config.branching_factor)
            tree_nodes = self._update_tree(tree_nodes)
            for node in tree_nodes:
                if node.is_terminating:
                    if self.terminating_nodes.get(self._get_term_key(node)) is None:
                        self.terminating_nodes[self._get_term_key(node)] = []
                    self.terminating_nodes[self._get_term_key(node)].append(node)

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
        def _collect_terminating_nodes(node,ignore=True,clear=False):
            # if node.is_terminating and (node.visible or node.merged_
            # value is not None):
            if node.is_terminating:
                node.is_submission = False
                node.executed = False
            if node.is_terminating and node.visible and node.level <= 20 and (node.raw_value is not None or not ignore):
                terminating_nodes.append(node)
            for child in node.children:
                _collect_terminating_nodes(child,ignore,clear)
            return terminating_nodes
        
        terminating_nodes = _collect_terminating_nodes(self.tree_root)
        if len(terminating_nodes) == 0:
            terminating_nodes = _collect_terminating_nodes(self.tree_root, False)
            if len(terminating_nodes) == 0:
                return None

        # for n in terminating_nodes:
        #     if n.is_submission:
        #         return n
        # raise Exception("SKIP")
        # count = 0
        # for n in terminating_nodes:
        #     if n.is_submission and n._pass:
        #         raise Exception("SKIP")
        #     if n._pass:
        #         count += 1
        # if count == 0:
        #     raise Exception("SKIP")
        
        unevaluated_terms = []
        for n in terminating_nodes:
            n.is_submission = False
            n.executed = False
            if n.raw_value is None:
                unevaluated_terms.append(n)
            # n.solution_summary = None # TMP
        
        
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
                        
        return self._recursive_tournament_voting_2(terminating_nodes)

        
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
                        best_node = self._get_best_terminating_node()
                        if best_node:
                            candidates = [best_node] # Get the best terminating node among all nodes in the current iteration. Generating new nodes won't help at this point, so we directly go for the best terminating node if there are no non-terminating nodes left to execute.
                    
                elif not self.config.defer_termination and self.n_submissions >= self.config.sub_thres and len(self.terminating_nodes) >= self.config.u_sub_thres:
                # if len(self.terminating_nodes) >= self.config.sub_thres: # Too harsh
                    # TODO: Should we just terminate or consider terminating actions from here?
                    # We are done exploring. Now check the tree if there is any terminating action. If multiple, choose the one with highest path value/reward. If none, choose the one with highest path value among all nodes and run sequentially from there until we reach a terminating node.   
                    best_node = self._get_best_terminating_node()
                    instance_logger.debug(">> Discovered enough solutions. Best terminating node: [{}] with merged value {}".format(best_node.id, best_node.merged_value))
                    self.frontier.clear()
                    self.node_map = {best_node.id: best_node}
                    candidates = [best_node]
                    
                self._update_frontier(candidates)
                
            if not best_node and self.n_expanded >= 10 + (self.itr-1) * 10 and self.itr <= self.config.itr_limit:
                self._update_iteration()
                
            if not best_node and self.itr > self.config.itr_limit and self.n_expanded + 1 >= 10 + (self.itr-2) * 10 + 10:
                self._update_iteration()

            if best_node:
                best_node = None
                
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
                    if best_node is None:
                        raise Exception("ERROR: No solution found at finish line.")
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
            
        if self.tree_node.last_action["command"] is not None:
            instance_logger.debug(f">> Executing selected action #{self.n_expanded + 1}: {self.tree_node.last_action['command'][:300]}{'...' if len(self.tree_node.last_action['command']) > 300 else ''}") # Log only the beginning of the command to avoid cluttering the logs
        else:
            instance_logger.debug(f">> Executing selected action #{self.n_expanded + 1}: {self.tree_node.last_action['command']}")
        self.tree_node.executed = True

    def _observe(self):
        observation = self.tree_node.observation
        if self.tree_node.is_terminating:
            raise Submitted("".join(observation))

        instance_logger.debug(f">> Observation: {observation[:200]}...") # Log only the beginning of the observation to avoid cluttering the logs
        self.add_message("user", observation)


    def step(self) -> dict:
        if self.tree_node.is_terminating:
            self._create_pseudo_root()
        
        self._expand()
        
        if self.mode == "evaluation":
            self.tree_node.visits += 1
            self.tree_node.itr = self.itr
            self.tree_node.order = self.n_expanded
        
        with open("debug_tree.json", "w", encoding="utf-8") as f:
            json.dump(self.tree_root.to_tree(), f, indent=4, ensure_ascii=False)

        with open("debug_nodes.json", "w", encoding="utf-8") as f:
            json.dump(self.tree_root.to_json(), f, indent=4, ensure_ascii=False)
        
        self.tree_node = self._select()
        self._act() # Dummy for tree search - As already executed in the environment. But we need to call it to update the messages and logs.
        self._observe() # Dummy for tree search - As already observed in the environment. But we need to call it to update the messages and logs.
        self.n_expanded += 1
                
        return self.tree_node.observation

    def _update_frontier(self, tree_nodes: List[TreeSearchNode]):
        if len(tree_nodes) == 0:
            instance_logger.debug(f">> Frontier size {self.frontier.length()}. No new nodes to add.")
            return
        
        instance_logger.debug(f"Frontier size {self.frontier.length()}. Adding new {len(tree_nodes)} actions...")    
        self._add_actions_to_frontier(tree_nodes)
