from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, TerminatingException, Submitted, ExecutionTimeoutError
from minisweagent.agents.tree_search_node import TreeSearchNode   
from minisweagent.agents.frontier import Frontier
from minisweagent.agents.action_analyzer import is_terminating
from typing import List, Any, Optional
from tabulate import tabulate
import time
import json
import logging
from pathlib import Path
import subprocess

from minisweagent.utils.log import instance_logger

class NoActionFound(Exception):
    """Raised when the agent has reached its cost or step limit."""
    
class SingleActionAgentConfig(AgentConfig):
    agent_role: str = "fixer"
    """Agent role: 'fixer' or 'reproducer'."""

    depth_limit: int = 20
    """The maximum depth allowed for any node."""
    
    max_history: int = None
    """The maximum number of messages to keep in the message history. Set to None for unlimited history."""
    
    context_compaction: bool = False
    """Whether to use context compaction to reduce message history length."""
    

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
        self.all_node_map = {} # OLD:
        self.task = None
    
    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.task = task
        self.messages = []
        
        self._reset()
        
        while True:
            try:
                self.step()
            # except NoActionFound as e:
            #     self.add_message("system", str(e))
            #     return type(e).__name__, str(e)
            except NonTerminatingException as e:
                self.add_message("user", str(e))
                self.tree_node.observation = str(e)
            except TerminatingException as e:
                if self.config.agent_role == "reproducer":
                    if self._should_block_reproduction_termination(e):
                        blocked_msg = (
                            "Reproduction submission rejected: you cannot stop before creating a new run_test.sh. "
                            "Create run_test.sh, run it to check expected behavior, and submit again."
                        )
                        self.add_message("user", blocked_msg)
                        self.tree_node.observation = blocked_msg
                        
                        if self.n_expanded + 1 >= self.config.step_limit:
                            instance_logger.debug("Max step limit reached while blocking reproduction submission. Forcing termination.")
                            return None, "Agent reached max step limit while blocking reproduction submission. Final output may be incomplete."
                        continue

                    validation_result = self._validate_reproduction_submission()
                    if validation_result is not None:
                        self.add_message("user", validation_result)
                        instance_logger.debug(f"Reproduction submission failed validation: {validation_result}")
                        self.tree_node.observation = validation_result
                        
                        if self.n_expanded + 1 >= self.config.step_limit:
                            instance_logger.debug("Max step limit reached while blocking reproduction submission. Forcing termination.")
                            return None, "Agent reached max step limit while blocking reproduction submission. Final output may be incomplete."
                        
                        continue

                self.add_message("user", str(e))
                self.tree_node.observation = str(e) 
                return type(e).__name__, str(e)

    def _should_block_reproduction_termination(self, exc: TerminatingException) -> bool:
        """For reproduction runs, block submission unless a new run_test.sh is present in staged diff."""
        if self.config.agent_role != "reproducer":
            return False
        if not isinstance(exc, Submitted):
            return False
        return not self._submitted_patch_has_new_run_test_script(str(exc))

    def _validate_reproduction_submission(self) -> str | None:
        """Run run_test.sh and require at least one failing test in test_status.json before accepting submission.

        Returns a rejection message if the submission should be blocked, otherwise None.
        """
        try:
            output = self.env.execute("bash run_test.sh")
            if output.get("returncode", 1) != 0:
                return (
                    "Reproduction submission rejected: run_test.sh failed to execute. "
                    "Fix the reproduction script and submit again."
                )

            try:
                status_data = json.loads(self.env.execute("cat test_status.json").get("output", ""))
            except json.JSONDecodeError:
                return (
                    "Reproduction submission rejected: test_status.json is not valid JSON. "
                    "Fix run_test.sh so it writes a valid status file."
                )
        except (TimeoutError, subprocess.TimeoutExpired) as e:
            return (
                "Reproduction submission rejected: run_test.sh execution timed out. "
                "Ensure run_test.sh can complete within the time limit and try again."
            )
            

        tests = status_data.get("tests", []) if isinstance(status_data, dict) else []
        failed_tests = [test for test in tests if str(test.get("status", "")).upper() == "FAILED"]
        if not failed_tests:
            return (
                "Reproduction submission rejected: after executing run_test.sh, the generated test_status.json contains no FAILED tests. "
                "This means the reproduction did not match a failing issue state. "
                "Adjust run_test.sh and try again."
            )
            
        # check if all the status are in PASSED|FAILED|SKIPPED|ERROR, if not, reject with error message
        valid_statuses = {"PASSED", "FAILED", "SKIPPED", "ERROR"}
        for test in tests:
            if str(test.get("status", "")).upper() not in valid_statuses:
                return (
                    f"Reproduction submission rejected: test '{test.get('name', '<unknown>')}' has invalid status '{test.get('status', '')}'. "
                    f"Statuses must be one of {', '.join(valid_statuses)}. Fix run_test.sh to produce valid statuses and try again."
                )

        instance_logger.debug(
            "Reproduction submission validated with %d failed test(s): %s",
            len(failed_tests),
            ", ".join(str(test.get("name", "<unknown>")) for test in failed_tests),
        )
        return None

    def _submitted_patch_has_new_run_test_script(self, patch: str) -> bool:
        """Return True when submitted patch contains a newly added run_test.sh file."""
        return (
            "diff --git a/run_test.sh b/run_test.sh" in patch
            and "new file mode" in patch
        )
            
    def _handle_max_steps(self):
        if self.n_expanded + 1 < self.config.step_limit:
            return None
        return self._make_terminating_action(self.tree_node)
    
    def _make_terminating_action(self, curr_node):
        node = self._create_node(
            last_action={
                "command": f"echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached",
                "thought": "THOUGHT: MAX STEPS REACHED\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached\n```",
                "extra": None,
                "type": "submit"
            },
        )
        node.value = node.merged_value = 0.0
        node.is_terminating = True
        curr_node.add_child(node)
        return node
    
    def _add_actions_to_frontier(self, actions: List[TreeSearchNode]):
        for new_node in actions:
            self.frontier.push(-new_node.merged_value, new_node)
            
    def _select_action(self):
        # 1. Handle max-step pruning
        node = self._handle_max_steps()
        if node: return node
        
        if self.frontier.length() > 0:
            neg_score, best_node = self.frontier.pop()
        else:
            best_node = self._make_terminating_action(self.tree_node)
            instance_logger.debug("Action queue empty. Forcing terminating action.")
        
        return best_node
    
    def _create_node(self, last_action: dict = None) -> TreeSearchNode:
        node = TreeSearchNode(
            last_action=last_action,
        )
        self.node_map[node.id] = node
        self.all_node_map[node.id] = node # OLD:
        return node
    
    def _reset(self):
        self.frontier.reset()
    
        self.USER_PROMPT = self.render_template(self.config.instance_template)
        self.SYSTEM_PROMPT = self.render_template(self.config.system_template) 
         
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
                instance_logger.debug(f"Generated action #{i+1}: {action['action']}")
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
                instance_logger.debug(f">> Invalid Response: {response['content']}")
                
            time.sleep(2)  # To avoid rate limiting
            
            new_node.value = new_node.merged_value = 0.0
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
            
        instance_logger.debug(f">> Executing selected action #{self.n_expanded + 1}: {self.tree_node.last_action['command']}")
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
    
    
    def _compact_history(self, messages: list, node: TreeSearchNode) -> list:    
        # incremental rolling compaction
        fixed   = messages[:2]
        history = messages[2:]
        max_steps = self.config.max_history
        min_steps = self.config.max_history // 2
        
        total_steps = len(history) // 2  # convert messages → steps
        x = total_steps // min_steps
        x = max(0, x-1)  # Don't summarize the first chunk until we have at least 2 full chunks to summarize
        recent_steps = total_steps - x*min_steps # Number of most recent messages to keep without summarization. We summarize the earlier part of the history and keep the recent part intact for better context.

        # Find the most recent summary in the node's ancestry and how many steps have been taken since then
        summary_node = node
        steps_after_last_summary = 0
        while True:
            if summary_node is None:
                break
            if summary_node.history_summary is not None:
                break
            steps_after_last_summary += 1
            summary_node = summary_node.parent
        
        if summary_node and summary_node.history_summary:
            # last compaction
            last_summary = summary_node.history_summary
        else:
            # No compaction so far
            last_summary = None
        # ===================================================================== 
                
        if recent_steps != min_steps and total_steps != recent_steps:
            if last_summary is not None and steps_after_last_summary >= min_steps:
                recent_msgs = history[-2*recent_steps:]
                return fixed + [
                    {   
                        "role": "user",      
                        "content": f"[CONTEXT SUMMARY - DO NOT REPEAT VERBATIM]\n" + last_summary,
                    }
                ] + recent_msgs, last_summary
            return fixed + history, last_summary

        # Case: recent_steps == min_steps
        old_msgs    = history[-2*max_steps:-2*min_steps]
        recent_msgs = history[-2*min_steps:]

        if steps_after_last_summary == min_steps and last_summary is not None:
            instance_logger.debug(f"Reusing existing summary from {steps_after_last_summary} steps ago for context compaction.")
            new_summary = last_summary # Case: Multiple Candidate actions generated at the same step, so we can reuse the summary from the first one without extra summarization cost.
        else:
            new_summary = self._summarize(old_msgs, last_summary)
            
        summary_node = node
        steps_to_cover = min_steps # convert messages → steps
        
        # skip recent steps and summarize before that
        for _ in range(steps_to_cover):
            summary_node = summary_node.parent
            
        summary_node.history_summary = new_summary
        
        return fixed + [
            {   
                "role": "user",      
                "content": f"[CONTEXT SUMMARY - DO NOT REPEAT VERBATIM]\n" + new_summary,
            }
        ] + recent_msgs, new_summary

    def _format_msgs(self, msgs):
        lines = []
        for m in msgs:
            role = m["role"]
            content = m["content"]
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _summarize(self, old_msgs: list, prev_summary: str = None) -> str:
        instance_logger.debug(f">> Compacting history of {len(old_msgs)} messages into summary. Previous summary length: {len(prev_summary) if prev_summary else 0} characters.")
    
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a memory compression module for an autonomous software engineering agent "
                    "solving a bug-fixing task. Your job is to summarize past steps "
                    "into a compact representation that preserves all critical technical information "
                    "needed for future debugging and code editing decisions.\n\n"
                    
                    "Focus on:\n"
                    "- The bug being investigated\n"
                    "- Relevant files, functions, and code regions\n"
                    "- Actions taken (e.g., code edits, searches, tests run)\n"
                    "- Key observations and outputs (e.g., errors, logs, test failures)\n"
                    "- What has been tried and whether it worked or failed\n"
                    "- Current hypotheses about the bug\n"
                    "- Important constraints or edge cases\n\n"
                    
                    "Do NOT restate the full bug description. "
                    "Assume it is already known. Focus only on incremental progress and findings.\n\n"
                    
                    "Be concise but information-dense. Avoid redundancy. "
                    "Do NOT include conversational fluff. Preserve technical details."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Bug Description:\n{self.task}\n\n"
                    f"{'[Previous summary]:\n' + prev_summary + '\n\n' if prev_summary else ''}"
                    "Steps to summarize:\n\n"
                    + self._format_msgs(old_msgs)
                )
            }
        ]
            
        response = self.model.query(messages)
        summary = response["content"].strip()   
         
         
        prev_context_len = (len(prev_summary) if prev_summary else 0) + len(self._format_msgs(old_msgs))
        new_context_len = len(summary)
        compression_ratio = new_context_len / prev_context_len if prev_context_len > 0 else 1.0
        
        instance_logger.debug(f"Generated summary of length {len(summary)} ({(1.0-compression_ratio)*100:.2f}% reduction) characters.")
        
        # save messages and summary to file for debugging
        with open("debug_summary.json", "w", encoding="utf-8") as f:
            json.dump([
                *messages,
                {
                    "role": "assistant",
                    "content": summary,
                }
            ], f, indent=4, ensure_ascii=False)
        return summary
        
    
    def get_messages(self, node) -> List[dict]:
        messages = []
        curr = node
        while curr.last_action is not None:
            messages.append(
                {
                    "role": "user", 
                    "content": curr.observation, 
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": curr.last_action["thought"],
                }
            )   
            curr = curr.parent
        
        messages.append({
            "role": "user",
            "content": self.USER_PROMPT
        })
        messages.append({
            "role": "system",
            "content": self.SYSTEM_PROMPT
        })
        messages.reverse()
        
        
        # ---- Context compaction if needed ----
        if self.config.context_compaction and self.config.max_history and len(messages) > 2*self.config.max_history:
            # Need to compact history
            # Starting from node, go back max_history//2 steps and check if node.history_summary is available. If not, summarize that portion of history and save it in the node for future use.
            messages, new_summary = self._compact_history(messages, node)
                
                
        # ---- Truncate history if needed ----
        # if self.config.max_history and len(messages) > 2*self.config.max_history+2:
        #     raise NotImplementedError("History truncation not implemented yet. Please set max_history to None or a sufficiently large value.")
        #     system_prompt = messages[0]
        #     user_prompt = messages[1]
        #     first_action = messages[2]
        #     first_observation = messages[3]
        #     last_msgs = messages[-2*(self.config.max_history-1):]

        #     messages = [
        #         system_prompt,
        #         user_prompt,
        #         first_action,
        #         first_observation,
        #         {
        #             "role": "system",
        #             "content": "⚠️ Middle conversation history was truncated due to context limits. Earlier steps occurred but are omitted."
        #         },
        #         *last_msgs
        #     ]
        
        # ---- Step limit warning ----
        if self.n_expanded > max(0.85 * self.config.step_limit,  self.config.step_limit - 5):
            warning_msg = f"⚠️ Warning: Approaching step limit ({self.config.step_limit - self.n_expanded} steps remaining). Consider finish editing and submitting soon to avoid forced termination."
            instance_logger.debug(warning_msg)
            messages.append({
                "role": "system",
                "content": warning_msg
            })
        return messages
    
    def query(self) -> dict:
        """Query the model and return the response."""
        if 0 < self.config.cost_limit <= self.model.cost:
            raise LimitsExceeded()
        
        messages = self.get_messages(self.tree_node)
        # save to file for debugging
        with open("debug_messages.json", "w", encoding="utf-8") as f:
            json.dump(messages, f, indent=4, ensure_ascii=False)
        response = self.model.query(messages)
        if "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in response["content"]:
            response["content"] = response["content"].replace("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached")
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
    
