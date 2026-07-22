"""Reward model for evaluating actions in tree search."""

import concurrent.futures
from time import sleep
from typing import Any, Optional, Dict
from dataclasses import dataclass
import abc
from minisweagent.agents.tree_search_node import TreeSearchNode
from minisweagent import Model
import random
import re
import time
import litellm

from minisweagent.utils.log import instance_logger

score_format_prompt = """
Output format requirement:

You may include explanations or other text if you want.
However, you MUST include EXACTLY ONE <score>...</score> block.

The <score> block must contain a single INTEGER between 0 and 100 (inclusive).

The final score will be read ONLY from inside the <score> block.
Anything outside the block will be ignored.

For example:
<score>INTEGER</score>
"""

_base_evaluation_prompt = """
>> Instruction
{task}

>> Previous Actions and Observations
{trajectory}

>> Current action
{action}

>> Observation
{observation}

"""

consistency_prompt = _base_evaluation_prompt + """
You are evaluating a debugging step.

Question:
Does the observation logically follow from the intent expressed in the thought? 

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how logically consistent the observation is with the intent expressed in the thought. A score of 0 means the observation clearly contradicts the intent expressed in the thought. A score of 100 means the observation fully and unambiguously supports the intent and follows logically from the thought. Use any integer between 0 and 100 to best reflect the degree of consistency.

{score_format_prompt}
"""

trajectory_alignment_prompt = _base_evaluation_prompt + """
You are evaluating whether a debugging step aligns with the overall intent.

Question:
Score how well this step moves toward the trajectory intent.

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how well this step moves the trajectory toward its intended goal. A score of 0 means the step clearly moves away from or contradicts the trajectory intent. A score of 100 means the step strongly and directly advances the trajectory intent toward the goal. Use any integer between 0 and 100 to best reflect the degree of alignment.

{score_format_prompt}
"""

knowledge_gain_prompt = _base_evaluation_prompt + """
You are evaluating whether a debugging step provided new and useful information.

Question:
Did this observation provide NEW and USEFUL information for fixing the issue?

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how much NEW and USEFUL information the observation provides. A score of 0 means the observation provides no useful new information and is completely redundant or irrelevant. A score of 100 means the observation provides highly valuable new information and reveals critical insights relevant to fixing the issue. Use any integer between 0 and 100 to best reflect the amount of knowledge gained.

{score_format_prompt}
"""

read_information_gain_prompt = _base_evaluation_prompt + """
You are evaluating a FILE READ or DIRECTORY INSPECTION step.

Question:
Did this file read or directory inspection provide NEW and USEFUL information for fixing the issue?

Scoring guidelines:
Assign an integer from 0 to 100 reflecting how much new, useful information the file read (or directory inspection) provides: 0 = no useful new information (redundant or irrelevant); 100 = highly valuable, actionable information that directly helps fix the issue. Use any integer between 0 and 100.

{score_format_prompt}
"""

search_information_gain_prompt = _base_evaluation_prompt + """
You are evaluating a SEARCH debugging step.

Question:
Did this search provide NEW and USEFUL information for fixing the issue?

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how much NEW and USEFUL information the search provides. A score of 0 means the search provides no useful new information and is completely redundant or irrelevant. A score of 100 means the search provides highly valuable new information and reveals critical insights relevant to fixing the issue. Use any integer between 0 and 100 to best reflect the amount of knowledge gained.

{score_format_prompt}
"""


code_edit_effectiveness_prompt = _base_evaluation_prompt + """
You are evaluating a CODE EDIT debugging step.

Question:
How effective was this code edit in addressing the underlying issue?

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how effective this code edit was. A score of 0 means the code edit is clearly harmful, irrelevant, or moves away from fixing the issue. A score of 100 means the code edit directly and substantially advances the fix or resolves the issue. Use any integer between 0 and 100 to best reflect the degree of effectiveness.

{score_format_prompt}
"""

test_feedback_gain_prompt = _base_evaluation_prompt + """
You are evaluating a TESTING debugging step.

Question:
Did this testing step provide meaningful and informative feedback for fixing the issue?

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how valuable the testing feedback is.

A score of 0 means the testing step provides no useful information (e.g., redundant test passes with no new coverage or insight).  
A score of 100 means the testing step provides highly informative feedback that significantly improves understanding of the issue (e.g., revealing new failures, isolating the bug, or substantially expanding test coverage).  
Use any integer between 0 and 100 to best reflect the degree of usefulness.

When assigning the score, consider:
- Whether the test outcome reveals new failures or confirms incorrect behavior.
- Whether the test outcome rules out hypotheses or narrows down the cause of the issue.
- Whether the testing scope meaningfully increases coverage (e.g., running an entire suite or directory vs. a single test).
- Whether passing tests are informative given prior test results (e.g., previously untested code paths vs. already-known passes).

{score_format_prompt}
"""

termination_readiness_prompt = _base_evaluation_prompt + """
You are evaluating a TERMINATION / SUBMISSION debugging step.

Question:
Is it appropriate for the agent to terminate and submit the solution at this point?

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how appropriate this termination decision is.

A score of 0 means the agent clearly terminates prematurely (e.g., unresolved errors, failing or unrun tests, unvalidated edits, or missing evidence of correctness).
A score of 100 means the agent terminates at an appropriate time, with strong evidence that the solution is correct and sufficiently validated.
Use any integer between 0 and 100 to best reflect the degree of readiness for termination.

When assigning the score, consider:
- Whether the core issue described in the instruction appears to be resolved.
- Whether recent code edits were followed by adequate testing or verification.
- Whether test results (if any) support correctness rather than uncertainty.
- Whether remaining plausible failure modes were reasonably ruled out.
- Whether additional debugging, testing, or refinement would still be necessary.

{score_format_prompt}
"""



# The observation reveals some new information that might be useful for debugging.
# <score>65</score>

# This action uncovers significant new insights, helping narrow down the root cause.
# <score>85</score>
# """

class RewardModel():
    def __init__(self, model: Model, use_combined_scoring: bool = True, max_retries: int = 3):
        """Initialize reward model.

        Args:
            model: The LLM model to use for scoring
            use_combined_scoring: If True, use single LLM call with CoT (default).
                                 If False, use legacy 3-call approach.
            max_retries: Maximum retry attempts for parsing errors (default: 3)
        """
        self.model = model
        # Combined scoring removed — always use legacy 3-call approach.
        self.use_combined_scoring = False
        self.max_retries = max_retries

    def _get_reward_weights(self, cmd_type: str) -> tuple[float, float, float]:
        weights = {
            "edit": (0.15, 0.55, 0.30),
            "test": (0.10, 0.50, 0.40),
            "submit": (0.20, 0.40, 0.40),
            "read": (0.30, 0.35, 0.35),
            "search": (0.15, 0.45, 0.40),
        }
        return weights.get(cmd_type, (0.25, 0.40, 0.35))


    def parse_score(self, text: str) -> int:
        matches = re.findall(r"<score>\s*(\d{1,3})\s*</score>", text)
        if len(matches) != 1:
            return None, f"Expected exactly one <score> </score> block, found {len(matches)}."

        val = int(matches[0])
        if 0 <= val <= 100:
            return val, None
        return None, f"Score {val} out of range [0, 100]."

    
    def format_patch(self, patch: str, max_chars: int = 5000) -> str:
        if len(patch) <= max_chars:
            return patch
        
        half = max_chars // 2
        elided = len(patch) - max_chars
        return (
            f"{patch[:half]}\n"
            f"....({elided} characters elided)....\n"
            f"{patch[-half:]}"
        )
        
    def format_observation(self, observation: str, max_chars: int = 5000) -> str:
        if not observation.startswith("diff --git"):
            return observation
            
        return self.format_patch(observation, max_chars=max_chars)


    def format_trajectory(self, trajectory: list[Dict[str, Any]], offset: int = 0, history_summary: str = None, n_steps: int = 5) -> str:
        if len(trajectory) == 0:
            return "<No previous actions or observations>\n\n"
        formatted_trajectory = ""
        if history_summary is not None:
            formatted_trajectory += f"Summary of earlier steps: {history_summary}\n\n"
        if len(trajectory) > n_steps:
            formatted_trajectory += "... (omitted earlier steps for brevity) ...\n\n"
        for i, step in enumerate(trajectory):
            if i + offset < len(trajectory) - n_steps:
                continue  # Only keep last {n_steps} steps for brevity
            formatted_trajectory += f"Action #{i+ offset + 1}: {step['thought']}\n"
            formatted_trajectory += f"Observation #{i+ offset + 1}: {step['observation']}\n\n"
                
        return formatted_trajectory.strip()
    
    def compute_reward_simple(
        self,
        node: TreeSearchNode,
        task: Optional[str] = None,
        cmd_type: str = "read"
    ) -> float:
        """Legacy compute reward with 3 separate LLM calls.
        
        Args:
            node: The current tree search node
            task: Optional task description for context
            
        Returns:
            A float reward value. Higher is better.
        """
        # return random.random()  # A random number from 0 to 1 for now; replace with proper evaluation later
        action = node.last_action['thought']
        observation = node.observation
        task = f"""
<pr_description>
Consider the following PR description:
{task}
</pr_description>

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
        """
        # Create plain trajectory text
        trajectory = []
        history_summary = None
        
        curr = node.parent
        offset = 0
        while curr.last_action is not None:
            if curr.history_summary is not None:
                history_summary = curr.history_summary
                offset = curr.level
                break
            trajectory.append(
                {
                    "thought": curr.last_action["thought"],
                    "action": curr.last_action["command"],
                    "observation": curr.observation
                }
            )
            curr = curr.parent
        trajectory.reverse()

        
        # Detailed scoring (independent dimensions can run in parallel)
        K_prompt = None
        if cmd_type == "edit":
            K_prompt = code_edit_effectiveness_prompt
        elif cmd_type == "test":
            K_prompt = test_feedback_gain_prompt
        elif cmd_type == "submit":
            K_prompt = termination_readiness_prompt
        elif cmd_type == "search":
            K_prompt = search_information_gain_prompt
        elif cmd_type == "read":
            K_prompt = read_information_gain_prompt
        else:
            K_prompt = knowledge_gain_prompt

        score_args = (task, offset, history_summary, trajectory, action, observation)
        
        K_score = self.score(
            K_prompt,
            *score_args,
            log_file="reward_model_scores_specific.log",
        )
        
        instance_logger.debug(
            f"[{cmd_type}] Reward scores - {K_score:.2f}"
        )
        
        return K_score
    
    def compute_reward(
        self,
        node: TreeSearchNode,
        task: Optional[str] = None,
        cmd_type: str = "read"
    ) -> float:
        """Legacy compute reward with 3 separate LLM calls.
        
        Args:
            node: The current tree search node
            task: Optional task description for context
            
        Returns:
            A float reward value. Higher is better.
        """
        # return random.random()  # A random number from 0 to 1 for now; replace with proper evaluation later
        action = node.last_action['thought']
        observation = node.observation
        task = f"""
<pr_description>
Consider the following PR description:
{task}
</pr_description>

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.
        """
        # Create plain trajectory text
        trajectory = []
        history_summary = None
        
        curr = node.parent
        offset = 0
        while curr.last_action is not None:
            if curr.history_summary is not None:
                history_summary = curr.history_summary
                offset = curr.level
                break
            trajectory.append(
                {
                    "thought": curr.last_action["thought"],
                    "action": curr.last_action["command"],
                    "observation": curr.observation
                }
            )
            curr = curr.parent
        trajectory.reverse()

        
        # Detailed scoring (independent dimensions can run in parallel)
        K_prompt = None
        if cmd_type == "edit":
            K_prompt = code_edit_effectiveness_prompt
        elif cmd_type == "test":
            K_prompt = test_feedback_gain_prompt
        elif cmd_type == "submit":
            K_prompt = termination_readiness_prompt
        elif cmd_type == "search":
            K_prompt = search_information_gain_prompt
        elif cmd_type == "read":
            K_prompt = read_information_gain_prompt
        else:
            K_prompt = knowledge_gain_prompt

        score_args = (task, offset, history_summary, trajectory, action, observation)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                "consistency": executor.submit(
                    self.score,
                    consistency_prompt,
                    *score_args,
                    log_file="reward_model_scores_consistency.log",
                ),
                "trajectory": executor.submit(
                    self.score,
                    trajectory_alignment_prompt,
                    *score_args,
                    log_file="reward_model_scores_trajectory.log",
                ),
                "specific": executor.submit(
                    self.score,
                    K_prompt,
                    *score_args,
                    log_file="reward_model_scores_specific.log",
                ),
            }

            C = futures["consistency"].result()
            T = futures["trajectory"].result()
            K = futures["specific"].result()

        instance_logger.debug(
            f"[{cmd_type}] Reward scores - Consistency: {C:.2f}, Knowledge Gain: {K:.2f}, Trajectory Alignment: {T:.2f}"
        )
        # Weighted sum using command-specific calibration
        w_c, w_k, w_t = self._get_reward_weights(cmd_type)
        R = w_c * C + w_k * K + w_t * T
        
        return R
            
    def score(
        self,
        prompt: str,
        task: str,
        offset: int,
        history_summary,
        trajectory: str,
        action: str,
        observation: str,
        log_file: str = "reward_model_scores.log",
    ) -> float:
        """Score a single dimension with retry logic.

        Returns:
            Score normalized to [0.0, 1.0]
        """
        n_steps = len(trajectory)
        formatted_prompt = prompt.format(
            task=task,
            trajectory=self.format_trajectory(trajectory, offset=offset, history_summary = history_summary, n_steps=n_steps),
            action=action,
            observation=self.format_observation(observation),
            score_format_prompt=score_format_prompt
        )

        curr_prompt = formatted_prompt

        with open(log_file, "w") as f:
            f.write(f"Prompt:\n{curr_prompt}")

        score, error = None, None

        for retry_attempt in range(self.max_retries):
            # Make LLM call with context window retry logic
            while True:
                try:
                    response = self.model.query(messages=[
                        {"role": "user", "content": curr_prompt}
                    ])
                    break
                except (litellm.exceptions.ContextWindowExceededError, litellm.exceptions.BadRequestError) as e:
                    if n_steps == 1:
                        instance_logger.debug(f"Final exception during model query with n_steps=1: {e}.")
                        with open("debug_error.log", 'w') as f:
                            f.write(f"Prompt ({time.time()}):\n{curr_prompt}\n\n")
                        raise e
                        # return 0.2  # Return a low score if even the minimal prompt exceeds context window
                    
                    n_steps = max(1, n_steps - 1)
                    instance_logger.debug(f"Exception during model query: {e}. Reducing trajectory steps to {n_steps} and retrying.")
                    formatted_prompt = prompt.format(
                        task=task,
                        trajectory=self.format_trajectory(trajectory, n_steps=n_steps),
                        action=action,
                        observation=self.format_observation(observation),
                        score_format_prompt=score_format_prompt
                    )
                    curr_prompt = formatted_prompt
                    with open(log_file, "a") as f:
                        f.write(f"Prompt:\n{curr_prompt}")
                    sleep(1)

            out = response["content"]
            score, error = self.parse_score(out)

            # If parsing succeeded, we're done
            # if error is None:
            # with open(log_file, "w") as f:
            #     f.write(f"Prompt:\n{curr_prompt}\n\nOutput:\n{out}\n\nParsed score: {score}")
            # break

            # If parsing failed and we have retries left, ask LLM to fix it
            # if retry_attempt < self.max_retries - 1:
            #     instance_logger.debug(f"Retry {retry_attempt + 1}/{self.max_retries}: Failed to parse score - {error}. Asking LLM to fix format.")
            #     curr_prompt = formatted_prompt + f"\n\n>> Previous output:\n{out}\n\n>> Error: {error}\n\nPlease fix the format and provide exactly one <score>INTEGER</score> block."
            #     with open("reward_model_scores.log", "a") as f:
            #         f.write(f"\n\n=== RETRY {retry_attempt + 1} ===\nPrompt:\n{curr_prompt}")
            # else:
            #     # Final retry failed
            #     instance_logger.debug(f"Final retry {retry_attempt + 1}/{self.max_retries}: Failed to parse score - {error}. Using random score as fallback.")
            #     with open("reward_model_scores.log", "w") as f:
            #         f.write(f"Prompt:\n{curr_prompt}\nOutput: {out}\nError: {error}\nUsing random score as fallback.")

        # If all retries failed, use random score
        if error is not None:
            instance_logger.debug(f"Error parsing score: {error}")
            
        if score is None:
            score = random.randint(0, 100)

        return score / 100.0  # Normalize to [0.0, 1.0]