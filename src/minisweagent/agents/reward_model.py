"""Reward model for evaluating actions in tree search."""

from time import sleep
from typing import Any, Optional, Dict
from dataclasses import dataclass
import abc
from minisweagent.agents.tree_search_node import TreeSearchNode
from minisweagent import Model
import random
import re
import time
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

consistency_prompt = """
You are evaluating a debugging step.

>> Instruction
{task}

>> Previous Actions and Observations
{trajectory}

>> Current Action
{action}

>> Observation
{observation}

Question:
Does the observation logically follow from the intent expressed in the thought? 

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how logically consistent the observation is with the intent expressed in the thought. A score of 0 means the observation clearly contradicts the intent expressed in the thought. A score of 100 means the observation fully and unambiguously supports the intent and follows logically from the thought. Use any integer between 0 and 100 to best reflect the degree of consistency.

{score_format_prompt}
"""

trajectory_alignment_prompt = """
You are evaluating whether a debugging step aligns with the overall intent.

>> Instruction
{task}

>> Previous Actions and Observations
{trajectory}

>> Current action
{action}

>> Observation
{observation}

Task:
Score how well this step moves toward the trajectory intent.

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how well this step moves the trajectory toward its intended goal. A score of 0 means the step clearly moves away from or contradicts the trajectory intent. A score of 100 means the step strongly and directly advances the trajectory intent toward the goal. Use any integer between 0 and 100 to best reflect the degree of alignment.

{score_format_prompt}
"""

knowledge_gain_prompt = """
You are evaluating whether a debugging step provided new and useful information.

>> Instruction
{task}

>> Previous Actions and Observations
{trajectory}

>> Current action
{action}

>> Observation
{observation}

Question:
Did this observation provide NEW and USEFUL information for fixing the issue?

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how much NEW and USEFUL information the observation provides. A score of 0 means the observation provides no useful new information and is completely redundant or irrelevant. A score of 100 means the observation provides highly valuable new information and reveals critical insights relevant to fixing the issue. Use any integer between 0 and 100 to best reflect the amount of knowledge gained.

{score_format_prompt}
"""


code_edit_effectiveness_prompt = """
You are evaluating a CODE EDIT debugging step.

>> Instruction
{task}

>> Previous Actions and Observations
{trajectory}

>> Current action (Code Edit)
{action}

>> Resulting Observation (e.g., test output, error message, behavior change)
{observation}

Question:
How effective was this code edit in addressing the underlying issue?

Scoring guidelines:
Assign an integer score from 0 to 100 that reflects how effective this code edit was. A score of 0 means the code edit is clearly harmful, irrelevant, or moves away from fixing the issue. A score of 100 means the code edit directly and substantially advances the fix or resolves the issue. Use any integer between 0 and 100 to best reflect the degree of effectiveness.

{score_format_prompt}
"""

test_feedback_gain_prompt = """
You are evaluating a TESTING debugging step.

>> Instruction
{task}

>> Previous Actions and Observations
{trajectory}

>> Current action (Testing)
{action}

>> Observation (e.g., test failures, passes, logs, coverage info)
{observation}

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

termination_readiness_prompt = """
You are evaluating a TERMINATION / SUBMISSION debugging step.

>> Instruction
{task}

>> Previous Actions and Observations
{trajectory}

>> Current action (Termination / Submission)
{action}

>> Observation
{observation}

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
    def __init__(self, model: Model):
        self.model = model
    
    
    def parse_score(self, text: str) -> int:
        matches = re.findall(r"<score>\s*(\d{1,3})\s*</score>", text)
        if len(matches) != 1:
            return None, f"Expected exactly one <score> </score> block, found {len(matches)}."

        val = int(matches[0])
        if 0 <= val <= 100:
            return val, None
        return None, f"Score {val} out of range [0, 100]."

    def format_trajectory(self, trajectory: list[Dict[str, Any]], n_steps: int = 5) -> str:
        if len(trajectory) == 0:
            return "<No previous actions or observations>\n\n"
        formatted_trajectory = ""
        if len(trajectory) > n_steps:
            formatted_trajectory += "... (omitted earlier steps for brevity) ...\n\n"
        for i, step in enumerate(trajectory):
            if i < len(trajectory) - n_steps:
                continue  # Only keep last {n_steps} steps for brevity
            formatted_trajectory += f"Action #{i+1}: {step['thought']}\n"
            formatted_trajectory += f"Observation #{i+1}: {step['observation']}\n\n"
        
        return formatted_trajectory.strip()


    def score(self, prompt: str, task: str, trajectory: str, action: str, observation: str) -> float:
        n_steps = len(trajectory)
        formatted_prompt = prompt.format(
            task=task,
            trajectory=self.format_trajectory(trajectory, n_steps=n_steps),
            action=action,
            observation=observation,
            score_format_prompt=score_format_prompt
        )
        
        curr_prompt = formatted_prompt
        
        with open("reward_model_scores.log", "w") as f:
            f.write(f"Prompt:\n{curr_prompt}")

        i = 0
        while True:  # Retry up to 5 times
            try:
                response = self.model.query(messages=[
                    {"role": "user", "content": curr_prompt}
                ])
                break
            except Exception as e:
                if n_steps == 1:
                    print(f"Final exception during model query with n_steps=1: {e}.")
                    # Save the prompt in file for error analysis
                    with open("debug_error.log", 'w') as f:
                        f.write(f"Prompt ({time.time()}):\n{curr_prompt}\n\n")
                    raise e
                n_steps = max(1, n_steps - 1)
                print(f"Exception during model query: {e}. Reducing trajectory steps to {n_steps} and retrying.")
                formatted_prompt = prompt.format(
                    task=task,
                    trajectory=self.format_trajectory(trajectory, n_steps=n_steps),
                    action=action,
                    observation=observation,
                    score_format_prompt=score_format_prompt
                )
                curr_prompt = formatted_prompt
                with open("reward_model_scores.log", "a") as f:
                    f.write(f"Prompt:\n{curr_prompt}")
                sleep(1)  # Brief pause before retrying
  
        out = response["content"]
        score, error = self.parse_score(out)
        # print(f"Warning: {error}")
        # Append last generation and error to prompt for clarification
        # curr_prompt = formatted_prompt + f"\n\n>> Previous output:\n{out}\n\n>> Error: {error}\nPlease try again."
        
        if error is not None:
            with open("reward_model_scores.log", "w") as f:
                f.write(f"Prompt:\n{curr_prompt}\nOutput: {out}\nError: {error}")
        else:
            with open("reward_model_scores.log", "w") as f:
                f.write(f"Prompt:\n{curr_prompt}\n\nOutput:\n{out}\n\nParsed score: {score}")
            
        if score is None:
            print(f"Final failure to parse score after retries. Using random score.")
            score = random.randint(0, 100)
            
        return score / 100.0  # Normalize to [0.0, 1.0]

    def compute_reward(
        self,
        node: TreeSearchNode,
        task: Optional[str] = None,
        cmd_type: str = "search" 
    ) -> float:
        """Compute reward for an action.
        
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
        
        curr = node.parent
        while curr.last_action is not None:
            trajectory.append(
                {
                    "thought": curr.last_action["thought"],
                    "action": curr.last_action["command"],
                    "observation": curr.observation
                }
            )
            curr = curr.parent
        trajectory.reverse()

        
        # Detailed scoring
        C = self.score(consistency_prompt, task, trajectory, action, observation)
        
        T = self.score(trajectory_alignment_prompt, task, trajectory, action, observation)
        if cmd_type == "edit":
            K = self.score(code_edit_effectiveness_prompt, task, trajectory, action, observation)
        elif cmd_type == "test":
            K = self.score(test_feedback_gain_prompt, task, trajectory, action, observation)
        elif cmd_type == "submit":
            K = self.score(termination_readiness_prompt, task, trajectory, action, observation)
        else:
            K = self.score(knowledge_gain_prompt, task, trajectory, action, observation)

        print(
            f"Reward scores - Consistency: {C:.2f}, Knowledge Gain: {K:.2f}, Trajectory Alignment: {T:.2f}"
        )
        # Weighted sum
        w_c, w_k, w_t = 0.25, 0.40, 0.35
        R = w_c * C + w_k * K + w_t * T
        
        return R
            
        
        
        

