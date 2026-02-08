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
from rank_bm25 import BM25Okapi
import pickle
import os
import numpy as np
from pathlib import Path
from minisweagent.agents.bash_parser import BashParser

import requests

class RewardGuidedAgentConfig(SingleActionAgentConfig):
    retrieval_template: str
    branching_factor: int = 3
    """The maximum number of branches to explore at each node."""
    
import json
import ast
from pathlib import PurePosixPath


def parse_python_content(file_content: str):
    """
    Equivalent to parse_python_file(...), but works from in-memory content.
    """
    try:
        parsed_data = ast.parse(file_content)
    except Exception:
        return [], [], file_content.splitlines()

    class_info = []
    function_names = []
    class_methods = set()
    lines = file_content.splitlines()

    for node in ast.walk(parsed_data):
        if isinstance(node, ast.ClassDef):
            methods = []
            for n in node.body:
                if isinstance(n, ast.FunctionDef):
                    methods.append(
                        {
                            "name": n.name,
                            "start_line": n.lineno,
                            "end_line": n.end_lineno,
                            "text": lines[n.lineno - 1 : n.end_lineno],
                        }
                    )
                    class_methods.add(n.name)

            class_info.append(
                {
                    "name": node.name,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno,
                    "text": lines[node.lineno - 1 : node.end_lineno],
                    "methods": methods,
                }
            )

        elif isinstance(node, ast.FunctionDef):
            if node.name not in class_methods:
                function_names.append(
                    {
                        "name": node.name,
                        "start_line": node.lineno,
                        "end_line": node.end_lineno,
                        "text": lines[node.lineno - 1 : node.end_lineno],
                    }
                )

    return class_info, function_names, lines


def result_to_structure(result: str):
    """
    Convert env.execute JSONL output into the same structure as create_structure(),
    except WITHOUT an artificial repo root.
    """
    structure = {}

    for line in result.splitlines():
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        path = PurePosixPath(obj["id"])

        # Remove the leading "path\\n"
        _, _, file_text = obj["content"].partition("\n")

        classes, functions, lines = parse_python_content(file_text)

        curr = structure

        # Build directory tree
        for part in path.parts[:-1]:
            curr = curr.setdefault(part, {})

        # Insert Python file payload
        curr[path.name] = {
            "classes": classes,
            "functions": functions,
            "text": lines,
        }

    return structure

parser = BashParser()

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
        self.candidates = []
        self.n_modifications = 0 # Number of nodes which have at least one write child
        result = self.env.execute("""
python3 - << 'EOF'
import json
import re
from pathlib import Path
import sys

ROOT = Path(".")  # change this to the folder you want to scan

def is_test(name, test_phrases=None):
    if test_phrases is None:
        test_phrases = ["test", "tests", "testing"]
    words = set(re.split(r" |_|\\/|\\.", name.lower()))
    return any(word in words for word in test_phrases)
    
# Your file reading function
def file_name_and_contents(filename, relative_path):
    text = relative_path + "\\n"
    with open(str(filename), encoding="utf-8", errors="replace") as f:
        text += f.read()
    return text

for filename in ROOT.rglob("*.py"):
    try:
        if is_test(filename.as_posix()):
            continue
        relative = filename.relative_to(ROOT).as_posix()
        content = file_name_and_contents(filename, relative)
        print(json.dumps({"id": relative, "content": content}))
    except Exception as e:
        print("ERROR:", filename, repr(e), file=sys.stderr)
        pass
EOF
""")
        print("Extracting Python files from the codebase..." + self.env.config.image)
        # print(result)
        image_ref = self.env.config.image
        image_name = image_ref.split("/")[-1].split(":")[0]
        # Check if documents/{image_name}.jsonl exists
        if not Path(f"retrieval/{image_name}/documents.jsonl").exists():
            Path(f"retrieval/{image_name}").mkdir(parents=True, exist_ok=True)
            with open(f"retrieval/{image_name}/documents.jsonl", "w") as f:
                f.write(result["output"])
        
        if not Path(f"retrieval/{image_name}/structure.json").exists():
            structure = result_to_structure(result["output"])
            with open(f"retrieval/{image_name}/structure.json", "w") as f:
                json.dump(structure, f, indent=2)
        
        documents = []
        self.file_ids = []

        with open(f"retrieval/{image_name}/documents.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                documents.append(obj["content"].split())  # tokenize by whitespace
                self.file_ids.append(obj["id"])
        
        index_path = f"retrieval/{image_name}/bm25_index.pkl"
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self.bm25 = pickle.load(f)
        else:
            self.bm25 = BM25Okapi(documents)
            with open(index_path, "wb") as f:
                pickle.dump(self.bm25, f)
                
        self.relevance_dict = {}
        
            
    def _get_commit_hash(self):
        """Get the current commit hash"""
        return self.env.execute("git rev-parse HEAD")["output"].strip()
    
    def _create_pseudo_root(self):
        if self._repo_has_changes():
            self.env.execute(f"git checkout -b ts-agent-root && git add -A && git commit -m 'Committing changes before starting tree search'")
            action = "git checkout -b ts-agent-root >/dev/null 2>&1 && git add -A >/dev/null 2>&1 && git commit -m 'Committing changes before starting tree search' >/dev/null 2>&1 && git rev-parse HEAD"
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
        output = self.env.execute("git add -A")
        if output.get("return_code", 0) != 0:
            raise Exception(">> Error staging changes:\n" + output.get("output", ""))
        output = self.env.execute(f'git commit -m "{message}"')
        if output.get("return_code", 0) != 0:
            raise Exception(">> Error committing changes:\n" + output.get("output", ""))
        
        output = self.env.execute("git rev-parse HEAD")
        self.add_message("system", f'THOUGHT: Commit changes of the last command.\n\n```bash\ngit add -A >/dev/null 2>&1 && git commit -m "{message}" >/dev/null 2>&1 && git rev-parse HEAD\n```')
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
        
        issue_tokens = self.task.split()
        scores = self.bm25.get_scores(issue_tokens)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        self.relevance_dict = dict(zip(self.file_ids, scores))
        # Print top 5 relevant files
        top_indices = np.argsort(scores)[-5:][::-1]
        print(">> Top 5 relevant files for the issue:")
        for idx in top_indices:
            print(f"- {self.file_ids[idx]} (score: {scores[idx]:.4f})")
            
        retrieved_docs = []
        for file_path, score in self.relevance_dict.items():
            if len(retrieved_docs) >= 10:
                break
            retrieved_docs.append({
                "file_path": file_path,
                "score": f"{score:.4f}",
            })
            
        self.candidates = [
            {
                "SYSTEM_PROMPT": self.render_template(self.config.system_template),
                "USER_PROMPT": self.render_template(self.config.instance_template),
            },
            {
                "SYSTEM_PROMPT": self.render_template(self.config.system_template),
                "USER_PROMPT": self.render_template(self.config.instance_template) + "\n\n" + self.render_template(self.config.retrieval_template, retrieved_docs=retrieved_docs),
            }
        ]
       
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
        has_write_child = False
        for i in range(n_actions):
            # get_observation action to get observation
            potential_termination = False
            
            self.SYSTEM_PROMPT = self.candidates[i % len(self.candidates)]["SYSTEM_PROMPT"]
            self.USER_PROMPT = self.candidates[i % len(self.candidates)]["USER_PROMPT"]
            
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
                    def is_git_command(cmd: str):
                        import re
                        GIT_CMD = re.compile(
                            r'(^|[;&|()]\s*)git(?=\s|$)'
                        )
                        if GIT_CMD.search(cmd):
                            return True
                        return False
                                                
                    if is_git_command(action["action"]):
                        print(">> Warning: git commands are not allowed in non-terminating actions. Skipping this action...")
                        new_node.observation = "Error: git commands are not allowed."
                        new_node.raw_observation = None
                        new_node.is_system_response = True
                        new_node.last_action["command"] = None
                        output = {"output": new_node.observation, "returncode": 1}
                        time.sleep(2)  # To avoid rate limiting
                    else:
                        # Be-aware of potential terminating actions
                        if action.get("action", "").strip() == "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT":
                            action['action'] += " && git add -A && git diff --cached"
                            potential_termination = True
                        else:
                            potential_termination = False
                            
                        if potential_termination:
                            self.env.execute(f"git checkout {self.tree_root.branch} && git restore --source {self.tree_node.commit} .")
                        
                        output = self.env.execute(action["action"])
                        # Check for terminating action
                        lines = output.get("output", "").lstrip().splitlines(keepends=True)
                        if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
                            print(">> Terminating action detected.")
                            new_node.is_terminating = True   
                            
                            if self.tree_node.commit == self.tree_root.commit:
                                print(">> Warning: Terminating action detected without any modifications.")
                                new_node.observation = "Error: Submission detected without any modifications."
                                new_node.raw_observation = output
                                new_node.is_system_response = True
                                new_node.is_terminating = False
                                new_node.invalid_termination = True
                                new_node.last_action["command"] = None
                            else:    
                                new_node.observation = "".join(lines[1:]) # 
                                new_node.raw_observation = output
                    
                        if potential_termination:
                            self.env.execute("git restore . && git checkout -")
                    observation = self.render_template(self.config.action_observation_template, output=output) 
                    raw_observation = output
                    
                except (TimeoutError, subprocess.TimeoutExpired) as e:
                    output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
                    observation = self.render_template(self.config.timeout_template, action=action, output=output)
                    raw_observation = None

                # Check for code modifications
                if self._repo_has_changes():
                    new_node.modifies_code = True
                    has_write_child = True
                    new_node.modified_files = self._get_modified_files()
                    # Rollback changes
                    # run tests
                    test_result = self.env.execute("pytest --maxfail=1 --disable-warnings -q")
                    if test_result.get("return_code", 0) != 0:
                        new_node.fails_tests = True
                    print(">> Write-action detected.")
                    self.env.execute("git reset --hard HEAD && git clean -fd")
                else: # Single line command (No here-doc or multi-line)
                    commands = parser.parse(action["action"])  # Check if it's a read action and can be parsed
                    for cmd in commands:
                        if cmd.get("command") in ["nl", "cat"]:
                            for arg in cmd.get("args", []):
                                if not arg.startswith('-'):
                                    new_node.read_files.append(arg)
                                    print(f">> Read-action detected. File: {arg}")
                # elif action['action'].startswith("nl"):
                    # import shlex
                    # cmd = action['action']
                    # tokens = shlex.split(cmd)
                    # filename = None
                    # if tokens[0] == "nl":
                    #     for token in tokens[1:]:
                    #         if not token.startswith('-'):
                    #             filename = token
                    #             break
                    
                    # if filename is not None:
                    #     new_node.read_files = [filename]
                    #     print(f">> Read-action detected. File: {filename}")

                if new_node.is_terminating != potential_termination:
                    print(">> Warning: Invalid terminating action detected. Skipping this action...")
                    time.sleep(2)  # To avoid rate limiting
                    continue   
                
            else:
                print(f"Generated action #{i+1}: <<Invalid Action>>")
                observation = error
                raw_observation = None
            
            if new_node.observation is None: # Q: When will it not be None here? A: When terminating action detected above
                new_node.observation = observation
                new_node.raw_observation = raw_observation
            nodes.append(new_node)

            time.sleep(2)  # To avoid rate limiting
            
        if has_write_child:
            self.n_modifications += 1
            
        return nodes
    
    def _repo_has_changes_with_main(self):
        if self.tree_node.parent.commit == self.tree_root.commit:
            return False
        output = self.env.execute(f"git diff {self.tree_root.commit}..{self.tree_node.parent.commit}")

        diff_text = output.get("output", "").strip()

        if not diff_text:
            print(f"No change between {self.tree_root.commit} and {self.tree_node.parent.commit}.")
            raise Exception(">> No changes detected to stage to main branch.")
        else:
            # print(">> Staging changes to main branch before submission...")
            # print(diff_text)
            return True

    def _repo_has_new_commit(self):
        output = self.env.execute("git rev-parse HEAD")
        current_commit = output.get("output", "").strip()
        if current_commit != self.tree_node.commit:
            print(f">> New commit detected: {current_commit} (previous: {self.tree_node.commit})")
            return True
        return False
            
    def _stage_to_main_branch(self):
        # self._repo_has_changes_with_main()
        self.env.execute(f"git checkout {self.tree_root.branch} && git restore --source {self.tree_node.parent.commit} . && git branch | grep '^  ts-agent' | sed 's/^  //' | xargs -r git branch -D")
        self.add_message("system", f"THOUGHT: Preparing final output before submission.\n\n```bash\ngit checkout {self.tree_root.branch} && git restore --source {self.tree_node.parent.commit} . && git branch | grep '^  ts-agent' | sed 's/^  //' | xargs -r git branch -D\n```")
            
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
    
    def _calculate_relevance(self, action, observation) -> float:
        # Example step from agent
        agent_step = f"Action: {action} | Observation: {observation}"
        # The issue we want to check
        issue_text = self.task
        # Get relevance score
        response = requests.post(os.environ["SENTENCE_TRANSFORMER_SERVER"] + "/v1/relevance", json={"model": "all-MiniLM-L6-v2", "text1": agent_step, "text2": issue_text})
        score = response.json().get("score", 0.0)
        print(f">> Relevance score for action '{action[:50] if action else '<<Invalid Action>>'}': {score:.4f}")
        return score
    
    def _get_trajectory(self, node: TreeSearchNode) -> List[dict]:
        trajectory = []
        curr = node
        while curr.last_action is not None:
            trajectory.append(
                {
                    "thought": curr.last_action["thought"],
                    "observation": curr.observation,
                }
            )
            curr = curr.parent
        trajectory.reverse()
        return trajectory
    
    def _format_trajectory(self, trajectory: List[dict], n_steps: int = 5) -> str:
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
    
    def _evaluate_nodes(self, node_list):
        for new_node in tqdm(node_list, desc="Evaluating nodes"):
            if new_node.value is None:
                cmd_type = "search"
                if new_node.last_action["command"] is not None:
                    if new_node.is_terminating or new_node.invalid_termination:
                        cmd_type = "submit"
                    elif new_node.modifies_code:
                        cmd_type = "edit"
                    elif new_node.last_action["command"].startswith("pytest"):
                        cmd_type = "test"
                new_node.value = self.reward_model.compute_reward(new_node, self.task, cmd_type=cmd_type)
                if new_node.last_action["command"] is None:
                    # Penalize invalid actions
                    penalty = 1
                    curr = new_node
                    while curr is not None:
                        if curr.last_action["command"] is None:
                            penalty *= 0.7
                        else:
                            break
                        curr = curr.parent
                        
                    new_value = penalty * new_node.value
                    print(f">> Invalid-action reward adjustment: {new_node.value:.4f} -> {new_value:.4f}")
                    new_node.value = new_value
                    
                elif new_node.raw_observation is not None and new_node.raw_observation.get("return_code", 0) != 0:
                    # Penalize actions with non-zero return code
                    new_value = 0.8 * new_node.value
                    print(f">> Non-zero return-code reward adjustment: {new_node.value:.4f} -> {new_value:.4f}")
                    new_node.value = new_value
                
                if len(new_node.modified_files) > 0:
                    # Boost nodes that modify code based on relevance
                    max_relevance = 0.0
                    for file in new_node.modified_files:
                        if file in self.relevance_dict:
                            max_relevance = max(max_relevance, self.relevance_dict[file])
                    
                    # Weighted average
                    new_value = (0.7 * new_node.value + 0.3 * max_relevance)
                    print(f">> Write-action reward adjustment: {new_node.value:.4f} -> {new_value:.4f}")
                    new_node.value = new_value
                    
                    if new_node.fails_tests:
                        # Penalize if tests fail
                        new_value = 0.6 * new_node.value
                        print(f">> Test-failure reward adjustment: {new_node.value:.4f} -> {new_value:.4f}")
                        new_node.value = new_value
                elif new_node.raw_observation is not None and new_node.raw_observation.get("output").strip() == "":
                    # Penalize read actions that produce no output
                    new_value = 0.7 * new_node.value
                    print(f">> Empty-output reward adjustment: {new_node.value:.4f} -> {new_value:.4f}")
                    new_node.value = new_value

                if len(new_node.read_files) > 0: 
                    # Slightly boost nodes that read files based on relevance
                    max_relevance = 0.0
                    for file in new_node.read_files:
                        if file in self.relevance_dict:
                            max_relevance = max(max_relevance, self.relevance_dict[file])
                    
                    # Weighted average
                    new_value = (0.9 * new_node.value + 0.1 * max_relevance)
                    print(f">> Read-action reward adjustment: {new_node.value:.4f} -> {new_value:.4f}")
                    new_node.value = new_value

                # For read-only actions, compute relevance score
                relevance_score = self._calculate_relevance(new_node.last_action["command"], new_node.observation)
                # Take weighted average of relevance score and current value
                new_value = (0.7 * new_node.value + 0.3 * relevance_score)
                print(f">> Similarity reward adjustment: {new_node.value:.4f} -> {new_value:.4f}")
                new_node.value = new_value
                            
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
            if new_node.is_terminating:
                self.n_submissions += 1
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