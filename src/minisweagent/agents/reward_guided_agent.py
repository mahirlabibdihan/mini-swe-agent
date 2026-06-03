from minisweagent.agents.default import AgentConfig, DefaultAgent, LimitsExceeded, NonTerminatingException, FormatError, TerminatingException, Submitted, ExecutionTimeoutError
from minisweagent.agents.tree_search_node import TreeSearchNode   
import minisweagent.agents.action_processor as action_processor
from minisweagent.agents.frontier import Frontier
from minisweagent.agents.action_analyzer import is_terminating
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from minisweagent.utils.log import instance_logger
from minisweagent.agents.repo_tree import result_to_structure, RepoNode, collect_rankable_nodes, dict_to_tree, collect_all_nodes, propagate_scores, remove_redundancy
import requests
import re
import math
import tempfile
import threading
import litellm

class RewardGuidedAgentConfig(SingleActionAgentConfig):
    retrieval_template: str
    reproduction_patch: str = ""
    """Patch that adds/updates reproduction artifacts (typically run_test.sh) for test status checks."""
    branching_factor: int = 3
    """The maximum number of branches to explore at each node."""
    shape_reward: bool = True

        
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
        self.commits = []
        self.mode = "evaluation"  # or "simulation"
        self.action_cache = {}
        self.node_creation_lock = threading.RLock()
        # instance_logger.debug(result)
        image_ref = self.env.config.image
        image_ref = self.env.config.image
        last_part = image_ref.split("/")[-1]
        if ":" in last_part and last_part.rsplit(":", 1)[1] == "latest":
            image_name = last_part.split(":", 1)[0]
        else:
            image_name = last_part.replace(":", "_")
        print(f">> Image name: {image_name}")
        # Check if documents/{image_name}.jsonl exists

        attempt = 0
        while True:
            if not Path(f"retrieval/{image_name}/documents.jsonl").exists():
                instance_logger.debug("Extracting Python files from the codebase..." + self.env.config.image)
                result = self.env.execute("""
python3 - << 'EOF' 2>/dev/null
import json
import re
from pathlib import Path
import sys
import subprocess

def is_gitignored(path):
    try:
        result = subprocess.run(
            ["git", "check-ignore", "-q", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0
    except Exception:
        return False

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

files = subprocess.check_output(
    ["git", "ls-files"],
    text=True
).splitlines()

ALLOWED_EXT = {".py", ".js", ".ts", ".go"}

for relative in files:
    try:
        filename = ROOT / relative

        if filename.suffix not in ALLOWED_EXT:
            continue
        if is_test(relative):
            continue

        content = file_name_and_contents(filename, relative)
        print(json.dumps({"id": relative, "content": content}))
    except Exception:
        pass
EOF
    """)
                dir_path = f"retrieval/{image_name}"    
                os.makedirs(dir_path, exist_ok=True)

                print(f"Found {len(result['output'].splitlines())} files in the codebase.")
                with open(f"{dir_path}/documents.jsonl", "w") as f:
                    f.write(result["output"])
                                
            repo_files = []
            try:
                with open(f"retrieval/{image_name}/documents.jsonl") as f:
                    for line in f:
                        obj = json.loads(line)
                        repo_files.append(obj)
            except Exception as e:
                instance_logger.debug(f">> Error reading documents.jsonl for {image_name}: {repr(e)}")
                repo_files = []
        
            if len(repo_files) == 0:
                # remove the file to trigger re-extraction in the next episode
                os.remove(f"retrieval/{image_name}/documents.jsonl")
                attempt += 1
                if attempt >= 3:
                    raise Exception(f">> Failed to extract any Python files after {attempt} attempts. Please check the environment and the extraction script.")
            else:
                break
                    
        while True:
            if not Path(f"retrieval/{image_name}/structure.json").exists():
                structure = result_to_structure(repo_files)
                with open(f"retrieval/{image_name}/structure.json", "w") as f:
                    json.dump(structure, f, indent=2)
            
            with open(f"retrieval/{image_name}/structure.json") as f:
                structure = json.load(f)
            
            if structure == {}:
                # remove the file to trigger re-extraction in the next episode
                os.remove(f"retrieval/{image_name}/structure.json")
            else:
                break
                
        while True:
            if not Path(f"retrieval/{image_name}/structure_opt.json").exists():
                structure_opt = structure
                remove_redundancy(structure_opt)
                with open(f"retrieval/{image_name}/structure_opt.json", "w") as f:
                    json.dump(structure_opt, f, indent=2)
   
            with open(f"retrieval/{image_name}/structure_opt.json") as f:
                structure_opt = json.load(f)
            
            if structure_opt == {}:
                # remove the file to trigger re-extraction in the next episode
                os.remove(f"retrieval/{image_name}/structure_opt.json")
            else:
                break
                
        self.repo_root = dict_to_tree(None, structure_opt)
        # self.repo_root.print()
        self.rank_nodes = collect_rankable_nodes(self.repo_root)
        
        index_path = f"retrieval/{image_name}/bm25_hindex.pkl"
        if os.path.exists(index_path):
            with open(index_path, "rb") as f:
                self.bm25_h = pickle.load(f)
        else:
            documents = ["\n".join([node.qualified_name()] + node.text).split() for node in self.rank_nodes]
            self.bm25_h = BM25Okapi(documents)
            with open(index_path, "wb") as f:
                pickle.dump(self.bm25_h, f)

        documents = []
        self.file_ids = []

        for obj in repo_files:
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
        
        self.env.execute('git config --global user.name "mahirlabibdihan" && git config --global user.email "mahirlabibdihan@gmail.com"')
    
    def _get_test_status(self):
        """Apply reproduction patch, run tests, and return parsed test entries from test_status.json."""
        raw_patch = self.config.reproduction_patch or ""
        if not raw_patch.strip():
            return None

        # Preserve patch body as-is; only normalize encoding/line endings.
        reproduction_patch = raw_patch.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        if not reproduction_patch.lstrip().startswith("diff --git"):
            instance_logger.debug(">> Skipping test status: reproduction patch is missing or not a git diff.")
            return None
        # git apply expects a newline-terminated patch file.
        patch_content = reproduction_patch if reproduction_patch.endswith("\n") else reproduction_patch + "\n"

        patch_file = ".mini_swe_reproduction.patch"
        patch_delim = "MINI_SWE_REPRO_PATCH_EOF"
        git_apply_cmds = [
            "git apply --verbose --reject",
            "patch --batch --fuzz=5 -p1 -i",
        ]

        try:
            copy_to_container = getattr(self.env, "copy_to_container", None)
            if callable(copy_to_container):
                with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False, encoding="utf-8") as tmp_patch:
                    tmp_patch.write(patch_content)
                    tmp_patch_path = Path(tmp_patch.name)
                try:
                    copy_to_container(tmp_patch_path, patch_file)
                finally:
                    tmp_patch_path.unlink(missing_ok=True)
            else:
                self.env.execute(
                    f"cat <<'{patch_delim}' > {patch_file}\n{patch_content}{patch_delim}"
                )

            applied_patch = False
            last_apply_output = ""
            for git_apply_cmd in git_apply_cmds:
                instance_logger.debug(f">> Attempting to apply reproduction patch with command: {git_apply_cmd} {patch_file}")
                apply_output = self.env.execute(f"{git_apply_cmd} {patch_file}")
                applied_patch = True
                # break
                if apply_output.get("returncode", 1) != 0:
                    instance_logger.debug(
                        ">> Failed to apply reproduction patch with '%s': %s",
                        git_apply_cmd,
                        apply_output.get("output", "").strip(),
                    )
                break
                # last_apply_output = apply_output.get("output", "").strip()

            if not applied_patch:
                instance_logger.debug(
                    ">> Failed to apply reproduction patch for test status: %s",
                    last_apply_output,
                )
                return None

            output = self.env.execute("bash run_test.sh")
            if output.get("returncode", 1) != 0:
                instance_logger.debug(
                    ">> run_test.sh exited with non-zero return code while collecting test status: %s",
                    output.get("output", "").strip(),
                )
                return None

            raw_status = self.env.execute("cat test_status.json").get("output", "")
            status_data = json.loads(raw_status)
            tests = status_data.get("tests", []) if isinstance(status_data, dict) else []
            if not isinstance(tests, list):
                return None
            return tests
        except json.JSONDecodeError:
            instance_logger.debug(">> test_status.json is not valid JSON while collecting test status.")
            return None
        except (TimeoutError, subprocess.TimeoutExpired):
            instance_logger.debug(">> run_test.sh timed out while collecting test status.")
            return None
        finally:
            self.env.execute(f"rm -f {patch_file} test_status.json run_test.sh")
            self.env.execute("git reset --hard HEAD && git clean -fd")

    def _calculate_relevance(self, action, observation) -> float:
        # Example step from agent
        agent_step = f"Action: {action} | Observation: {observation}"
        # The issue we want to check
        issue_text = self.task
        # Get relevance score
        for _ in range(3):  # Retry mechanism in case of transient errors
            try:
                response = requests.post(
                    os.environ["SENTENCE_TRANSFORMER_SERVER"] + "/v1/relevance", 
                    json = {
                        "model": "all-mpnet-base-v2", 
                        "text1": agent_step, 
                        "text2": issue_text
                    }
                )
                # all-mpnet-base-v2, all-MiniLM-L6-v2
                score = response.json().get("score", 0.0)
                instance_logger.debug(f">> Relevance score for action '{action[:50] if action else '<<Invalid Action>>'}': {score:.4f}")
                break
            except Exception as e:
                instance_logger.debug(f">> Error calculating relevance score: {repr(e)}. Retrying...")
                score = 0.0
                time.sleep(1)
            
        return score
    
    def is_test_failure(self, node: TreeSearchNode, returncode: int) -> bool:
        if node.last_action["type"] == "test" and returncode == 1:
            return True
        return False

    def _normalize_test_status_entries(self, tests) -> dict[str, str]:
        """Normalize test entries to a name->status map with known statuses only."""
        valid_statuses = {"PASSED", "FAILED", "SKIPPED", "ERROR"}
        normalized = {}
        if not isinstance(tests, list):
            return normalized

        for test in tests:
            if not isinstance(test, dict):
                continue
            name = str(test.get("name", "")).strip()
            status = str(test.get("status", "")).upper().strip()
            if not name or status not in valid_statuses:
                continue
            normalized[name] = status
        return normalized

    def _baseline_failure_count(self) -> int:
        """Return the number of failing tests in the baseline reproduction status."""
        baseline_tests = []
        if getattr(self, "tree_root", None) is not None and getattr(self.tree_root, "children", None):
            baseline_tests = getattr(self.tree_root.children[0], "test_status", []) or []

        baseline_map = self._normalize_test_status_entries(baseline_tests)
        failure_count = sum(1 for status in baseline_map.values() if status in {"FAILED", "ERROR"})
        return max(1, failure_count)

    def _compare_test_statuses(self, previous_tests, current_tests) -> float:
        """Return normalized signed status delta: positive means better, negative means worse."""
        prev_map = self._normalize_test_status_entries(previous_tests)
        curr_map = self._normalize_test_status_entries(current_tests)
        if not prev_map or not curr_map:
            return 0.0

        common_tests = set(prev_map.keys()) & set(curr_map.keys())
        if not common_tests:
            return 0.0

        status_score = {
            "ERROR": 0,
            "FAILED": 1,
            "SKIPPED": 2,
            "PASSED": 3,
        }

        delta = 0
        for test_name in common_tests:
            delta += status_score[curr_map[test_name]] - status_score[prev_map[test_name]]
        return delta / self._baseline_failure_count()

    def _test_status_component(self, current_tests) -> float:
        """Return a normalized test-status component in [0, 1]."""
        status_score = {
            "ERROR": 0,
            "FAILED": 1,
            "SKIPPED": 2,
            "PASSED": 3,
        }

        curr_map = self._normalize_test_status_entries(current_tests)
        if not curr_map:
            return 0.5

        # For terminating actions, use absolute quality only.
        return sum(status_score[s] for s in curr_map.values()) / (3.0 * len(curr_map))
    
    def _extract_changed_files(self, patch_text: str):
        files = []
        for line in patch_text.splitlines():
            if line.startswith("diff --git"):
                parts = line.split()
                if len(parts) >= 4:
                    # take the "b/..." path
                    files.append(parts[3][2:])
        return files
    
    def _adjust_terminating_reward(self, node, value):
        test_component = self._test_status_component(node.test_status)
        final_value = 0.9 * value + 0.1 * test_component
        instance_logger.debug(
            f">> Terminating test-status adjustment: {value:.4f} + test({test_component:.4f}) -> {final_value:.4f}"
        )
        new_value = final_value
        
        patch = node.observation
        modified_files = self._extract_changed_files(patch)
        # Now calculate average reward of modified files based on relevance dict
        if len(modified_files) > 0:
            file_rewards = [
                self.relevance_dict[f]
                for f in modified_files
                if f in self.relevance_dict
            ]
            
            if len(file_rewards) > 0:
                avg_file_reward = sum(file_rewards) / len(file_rewards)
                new_value = 0.7 * new_value + 0.3 * avg_file_reward
                
        # relevance_score = self._calculate_relevance(patch, self.task)
        # new_value = 0.7 * new_value + 0.3 * relevance_score
                
        return new_value
    
    def _evaluate_node(self, node):
        # Current active node should be node.parent
        if node.parent is None:
            raise Exception("Can't evaluate nodes without parent")
        
        # TEMP: Force re-evaluate  
        # if node.value is not None:
        #     return node.value
        
        if self.config.branching_factor == 1:
            return 0.0 # For single-branch case, we can skip reward computation and directly evaluate the action's relevance to the task
        
        # track the time taken for reward computation
        start_time = time.time()
        cmd_type = node.last_action['type']
        if not node.raw_value:
            node.raw_value = node.value = self.reward_model.compute_reward(node, self.task, cmd_type=cmd_type)
        else:
            node.value = node.raw_value
        
        if not self.config.shape_reward:
            new_value = node.value
            end_time = time.time()
            instance_logger.debug(f"=>> Reward: {new_value:.4f} | Time taken: {end_time - start_time:.2f} seconds")
            return new_value
        
        if node.last_action["command"] is None or node.invalid_termination:
            # Penalize invalid actions
            penalty = 1
            curr = node
            while curr is not None and curr.last_action is not None:
                if curr.last_action["command"] is None or node.invalid_termination:
                    penalty *= 0.9
                else:
                    break
                curr = curr.parent
                
            new_value = 0.9 * node.value
            # new_value = penalty * node.value
            instance_logger.debug(f">> Invalid-action reward adjustment: {node.value:.4f} -> {new_value:.4f}")
            node.value = new_value
            
        # NEW: If we give big penalty, then those path will die there
        elif node.raw_observation is not None and not self.is_test_failure(node, node.raw_observation.get("returncode", 0)) and node.raw_observation.get("returncode", 0) != 0:
            penalty = 1
            curr = node
            while curr is not None and curr.last_action is not None:
                if curr.raw_observation is not None and not self.is_test_failure(curr, curr.raw_observation.get("returncode", 0)) and curr.raw_observation.get("returncode", 0) != 0:
                    penalty *= 0.95
                else:
                    break
                curr = curr.parent
            # Penalize actions with non-zero return code
            new_value = 0.95 * node.value
            # new_value = penalty * node.value
            instance_logger.debug(f">> Non-zero return-code reward adjustment: {node.value:.4f} -> {new_value:.4f}")
            node.value = new_value
        elif node.is_timeout:
            # Penalize actions that time out
            new_value = 0.9 * node.value
            instance_logger.debug(f">> Timeout reward adjustment: {node.value:.4f} -> {new_value:.4f}")
            node.value = new_value
        elif node.is_repeat:
            # Penalize repeat actions, but less than invalid actions
            new_value = 0.95 * node.value
            instance_logger.debug(f">> Repeat-action reward adjustment: {node.value:.4f} -> {new_value:.4f}")
            node.value = new_value
        
        if len(node.modified_files) > 0:
            # Boost nodes that modify code based on relevance
            max_relevance = 0.0
            
            # Option 1: File-level relevance
            for file in node.modified_files:
                if file in self.relevance_dict:
                    max_relevance = max(max_relevance, self.relevance_dict[file])
            
            # Weighted average
            new_value = (0.7 * node.value + 0.3 * max_relevance)
            
            instance_logger.debug(f">> Write-action reward adjustment: {node.value:.4f} -> {new_value:.4f}")
            node.value = new_value
            
            if node.fails_tests:
                # Penalize if tests fail
                new_value = 0.6 * node.value
                # Red color instance_logger.debug to indicate test failure
                instance_logger.debug(f">> Test-failure reward adjustment: {node.value:.4f} -> {new_value:.4f}")
                node.value = new_value
                
            
            if node.diff_size is not None and node.diff_size > 50:
                penalty = math.exp(-(node.diff_size - 50) / 100.0)
                new_value = node.value * penalty
                instance_logger.debug(f">> Excessive-diff reward adjustment: {node.value:.4f} -> {new_value:.4f}")
                node.value = new_value
            # TODO: Penalize if the number of modified files is excessive (e.g., more than 5 files), as it may indicate unfocused modifications. We can use an exponential penalty based on the number of modified files to avoid hard thresholds.
            elif len(node.modified_files) > 5:
                penalty = math.exp(-(len(node.modified_files) - 5) / 5.0)
                new_value = node.value * penalty
                instance_logger.debug(f">> Excessive-modified-files reward adjustment: {node.value:.4f} -> {new_value:.4f}")
                node.value = new_value
                
            status_delta = self._compare_test_statuses(node.parent.test_status, node.test_status)
            if status_delta < 0:
                penalty = max(0.9, 1.0 + 0.1 * status_delta)
                new_value = node.value * penalty
                instance_logger.debug(
                    f">> Test-status regression adjustment (delta={status_delta}): {node.value:.4f} -> {new_value:.4f}"
                )
                node.value = new_value
            elif status_delta > 0:
                boost = min(1.1, 1.0 + 0.1 * status_delta)
                new_value = node.value * boost
                instance_logger.debug(
                    f">> Test-status improvement adjustment (delta={status_delta}): {node.value:.4f} -> {new_value:.4f}"
                )
                node.value = new_value
        
        elif node.raw_observation is not None and node.raw_observation.get("output").strip() == "":
            # Penalize read actions that produce no output
            new_value = 0.7 * node.value
            instance_logger.debug(f">> Empty-output reward adjustment: {node.value:.4f} -> {new_value:.4f}")
            node.value = new_value

        elif (node.last_action["command"] is None or node.last_action["type"] != "test") and node.raw_observation is not None and len(node.raw_observation.get("output").strip()) > 5000:
            # Penalize read actions that produce excessive output
            penalty = math.exp(-(len(node.raw_observation.get("output").strip()) - 5000) / 3000.0)
            new_value = max(0.8, penalty) * node.value
            instance_logger.debug(f">> Excessive-output reward adjustment ({len(node.raw_observation.get("output").strip())} chars): {node.value:.4f} -> {new_value:.4f}")
            # Excessive-output reward adjustment (5377 chars): 0.6325 -> 0.5060
            node.value = new_value

        if len(node.read_files) > 0: 
            max_relevance = 0.0
            for file in node.read_files:
                if file in self.relevance_dict:
                    print(f">> Found relevance score for read file {file}: {self.relevance_dict[file]:.4f}")
                    max_relevance = max(max_relevance, self.relevance_dict[file])
            
            # Weighted average
            new_value = (0.9 * node.value + 0.1 * max_relevance)
            instance_logger.debug(f">> Read-action reward adjustment: {node.value:.4f} -> {new_value:.4f}")
            node.value = new_value

        # compute relevance score
        relevance_score = self._calculate_relevance(node.last_action["command"], node.observation)
        # Take weighted average of relevance score and current value
        new_value = (0.7 * node.value + 0.3 * relevance_score)
        instance_logger.debug(f">> Similarity reward adjustment: {node.value:.4f} -> {new_value:.4f}")

        if node.is_terminating and not node.invalid_termination:
            new_value = self._adjust_terminating_reward(node, new_value)
            
        end_time = time.time()
        instance_logger.debug(f"=>> Reward: {new_value:.4f} | Time taken: {end_time - start_time:.2f} seconds")
        return new_value
    
    def _get_commit_hash(self):
        """Get the current commit hash"""
        return self.env.execute("git rev-parse HEAD")["output"].strip()
    
    def _create_pseudo_root(self):
        if self._repo_has_changes():
            self.env.execute(f"git add -A && git commit -m 'Committing changes before starting tree search' --no-verify")
            action = "git add -A >/dev/null 2>&1 && git commit -m 'Committing changes before starting tree search' --no-verify >/dev/null 2>&1 && git rev-parse HEAD"
            self.add_message("system", f"THOUGHT: Need to commit changes before starting tree search.\n\n```bash\n{action}\n```")
            instance_logger.debug(">> Warning: Uncommitted changes detected at the start of tree search. Committing changes before starting tree search...")
            
            if self._repo_has_changes():
                if self._repo_has_submodules():
                    self.env.execute("git submodule foreach --recursive git reset --hard")
                    self.env.execute("git submodule foreach --recursive git clean -fd")

        output = self.env.execute("git rev-parse HEAD")    
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        
        new_node = self._create_node()
        self.tree_node.add_child(
            new_node
        )

        new_node.commit = output["output"].strip() # Commit hash of the pseudo root node
        self.tree_node.executed = True
        self.tree_node = new_node
        self.tree_node.executed = True
        
    def _commit_changes(self, message="Automated commit") -> tuple[str, bool]:
        """Stage all changes and commit"""
        instance_logger.debug(">> Committing changes to the repository...")
        is_submodule_commit = False
        output = self.env.execute("git add -A")
        if output.get("returncode", 0) != 0:
            instance_logger.debug(">> Error staging changes:\n" + output.get("output", ""))
        output = self.env.execute (f'git commit -m "{message}" --no-verify')
        if output.get("returncode", 0) != 0:
            if self._repo_has_submodules():
                # Step 1: commit each submodule if dirty
                self.env.execute(
                    "git submodule foreach --recursive '"
                    "if [ -n \"$(git status --porcelain)\" ]; then "
                    "git add -A && git commit --no-verify -m \"{}\"; "
                    "fi'".format(message)
                )
                # Step 2: stage updated submodule pointers in parent repo
                self.env.execute("git add -A")
                # Step 3: commit in parent repo
                output = self.env.execute(f'git commit -m "{message}" --no-verify')
                if output.get("returncode", 0) != 0:
                    raise Exception(">> Still could not commit changes:\n" + output.get("output", ""))
                # clear frontier to avoid stale nodes with old commit hashes
                is_submodule_commit = True
            else:
                raise Exception(">> Error committing changes:\n" + output.get("output", ""))
        
        output = self.env.execute("git rev-parse HEAD")
        self.add_message("system", f'THOUGHT: Commit changes of the last command.\n\n```bash\ngit add -A >/dev/null 2>&1 && git commit -m "{message}" --no-verify >/dev/null 2>&1 && git rev-parse HEAD\n```')
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        if self._repo_has_changes():
            raise Exception(">> Warning: Changes still detected after commit.")
        
        commit = output["output"].strip()
        self.commits.append(commit)

        return commit, is_submodule_commit
    
    def _reset(self):
        super()._reset()
        
        if self.env.config.checkpoint:
            with open(self.env.config.checkpoint, "r", encoding="utf-8") as f:
                tree_json = json.load(f)
            self.tree_root.from_tree(tree_json)
            self.mode = "simulation"
            self.tree_node = self.tree_root.children[0]
            instance_logger.info(f">> Loaded tree from checkpoint {self.env.config.checkpoint}. Starting in simulation mode.")
        else:
            self.tree_root.commit = self._get_commit_hash()
            self._create_pseudo_root()
            self.tree_node.test_status = self._get_test_status()
            self.tree_node.state_hash = self._get_state_hash(self.tree_node)
            if self.tree_node.test_status == None:
                self.tree_node.test_status = []        
    
        issue_tokens = self.task.split()
        scores = self.bm25.get_scores(issue_tokens)
        scores = (scores - scores.min()) / (scores.max() - scores.min())
        self.relevance_dict = dict(zip(self.file_ids, scores))
        
        # Print top 10 relevant files
        top_indices = np.argsort(scores)[-10:][::-1]
        instance_logger.debug(">> Top 10 relevant files for the issue:")
        for idx in top_indices:
            instance_logger.debug(f"- {self.file_ids[idx]} (score: {scores[idx]:.4f})")
            
        sorted_items = sorted(self.relevance_dict.items(), key=lambda x: x[1], reverse=True)

        retrieved_docs = []
        for file_path, score in sorted_items[:10]:
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
            instance_logger.debug(">> Repository has unstaged or uncommitted changes.")
            instance_logger.debug(observation["output"])
        return bool(observation["output"])
    
    def _repo_has_submodules(self):
        """Check if the repo has any submodules"""
        observation = self.env.execute("git submodule status")
        return bool(observation["output"].strip())
    
    # Issue: Doesn't capture untracked files
    # def _get_modified_files(self):
    #     """Get the list of modified files in the repo"""
    #     observation = self.env.execute("git diff --name-only")
    #     return observation["output"].splitlines()
    
    def _get_modified_files(self):
        """Get the list of modified and untracked files in the repo"""
        observation = self.env.execute("git status --porcelain")
        # Each line starts with a 2-char status, then a space, then the file name
        files = [line[3:] for line in observation["output"].splitlines() if line.strip()]
        return files

    def parse_git_diff(self):
        observation = self.env.execute("git add -A && git diff --cached && git reset HEAD") # TODO: Doesn't work for new files. As new file isn't staged.
        lines = observation["output"].splitlines()
        
        instance_logger.debug(">> Parsing git diff output...")
        instance_logger.debug(f"Number of lines in git diff: {len(lines)}")
        instance_logger.debug("\n".join(lines[:10]))  # Print the first 10 lines of the diff for debugging

        file_name = None
        change_type = "modified"
        changes = []

        hunk_pattern = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")

        for line in lines:
            # Start of a new file section
            if line.startswith("diff --git"):
                file_name = None
                change_type = "modified"
                continue

            # Detect new file
            if line.startswith("new file mode"):
                change_type = "new"
                continue

            # Detect deleted file
            if line.startswith("deleted file mode"):
                change_type = "deleted"
                continue

            # Extract file name
            if line.startswith("+++ "):
                if line.startswith("+++ /dev/null"):
                    continue
                file_name = line.replace("+++ b/", "").strip()
                if change_type in ("new", "deleted") and file_name:
                    changes.append((file_name, change_type, None, None))
                continue

            # Extract hunk info (only for modified files)
            hunk_match = hunk_pattern.match(line)
            if hunk_match and file_name and change_type == "modified":
                old_start = int(hunk_match.group(1))
                old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                old_end = old_start + old_count - 1

                changes.append((file_name, change_type, old_start, old_end))


        return changes, len(lines)
    
    def _is_repeat_action(self, node):
        if node.last_action is None or node.modifies_code:
            return False
        curr = node.parent
        while curr is not None and curr.last_action is not None:
            if curr.modifies_code:
                break
            if curr.last_action["command"] and curr.last_action["command"] == node.last_action["command"]:
                return True
            curr = curr.parent
        return False
        
    def _query_safely(self, messages):
        remove_steps = 0
        while True:
            try:
                response = self.query(messages)
                return response
            except (litellm.exceptions.ContextWindowExceededError, litellm.exceptions.BadRequestError) as e:
                remove_steps += 1
                messages = messages[:2] + messages[2+2*remove_steps:]
                if len(messages) <= 2:
                    raise Exception(">> Error: Context window too small to fit any messages.")
                instance_logger.debug(f">> Context window exceeded. Removing the oldest {remove_steps} steps and retrying...")
            
    def _generate_action(self):
        """
        Generate an action from the model and parse it
        
        Returns:
            response (dict): The raw response from the model
            action (dict): The parsed action
            error (str | None): The error message if parsing failed
        """
        messages = self.get_messages(self.tree_node)
        max_retries = 3
        
        def is_git_command(cmd: str):
            import re
            GIT_CMD = re.compile(
                r'(^|[;&|()]\s*)git(?=\s|$)'
            )
            if GIT_CMD.search(cmd):
                return True
            return False
                
        for i in range(max_retries):  # Retry mechanism in case of parsing errors
            response = self._query_safely(messages)
            try:
                action = self.parse_action(response)
                potential_termination = is_terminating(action["action"])             
                if not potential_termination and is_git_command(action["action"]):
                    error = "Error: git commands are not allowed." + ("Try 'applypatch' instead of 'git apply' for applying patches." if "git apply" in action["action"] else "")
                    messages.append({
                        "role": "assistant",
                        "content": response["content"]
                    })
                    messages.append({
                        "role": "user",
                        "content": error
                    })
                    if i == max_retries - 1:
                        instance_logger.debug(f">> Failed to parse action after {max_retries} attempts. Returning error.")
                        return response, None, error, i
                    else:
                        instance_logger.debug(f">> Error parsing action. Retrying #{i+1}...")
                    continue
                    # time.sleep(2)  # To avoid rate limiting
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
                    
    def _get_root_commit(self) -> str:
        if self.env.config.clean_start:
            return self.tree_root.children[0].commit
        return self.tree_root.commit
    
    def _get_type(self, node):
        cmd_type = "read"
        if node.last_action["command"] is not None:
            if node.is_terminating or node.invalid_termination:
                cmd_type = "submit"
            elif "[EDIT]" in node.last_action["thought"]: #  or node.modifies_code --- testing may have side effects. Can ignore that.
                cmd_type = "edit"
            elif "[SEARCH]" in node.last_action["thought"]:
                cmd_type = "search"
            elif "[TEST]" in node.last_action["thought"] or ("[READ]" not in node.last_action["thought"] and self.is_test_command(node)): # TEMP: Since test command detection is not very robust, we can also use a heuristic based on the thought content to identify potential test commands for better reward adjustment.
                cmd_type = "test"
            elif node.modifies_code:
                cmd_type = "edit"
        else:
            if "[SUBMIT]" in node.last_action["thought"]:
                cmd_type = "submit"
            elif "[EDIT]" in node.last_action["thought"]:
                cmd_type = "edit"
            elif "[SEARCH]" in node.last_action["thought"]:
                cmd_type = "search"
            elif "[TEST]" in node.last_action["thought"]:
                cmd_type = "test"
        
        gen_type = (
            'edit' if '[EDIT]' in node.last_action['thought']
            else 'search' if '[SEARCH]' in node.last_action['thought']
            else 'test' if '[TEST]' in node.last_action['thought']
            else 'submit' if '[SUBMIT]' in node.last_action['thought']
            else 'read'
        )
        if gen_type != cmd_type:
            instance_logger.debug(f">> Warning: Command type mismatch. Thought indicates {gen_type} but detected as {cmd_type}.")
            
        return cmd_type
    
    def _get_state_hash(self, node):
        if self.mode == "evaluation":
            if self._get_root_commit() == node.commit:
                return "empty"
            if not node.modifies_code and node.last_action:
                return node.parent.state_hash
            # Stage changes
            response = self.env.execute(f"git checkout {self._get_root_commit()} && git restore --source {node.commit} . && git add -A")
                    
            if response.get("returncode", 0) != 0:
                instance_logger.debug(">> Warning: Failed to stage changes to main branch before submission.")
                instance_logger.debug(f"Error details: {response}")
            
            # Check if staging area is empty
            response = self.env.execute("git diff --cached --quiet")
            if response.get("returncode", 0) == 0:
                state_hash = "empty"
            else:
                # Calculate state hash based on staged changes
                response = self.env.execute("git write-tree")
                state_hash = response.get("output", "").strip()
            
            self.env.execute(f"git reset --hard HEAD && git clean -fd  && git checkout {node.commit}")
            if state_hash == node.parent.state_hash:
                instance_logger.debug(">> ERROR: State hash is the same as parent node, which may indicate an issue with state hash calculation.")
                for file, ctype, start, end in node.changes:
                    instance_logger.debug(f">> Modified file: {file} | Change type: {ctype} | Lines: {start}-{end if end is not None else ''}")

            return state_hash
        
        return node.state_hash
                   
    def _action_to_node(self, response, action, error, current_node, retries = 0):
        # get_observation action to get observation
        potential_termination = False
        
        if error is None:
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
            
        new_node.retries = retries
        new_node.parent = current_node
        
        cache_node = self._find_node_from_cache(current_node.state_hash, new_node.last_action["command"])
        
        if cache_node:
        # if False:
            instance_logger.debug(">> Cache hit for action: " + new_node.last_action["command"] + " at parent state: " + current_node.state_hash)
            new_node.observation = cache_node.observation
            new_node.raw_observation = cache_node.raw_observation
            new_node.modifies_code = cache_node.modifies_code
            new_node.modified_files = cache_node.modified_files
            new_node.changes = cache_node.changes
            new_node.diff_size = cache_node.diff_size
            new_node.test_status = cache_node.test_status
            new_node.fails_tests = cache_node.fails_tests
            new_node.is_terminating = cache_node.is_terminating
            new_node.invalid_termination = cache_node.invalid_termination
            new_node.is_timeout = cache_node.is_timeout
            if cache_node.modifies_code:
                new_node.commit = cache_node.commit
            else:
                new_node.commit = current_node.commit
            new_node.read_files = cache_node.read_files
            new_node.is_system_response = cache_node.is_system_response
            new_node.cache_hit = cache_node.id
            new_node.state_hash = cache_node.state_hash
            new_node.parent = current_node
            new_node.last_action["type"] = self._get_type(new_node)
            
        else:
            if error is None:
                try:
                    potential_termination = is_terminating(new_node.last_action["command"])     
                    # Be-aware of potential terminating actions
                    if potential_termination:
                        res = self.env.execute(f"git checkout {self._get_root_commit()} && git restore --source {current_node.commit} .")
                            
                        if res.get("returncode", 0) != 0:
                            instance_logger.debug(">> Warning: Failed to restore to the current node's commit before executing potential terminating action.")
                            instance_logger.debug(f"Error details: {res}")  
                            
                    output = self.env.execute(new_node.last_action["command"])
                    # Check for terminating action
                    lines = output.get("output", "").lstrip().splitlines(keepends=True)
                    if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
                        instance_logger.debug(">> Terminating action detected.")
                        new_node.is_terminating = True   
                        patch = "".join(lines[1:])
                        if current_node.commit == self._get_root_commit() or not patch.strip(): # After some modifications, final patch may still be empty. May be agent reverted the changes.
                            instance_logger.debug(">> Warning: Terminating action detected without any modifications.")
                            new_node.observation = "Error: Submission detected without any modifications. Make sure to modify the code before submission."
                            new_node.raw_observation = output
                            new_node.is_system_response = True
                            new_node.is_terminating = False
                            new_node.invalid_termination = True
                            new_node.last_action["thought"] = new_node.last_action["thought"].replace("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
                            new_node.last_action["command"] = new_node.last_action["command"].replace("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")
                            # new_node.last_action["command"] = None
                        else:    
                            new_node.observation = patch # 
                            new_node.raw_observation = output
                
                    if potential_termination:
                        self.env.execute(f"git reset --hard HEAD && git clean -fd  && git checkout {current_node.commit}") # Restore to current node's commit after executing potential terminating action to avoid affecting other branches, as we will use staged changes to calculate state hash for terminating nodes.
                    observation = self.render_template(self.config.action_observation_template, output=output) 
                    raw_observation = output
                    
                except (TimeoutError, subprocess.TimeoutExpired) as e:
                    output = e.output.decode("utf-8", errors="replace") if getattr(e, "output", None) else ""
                    observation = self.render_template(self.config.timeout_template, action=action, output=output)
                    raw_observation = None
                    new_node.is_timeout = True
                    if potential_termination:
                        self.env.execute(f"git reset --hard HEAD && git clean -fd && git checkout {current_node.commit}")
                    
                # Check for code modifications
                if self._repo_has_changes():
                    new_node.modifies_code = True
                    has_write_child = True
                    new_node.modified_files = self._get_modified_files()
                    print(f">> Modified files: {new_node.modified_files[:10]}")  # Print the first 10 modified files for debugging
                    new_node.changes, new_node.diff_size = self.parse_git_diff()
                    # Print first 5 and last 5 modified files in case of large number of changes
                    if len(new_node.changes) > 10:
                        instance_logger.debug(f">> More than 10 modified files detected. Showing first 5 and last 5 modified files:")
                        for file, ctype, start, end in new_node.changes[:5]:
                            instance_logger.debug(f">> Modified file: {file} | Change type: {ctype} | Lines: {start}-{end if end is not None else ''}")
                        instance_logger.debug("...")
                        for file, ctype, start, end in new_node.changes[-5:]:
                            instance_logger.debug(f">> Modified file: {file} | Change type: {ctype} | Lines: {start}-{end if end is not None else ''}")
                    else:
                        for file, ctype, start, end in new_node.changes:
                            instance_logger.debug(f">> Modified file: {file} | Change type: {ctype} | Lines: {start}-{end if end is not None else ''}")
                    
                    instance_logger.debug(">> Write-action detected.")
                    
                    # self.env.execute("git reset --hard HEAD && git clean -fd") # OLD
                    # NEW
                    new_node.commit, _ = self._commit_changes()
                    instance_logger.debug(f">> New commit created: {self.tree_node.commit}")
                    
                    # double check: submodule case
                    if self._repo_has_changes():
                        if self._repo_has_submodules():
                            self.env.execute("git submodule foreach --recursive git reset --hard")
                            self.env.execute("git submodule foreach --recursive git clean -fd")
                            instance_logger.debug(">> Warning: SWE-Bench eval doesn't support submodule modifications. Skipping this action...")
                            return None
                        else:
                            raise Exception(">> Error: Changes still detected after reset and clean.")
                    
                    new_node.state_hash = self._get_state_hash(new_node)
                    if self._repo_has_changes():
                        raise Exception(">> Error: Changes detected after state hashing.")
                    
                    new_node.test_status = self._get_test_status()
                    if new_node.test_status == None:
                        instance_logger.debug(">> Warning: Failed to get test status after code modifications. Marking all tests as ERROR for this node.")
                        new_node.test_status = [
                            {**test, "status": "ERROR"}
                            for test in current_node.test_status
                        ]

                    self.env.execute(f"git checkout {current_node.commit}") # Move back to previous commit after evaluation to avoid affecting other branches
                    
                elif new_node.last_action["command"] is not None:
                    commands = parser.parse(new_node.last_action["command"])  # Check if it's a read action and can be parsed
                    # Slightly boost nodes that read files based on relevance
                    def normalize_path(p: str) -> str:
                        p = p.lstrip("./")  # remove leading ./ or /
                        if p.startswith("testbed/"):
                            p = p[len("testbed/"):]
                        return p
                    for cmd in commands:
                        if cmd.get("command") in ["nl", "cat", "head", "tail"]:
                            args = cmd.get("args", [])
                            if args and ("/" in args[-1] or "." in args[-1]):  # crude check for file path
                                arg = normalize_path(args[-1])  # consider the last argument as the file path
                                if arg not in new_node.read_files:
                                    new_node.read_files.append(arg)
                                    instance_logger.debug(f">> Read-action detected. File: {arg}")
                            # for arg in cmd.get("args", []):
                            #     if not arg.startswith('-'):
                    new_node.test_status = current_node.test_status
                
                if not new_node.invalid_termination and new_node.is_terminating != potential_termination:
                    instance_logger.debug(">> Warning: Invalid terminating action detected. Skipping this action...")
                    instance_logger.debug(f"Action: {new_node.last_action['command']}")
                    instance_logger.debug(f"Observation: {observation[:200]}{'...' if len(observation) > 200 else ''}")
                    time.sleep(2)  # To avoid rate limiting
                    return None     
            else:
                observation = error
                raw_observation = None
            
            if new_node.observation is None: # Q: When will it not be None here? A: When terminating action detected above
                new_node.observation = observation
                new_node.raw_observation = raw_observation
            
            
            if not new_node.modifies_code:
                new_node.commit = current_node.commit
                new_node.test_status = current_node.test_status
                new_node.state_hash = current_node.state_hash

            if new_node.last_action["command"] is not None:
                self.action_cache[f"{new_node.state_hash}_{new_node.last_action['command']}"] = new_node.id
                    
        if self._is_repeat_action(new_node):
            instance_logger.debug(">> Warning: Repeat action detected.")
            # new_node.observation = "<warning>Repeat action detected. This action-observation already exists in the context. No information gain expected from this action.</warning>\n" + new_node.observation # TODO: Issue with caching
            new_node.is_repeat = True

        
        new_node.last_action["type"] = self._get_type(new_node)
        
        return new_node
      
    def _generate_new_node(self, i) -> TreeSearchNode:
        # with self.node_creation_lock:
        self.SYSTEM_PROMPT = self.candidates[i % len(self.candidates)]["SYSTEM_PROMPT"]
        self.USER_PROMPT = self.candidates[i % len(self.candidates)]["USER_PROMPT"]
        response, action, error, retries = self._generate_action()
        if error is None:
            instance_logger.debug(f"Generated action #{i+1}: {action['action'][:200]}{'...' if len(action['action']) > 200 else ''}")
        else:
            instance_logger.debug(f"Generated action #{i+1}: <<Invalid Action>>")
        return self._action_to_node(response, action, error, self.tree_node, retries)
    
    def _find_node_from_cache(self, state_hash, command):
        if command is None:
            return None
        key = f"{state_hash}_{command}"
        node_id = self.action_cache.get(key)
        if node_id is not None:
            return self.all_node_map.get(node_id)
        return None
    
    def _generate_new_nodes(self, n_actions) -> List[TreeSearchNode]:
        nodes = []
        futures = []

        max_workers = 4

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for i in tqdm(range(n_actions), desc="Generating nodes"):
                new_node = self._generate_new_node(i)
                if new_node is None:
                    continue

                nodes.append(new_node)
                # if not new_node.cache_hit and new_node.last_action["command"] is not None:
                #     self.action_cache[f"{self.tree_node.state_hash}_{new_node.last_action['command']}"] = new_node.id
                    
                futures.append((new_node, executor.submit(self._evaluate_node, new_node)))
                # time.sleep(1)

            if futures:
                for new_node, future in tqdm(futures, total=len(futures), desc="Waiting for node scores"):
                    new_node.value = future.result()

        has_write_child = any(node.modifies_code for node in nodes)
        if has_write_child:
            self.n_modifications += 1
            
        return nodes

    def _repo_has_new_commit(self):
        output = self.env.execute("git rev-parse HEAD")
        current_commit = output.get("output", "").strip()
        if current_commit != self.tree_node.commit:
            instance_logger.debug(f">> New commit detected: {current_commit} (previous: {self.tree_node.commit})")
            return True
        return False
 
    def _stage_to_main_branch(self):
        if self.mode == "evaluation":
            response = self.env.execute(f"git checkout {self._get_root_commit()} && git restore --source {self.tree_node.commit} .")
                    
            if response.get("returncode", 0) != 0:
                instance_logger.debug(">> Warning: Failed to stage changes to main branch before submission.")
                instance_logger.debug(f"Error details: {response}")

            # Check for repo changes
            if not self._repo_has_changes():
                instance_logger.error(">> No changes detected to stage to main branch before submission.")
        
        self.add_message("system", f"THOUGHT: Preparing final output before submission.\n\n```bash\ngit checkout {self._get_root_commit()} && git restore --source {self.tree_node.commit} .\n```")
                     
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
            self.tree_node.is_submission = True
            self.tree_node.commit = self.tree_node.parent.commit

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

        if self.tree_node.level == self.config.depth_limit:
            instance_logger.debug(f">> Reached max depth limit at node with action: {self.tree_node.last_action['command']}. Marking as leaf node.")
            raise Exception(">> Reached max depth limit. Stopping expansion at this node.")
            
        return self.tree_node.observation
    
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
    
    def is_test_command(self, node: str) -> bool:
        if node.last_action["command"] is None:
            return False
        import re
        TEST_CMD = re.compile(
            r'(^|\s)(pytest|python\s+-m\s+pytest|python\s+-m\s+unittest|unittest|runtests)(\s|$)'
        )
        return bool(TEST_CMD.search(node.last_action["command"]))
            
    def _print_candidates(self, nodes):
        reward_data = []
        for new_node in nodes:
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
            instance_logger.debug(
                tabulate(
                    reward_data,
                    headers=["Action", "Reward", "Merged"],
                    tablefmt="grid",
                    colalign=("left", "center", "center"),
                )
            )
                         
    def _process_nodes(self, tree_nodes: List[str]) -> List[TreeSearchNode]:
        self.n_actions += len(self.tree_node.children)
        instance_logger.debug(f"# {len(tree_nodes)} new nodes generated at level {self.tree_node.level}:")
        for node in tree_nodes:
            instance_logger.debug(f"- {node.last_action['command']}")
            
        tree_nodes = action_processor.merge_nodes(tree_nodes)
        self._print_candidates(tree_nodes)
            
        return tree_nodes
    
    def _update_frontier(self, tree_nodes: List[TreeSearchNode]):  
        if len(tree_nodes) == 0:
            return                    
        self._add_actions_to_frontier(tree_nodes)