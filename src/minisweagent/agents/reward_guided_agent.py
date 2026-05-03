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

class RewardGuidedAgentConfig(SingleActionAgentConfig):
    retrieval_template: str
    reproduction_patch: str = ""
    """Patch that adds/updates reproduction artifacts (typically run_test.sh) for test status checks."""
    branching_factor: int = 3
    """The maximum number of branches to explore at each node."""
    

        
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
        
    def _get_commit_hash(self):
        """Get the current commit hash"""
        return self.env.execute("git rev-parse HEAD")["output"].strip()
    
    def _create_pseudo_root(self):
        if self._repo_has_changes():
            # if self.env.config.clean_start:
            #     self.env.execute("git reset --hard HEAD && git clean -fd")
            #     action = "git reset --hard HEAD >/dev/null 2>&1 && git clean -fd >/dev/null 2>&1 && git rev-parse HEAD"
            #     self.add_message("system", f"THOUGHT: Starting with a clean state for tree search.\n\n```bash\n{action}\n```")
            #     instance_logger.debug(">> Warning: Uncommitted changes detected at the start of tree search. Cleaning changes and starting tree search with a clean state...")
            # else:
            self.env.execute(f"git add -A && git commit -m 'Committing changes before starting tree search' --no-verify")
            action = "git add -A >/dev/null 2>&1 && git commit -m 'Committing changes before starting tree search' --no-verify >/dev/null 2>&1 && git rev-parse HEAD"
            self.add_message("system", f"THOUGHT: Need to commit changes before starting tree search.\n\n```bash\n{action}\n```")
            instance_logger.debug(">> Warning: Uncommitted changes detected at the start of tree search. Committing changes before starting tree search...")
            
            if self._repo_has_changes():
                if self._repo_has_submodules():
                    self.env.execute("git submodule foreach --recursive git reset --hard")
                    self.env.execute("git submodule foreach --recursive git clean -fd")
        # else:
            # self.env.execute(f"git checkout -b ts-agent-root")
            # action = "git checkout -b ts-agent-root >/dev/null 2>&1 && git rev-parse HEAD"
            # self.add_message("system", f"THOUGHT: Switching to new branch before starting tree search.\n\n```bash\n{action}\n```")
            
        output = self.env.execute("git rev-parse HEAD")    
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        
        new_node = self._create_node()
        self.tree_node.add_child(
            new_node
        )
        # new_node.branch = f"ts-agent-root"
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
        return output["output"].strip(), is_submodule_commit
    
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
        
    def _reset(self):
        super()._reset()
        # checkout to ts-main branch (new)
        # self.tree_root.branch = "ts-main"
        # self.env.execute("git checkout -b ts-main")
        self.tree_root.commit = self._get_commit_hash()
        self._create_pseudo_root()
        self.tree_node.test_status = self._get_test_status()
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
        
        # Hierarchy
        # scores = self.bm25_h.get_scores(issue_tokens)
        # scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # for node, score in zip(self.rank_nodes, scores):
        #     node.self_score = score
        # propagate_scores(self.repo_root)
        # repo_nodes = collect_all_nodes(self.repo_root)
        # file_scores = [node.score for node in repo_nodes if node.type == "file"]
        # qualified_names = [node.qualified_name() for node in repo_nodes if node.type == "file"]
        # # instance_logger.debug(f">> Found {len(file_scores)} files.")
        # top_indices = np.argsort(file_scores)[-5:][::-1]
        # instance_logger.debug(">> Top 5 relevant files for the issue (H):")
        # for idx in top_indices:
        #     instance_logger.debug(f"- {qualified_names[idx]} (score: {file_scores[idx]:.4f})")
        

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
        instance_logger.debug(lines[:10])  # Print the first 10 lines of the diff for debugging

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
        
    def _find_file_by_path(self, node, path_parts):
        if not path_parts:
            return node
        for child in node.children:
            if child.name == path_parts[0]:
                return self._find_file_by_path(child, path_parts[1:])
        return None

    def _calculate_hierarchical_score(self, changes):
        # total_score = 0.0
        max_score = 0.0
        count = 0
        
        for file, ctype, start, end in changes:
            if ctype != "modified":
                continue
            
            flag = False
            
            path = Path(file)
            node = self._find_file_by_path(self.repo_root, path.parts)
            for child in node.children:
                if start >= child.start_line and start <= child.end_line:
                    if child.type == "function":
                        # total_score += child.self_score
                        max_score = max(max_score, child.self_score)
                        instance_logger.debug(f">> Found relevant function: {child.qualified_name()} with score {child.self_score:.4f}")
                        flag = True
                        break
                    elif child.type == "class":
                        for method in child.children:
                            if start >= method.start_line and start <= method.end_line:
                                # total_score += method.self_score
                                max_score = max(max_score, method.self_score)
                                instance_logger.debug(f">> Found relevant method: {method.qualified_name()} with score {method.self_score:.4f}")
                                flag = True
                                break
                        if not flag:
                            # total_score += child.self_score
                            max_score = max(max_score, child.self_score)
                            instance_logger.debug(f">> Found relevant class: {child.qualified_name()} with score {child.self_score:.4f}")
                            flag = True
                            break
            
            if not flag:
                # total_score += node.self_score
                max_score = max(max_score, node.self_score)
                instance_logger.debug(f">> Found relevant file: {node.qualified_name()} with score {node.self_score:.4f}")
                flag = True
                
            if flag:
                count += 1
        # return total_score / count if count > 0 else 0.0
        return max_score
        
    def _get_root_commit(self) -> str:
        if self.env.config.clean_start:
            return self.tree_root.children[0].commit
        return self.tree_root.commit
    
    def _generate_new_node(self, i) -> TreeSearchNode:
        # get_observation action to get observation
        potential_termination = False
        
        self.SYSTEM_PROMPT = self.candidates[i % len(self.candidates)]["SYSTEM_PROMPT"]
        self.USER_PROMPT = self.candidates[i % len(self.candidates)]["USER_PROMPT"]
        
        response, action, error = self._generate_action()
        if error is None:
            instance_logger.debug(f"Generated action #{i+1}: {action['action']}")
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
                                            
                potential_termination = is_terminating(action["action"])                       
                if not potential_termination and is_git_command(action["action"]):
                    instance_logger.debug(">> Warning: git commands are not allowed in non-terminating actions. Skipping this action...")
                    new_node.observation = "Error: git commands are not allowed."
                    new_node.raw_observation = None
                    new_node.is_system_response = True
                    new_node.last_action["command"] = None
                    output = {"output": new_node.observation, "returncode": 1}
                    # time.sleep(2)  # To avoid rate limiting
                else:
                    # Be-aware of potential terminating actions
                    if potential_termination:
                        res = self.env.execute(f"git checkout {self._get_root_commit()} && git restore --source {self.tree_node.commit} .")
                            
                        if res.get("returncode", 0) != 0:
                            instance_logger.debug(">> Warning: Failed to restore to the current node's commit before executing potential terminating action.")
                            instance_logger.debug(f"Error details: {res}")  
                            
                    output = self.env.execute(action["action"])
                    # Check for terminating action
                    lines = output.get("output", "").lstrip().splitlines(keepends=True)
                    if lines and lines[0].strip() in ["MINI_SWE_AGENT_FINAL_OUTPUT", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"]:
                        instance_logger.debug(">> Terminating action detected.")
                        new_node.is_terminating = True   
                        
                        if self.tree_node.commit == self._get_root_commit():
                            instance_logger.debug(">> Warning: Terminating action detected without any modifications.")
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
                        
                new_node.test_status = self._get_test_status()
                if new_node.test_status == None:
                    new_node.test_status = self.tree_node.test_status
                # Rollback changes
                # run tests
                # test_result = self.env.execute("pytest --maxfail=1 --disable-warnings -q")
                # if test_result.get("returncode", 0) != 0:
                #     new_node.fails_tests = True
                #     print(">> pytest --maxfail=1 --disable-warnings -q")
                #     print(test_result.get("output", ""))
                instance_logger.debug(">> Write-action detected.")
                self.env.execute("git reset --hard HEAD && git clean -fd")
                
                # double check: submodule case
                if self._repo_has_changes():
                    if self._repo_has_submodules():
                        self.env.execute("git submodule foreach --recursive git reset --hard")
                        self.env.execute("git submodule foreach --recursive git clean -fd")
                        instance_logger.debug(">> Warning: SWE-Bench eval doesn't support submodule modifications. Skipping this action...")
                        return None
                    else:
                        raise Exception(">> Error: Changes still detected after reset and clean.")
            else:
                commands = parser.parse(action["action"])  # Check if it's a read action and can be parsed
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
                new_node.test_status = self.tree_node.test_status
                                
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
                #     instance_logger.debug(f">> Read-action detected. File: {filename}")
            
            if not new_node.invalid_termination and new_node.is_terminating != potential_termination:
                instance_logger.debug(">> Warning: Invalid terminating action detected. Skipping this action...")
                time.sleep(2)  # To avoid rate limiting
                return None     
            
        else:
            instance_logger.debug(f"Generated action #{i+1}: <<Invalid Action>>")
            observation = error
            raw_observation = None
        
        if new_node.observation is None: # Q: When will it not be None here? A: When terminating action detected above
            new_node.observation = observation
            new_node.raw_observation = raw_observation
            
        # if not new_node.is_terminating and new_node.level >= self.config.depth_limit:
        #     instance_logger.debug(f"Non-terminating Node {new_node.last_action['command']} exceeded max depth {self.config.depth_limit}, skipping...")
        #     new_node.prune()
        #     return None
        
        # new_node.value = self._evaluate_node(new_node)
        new_node.parent = self.tree_node
        return new_node
            
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
                futures.append((new_node, executor.submit(self._evaluate_node, new_node)))
                time.sleep(2)

            if futures:
                for new_node, future in tqdm(futures, total=len(futures), desc="Waiting for node scores"):
                    new_node.value = future.result()

        has_write_child = any(node.modifies_code for node in nodes)
        if has_write_child:
            self.n_modifications += 1
            
        return nodes
    
    def _repo_has_changes_with_main(self):
        if self.tree_node.parent.commit == self._get_root_commit():
            return False
        output = self.env.execute(f"git diff {self._get_root_commit()}..{self.tree_node.parent.commit}")

        diff_text = output.get("output", "").strip()

        if not diff_text:
            instance_logger.debug(f"No change between {self._get_root_commit()} and {self.tree_node.parent.commit}.")
            raise Exception(">> No changes detected to stage to main branch.")
        else:
            # instance_logger.debug(">> Staging changes to main branch before submission...")
            # instance_logger.debug(diff_text)
            return True

    def _repo_has_new_commit(self):
        output = self.env.execute("git rev-parse HEAD")
        current_commit = output.get("output", "").strip()
        if current_commit != self.tree_node.commit:
            instance_logger.debug(f">> New commit detected: {current_commit} (previous: {self.tree_node.commit})")
            return True
        return False
            
    def _stage_to_main_branch(self):
        # self._repo_has_changes_with_main()
        response = self.env.execute(f"git checkout {self._get_root_commit()} && git restore --source {self.tree_node.parent.commit} .")
        self.add_message("system", f"THOUGHT: Preparing final output before submission.\n\n```bash\ngit checkout {self._get_root_commit()} && git restore --source {self.tree_node.parent.commit} .\n```")
        
        if response.get("returncode", 0) != 0:
            instance_logger.debug(">> Warning: Failed to stage changes to main branch before submission.")
            instance_logger.debug(f"Error details: {response}")
            
        # output = self.env.execute(f"git fsck --unreachable")
        # instance_logger.debug(f">> Unreachable commits:\n{output.get('output', '')}")
            
        # Check for repo changes
        if not self._repo_has_changes():
            instance_logger.error(">> No changes detected to stage to main branch before submission.")
            
        
            
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
            # self.tree_node.branch = self.tree_node.parent.branch
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
        # self.tree_node.branch = self.tree_node.parent.branch
        if self.tree_node.modifies_code:
            self.tree_node.commit, _ = self._commit_changes()
            instance_logger.debug(f">> New commit created: {self.tree_node.commit}")
        else:
            self.tree_node.commit = self._get_commit_hash() #  Can't we just keep the same commit if code isn't modified?
            instance_logger.debug(f">> No changes detected, staying on commit: {self.tree_node.commit}")

        if self.tree_node.level == self.config.depth_limit:
            instance_logger.debug(f">> Reached max depth limit at node with action: {self.tree_node.last_action['command']}. Marking as leaf node.")
            raise Exception(">> Reached max depth limit. Stopping expansion at this node.")
            
        return self.tree_node.observation
    
    def _calculate_relevance(self, action, observation) -> float:
        # Example step from agent
        agent_step = f"Action: {action} | Observation: {observation}"
        # The issue we want to check
        issue_text = self.task
        # Get relevance score
        for _ in range(3):  # Retry mechanism in case of transient errors
            try:
                response = requests.post(os.environ["SENTENCE_TRANSFORMER_SERVER"] + "/v1/relevance", json={"model": "all-mpnet-base-v2", "text1": agent_step, "text2": issue_text})
                # all-mpnet-base-v2, all-MiniLM-L6-v2
                score = response.json().get("score", 0.0)
                instance_logger.debug(f">> Relevance score for action '{action[:50] if action else '<<Invalid Action>>'}': {score:.4f}")
                break
            except Exception as e:
                instance_logger.debug(f">> Error calculating relevance score: {repr(e)}. Retrying...")
                score = 0.0
                time.sleep(1)
            
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
    
    def is_test_command(self, node: str) -> bool:
        import re
        TEST_CMD = re.compile(
            r'(^|\s)(pytest|python\s+-m\s+pytest|python\s+-m\s+unittest|unittest|runtests)(\s|$)'
        )
        return bool(TEST_CMD.search(node.last_action["command"]))
    
    def is_test_failure(self, node: TreeSearchNode, returncode: int) -> bool:
        # OLD:
        if self.is_test_command(node) and returncode == 1:
            return True
        # NEW:
        # if node.last_action["type"] == "test" and returncode == 1:
        #     return True
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
        final_value = 0.7 * value + 0.3 * test_component
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
                
        return new_value
    
    def _evaluate_node(self, node):
        if node.value is not None:
            return node.value
        
        if self.config.branching_factor == 1:
            return 0.0 # For single-branch case, we can skip reward computation and directly evaluate the action's relevance to the task
        
        # track the time taken for reward computation
        start_time = time.time()
        cmd_type = "read"
        if node.last_action["command"] is not None:
            if node.is_terminating or node.invalid_termination:
                cmd_type = "submit"
            elif "[EDIT]" in node.last_action["thought"]: #  or node.modifies_code --- testing may have side effects. Can ignore that.
                cmd_type = "edit"
            # elif self.is_test_command(node.last_action["command"]):
            elif "[TEST]" in node.last_action["thought"] or ("[READ]" not in node.last_action["thought"] and self.is_test_command(node.last_action["command"])): # TEMP: Since test command detection is not very robust, we can also use a heuristic based on the thought content to identify potential test commands for better reward adjustment.
                cmd_type = "test"
            elif node.modifies_code:
                cmd_type = "edit"
        else:
            if "[SUBMIT]" in node.last_action["thought"]:
                cmd_type = "submit"
            elif "[EDIT]" in node.last_action["thought"]:
                cmd_type = "edit"
            elif "[TEST]" in node.last_action["thought"]:
                cmd_type = "test"
        
        gen_type = 'edit' if '[EDIT]' in node.last_action['thought'] else 'test' if '[TEST]' in node.last_action['thought'] else 'submit' if '[SUBMIT]' in node.last_action['thought'] else 'read'    
        if gen_type != cmd_type:
            instance_logger.debug(f">> Warning: Command type mismatch. Thought indicates {gen_type} but detected as {cmd_type}.")
            
        node.last_action['type'] = cmd_type
        node.value = self.reward_model.compute_reward(node, self.task, cmd_type=cmd_type)
        if node.last_action["command"] is None:
            # Penalize invalid actions
            penalty = 1
            curr = node
            while curr is not None and curr.last_action is not None:
                if curr.last_action["command"] is None:
                    penalty *= 0.7
                else:
                    break
                curr = curr.parent
                
            new_value = penalty * node.value
            instance_logger.debug(f">> Invalid-action reward adjustment: {node.value:.4f} -> {new_value:.4f}")
            node.value = new_value
            
        elif node.raw_observation is not None and not self.is_test_failure(node, node.raw_observation.get("returncode", 0)) and node.raw_observation.get("returncode", 0) != 0:
            penalty = 1
            curr = node
            while curr is not None:
                if curr.raw_observation is not None and not self.is_test_failure(curr, curr.raw_observation.get("returncode", 0)) and curr.raw_observation.get("returncode", 0) != 0:
                    penalty *= 0.8
                else:
                    break
                curr = curr.parent
            # Penalize actions with non-zero return code
            new_value = penalty * node.value
            instance_logger.debug(f">> Non-zero return-code reward adjustment: {node.value:.4f} -> {new_value:.4f}")
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
            
            # Option 2: Hierarchical relevance
            # max_relevance = self._calculate_hierarchical_score(node.changes) 
            # new_value = (0.9 * new_value + 0.1 * max_relevance)
            
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
                
            status_delta = self._compare_test_statuses(self.tree_node.test_status, node.test_status)
            if status_delta < 0:
                penalty = max(0.7, 1.0 + 0.3 * status_delta)
                new_value = node.value * penalty
                instance_logger.debug(
                    f">> Test-status regression adjustment (delta={status_delta}): {node.value:.4f} -> {new_value:.4f}"
                )
                node.value = new_value
            elif status_delta > 0:
                boost = min(1.4, 1.0 + 0.3 * status_delta)
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
        
        # OLD:
        elif (node.last_action["command"] is None or not self.is_test_command(node)) and node.raw_observation is not None and len(node.raw_observation.get("output").strip()) > 5000:
        # NEW:
        # elif (node.last_action["command"] is None or node.last_action["type"] != "test") and node.raw_observation is not None and len(node.raw_observation.get("output").strip()) > 5000:
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
        
    def _evaluate_nodes(self, node_list):
        for new_node in tqdm(node_list, desc="Evaluating nodes"):
            new_node.value = self._evaluate_node(new_node)
                            
    def _process_nodes(self, tree_nodes: List[str]) -> List[TreeSearchNode]:
        self.n_actions += len(self.tree_node.children)
        instance_logger.debug(f"# {len(tree_nodes)} new nodes generated at level {self.tree_node.level}:")
        for node in tree_nodes:
            instance_logger.debug(f"- {node.last_action['command']}")
            
        # self._evaluate_nodes(tree_nodes)
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
            instance_logger.debug(
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