import json
import re
import requests
import traceback
from importlib import resources
import swebench.resources
from minisweagent.agents.tree_search_node import TreeSearchNode
from argparse import ArgumentTypeError
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import Dataset, load_dataset, load_from_disk
from dotenv import load_dotenv
from pathlib import Path
from typing import cast
from swebench.harness.constants import (
    SWEbenchInstance,
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
)
from unidiff import PatchSet

load_dotenv()


class EvaluationError(Exception):
    def __init__(self, instance_id, message, logger):
        super().__init__(message)
        self.instance_id = instance_id
        self.log_file = logger.log_file
        self.logger = logger

    def __str__(self):
        log_msg = traceback.format_exc()
        self.logger.info(log_msg)
        return (
            f"{self.instance_id}: {super().__str__()}\n"
            f"Check ({self.log_file}) for more information."
        )


def get_predictions_from_file(predictions_path: str, dataset_name: str, split: str):
    if predictions_path == "gold":
        print("Using gold predictions")
        dataset = load_swebench_dataset(dataset_name, split)
        return [
            {
                KEY_INSTANCE_ID: datum[KEY_INSTANCE_ID],
                KEY_PREDICTION: datum["patch"],
                KEY_MODEL: "gold",
            }
            for datum in dataset
        ]
    if predictions_path.endswith(".json"):
        with open(predictions_path, "r") as f:
            predictions = json.load(f)
            if isinstance(predictions, dict):
                predictions = list(
                    predictions.values()
                )  # compatible with SWE-agent predictions
            if not isinstance(predictions, list):
                raise ValueError(
                    "Predictions must be a list[prediction] or a dictionary[instance_id: prediction]"
                )
    elif predictions_path.endswith(".jsonl"):
        with open(predictions_path, "r") as f:
            predictions = [json.loads(line) for line in f]
    else:
        raise ValueError("Predictions path must be .json or .jsonl")

    # Validate that each prediction has an instance_id
    for pred in predictions:
        if not isinstance(pred, dict):
            raise ValueError(f"Each prediction must be a dictionary, got {type(pred)}")
        if KEY_INSTANCE_ID not in pred:
            raise ValueError(f"Each prediction must contain '{KEY_INSTANCE_ID}'")

    return predictions


def get_predictions_from_tree_dir(predictions_dir: str, model_name: str = "tree_model"):
    """
    Load predictions from a directory containing instance_id subdirectories with *.tree.json files.
    
    Directory structure:
        predictions_dir/
            instance_id_1/
                *.tree.json (or similar tree file)
            instance_id_2/
                *.tree.json
            ...
    
    For each tree file, traverses the tree and extracts patches from nodes with 
    is_terminating=true. Each terminating node generates a prediction.
    
    Args:
        predictions_dir (str): Path to the directory containing instance_id subdirectories
        model_name (str): Name to assign to the model for all predictions
    
    Returns:
        list: List of prediction dictionaries with keys: instance_id, model_name_or_path, model_patch
        and additional metadata for aggregation
    """
    predictions = []
    predictions_path = Path(predictions_dir)
    
    if not predictions_path.is_dir():
        raise ValueError(f"Predictions directory not found: {predictions_dir}")
    
    # Iterate through each task subdirectory
    for task_dir in predictions_path.iterdir():
        if not task_dir.is_dir():
            continue
        
        instance_id = task_dir.name
        
        # Find *.tree.json files in the task directory
        tree_files = list(task_dir.glob("*.tree.json"))
        
        if not tree_files:
            print(f"Warning: No .tree.json files found in {task_dir}")
            continue
        
        # Use the first tree file found
        tree_file = tree_files[0]
        
        try:
            with open(tree_file, 'r') as f:
                tree = json.load(f)
            
            tree_root = TreeSearchNode(last_action=None)
            tree_root.from_tree(tree)
            
            # Traverse the tree and extract patches from terminating nodes
            terminating_nodes = []
            _collect_terminating_nodes(tree_root, terminating_nodes)
            
            if not terminating_nodes:
                print(f"Warning: No terminating nodes found in {tree_file}")
                continue
            
            # Take top 4
            # best_node = sorted(
            #     terminating_nodes,
            #     key=lambda x: (
            #         # x.merged_value, # OLD
            #         0.8 * x.merged_value + (1 - (x.parent.order / self.config.step_limit)) * 0.1 + 0.1 * (not x.system_generated), # NEW:  Should give priority to early discovered solutions
            #         # NEW: Give priority to AI generated nodes
            #         x.get_path_value(0.85) # NEW: In case of tie
            #     ),
            #     reverse=True
            # )[:4]
            terminating_nodes = sorted(
                terminating_nodes,
                key=lambda x: (
                    0.8 * x.merged_value + (1 - (x.parent.order / 50)) * 0.1 + 0.1 * (not x.system_generated), # NEW:  Should give priority to early discovered solutions
                    x.get_path_value(0.85)
                ),
                reverse=True,
            )
            
            # Create predictions for each terminating node
            for node_idx, node in enumerate(terminating_nodes):
                patch = node.observation
                
                # Create a unique model name that includes the node index for aggregation
                # Format: model_name_node_0, model_name_node_1, etc.
                node_model_name = f"{model_name}_node_{node_idx}"
                
                pred = {
                    KEY_INSTANCE_ID: instance_id,
                    KEY_MODEL: node_model_name,
                    KEY_PREDICTION: patch,
                    "node_id": node.id,
                    "node_index": node_idx,
                    "pass": node._pass,
                    "is_submission": node.is_submission,
                    "tree_file": str(tree_file),
                    "original_model": model_name,  # Keep original model name for aggregation
                }
                predictions.append(pred)
        
        except Exception as e:
            print(f"Error loading tree from {tree_file}: {e}")
            traceback.print_exc()
            continue
    
    if not predictions:
        raise ValueError(f"No predictions found in {predictions_dir}")
    
    print(f"Loaded {len(predictions)} predictions from tree files")
    return predictions



# def _collect_terminating_nodes(node: dict, terminating_nodes: list, seen: set | None = None):
#     """
#     Recursively traverse the tree and collect all nodes with is_terminating=true.
    
#     Args:
#         node (dict): Current node in the tree
#         terminating_nodes (list): List to accumulate terminating nodes
#     """
#     if seen is None:
#         seen = set()

#     if node is not None:
#         node_key = node.get("id") if isinstance(node, dict) else None
#         if node_key is None:
#             node_key = id(node)
#         if node_key in seen:
#             return
#         seen.add(node_key)

#     if node is not None and (node.get("is_terminating", False) or (node.get("observation", "") is not None and node.get("observation", "").startswith("diff --git"))) and node.get("merged_value") is not None:
#         terminating_nodes.append(node)
    
#     # Recursively traverse children
#     if node is not None and "children" in node and node["children"]:
#         for child in node["children"]:
#             if child.get("visible", False):  # Only traverse visible nodes to avoid duplicates
#                 _collect_terminating_nodes(child, terminating_nodes, seen)

def _collect_terminating_nodes(node: TreeSearchNode, terminating_nodes: list[TreeSearchNode]):
    """
    Recursively traverse the tree and collect all nodes with is_terminating=true.
    
    Args:
        node (TreeSearchNode): Current node in the tree
        terminating_nodes (list): List to accumulate terminating nodes
    """
    if node is None:
        return

    if (node.is_terminating or (node.observation and node.observation.startswith("diff --git"))) and node.merged_value is not None:
        terminating_nodes.append(node)
    
    # Recursively traverse children
    for child in node.children:
        if child.visible or child.merged_value is not None:  # Only traverse visible nodes to avoid duplicates
            _collect_terminating_nodes(child, terminating_nodes)

def _extract_patch_from_node(node: dict) -> str:
    """
    Extract patch content from a terminating node.
    
    The patch should be in the 'observation' field. This function extracts the 
    patch content, handling various possible formats.
    
    Args:
        node (dict): A tree node (typically with is_terminating=true)
    
    Returns:
        str: The patch content, or empty string if not found
    """
    observation = node.get("observation", "")
    
    if not observation:
        return ""
    
    if isinstance(observation, str):
        # If observation is a string, check if it contains patch-like content
        # The patch might be embedded in XML-like responses or could be direct diff content
        
        # If it starts with --- or +++, it's likely a raw diff
        if observation.startswith("---") or observation.startswith("+++"):
            return observation
        
        # If it contains <diff> tags, extract the content
        if "<diff>" in observation:
            start = observation.find("<diff>") + len("<diff>")
            end = observation.find("</diff>")
            if end > start:
                patch = observation[start:end].strip()
                return patch
        
        # If it contains diff content between other tags, try to extract it
        if "<returncode>0</returncode>" in observation:
            # This might be output with embedded diff
            lines = observation.split('\n')
            diff_lines = []
            in_diff = False
            for line in lines:
                if line.startswith("---") or line.startswith("+++"):
                    in_diff = True
                if in_diff:
                    diff_lines.append(line)
            if diff_lines:
                return '\n'.join(diff_lines)
    
    # Return the observation as-is (could be a dict or string)
    if isinstance(observation, dict):
        return json.dumps(observation)
    
    return observation


def run_threadpool(func, payloads, max_workers):
    """
    Run a function with a list of payloads using ThreadPoolExecutor.

    Args:
        func: Function to run for each payload
        payloads: List of payloads to process
        max_workers: Maximum number of worker threads

    Returns:
        tuple: (succeeded, failed) lists of payloads
    """
    if max_workers <= 0:
        return run_sequential(func, payloads)
    succeeded, failed = [], []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create a future for running each instance
        futures = {executor.submit(func, *payload): payload for payload in payloads}
        # Wait for each future to complete
        for future in as_completed(futures):
            try:
                # Check if instance ran successfully
                future.result()
                succeeded.append(futures[future])
            except Exception as e:
                print(f"{type(e)}: {e}")
                traceback.print_exc()
                failed.append(futures[future])
    return succeeded, failed


def run_sequential(func, payloads):
    """
    Run a function with a list of payloads sequentially.

    Args:
        func: Function to run for each payload
        payloads: List of payloads to process

    Returns:
        tuple: (succeeded, failed) lists of payloads
    """
    succeeded, failed = [], []
    for payload in payloads:
        try:
            func(*payload)
            succeeded.append(payload)
        except Exception:
            traceback.print_exc()
            failed.append(payload)
    return succeeded, failed


def load_swebench_dataset(
    name="SWE-bench/SWE-bench", split="test", instance_ids=None
) -> list[SWEbenchInstance]:
    """
    Load SWE-bench dataset from Hugging Face Datasets or local .json/.jsonl file
    """
    # check that all instance IDs are in the dataset
    if instance_ids:
        instance_ids = set(instance_ids)
    # Load from local .json/.jsonl file
    if name.endswith(".json"):
        dataset = json.loads(Path(name).read_text())
    elif name.endswith(".jsonl"):
        dataset = [json.loads(line) for line in Path(name).read_text().splitlines()]
    else:
        # Load from Hugging Face Datasets
        if name.lower() in {"swe-bench", "swebench", "swe_bench"}:
            name = "SWE-bench/SWE-bench"
        elif name.lower() in {
            "swe-bench-lite",
            "swebench-lite",
            "swe_bench_lite",
            "swe-bench_lite",
            "lite",
        }:
            name = "SWE-bench/SWE-bench_Lite"
        if (Path(name) / split / "dataset_info.json").exists():
            dataset = cast(Dataset, load_from_disk(Path(name) / split))
        else:
            dataset = cast(Dataset, load_dataset(name, split=split))
    dataset_ids = {instance[KEY_INSTANCE_ID] for instance in dataset}
    if instance_ids:
        if instance_ids - dataset_ids:
            raise ValueError(
                (
                    "Some instance IDs not found in dataset!"
                    f"\nMissing IDs:\n{' '.join(instance_ids - dataset_ids)}"
                )
            )
        dataset = [
            instance
            for instance in dataset
            if instance[KEY_INSTANCE_ID] in instance_ids
        ]
    return [cast(SWEbenchInstance, instance) for instance in dataset]


### MARK - Patch Correction
PATCH_PATTERN = re.compile(
    r"(?:diff[\w\_\.\ \/\-]+\n)?\-\-\-\s+a\/(?:.*?)\n\+\+\+\s+b\/(?:.*?)(?=diff\ |\-\-\-\ a\/|\Z)",
    re.DOTALL,
)
PATCH_FILE_PATTERN = re.compile(r"\-\-\-\s+a\/(?:.+)\n\+\+\+\s+b\/(?:.+)")
PATCH_HUNK_PATTERN = re.compile(
    r"\@\@\s+\-(\d+),(\d+)\s+\+(\d+),(\d+)\s+\@\@(.+?)(?=diff\ |\-\-\-\ a\/|\@\@\ \-|\Z)",
    re.DOTALL,
)


def get_first_idx(charlist):
    """Get index of first occurrence of "-" or "+" in charlist"""
    first_min = charlist.index("-") if "-" in charlist else len(charlist)
    first_plus = charlist.index("+") if "+" in charlist else len(charlist)
    return min(first_min, first_plus)


def get_last_idx(charlist):
    """Get index of last occurrence of "-" or "+" in charlist"""
    char_idx = get_first_idx(charlist[::-1])
    last_idx = len(charlist) - char_idx
    return last_idx + 1


def strip_content(hunk):
    """Remove trailing non +/- lines and trailing whitespace per line per hunk"""
    first_chars = list(map(lambda x: None if not len(x) else x[0], hunk.split("\n")))
    first_idx = get_first_idx(first_chars)
    last_idx = get_last_idx(first_chars)
    new_lines = list(map(lambda x: x.rstrip(), hunk.split("\n")[first_idx:last_idx]))
    # should leave one space for empty context lines
    new_lines = [line if line.strip() else " " for line in new_lines]
    new_hunk = "\n" + "\n".join(new_lines) + "\n"
    return new_hunk, first_idx - 1


def get_hunk_stats(pre_start, pre_len, post_start, post_len, hunk, total_delta):
    """Recalculate hunk start/end position and diff delta"""
    stats = {"context": 0, "added": 0, "subtracted": 0}
    hunk = hunk.split("\n", 1)[-1].strip("\n")
    for line in hunk.split("\n"):
        if line.startswith("-"):
            stats["subtracted"] += 1
        elif line.startswith("+"):
            stats["added"] += 1
        else:
            stats["context"] += 1
    context = stats["context"]
    added = stats["added"]
    subtracted = stats["subtracted"]
    pre_len = context + subtracted
    post_start = pre_start + total_delta
    post_len = context + added
    total_delta = total_delta + (post_len - pre_len)
    return pre_start, pre_len, post_start, post_len, total_delta


def extract_minimal_patch(model_patch):
    """
    Wrapper function that takes hunk and
    * Removes trailing non +/- lines and trailing whitespace per line per hunk
    * Recalculates hunk start/end position and diff delta
    * Returns new patch
    """
    model_patch = model_patch.lstrip("\n")
    new_patch = ""
    for patch in PATCH_PATTERN.findall(model_patch):
        total_delta = 0
        patch_header = PATCH_FILE_PATTERN.findall(patch)[0]
        if patch_header:
            new_patch += patch_header + "\n"
        for hunk in PATCH_HUNK_PATTERN.findall(patch):
            pre_start, pre_len, post_start, post_len, content = hunk
            pre_start, pre_len, post_start, post_len, content = list(
                map(lambda x: int(x) if x.isnumeric() else x, hunk)
            )
            content, adjust_pre_start = strip_content(content)
            pre_start += adjust_pre_start
            pre_start, pre_len, post_start, post_len, total_delta = get_hunk_stats(
                pre_start, pre_len, post_start, post_len, content, total_delta
            )
            new_patch += (
                f"@@ -{pre_start},{pre_len} +{post_start},{post_len} @@{content}"
            )
    return new_patch


def has_attribute_or_import_error(log_before):
    """
    Check to see if Attribute/Import-prefix is in log text

    Args:
        log_before (str): Validation log text before patch application
    """
    log_before = log_before.lower()

    if any([x in log_before for x in ["attribute", "import"]]):

        def get_lines_with_word(text, target_word):
            # Function to extract line(s) that contains target_word
            text, target_word = text.lower(), target_word.lower()
            lines, hits = text.split("\n")[::-1], []
            for line in lines:
                if target_word in line:
                    hits.append(line)
            return hits

        # Get line with Attribute/Import error
        lines_1 = get_lines_with_word(log_before, "attribute")
        lines_2 = get_lines_with_word(log_before, "import")
        lines_1 = " ".join(lines_1)
        lines_2 = " ".join(lines_2)

        if any([(x in lines_1 or x in lines_2) for x in ["error", "fail"]]):
            return True
    return False


def str2bool(v):
    """
    Minor helper function to convert string to boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def optional_str(value: str) -> str | None:
    """
    Convert special string values to None, otherwise return the string as-is.
    """
    if value.lower() in ("none", "null", ""):
        return None
    return value


def get_repo_file(repo, commit, filepath):
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{filepath}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        return None
    except:
        return None


def get_modified_files(patch: str) -> list[str]:
    """
    Get the list of modified files in a patch
    """
    source_files = []
    for file in PatchSet(patch):
        if file.source_file != "/dev/null":
            source_files.append(file.source_file)
    source_files = [x[2:] for x in source_files if x.startswith("a/")]
    return source_files


def ansi_escape(text: str) -> str:
    """
    Remove ANSI escape sequences from text
    """
    return re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])").sub("", text)


def load_cached_environment_yml(instance_id: str) -> str:
    """
    Load environment.yml from cache
    """
    try:
        repo, number = instance_id.rsplit("-", 1)
    except ValueError:
        return None
    try:
        return (
            resources.files(swebench.resources)
            .joinpath(f"swebench-og/{repo}/{number}/environment.yml")
            .read_text()
        )
    except FileNotFoundError:
        return None
