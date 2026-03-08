from minisweagent.agents.repo_tree import result_to_structure, RepoNode, collect_rankable_nodes, dict_to_tree, collect_all_nodes, propagate_scores, remove_redundancy
import json
from datasets import load_dataset
from pathlib import Path
from rank_bm25 import BM25Okapi
import pickle
import numpy as np
import os 

ds = load_dataset("SWE-bench/SWE-bench_Verified", split="test")

def patch_to_files(patch):
    files = set()
    for line in patch.splitlines():
        line = line.strip()
        if line.startswith("diff --git"):
            parts = line.split()
            if len(parts) >= 4:
                file_path = parts[2][2:]  # Remove the "a/" prefix
                files.add(file_path)
    return files

gold_files = {}

def get_swebench_docker_image_name(instance: dict) -> str:
    """Get the image name for a SWEBench instance."""
    image_name = instance.get("image_name", None)
    if image_name is None:
        # Docker doesn't allow double underscore, so we replace them with a magic token
        iid = instance["instance_id"]
        id_docker_compatible = iid.replace("__", "_1776_")
        image_name = f"sweb.eval.x86_64.{id_docker_compatible}".lower()
    return image_name

metrics = {}

from tqdm import tqdm

for item in tqdm(ds):
    instance_id = item["instance_id"]
    gold_patch = item["patch"]
    gold_files[instance_id] = patch_to_files(gold_patch)
    
    image_name = get_swebench_docker_image_name(item)
    repo_files = []
    
    if not Path(f"retrieval/{image_name}/documents.jsonl").exists():
        # print(f"Missing retrieval results for {image_name}")
        continue
    with open(f"retrieval/{image_name}/documents.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            repo_files.append(obj)
    
    if not Path(f"retrieval/{image_name}/structure.json").exists():
        structure = result_to_structure(repo_files)
        with open(f"retrieval/{image_name}/structure.json", "w") as f:
            json.dump(structure, f, indent=2)
    else:
        with open(f"retrieval/{image_name}/structure.json") as f:
            structure = json.load(f)
    
    if not Path(f"retrieval/{image_name}/structure_opt.json").exists():
        structure_opt = structure
        remove_redundancy(structure_opt)
        with open(f"retrieval/{image_name}/structure_opt.json", "w") as f:
            json.dump(structure_opt, f, indent=2)
    else:
        with open(f"retrieval/{image_name}/structure_opt.json") as f:
            structure_opt = json.load(f)

    # Plain BM25
    documents = []
    file_ids = []

    for obj in repo_files:
        documents.append(obj["content"].split())  # tokenize by whitespace
        file_ids.append(obj["id"])
    
    index_path = f"retrieval/{image_name}/bm25_index.pkl"
    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            bm25 = pickle.load(f)
    else:
        bm25 = BM25Okapi(documents)
        with open(index_path, "wb") as f:
            pickle.dump(bm25, f)
            
            
    # Hierarchical BM25    
    repo_root = dict_to_tree(None, structure_opt)
    rank_nodes = collect_rankable_nodes(repo_root)
    index_path = f"retrieval/{image_name}/bm25_hindex.pkl"
    documents = ["\n".join([node.qualified_name()] + node.text).split() for node in rank_nodes]
    if os.path.exists(index_path):
        with open(index_path, "rb") as f:
            bm25_h = pickle.load(f)
    else:
        bm25_h = BM25Okapi(documents)
        with open(index_path, "wb") as f:
            pickle.dump(bm25_h, f)
        
    issue_tokens = item["problem_statement"].split()
    scores = bm25_h.get_scores(issue_tokens)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    
    for node, score in zip(rank_nodes, scores):
        node.self_score = score
        
    propagate_scores(repo_root)
    repo_nodes = collect_all_nodes(repo_root)
    file_scores = [node.score for node in repo_nodes if node.type == "file"]
    qualified_names = [node.qualified_name() for node in repo_nodes if node.type == "file"]
    top_indices = np.argsort(file_scores)[-10:][::-1]
    pred_files = {qualified_names[idx] for idx in top_indices}
    
    # scores = bm25.get_scores(issue_tokens)
    # scores = (scores - scores.min()) / (scores.max() - scores.min())
    # relevance_dict = dict(zip(file_ids, scores))
    # top_indices = np.argsort(scores)[-10:][::-1]
    # pred_files = {file_ids[idx] for idx in top_indices}
    
    gold_files_set = gold_files.get(instance_id, set())
    
    # Recall
    recall = (
        len(pred_files & gold_files_set) / len(gold_files_set)
        if gold_files_set else None
    )

    # Precision
    precision = (
        len(pred_files & gold_files_set) / len(pred_files)
        if pred_files else None
    )

    # F1
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None

    # Jaccard
    union = pred_files | gold_files_set
    jaccard = len(pred_files & gold_files_set) / len(union) if union else None

    # Exact match
    exact_match = pred_files == gold_files_set

    # Over / under editing
    over_edit = len(pred_files - gold_files_set)
    under_edit = len(gold_files_set - pred_files)

    metrics[instance_id] = {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "jaccard": jaccard,
        "exact_match": exact_match,
        "over_edit": over_edit,
        "under_edit": under_edit,
        "pred_file_count": len(pred_files),
        "gold_file_count": len(gold_files_set),
        "pred_empty": len(pred_files) == 0,
        "gold_empty": len(gold_files_set) == 0,
    }
    
def avg(key):
    vals = [v[key] for v in metrics.values() if v[key] is not None]
    return sum(vals) / len(vals) if vals else 0.0

print(f"Total instances: {len(metrics)}")
print(f"Recall:    {avg('recall'):.4f}")
print(f"Precision: {avg('precision'):.4f}")
print(f"F1:        {avg('f1'):.4f}")
print(f"Jaccard:   {avg('jaccard'):.4f}")

em_rate = sum(v["exact_match"] for v in metrics.values()) / len(metrics)
print(f"Exact Match: {em_rate:.4f}")

print(f"Avg over-edit:  {avg('over_edit'):.2f}")
print(f"Avg under-edit: {avg('under_edit'):.2f}")

empty_pred_rate = sum(v["pred_empty"] for v in metrics.values()) / len(metrics)
print(f"Empty prediction rate: {empty_pred_rate:.4f}")
