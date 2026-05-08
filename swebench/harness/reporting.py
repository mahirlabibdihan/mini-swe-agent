import docker
import json
from pathlib import Path
from typing import Optional

from swebench.harness.constants import (
    KEY_INSTANCE_ID,
    KEY_MODEL,
    KEY_PREDICTION,
    RUN_EVALUATION_LOG_DIR,
    RUN_EVALUATION_TS_LOG_DIR,
    LOG_REPORT,
)
from swebench.harness.docker_utils import list_images
from swebench.harness.test_spec.test_spec import make_test_spec


def _iter_tree_nodes(tree: object):
    """Yield all dict nodes from a nested tree structure."""
    stack = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, list):
            stack.extend(reversed(node))
            continue
        if not isinstance(node, dict):
            continue
        yield node
        children = node.get("children", []) or []
        if isinstance(children, list):
            stack.extend(reversed(children))


def _build_verbose_tree_rows(predictions_by_instance: dict, full_dataset: list, pass_at_1_ids: set) -> list[dict]:
    """Collect per-instance tree statistics for verbose reporting."""
    rows = []
    for instance in full_dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        instance_predictions = predictions_by_instance.get(instance_id, [])

        tree_file = None
        for prediction in instance_predictions:
            candidate_tree_file = prediction.get("tree_file")
            if candidate_tree_file:
                tree_file = candidate_tree_file
                break

        max_order = 0
        max_itr = 0
        total_pass_solutions = 0
        total_fail_solutions = 0

        if tree_file:
            tree_path = Path(tree_file)
            if tree_path.exists():
                try:
                    with tree_path.open("r", encoding="utf-8") as f:
                        tree = json.load(f)

                    for node in _iter_tree_nodes(tree):
                        order = node.get("order")
                        itr = node.get("itr")
                        if isinstance(order, int):
                            max_order = max(max_order, order)
                        if isinstance(itr, int):
                            max_itr = max(max_itr, itr)

                        if node.get("is_terminating", False):
                            if node.get("pass") is True:
                                total_pass_solutions += 1
                            elif node.get("pass") is False:
                                total_fail_solutions += 1
                except (OSError, json.JSONDecodeError, TypeError):
                    pass

        rows.append(
            {
                "instance_id": instance_id,
                "total_steps": max_order,
                "total_iterations": max_itr,
                "total_pass_solutions": total_pass_solutions,
                "total_fail_solutions": total_fail_solutions,
                "pass_at_1": 1 if instance_id in pass_at_1_ids else 0,
            }
        )

    return rows


def _print_verbose_tree_table(rows: list[dict]) -> None:
    """Print a compact per-instance summary table."""
    headers = [
        "instance_id",
        "total_steps",
        "total_iterations",
        "total_pass_solutions",
        "total_fail_solutions",
        "pass@1",
    ]
    if not rows:
        print("No per-instance tree metrics available.")
        return

    display_rows = []
    for row in rows:
        display_rows.append(
            {
                "instance_id": str(row["instance_id"]),
                "total_steps": str(row["total_steps"]),
                "total_iterations": str(row["total_iterations"]),
                "total_pass_solutions": str(row["total_pass_solutions"]),
                "total_fail_solutions": str(row["total_fail_solutions"]),
                "pass@1": str(row["pass_at_1"]),
            }
        )

    widths = {header: len(header) for header in headers}
    for row in display_rows:
        for header in headers:
            widths[header] = max(widths[header], len(row[header]))

    def format_row(row: dict) -> str:
        return " | ".join(
            row[header].ljust(widths[header]) if header == "instance_id" else row[header].rjust(widths[header])
            for header in headers
        )

    print("Per-instance tree summary:")
    print(format_row({header: header for header in headers}))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in display_rows:
        print(format_row(row))


def _handle_regular_predictions(
    predictions: dict,
    full_dataset: list,
    run_id: str,
    completed_ids: set,
    resolved_ids: set,
    error_ids: set,
    unresolved_ids: set,
    incomplete_ids: set,
    empty_patch_ids: set,
):
    """
    Handle regular predictions (non-tree-based).
    Original logic for handling predictions.
    """
    # iterate through dataset and check if the instance has been run
    for instance in full_dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id not in predictions:
            # skip instances without predictions
            incomplete_ids.add(instance_id)
            continue
        prediction = predictions[instance_id]
        if prediction.get(KEY_PREDICTION, None) in ["", None]:
            empty_patch_ids.add(instance_id)
            continue
        report_file = (
            RUN_EVALUATION_LOG_DIR
            / run_id
            / prediction[KEY_MODEL].replace("/", "__")
            / prediction[KEY_INSTANCE_ID]
            / LOG_REPORT
        )
        if report_file.exists():
            completed_ids.add(instance_id)
            try:
                content = report_file.read_text().strip()
                if not content:  # Empty file
                    error_ids.add(instance_id)
                    continue

                report = json.loads(content)
                if report[instance_id]["resolved"]:
                    # Record if the instance was resolved
                    resolved_ids.add(instance_id)
                else:
                    unresolved_ids.add(instance_id)
            except (json.JSONDecodeError, KeyError):
                # If the report file is not valid JSON or missing keys, treat as error
                error_ids.add(instance_id)
        else:
            # Otherwise, the instance was not run successfully
            error_ids.add(instance_id)


def _handle_tree_based_predictions(
    predictions: dict,
    full_dataset: list,
    run_id: str,
    completed_ids: set,
    resolved_ids: set,
    pass_at_1_ids: set,
    pass_at_k_ids: set,
    error_ids: set,
    unresolved_ids: set,
    incomplete_ids: set,
    empty_patch_ids: set,
):
    """
    Handle tree-based predictions with aggregation.
    
    For tree-based predictions:
    - If ANY terminating node for an instance passes, the instance is marked as resolved
    - If ALL terminating nodes fail, the instance is marked as unresolved
    """
    # Group predictions by instance_id
    predictions_by_instance = {}
    for pred_key, prediction in predictions.items():
        if not isinstance(prediction, dict):
            continue
        instance_id = prediction.get(KEY_INSTANCE_ID)
        if instance_id:
            if instance_id not in predictions_by_instance:
                predictions_by_instance[instance_id] = []
            predictions_by_instance[instance_id].append(prediction)
    
    # iterate through dataset and check if the instance has been run
    for instance in full_dataset:
        instance_id = instance[KEY_INSTANCE_ID]
        if instance_id not in predictions_by_instance:
            # skip instances without predictions
            incomplete_ids.add(instance_id)
            continue
        
        instance_predictions = predictions_by_instance[instance_id]
        
        # Check if any prediction is empty
        if all(pred.get(KEY_PREDICTION, None) in ["", None] for pred in instance_predictions):
            empty_patch_ids.add(instance_id)
            continue
        
        # Check all report files for this instance across all node predictions
        any_resolved = False
        submission_resolved = False
        all_completed = True
        any_error = False
        
        for prediction in instance_predictions:
            report_file = (
                RUN_EVALUATION_TS_LOG_DIR
                / run_id
                / prediction[KEY_MODEL].replace("/", "__")
                / prediction[KEY_INSTANCE_ID]
                / LOG_REPORT
            )
            
            if report_file.exists():
                try:
                    content = report_file.read_text().strip()
                    if not content:  # Empty file
                        any_error = True
                        continue

                    report = json.loads(content)
                    if report[instance_id]["resolved"]:
                        # Any terminating node resolved the instance
                        any_resolved = True
                        if prediction.get("is_submission", False):
                            submission_resolved = True
                    # Don't break here, continue checking other nodes
                except (json.JSONDecodeError, KeyError):
                    # If the report file is not valid JSON or missing keys, treat as error
                    any_error = True
            else:
                # If the tree node already has a pass flag, treat it as previously evaluated.
                if prediction.get("pass") is not None:
                    if bool(prediction.get("pass")):
                        any_resolved = True
                        if prediction.get("is_submission", False):
                            submission_resolved = True
                    continue

                # This node's prediction was not run successfully
                all_completed = False
        
        # Determine final status for this instance
        if any_resolved:
            resolved_ids.add(instance_id)
            pass_at_k_ids.add(instance_id)
            if submission_resolved:
                pass_at_1_ids.add(instance_id)
            completed_ids.add(instance_id)
        elif all_completed:
            # All nodes completed but none resolved
            unresolved_ids.add(instance_id)
            completed_ids.add(instance_id)
        else:
            # Match regular-report semantics:
            # submitted instances without complete reports are errors, not incomplete.
            error_ids.add(instance_id)


def make_run_report(
    predictions: dict,
    full_dataset: list,
    run_id: str,
    client: Optional[docker.DockerClient] = None,
    namespace: str = None,
    instance_image_tag: str = "latest",
    env_image_tag: str = "latest",
    report_dir: Path = Path("./evaluation"),
    verbose: bool = False,
) -> Path:
    """
    Make a final evaluation and run report of the instances that have been run.
    Also reports on images and containers that may still running if client is provided.

    Args:
        predictions (dict): Predictions dict generated by the model
        full_dataset (list): List of all instances
        run_id (str): Run ID
        client (docker.DockerClient): Docker client (optional)

    Returns:
        Path to report file
    """
    # instantiate sets to store IDs of different outcomes
    completed_ids = set()
    resolved_ids = set()
    pass_at_1_ids = set()
    pass_at_k_ids = set()
    error_ids = set()
    unstopped_containers = set()
    unremoved_images = set()
    unresolved_ids = set()
    incomplete_ids = set()
    # get instances with empty patches
    empty_patch_ids = set()
    
    # Check if this is tree-based predictions (contains original_model and node_index)
    is_tree_based = (
        any("original_model" in v for v in predictions.values() if isinstance(v, dict))
        or any("node_index" in v for v in predictions.values() if isinstance(v, dict))
    )

    predictions_by_instance = {}
    if is_tree_based:
        for prediction in predictions.values():
            if not isinstance(prediction, dict):
                continue
            instance_id = prediction.get(KEY_INSTANCE_ID)
            if not instance_id:
                continue
            predictions_by_instance.setdefault(instance_id, []).append(prediction)
    
    if is_tree_based:
        # Handle tree-based predictions with aggregation
        _handle_tree_based_predictions(
            predictions,
            full_dataset,
            run_id,
            completed_ids,
            resolved_ids,
            pass_at_1_ids,
            pass_at_k_ids,
            error_ids,
            unresolved_ids,
            incomplete_ids,
            empty_patch_ids,
        )
    else:
        # Handle regular predictions (original logic)
        _handle_regular_predictions(
            predictions,
            full_dataset,
            run_id,
            completed_ids,
            resolved_ids,
            error_ids,
            unresolved_ids,
            incomplete_ids,
            empty_patch_ids,
        )
        # For non-tree runs, pass@1 and pass@k are equivalent to resolved.
        pass_at_1_ids = set(resolved_ids)
        pass_at_k_ids = set(resolved_ids)

    if client:
        # get remaining images and containers
        images = list_images(client)
        test_specs = list(
            map(
                lambda x: make_test_spec(
                    x,
                    namespace=namespace,
                    instance_image_tag=instance_image_tag,
                    env_image_tag=env_image_tag,
                ),
                full_dataset,
            )
        )
        for spec in test_specs:
            image_name = spec.instance_image_key
            if image_name in images:
                unremoved_images.add(image_name)
        containers = client.containers.list(all=True)
        for container in containers:
            if run_id in container.name:
                unstopped_containers.add(container.name)

    # submitted IDs should be counted by instance_id in tree mode.
    if is_tree_based:
        submitted_ids = {
            pred.get(KEY_INSTANCE_ID)
            for pred in predictions.values()
            if isinstance(pred, dict) and pred.get(KEY_INSTANCE_ID)
        }
    else:
        submitted_ids = set(predictions.keys())

    # print final report
    dataset_ids = {i[KEY_INSTANCE_ID] for i in full_dataset}
    print(f"Total instances: {len(full_dataset)}")
    print(f"Instances submitted: {len(submitted_ids & dataset_ids)}")
    print(f"Instances completed: {len(completed_ids)}")
    print(f"Instances incomplete: {len(incomplete_ids)}")
    print(f"Pass@1: {len(pass_at_1_ids)}")
    print(f"Pass@k: {len(pass_at_k_ids)}")
    print(f"Instances resolved: {len(resolved_ids)}")
    print(f"Instances unresolved: {len(unresolved_ids)}")
    print(f"Instances with empty patches: {len(empty_patch_ids)}")
    print(f"Instances with errors: {len(error_ids)}")
    if client:
        print(f"Unstopped containers: {len(unstopped_containers)}")
        print(f"Unremoved images: {len(unremoved_images)}")

    if verbose and is_tree_based:
        print()
        verbose_rows = _build_verbose_tree_rows(predictions_by_instance, full_dataset, pass_at_1_ids)
        _print_verbose_tree_table(verbose_rows)

    # write report to file
    report = {
        "total_instances": len(full_dataset),
        "submitted_instances": len(submitted_ids),
        "completed_instances": len(completed_ids),
        "pass_at_1": len(pass_at_1_ids),
        "pass_at_k": len(pass_at_k_ids),
        "resolved_instances": len(resolved_ids),
        "unresolved_instances": len(unresolved_ids),
        "empty_patch_instances": len(empty_patch_ids),
        "error_instances": len(error_ids),
        "completed_ids": list(sorted(completed_ids)),
        "incomplete_ids": list(sorted(incomplete_ids)),
        "empty_patch_ids": list(sorted(empty_patch_ids)),
        "submitted_ids": list(sorted(submitted_ids)),
        "pass_at_1_ids": list(sorted(pass_at_1_ids)),
        "pass_at_k_ids": list(sorted(pass_at_k_ids)),
        "resolved_ids": list(sorted(resolved_ids)),
        "unresolved_ids": list(sorted(unresolved_ids)),
        "error_ids": list(sorted(error_ids)),
        "schema_version": 2,
    }
    if not client:
        report.update(
            {
                "unstopped_instances": len(unstopped_containers),
                "unstopped_containers": list(sorted(unstopped_containers)),
                "unremoved_images": list(sorted(unremoved_images)),
            }
        )

    # write report to json file
    if not report_dir.exists():
        report_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report file path
    if predictions:
        first_pred = next(iter(predictions.values()))
        if isinstance(first_pred, dict):
            model_name = first_pred.get(KEY_MODEL, "unknown").replace("/", "__")
        else:
            model_name = "unknown"
    else:
        model_name = "unknown"
    
    # Add .ts suffix for tree-based predictions
    suffix = ".ts" if is_tree_based else ""
    report_file = report_dir / f"{model_name}.{run_id}{suffix}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"Report written to {report_file}")
    return report_file
