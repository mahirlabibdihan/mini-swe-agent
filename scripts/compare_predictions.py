#!/usr/bin/env python3
"""
Compare two prediction directories and show pass/fail mismatches.

Usage:
    python compare_predictions.py <dir1> <dir2>

This script compares prediction results from two directories and identifies:
1. Instances that pass in dir1 but fail in dir2 (regressions)
2. Instances that fail in dir1 but pass in dir2 (improvements)
3. Instances with different status across different prediction files
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, Tuple


def load_predictions(directory: str) -> Dict[str, str]:
    """
    Load all predictions from a directory.
    
    Returns a dict mapping instance_id -> 'pass' or 'fail'
    """
    directory = Path(directory)
    if not directory.exists():
        raise ValueError(f"Directory not found: {directory}")
    
    predictions = {}
    json_files = list(directory.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {directory}")
    
    print(f"Loading predictions from {len(json_files)} files in {directory}")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get resolved (pass) and unresolved (fail) instances
            resolved = set(data.get("resolved_instances", []))
            unresolved = set(data.get("unresolved_instances", []))
            
            for instance_id in resolved:
                predictions[instance_id] = 'pass'
            
            for instance_id in unresolved:
                predictions[instance_id] = 'fail'
                
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}")
    
    print(f"Loaded {len(predictions)} total predictions from {directory}\n")
    return predictions


def compare_predictions(pred1: Dict[str, str], pred2: Dict[str, str]) -> Dict[str, list]:
    """
    Compare two prediction dictionaries and return mismatches.
    
    Returns:
        {
            'pass_to_fail': [(instance_id, pred1_status, pred2_status), ...],
            'fail_to_pass': [...],
            'only_in_dir1': [...],
            'only_in_dir2': [...]
        }
    """
    mismatches = {
        'pass_to_fail': [],  # pass in dir1, fail in dir2 (regression)
        'fail_to_pass': [],  # fail in dir1, pass in dir2 (improvement)
        'only_in_dir1': [],  # instance only in dir1
        'only_in_dir2': [],  # instance only in dir2
    }
    
    all_instances = set(pred1.keys()) | set(pred2.keys())
    
    for instance_id in sorted(all_instances):
        status1 = pred1.get(instance_id)
        status2 = pred2.get(instance_id)
        
        # Only in one directory
        if status1 is None:
            mismatches['only_in_dir2'].append(instance_id)
            continue
        if status2 is None:
            mismatches['only_in_dir1'].append(instance_id)
            continue
        
        # Different status between directories
        if status1 != status2:
            if status1 == 'pass' and status2 == 'fail':
                mismatches['pass_to_fail'].append((instance_id, status1, status2))
            elif status1 == 'fail' and status2 == 'pass':
                mismatches['fail_to_pass'].append((instance_id, status1, status2))
    
    return mismatches


def print_results(mismatches: Dict[str, list], dir1: str, dir2: str):
    """Print comparison results in a formatted way."""
    
    print("=" * 80)
    print(f"PREDICTION MISMATCH REPORT")
    print(f"Directory 1: {dir1}")
    print(f"Directory 2: {dir2}")
    print("=" * 80)
    print()
    
    # Regressions (pass -> fail)
    if mismatches['pass_to_fail']:
        print(f"❌ REGRESSIONS (Pass → Fail): {len(mismatches['pass_to_fail'])} instances")
        print("-" * 80)
        for instance_id, status1, status2 in mismatches['pass_to_fail']:
            print(f"  {instance_id}")
        print()
    
    # Improvements (fail -> pass)
    if mismatches['fail_to_pass']:
        print(f"✅ IMPROVEMENTS (Fail → Pass): {len(mismatches['fail_to_pass'])} instances")
        print("-" * 80)
        for instance_id, status1, status2 in mismatches['fail_to_pass']:
            print(f"  {instance_id}")
        print()
    
    # Only in dir1
    if mismatches['only_in_dir1']:
        print(f"📌 ONLY IN DIR1: {len(mismatches['only_in_dir1'])} instances")
        print("-" * 80)
        for instance_id in mismatches['only_in_dir1'][:20]:
            print(f"  {instance_id}")
        if len(mismatches['only_in_dir1']) > 20:
            print(f"  ... and {len(mismatches['only_in_dir1']) - 20} more")
        print()
    
    # Only in dir2
    if mismatches['only_in_dir2']:
        print(f"📌 ONLY IN DIR2: {len(mismatches['only_in_dir2'])} instances")
        print("-" * 80)
        for instance_id in mismatches['only_in_dir2'][:20]:
            print(f"  {instance_id}")
        if len(mismatches['only_in_dir2']) > 20:
            print(f"  ... and {len(mismatches['only_in_dir2']) - 20} more")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("-" * 80)
    total_mismatches = (len(mismatches['pass_to_fail']) + 
                       len(mismatches['fail_to_pass']) + 
                       len(mismatches['only_in_dir1']) + 
                       len(mismatches['only_in_dir2']))
    print(f"Total Regressions (Pass → Fail):  {len(mismatches['pass_to_fail'])}")
    print(f"Total Improvements (Fail → Pass): {len(mismatches['fail_to_pass'])}")
    print(f"Only in Dir1:                     {len(mismatches['only_in_dir1'])}")
    print(f"Only in Dir2:                     {len(mismatches['only_in_dir2'])}")
    print(f"Total Mismatches:                 {total_mismatches}")
    print("=" * 80)


def save_results_to_file(mismatches: Dict[str, list], output_file: str):
    """Save detailed results to a JSON file."""
    results = {
        'regressions': mismatches['pass_to_fail'],
        'improvements': mismatches['fail_to_pass'],
        'only_in_dir1': mismatches['only_in_dir1'],
        'only_in_dir2': mismatches['only_in_dir2'],
        'summary': {
            'regressions_count': len(mismatches['pass_to_fail']),
            'improvements_count': len(mismatches['fail_to_pass']),
            'only_in_dir1_count': len(mismatches['only_in_dir1']),
            'only_in_dir2_count': len(mismatches['only_in_dir2']),
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    
    try:
        # Load predictions from both directories
        print(f"Loading predictions from {dir1}...")
        pred1 = load_predictions(dir1)
        
        print(f"Loading predictions from {dir2}...")
        pred2 = load_predictions(dir2)
        
        # Compare predictions
        print("Comparing predictions...")
        mismatches = compare_predictions(pred1, pred2)
        
        # Print results
        print_results(mismatches, dir1, dir2)
        
        # Optionally save to file
        output_file = "prediction_mismatches.json"
        save_results_to_file(mismatches, output_file)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
