#!/usr/bin/env python3
"""
Check existing InferDPT and SANTEXT+ results for MMLU datasets.

Queries the database to see what InferDPT and SANTEXT+ results already exist
for each MMLU dataset with epsilon=2.0.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpprivqa.utils.config import load_config


DATASET_CONFIG = {
    'mmlu_professional_law': {
        'display_name': 'Professional Law',
        'expected_questions': 200
    },
    'mmlu_professional_medicine': {
        'display_name': 'Professional Medicine',
        'expected_questions': 272
    },
    'mmlu_clinical_knowledge': {
        'display_name': 'Clinical Knowledge',
        'expected_questions': 265
    },
    'mmlu_college_medicine': {
        'display_name': 'College Medicine',
        'expected_questions': 173
    }
}


def check_dataset_results(conn: sqlite3.Connection, dataset_name: str, expected_questions: int, epsilon: float = 2.0) -> Dict[str, any]:
    """Check results for a specific dataset."""
    table_name = f"{dataset_name}_epsilon_dependent_results"
    
    # Check if table exists
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name = ?
    """, (table_name,))
    
    if not cursor.fetchone():
        return {
            'inferdpt': {'count': 0, 'missing': set(range(expected_questions))},
            'santext': {'count': 0, 'missing': set(range(expected_questions))}
        }
    
    results = {}
    
    for mechanism in ['inferdpt', 'santext']:
        # Get all question indices for this mechanism and epsilon
        cursor = conn.execute(f"""
            SELECT DISTINCT question_idx
            FROM {table_name}
            WHERE mechanism = ? AND epsilon = ?
        """, (mechanism, epsilon))
        
        existing_indices = set(row[0] for row in cursor.fetchall())
        all_indices = set(range(expected_questions))
        missing_indices = all_indices - existing_indices
        
        results[mechanism] = {
            'count': len(existing_indices),
            'missing': missing_indices,
            'existing': existing_indices
        }
    
    return results


def format_missing_indices(missing: Set[int]) -> str:
    """Format missing indices as ranges."""
    if not missing:
        return "none"
    
    missing_list = sorted(missing)
    if len(missing_list) == len(missing):
        # Check if it's a continuous range
        if missing_list[-1] - missing_list[0] + 1 == len(missing_list):
            if len(missing_list) == 1:
                return str(missing_list[0])
            return f"{missing_list[0]}-{missing_list[-1]}"
    
    # Format as ranges
    ranges = []
    start = missing_list[0]
    end = missing_list[0]
    
    for i in range(1, len(missing_list)):
        if missing_list[i] == end + 1:
            end = missing_list[i]
        else:
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{end}")
            start = missing_list[i]
            end = missing_list[i]
    
    # Add last range
    if start == end:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{end}")
    
    if len(ranges) <= 3:
        return ", ".join(ranges)
    else:
        return f"{ranges[0]}, ..., {ranges[-1]} ({len(missing)} total)"


def main():
    config = load_config()
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    
    if not Path(db_path).exists():
        print(f"Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    epsilon = 2.0
    
    print("=" * 80)
    print("Checking InferDPT and SANTEXT+ Results (ε=2.0)")
    print("=" * 80)
    print()
    
    all_results = {}
    
    for dataset_name, info in DATASET_CONFIG.items():
        display_name = info['display_name']
        expected = info['expected_questions']
        
        print(f"Dataset: {display_name} ({dataset_name})")
        print(f"  Expected questions: {expected}")
        
        results = check_dataset_results(conn, dataset_name, expected, epsilon)
        all_results[dataset_name] = results
        
        for mechanism in ['inferdpt', 'santext']:
            mech_display = mechanism.upper() if mechanism == 'inferdpt' else 'SANTEXT+'
            count = results[mechanism]['count']
            missing = results[mechanism]['missing']
            missing_str = format_missing_indices(missing)
            
            status = "✅ COMPLETE" if count == expected else f"⚠️  INCOMPLETE ({count}/{expected})"
            print(f"  {mech_display} (ε={epsilon}): {status}")
            print(f"    Completed: {count}/{expected} questions")
            if missing:
                print(f"    Missing indices: {missing_str}")
        
        print()
    
    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    total_expected = sum(info['expected_questions'] for info in DATASET_CONFIG.values())
    total_inferdpt = sum(r['inferdpt']['count'] for r in all_results.values())
    total_santext = sum(r['santext']['count'] for r in all_results.values())
    
    print(f"Total questions expected: {total_expected} × 2 mechanisms = {total_expected * 2} runs")
    print(f"InferDPT completed: {total_inferdpt}/{total_expected} ({100*total_inferdpt/total_expected:.1f}%)")
    print(f"SANTEXT+ completed: {total_santext}/{total_expected} ({100*total_santext/total_expected:.1f}%)")
    print()
    
    # Check which datasets need to be run
    datasets_to_run = []
    for dataset_name, results in all_results.items():
        display_name = DATASET_CONFIG[dataset_name]['display_name']
        needs_inferdpt = len(results['inferdpt']['missing']) > 0
        needs_santext = len(results['santext']['missing']) > 0
        
        if needs_inferdpt or needs_santext:
            datasets_to_run.append({
                'name': dataset_name,
                'display': display_name,
                'needs_inferdpt': needs_inferdpt,
                'needs_santext': needs_santext
            })
    
    if datasets_to_run:
        print("Datasets that need experiments:")
        for ds in datasets_to_run:
            mechanisms = []
            if ds['needs_inferdpt']:
                mechanisms.append('InferDPT')
            if ds['needs_santext']:
                mechanisms.append('SANTEXT+')
            print(f"  - {ds['display']}: {', '.join(mechanisms)}")
    else:
        print("✅ All datasets are complete!")
    
    conn.close()


if __name__ == "__main__":
    main()

