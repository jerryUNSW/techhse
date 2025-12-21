#!/usr/bin/env python3
"""
Verify InferDPT and SANTEXT+ results for MMLU datasets.

Checks that all expected results are present in the database and calculates accuracy.
"""

import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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


def verify_dataset_results(conn: sqlite3.Connection, dataset_name: str, expected_questions: int, epsilon: float = 2.0) -> Dict[str, any]:
    """Verify results for a specific dataset."""
    table_name = f"{dataset_name}_epsilon_dependent_results"
    
    # Check if table exists
    cursor = conn.execute("""
        SELECT name FROM sqlite_master 
        WHERE type='table' AND name = ?
    """, (table_name,))
    
    if not cursor.fetchone():
        return {
            'inferdpt': {'count': 0, 'correct': 0, 'accuracy': 0.0, 'complete': False},
            'santext': {'count': 0, 'correct': 0, 'accuracy': 0.0, 'complete': False}
        }
    
    results = {}
    
    for mechanism in ['inferdpt', 'santext']:
        # Get count and accuracy
        cursor = conn.execute(f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
                COUNT(DISTINCT question_idx) as unique_questions
            FROM {table_name}
            WHERE mechanism = ? AND epsilon = ?
        """, (mechanism, epsilon))
        
        row = cursor.fetchone()
        if row:
            total, correct, unique_questions = row
            accuracy = (correct / total * 100) if total > 0 else 0.0
            complete = unique_questions == expected_questions
            
            results[mechanism] = {
                'count': unique_questions,
                'total_rows': total,
                'correct': correct,
                'accuracy': accuracy,
                'complete': complete
            }
        else:
            results[mechanism] = {
                'count': 0,
                'total_rows': 0,
                'correct': 0,
                'accuracy': 0.0,
                'complete': False
            }
        
        # Check for missing question indices
        cursor = conn.execute(f"""
            SELECT DISTINCT question_idx
            FROM {table_name}
            WHERE mechanism = ? AND epsilon = ?
            ORDER BY question_idx
        """, (mechanism, epsilon))
        
        existing_indices = set(row[0] for row in cursor.fetchall())
        all_indices = set(range(expected_questions))
        missing_indices = sorted(all_indices - existing_indices)
        
        results[mechanism]['missing_indices'] = missing_indices
        results[mechanism]['existing_indices'] = sorted(existing_indices)
    
    return results


def format_missing_indices(missing: List[int], max_display: int = 10) -> str:
    """Format missing indices."""
    if not missing:
        return "none"
    
    if len(missing) <= max_display:
        return ", ".join(str(i) for i in missing)
    else:
        return f"{', '.join(str(i) for i in missing[:max_display])}, ... ({len(missing)} total)"


def main():
    config = load_config()
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    
    if not Path(db_path).exists():
        print(f"❌ Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    epsilon = 2.0
    
    print("=" * 100)
    print("Verifying InferDPT and SANTEXT+ Results (ε=2.0)")
    print("=" * 100)
    print()
    
    # Print header
    print(f"{'Dataset':<25} | {'Mechanism':<12} | {'Questions':<12} | {'Correct':<10} | {'Accuracy':<10} | {'Status'}")
    print("-" * 100)
    
    all_verified = True
    summary_data = []
    
    for dataset_name, info in DATASET_CONFIG.items():
        display_name = info['display_name']
        expected = info['expected_questions']
        
        results = verify_dataset_results(conn, dataset_name, expected, epsilon)
        
        for mechanism in ['inferdpt', 'santext']:
            mech_display = mechanism.upper() if mechanism == 'inferdpt' else 'SANTEXT+'
            result = results[mechanism]
            
            count = result['count']
            correct = result['correct']
            accuracy = result['accuracy']
            complete = result['complete']
            
            status = "✅ COMPLETE" if complete else f"⚠️  INCOMPLETE ({count}/{expected})"
            
            if not complete:
                all_verified = False
            
            print(f"{display_name:<25} | {mech_display:<12} | {count}/{expected:<10} | {correct:<10} | {accuracy:>6.1f}%    | {status}")
            
            summary_data.append({
                'dataset': display_name,
                'mechanism': mech_display,
                'count': count,
                'expected': expected,
                'correct': correct,
                'accuracy': accuracy,
                'complete': complete
            })
    
    print("-" * 100)
    print()
    
    # Summary statistics
    print("=" * 100)
    print("Summary Statistics")
    print("=" * 100)
    
    total_expected = sum(info['expected_questions'] for info in DATASET_CONFIG.values())
    total_inferdpt = sum(d['count'] for d in summary_data if d['mechanism'] == 'INFERDPT')
    total_santext = sum(d['count'] for d in summary_data if d['mechanism'] == 'SANTEXT+')
    
    inferdpt_correct = sum(d['correct'] for d in summary_data if d['mechanism'] == 'INFERDPT')
    santext_correct = sum(d['correct'] for d in summary_data if d['mechanism'] == 'SANTEXT+')
    
    inferdpt_accuracy = (inferdpt_correct / total_inferdpt * 100) if total_inferdpt > 0 else 0.0
    santext_accuracy = (santext_correct / total_santext * 100) if total_santext > 0 else 0.0
    
    print(f"Total questions expected: {total_expected} × 2 mechanisms = {total_expected * 2} runs")
    print()
    print(f"InferDPT:")
    print(f"  Completed: {total_inferdpt}/{total_expected} ({100*total_inferdpt/total_expected:.1f}%)")
    print(f"  Correct: {inferdpt_correct}/{total_inferdpt}")
    print(f"  Accuracy: {inferdpt_accuracy:.1f}%")
    print()
    print(f"SANTEXT+:")
    print(f"  Completed: {total_santext}/{total_expected} ({100*total_santext/total_expected:.1f}%)")
    print(f"  Correct: {santext_correct}/{total_santext}")
    print(f"  Accuracy: {santext_accuracy:.1f}%")
    print()
    
    # Check for missing indices
    print("=" * 100)
    print("Missing Question Indices")
    print("=" * 100)
    
    has_missing = False
    for dataset_name, info in DATASET_CONFIG.items():
        display_name = info['display_name']
        expected = info['expected_questions']
        
        results = verify_dataset_results(conn, dataset_name, expected, epsilon)
        
        for mechanism in ['inferdpt', 'santext']:
            mech_display = mechanism.upper() if mechanism == 'inferdpt' else 'SANTEXT+'
            missing = results[mechanism]['missing_indices']
            
            if missing:
                has_missing = True
                missing_str = format_missing_indices(missing)
                print(f"{display_name} - {mech_display}: {missing_str}")
    
    if not has_missing:
        print("✅ No missing question indices found!")
    
    print()
    
    # Final status
    if all_verified:
        print("=" * 100)
        print("✅ ALL EXPERIMENTS COMPLETE AND VERIFIED!")
        print("=" * 100)
    else:
        print("=" * 100)
        print("⚠️  SOME EXPERIMENTS ARE INCOMPLETE")
        print("=" * 100)
        print("Check the missing indices above and re-run experiments if needed.")
    
    conn.close()


if __name__ == "__main__":
    main()

