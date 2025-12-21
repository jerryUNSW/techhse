#!/usr/bin/env python3
"""
Delete incorrect SANTEXT+ results that used simplified implementations.
"""

import sqlite3
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpprivqa.utils.config import load_config

DATASETS = [
    'mmlu_professional_law',
    'mmlu_professional_medicine',
    'mmlu_clinical_knowledge',
    'mmlu_college_medicine'
]

def delete_santext_results():
    """Delete all SANTEXT+ results from database."""
    config = load_config()
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    total_deleted = 0
    
    for dataset_name in DATASETS:
        table_name = f"{dataset_name}_epsilon_dependent_results"
        
        # Check if table exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name = ?
        """, (table_name,))
        
        if not cursor.fetchone():
            print(f"  {dataset_name}: Table does not exist, skipping")
            continue
        
        # Count before deletion
        cursor.execute(f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE mechanism = 'santext' AND epsilon = 2.0
        """)
        count_before = cursor.fetchone()[0]
        
        if count_before == 0:
            print(f"  {dataset_name}: No SANTEXT+ results to delete")
            continue
        
        # Delete SANTEXT+ results
        cursor.execute(f"""
            DELETE FROM {table_name}
            WHERE mechanism = 'santext' AND epsilon = 2.0
        """)
        
        deleted = cursor.rowcount
        total_deleted += deleted
        
        print(f"  {dataset_name}: Deleted {deleted} SANTEXT+ results (ε=2.0)")
    
    # Also delete experiment records that only have SANTEXT+ results
    cursor.execute("""
        SELECT id, dataset_name FROM experiments
        WHERE experiment_type = 'epsilon_dependent'
        AND mechanisms LIKE '%santext%'
    """)
    
    exp_ids_to_check = cursor.fetchall()
    for exp_id, dataset_name in exp_ids_to_check:
        # Check if this experiment has any remaining results
        table_name = f"{dataset_name}_epsilon_dependent_results"
        cursor.execute(f"""
            SELECT COUNT(*) FROM {table_name}
            WHERE experiment_id = ?
        """, (exp_id,))
        remaining = cursor.fetchone()[0]
        
        if remaining == 0:
            cursor.execute("DELETE FROM experiments WHERE id = ?", (exp_id,))
            print(f"  Deleted experiment record {exp_id} for {dataset_name} (no remaining results)")
    
    conn.commit()
    conn.close()
    
    print(f"\n✅ Total deleted: {total_deleted} SANTEXT+ results")
    print("✅ Database cleanup completed!")


if __name__ == "__main__":
    print("=" * 80)
    print("Deleting Incorrect SANTEXT+ Results")
    print("=" * 80)
    print()
    delete_santext_results()

