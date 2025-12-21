#!/usr/bin/env python3
"""
Generate UpSet Plots for Local, Local+CoT, and Remote Results
==============================================================

Creates UpSet plots showing all intersections between three scenarios:
- S1 = Local: Direct local model answers (no CoT)
- S2 = Local+CoT: Local model with Chain-of-Thought guidance
- S3 = Remote: Purely remote model answers

UpSet plots show:
- Horizontal bars: Size of each set (S1, S2, S3)
- Vertical bars: Size of each intersection
- Dot matrix: Which sets are in that intersection
"""

import sqlite3
import os
import sys
from typing import Set, Dict, List, Any
import matplotlib.pyplot as plt
try:
    from upsetplot import UpSet, from_contents
    HAS_UPSET = True
except ImportError:
    HAS_UPSET = False
    print("Warning: upsetplot not available. Install with: pip install upsetplot")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DB_PATH = os.path.join(PROJECT_ROOT, "exp-results", "results.db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "exploratory", "plots")

DATASET_CONFIGS = [
    {
        "name": "professional_law",
        "display_name": "Professional Law",
        "table": "mmlu_professional_law_epsilon_independent_results",
        "local_exp_id": 7,
        "cot_exp_id": 21,
        "remote_exp_id": 17,
        "total_questions": 100
    },
    {
        "name": "professional_medicine",
        "display_name": "Professional Medicine",
        "table": "mmlu_professional_medicine_epsilon_independent_results",
        "local_exp_id": 8,
        "cot_exp_id": 22,
        "remote_exp_id": 18,
        "total_questions": 272
    },
    {
        "name": "clinical_knowledge",
        "display_name": "Clinical Knowledge",
        "table": "mmlu_clinical_knowledge_epsilon_independent_results",
        "local_exp_id": 9,
        "cot_exp_id": 23,
        "remote_exp_id": 19,
        "total_questions": 265
    },
    {
        "name": "college_medicine",
        "display_name": "College Medicine",
        "table": "mmlu_college_medicine_epsilon_independent_results",
        "local_exp_id": 10,
        "cot_exp_id": 24,
        "remote_exp_id": 20,
        "total_questions": 173
    }
]


class UpSetPlotGenerator:
    """Generate UpSet plots for Local, Local+CoT, and Remote comparison."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at {db_path}")
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
    
    def get_correct_questions(self, table: str, experiment_id: int, scenario: str) -> Set[int]:
        """Get question indices where a scenario answered correctly."""
        query = f"""
            SELECT DISTINCT question_idx
            FROM {table}
            WHERE experiment_id = ?
                AND scenario = ?
                AND is_correct = 1
        """
        cursor = self.conn.execute(query, (experiment_id, scenario))
        return {row['question_idx'] for row in cursor.fetchall()}
    
    def get_all_question_indices(self, table: str, experiment_id: int) -> Set[int]:
        """Get all question indices for an experiment."""
        query = f"""
            SELECT DISTINCT question_idx
            FROM {table}
            WHERE experiment_id = ?
        """
        cursor = self.conn.execute(query, (experiment_id,))
        return {row['question_idx'] for row in cursor.fetchall()}
    
    def generate_upset_plot(self, config: Dict[str, Any]):
        """Generate UpSet plot visualization."""
        if not HAS_UPSET:
            raise ImportError("upsetplot is required. Install with: pip install upsetplot")
        
        table = config['table']
        
        # Get correct question sets
        s1_local = self.get_correct_questions(table, config['local_exp_id'], 'local')
        s2_local_cot = self.get_correct_questions(table, config['cot_exp_id'], 'local_cot')
        s3_remote = self.get_correct_questions(table, config['remote_exp_id'], 'remote')
        
        # Prepare data for UpSet plot
        # Format: {set_name: set_of_question_indices}
        contents = {
            'S1: Local': s1_local,
            'S2: Local+CoT': s2_local_cot,
            'S3: Remote': s3_remote
        }
        
        # Create UpSet data structure
        upset_data = from_contents(contents)
        
        # Create figure
        fig = plt.figure(figsize=(14, 8))
        
        # Create UpSet plot
        upset = UpSet(
            upset_data,
            subset_size='count',
            show_counts=True,
            sort_by='cardinality',
            sort_categories_by=None,
            facecolor='#8da0cb',  # Use blue color scheme
            orientation='horizontal'
        )
        
        # Plot
        upset.plot(fig=fig)
        
        # Add title
        fig.suptitle(
            f'UpSet Plot: {config["display_name"]}\nS1: Local, S2: Local+CoT, S3: Remote',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
        
        # Save figure with self-explanatory filename
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(OUTPUT_DIR, f"upset_plot_{config['name']}_local_local_cot_remote.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved UpSet plot to: {output_file}")
        
        plt.close()
        
        # Calculate statistics for summary
        all_questions = self.get_all_question_indices(table, config['local_exp_id'])
        
        # Calculate all intersections
        s1_only = s1_local - s2_local_cot - s3_remote
        s2_only = s2_local_cot - s1_local - s3_remote
        s3_only = s3_remote - s1_local - s2_local_cot
        s1_s2 = (s1_local & s2_local_cot) - s3_remote
        s1_s3 = (s1_local & s3_remote) - s2_local_cot
        s2_s3 = (s2_local_cot & s3_remote) - s1_local
        all_three = s1_local & s2_local_cot & s3_remote
        
        stats = {
            'total_questions': len(all_questions),
            's1_accuracy': len(s1_local) / len(all_questions) * 100 if all_questions else 0,
            's2_accuracy': len(s2_local_cot) / len(all_questions) * 100 if all_questions else 0,
            's3_accuracy': len(s3_remote) / len(all_questions) * 100 if all_questions else 0,
            'intersections': {
                's1_only': len(s1_only),
                's2_only': len(s2_only),
                's3_only': len(s3_only),
                's1_s2': len(s1_s2),
                's1_s3': len(s1_s3),
                's2_s3': len(s2_s3),
                'all_three': len(all_three)
            }
        }
        
        return stats
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Generate UpSet plots for all datasets."""
    print("="*80)
    print("Generating UpSet Plots: S1 (Local), S2 (Local+CoT), S3 (Remote)")
    print("="*80)
    
    if not HAS_UPSET:
        print("ERROR: upsetplot is required. Install with: pip install upsetplot")
        return
    
    generator = UpSetPlotGenerator(DB_PATH)
    
    all_stats = {}
    
    for config in DATASET_CONFIGS:
        print(f"\nProcessing {config['display_name']}...")
        stats = generator.generate_upset_plot(config)
        all_stats[config['name']] = stats
    
    generator.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    for config in DATASET_CONFIGS:
        name = config['name']
        if name in all_stats:
            stats = all_stats[name]
            print(f"\n{config['display_name']}:")
            print(f"  Total questions: {stats['total_questions']}")
            print(f"  S1 (Local) accuracy: {stats['s1_accuracy']:.1f}%")
            print(f"  S2 (Local+CoT) accuracy: {stats['s2_accuracy']:.1f}%")
            print(f"  S3 (Remote) accuracy: {stats['s3_accuracy']:.1f}%")
            print(f"  Intersections:")
            intersections = stats['intersections']
            print(f"    S1 only: {intersections['s1_only']}")
            print(f"    S2 only: {intersections['s2_only']}")
            print(f"    S3 only: {intersections['s3_only']}")
            print(f"    S1 ∩ S2: {intersections['s1_s2']}")
            print(f"    S1 ∩ S3: {intersections['s1_s3']}")
            print(f"    S2 ∩ S3: {intersections['s2_s3']}")
            print(f"    All three: {intersections['all_three']}")
    
    print(f"\nAll UpSet plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()


