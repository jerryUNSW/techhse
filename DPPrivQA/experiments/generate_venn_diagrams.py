#!/usr/bin/env python3
r"""
Generate Venn Diagrams for Local, Local+CoT, and Remote Results
================================================================

This script analyzes the overlap between three mechanisms:
- S1 = Local: Direct local model answers (no CoT)
- S2 = Local+CoT: Local model with Chain-of-Thought guidance
- S3 = Remote: Purely remote model answers

The Venn diagram helps identify:
1. Questions all three mechanisms answer correctly (core competency)
2. Questions only one mechanism answers correctly (unique strengths)
3. S1 \ S2: Local correct but Local+CoT incorrect (degradation cases)
4. S2 \ S3: Local+CoT correct but Remote incorrect (CoT advantage cases)
"""

import sqlite3
import json
import os
import sys
from typing import Set, Dict, List, Any
import matplotlib.pyplot as plt
try:
    from matplotlib_venn import venn3, venn3_circles
    HAS_VENN = True
except ImportError:
    HAS_VENN = False
    print("Warning: matplotlib-venn not available. Install with: pip install matplotlib-venn")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_CONFIGS = [
    {
        "name": "Professional Law",
        "table": "mmlu_professional_law_epsilon_independent_results",
        "local_exp_id": 7,
        "cot_exp_id": 21,
        "remote_exp_id": 17,
        "total_questions": 100
    },
    {
        "name": "Professional Medicine",
        "table": "mmlu_professional_medicine_epsilon_independent_results",
        "local_exp_id": 8,
        "cot_exp_id": 22,
        "remote_exp_id": 18,
        "total_questions": 272
    },
    {
        "name": "Clinical Knowledge",
        "table": "mmlu_clinical_knowledge_epsilon_independent_results",
        "local_exp_id": 9,
        "cot_exp_id": 23,
        "remote_exp_id": 19,
        "total_questions": 265
    },
    {
        "name": "College Medicine",
        "table": "mmlu_college_medicine_epsilon_independent_results",
        "local_exp_id": 10,
        "cot_exp_id": 24,
        "remote_exp_id": 20,
        "total_questions": 173
    }
]


class VennDiagramGenerator:
    """Generate Venn diagrams for Local, Local+CoT, and Remote comparison."""
    
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
    
    def analyze_overlap(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overlap between the three mechanisms."""
        table = config['table']
        
        # Get correct question sets
        s1_local = self.get_correct_questions(table, config['local_exp_id'], 'local')
        s2_local_cot = self.get_correct_questions(table, config['cot_exp_id'], 'local_cot')
        s3_remote = self.get_correct_questions(table, config['remote_exp_id'], 'remote')
        
        # Get all questions (use local experiment as reference)
        all_questions = self.get_all_question_indices(table, config['local_exp_id'])
        
        # Calculate overlaps
        all_three = s1_local & s2_local_cot & s3_remote
        
        # Individual only sets
        s1_only = s1_local - s2_local_cot - s3_remote
        s2_only = s2_local_cot - s1_local - s3_remote
        s3_only = s3_remote - s1_local - s2_local_cot
        
        # Pair overlaps (excluding third)
        s1_s2_only = (s1_local & s2_local_cot) - s3_remote
        s1_s3_only = (s1_local & s3_remote) - s2_local_cot
        s2_s3_only = (s2_local_cot & s3_remote) - s1_local
        
        # Key sets of interest
        s1_minus_s2 = s1_local - s2_local_cot  # Local correct but Local+CoT incorrect
        s2_minus_s3 = s2_local_cot - s3_remote  # Local+CoT correct but Remote incorrect
        
        # All fail
        all_fail = all_questions - s1_local - s2_local_cot - s3_remote
        
        return {
            's1_local': s1_local,
            's2_local_cot': s2_local_cot,
            's3_remote': s3_remote,
            'all_questions': all_questions,
            'overlaps': {
                'all_three': all_three,
                's1_only': s1_only,
                's2_only': s2_only,
                's3_only': s3_only,
                's1_s2_only': s1_s2_only,
                's1_s3_only': s1_s3_only,
                's2_s3_only': s2_s3_only,
                'all_fail': all_fail
            },
            'key_sets': {
                's1_minus_s2': s1_minus_s2,  # Degradation cases
                's2_minus_s3': s2_minus_s3   # CoT advantage cases
            },
            'statistics': {
                'total_questions': len(all_questions),
                's1_accuracy': len(s1_local) / len(all_questions) * 100 if all_questions else 0,
                's2_accuracy': len(s2_local_cot) / len(all_questions) * 100 if all_questions else 0,
                's3_accuracy': len(s3_remote) / len(all_questions) * 100 if all_questions else 0,
            }
        }
    
    def generate_venn_diagram(self, config: Dict[str, Any], output_dir: str = "exp-results/venn_diagrams"):
        """Generate Venn diagram visualization."""
        if not HAS_VENN:
            raise ImportError("matplotlib-venn is required. Install with: pip install matplotlib-venn")
        
        analysis = self.analyze_overlap(config)
        
        s1_local = analysis['s1_local']
        s2_local_cot = analysis['s2_local_cot']
        s3_remote = analysis['s3_remote']
        overlaps = analysis['overlaps']
        key_sets = analysis['key_sets']
        stats = analysis['statistics']
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figure with space for side text
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], hspace=0.3, wspace=0.3)
        ax_venn = fig.add_subplot(gs[0])
        ax_text = fig.add_subplot(gs[1])
        ax_text.axis('off')
        
        # Generate Venn diagram
        v = venn3(
            [s1_local, s2_local_cot, s3_remote],
            set_labels=('S1: Local', 'S2: Local+CoT', 'S3: Remote'),
            set_colors=('#FF6B6B', '#4ECDC4', '#45B7D1'),
            alpha=0.7,
            ax=ax_venn
        )
        
        # Customize labels with counts
        label_fontsize = 8
        if v.get_label_by_id('100'):
            v.get_label_by_id('100').set_text(f"S1 only\n{len(overlaps['s1_only'])}")
            v.get_label_by_id('100').set_fontsize(label_fontsize)
        if v.get_label_by_id('010'):
            v.get_label_by_id('010').set_text(f"S2 only\n{len(overlaps['s2_only'])}")
            v.get_label_by_id('010').set_fontsize(label_fontsize)
        if v.get_label_by_id('001'):
            v.get_label_by_id('001').set_text(f"S3 only\n{len(overlaps['s3_only'])}")
            v.get_label_by_id('001').set_fontsize(label_fontsize)
        if v.get_label_by_id('110'):
            v.get_label_by_id('110').set_text(f"S1 ∩ S2\n{len(overlaps['s1_s2_only'])}")
            v.get_label_by_id('110').set_fontsize(label_fontsize)
        if v.get_label_by_id('101'):
            v.get_label_by_id('101').set_text(f"S1 ∩ S3\n{len(overlaps['s1_s3_only'])}")
            v.get_label_by_id('101').set_fontsize(label_fontsize)
        if v.get_label_by_id('011'):
            v.get_label_by_id('011').set_text(f"S2 ∩ S3\n{len(overlaps['s2_s3_only'])}")
            v.get_label_by_id('011').set_fontsize(label_fontsize)
        if v.get_label_by_id('111'):
            v.get_label_by_id('111').set_text(f"All three\n{len(overlaps['all_three'])}")
            v.get_label_by_id('111').set_fontsize(label_fontsize)
        
        # Set label font sizes for set labels
        for label in v.set_labels:
            if label:
                label.set_fontsize(11)
                label.set_fontweight('bold')
        
        # Add title
        title = f"Venn Diagram: {config['name']}\nS1: Local, S2: Local+CoT, S3: Remote"
        ax_venn.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Add statistics text box
        stats_text = (
            f"Accuracy Statistics:\n"
            f"  S1 (Local): {stats['s1_accuracy']:.1f}%\n"
            f"  S2 (Local+CoT): {stats['s2_accuracy']:.1f}%\n"
            f"  S3 (Remote): {stats['s3_accuracy']:.1f}%\n"
            f"\nTotal Questions: {stats['total_questions']}\n"
            f"All Fail: {len(overlaps['all_fail'])} questions\n"
        )
        
        # Add key insights box with highlighted sets
        insights_text = (
            f"Key Sets:\n"
            f"  All three correct: {len(overlaps['all_three'])}\n"
            f"  S1 only: {len(overlaps['s1_only'])}\n"
            f"  S2 only: {len(overlaps['s2_only'])}\n"
            f"  S3 only: {len(overlaps['s3_only'])}\n"
            f"\n"
            f"  S1 ∩ S2 (not S3): {len(overlaps['s1_s2_only'])}\n"
            f"  S1 ∩ S3 (not S2): {len(overlaps['s1_s3_only'])}\n"
            f"  S2 ∩ S3 (not S1): {len(overlaps['s2_s3_only'])}\n"
            f"\n"
            f"[RED] S1 \\ S2 (Degradation):\n"
            f"   {len(key_sets['s1_minus_s2'])} questions\n"
            f"   Local correct but Local+CoT incorrect\n"
            f"\n"
            f"[GREEN] S2 \\ S3 (CoT Advantage):\n"
            f"   {len(key_sets['s2_minus_s3'])} questions\n"
            f"   Local+CoT correct but Remote incorrect\n"
        )
        
        # Combine text
        full_text = stats_text + "\n" + insights_text
        
        ax_text.text(0.05, 0.95, full_text, transform=ax_text.transAxes,
                    fontsize=10, verticalalignment='top', family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Save figure
        output_file = os.path.join(output_dir, f"venn_{config['name'].lower().replace(' ', '_')}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved Venn diagram to: {output_file}")
        
        # Also save detailed analysis to JSON
        json_output = os.path.join(output_dir, f"venn_analysis_{config['name'].lower().replace(' ', '_')}.json")
        json_data = {
            'dataset': config['name'],
            'statistics': stats,
            'set_sizes': {
                's1_local': len(s1_local),
                's2_local_cot': len(s2_local_cot),
                's3_remote': len(s3_remote)
            },
            'overlap_counts': {k: len(v) for k, v in overlaps.items()},
            'key_set_counts': {k: len(v) for k, v in key_sets.items()},
            'key_set_indices': {
                's1_minus_s2': sorted(list(key_sets['s1_minus_s2'])),
                's2_minus_s3': sorted(list(key_sets['s2_minus_s3']))
            }
        }
        with open(json_output, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved analysis to: {json_output}")
        
        plt.close()
        
        return analysis
    
    def print_summary(self, config: Dict[str, Any], analysis: Dict[str, Any]):
        """Print summary statistics."""
        print(f"\n{'='*80}")
        print(f"Dataset: {config['name']}")
        print(f"{'='*80}")
        
        stats = analysis['statistics']
        overlaps = analysis['overlaps']
        key_sets = analysis['key_sets']
        
        print(f"\nAccuracy:")
        print(f"  S1 (Local): {stats['s1_accuracy']:.1f}% ({len(analysis['s1_local'])}/{stats['total_questions']})")
        print(f"  S2 (Local+CoT): {stats['s2_accuracy']:.1f}% ({len(analysis['s2_local_cot'])}/{stats['total_questions']})")
        print(f"  S3 (Remote): {stats['s3_accuracy']:.1f}% ({len(analysis['s3_remote'])}/{stats['total_questions']})")
        
        print(f"\nOverlap Analysis:")
        print(f"  All three correct: {len(overlaps['all_three'])}")
        print(f"  S1 only: {len(overlaps['s1_only'])}")
        print(f"  S2 only: {len(overlaps['s2_only'])}")
        print(f"  S3 only: {len(overlaps['s3_only'])}")
        print(f"  S1 ∩ S2 (not S3): {len(overlaps['s1_s2_only'])}")
        print(f"  S1 ∩ S3 (not S2): {len(overlaps['s1_s3_only'])}")
        print(f"  S2 ∩ S3 (not S1): {len(overlaps['s2_s3_only'])}")
        print(f"  All fail: {len(overlaps['all_fail'])}")
        
        print(f"\n[RED] Key Set: S1 \\ S2 (Degradation Cases)")
        print(f"   Count: {len(key_sets['s1_minus_s2'])}")
        print(f"   Question indices: {sorted(list(key_sets['s1_minus_s2']))[:20]}{'...' if len(key_sets['s1_minus_s2']) > 20 else ''}")
        
        print(f"\n[GREEN] Key Set: S2 \\ S3 (CoT Advantage Cases)")
        print(f"   Count: {len(key_sets['s2_minus_s3'])}")
        print(f"   Question indices: {sorted(list(key_sets['s2_minus_s3']))[:20]}{'...' if len(key_sets['s2_minus_s3']) > 20 else ''}")
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Generate Venn diagrams for all datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Venn diagrams for Local, Local+CoT, and Remote")
    parser.add_argument("--db-path", type=str, default="exp-results/results.db",
                       help="Path to SQLite database")
    parser.add_argument("--output-dir", type=str, default="exp-results/venn_diagrams",
                       help="Output directory for diagrams")
    parser.add_argument("--dataset", type=str, choices=["all", "professional_law", "professional_medicine", 
                                                         "clinical_knowledge", "college_medicine"],
                       default="all", help="Which dataset to process")
    
    args = parser.parse_args()
    
    if not HAS_VENN:
        print("ERROR: matplotlib-venn is required. Install with: pip install matplotlib-venn")
        return
    
    generator = VennDiagramGenerator(args.db_path)
    
    datasets_to_process = DATASET_CONFIGS
    if args.dataset != "all":
        name_map = {
            "professional_law": "Professional Law",
            "professional_medicine": "Professional Medicine",
            "clinical_knowledge": "Clinical Knowledge",
            "college_medicine": "College Medicine"
        }
        datasets_to_process = [c for c in DATASET_CONFIGS if c['name'] == name_map[args.dataset]]
    
    print("="*80)
    print("Venn Diagram Generation: S1 (Local), S2 (Local+CoT), S3 (Remote)")
    print("="*80)
    
    all_analyses = {}
    
    for config in datasets_to_process:
        print(f"\nProcessing {config['name']}...")
        analysis = generator.generate_venn_diagram(config, args.output_dir)
        generator.print_summary(config, analysis)
        all_analyses[config['name']] = analysis
    
    generator.close()
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_s1_minus_s2 = sum(len(a['key_sets']['s1_minus_s2']) for a in all_analyses.values())
    total_s2_minus_s3 = sum(len(a['key_sets']['s2_minus_s3']) for a in all_analyses.values())
    
    print(f"\nTotal degradation cases (S1 \\ S2) across all datasets: {total_s1_minus_s2}")
    print(f"Total CoT advantage cases (S2 \\ S3) across all datasets: {total_s2_minus_s3}")
    
    print(f"\nPer-dataset breakdown:")
    for name, analysis in all_analyses.items():
        s1_minus_s2 = len(analysis['key_sets']['s1_minus_s2'])
        s2_minus_s3 = len(analysis['key_sets']['s2_minus_s3'])
        print(f"  {name}:")
        print(f"    S1 \\ S2: {s1_minus_s2}")
        print(f"    S2 \\ S3: {s2_minus_s3}")


if __name__ == "__main__":
    main()

