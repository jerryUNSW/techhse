#!/usr/bin/env python3
r"""
Generate Venn Diagrams for Local, Local+CoT, and Remote Results
================================================================

Creates Venn diagrams showing overlap between three scenarios:
- S1 = Local: Direct local model answers (no CoT)
- S2 = Local+CoT: Local model with Chain-of-Thought guidance
- S3 = Remote: Purely remote model answers

For all 4 MMLU datasets with self-explanatory filenames.
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
    
    def generate_venn_diagram(self, config: Dict[str, Any]):
        """Generate Venn diagram visualization with self-explanatory filename."""
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
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Create figure with space for side text
        fig = plt.figure(figsize=(14, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8], hspace=0.3, wspace=0.3)
        ax_venn = fig.add_subplot(gs[0])
        ax_text = fig.add_subplot(gs[1])
        ax_text.axis('off')
        
        # Generate Venn diagram with improved colors and transparency
        # Using ColorBrewer perceptually uniform colors with low opacity
        # Standard venn3 layout: set 1 = top-left, set 2 = top-right, set 3 = bottom
        # Reorder sets to match reference: S2 (blue) top-left, S3 (teal) top-right, S1 (coral) bottom
        # S1: coral (#fc8d62), S2: blue (#8da0cb), S3: teal (#66c2a5)
        v = venn3(
            [s2_local_cot, s3_remote, s1_local],  # Reordered: S2, S3, S1 to match layout
            set_labels=('S2: Local+CoT', 'S3: Remote', 'S1: Local'),  # Labels match new order
            set_colors=('#8da0cb', '#66c2a5', '#fc8d62'),  # Blue, Teal, Coral
            alpha=0.25,  # 25% opacity for transparent circles
            ax=ax_venn
        )
        
        # Add border strokes to circles for better definition
        # Use full-saturation versions of colors for borders
        border_colors = ['#8da0cb', '#66c2a5', '#fc8d62']  # Blue, Teal, Coral
        circles = venn3_circles([s2_local_cot, s3_remote, s1_local], ax=ax_venn)  # Same reordering
        # circles is a tuple of Circle objects, one per set
        for i, circle in enumerate(circles):
            circle.set_edgecolor(border_colors[i])
            circle.set_linewidth(2.5)
            circle.set_alpha(1.0)  # Full opacity for borders
        
        # Make the circle patches more transparent
        for patch in v.patches:
            if patch:
                patch.set_alpha(0.25)
        
        # Customize labels with counts and add white background for readability
        label_fontsize = 8
        label_bg_alpha = 0.7  # White background for text readability
        
        def set_label_with_bg(label_id, text):
            label = v.get_label_by_id(label_id)
            if label:
                label.set_text(text)
                label.set_fontsize(label_fontsize)
                label.set_fontweight('bold')
                # Add white background for better readability
                label.set_bbox(dict(boxstyle='round,pad=0.3', facecolor='white', 
                                  edgecolor='none', alpha=label_bg_alpha))
        
        if v.get_label_by_id('100'):
            set_label_with_bg('100', f"S1 only\n{len(overlaps['s1_only'])}")
        if v.get_label_by_id('010'):
            set_label_with_bg('010', f"S2 only\n{len(overlaps['s2_only'])}")
        if v.get_label_by_id('001'):
            set_label_with_bg('001', f"S3 only\n{len(overlaps['s3_only'])}")
        if v.get_label_by_id('110'):
            set_label_with_bg('110', f"S1 ∩ S2\n{len(overlaps['s1_s2_only'])}")
        if v.get_label_by_id('101'):
            set_label_with_bg('101', f"S1 ∩ S3\n{len(overlaps['s1_s3_only'])}")
        if v.get_label_by_id('011'):
            set_label_with_bg('011', f"S2 ∩ S3\n{len(overlaps['s2_s3_only'])}")
        if v.get_label_by_id('111'):
            set_label_with_bg('111', f"All three\n{len(overlaps['all_three'])}")
        
        # Set label font sizes for set labels
        for label in v.set_labels:
            if label:
                label.set_fontsize(11)
                label.set_fontweight('bold')
        
        # Add title
        title = f"Venn Diagram: {config['display_name']}\nS1: Local, S2: Local+CoT, S3: Remote"
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
        
        # Add key insights box
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
        
        # Save figure with self-explanatory filename
        output_file = os.path.join(OUTPUT_DIR, f"venn_diagram_{config['name']}_local_local_cot_remote.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved Venn diagram to: {output_file}")
        
        # Also save detailed analysis to JSON with self-explanatory filename
        json_output = os.path.join(OUTPUT_DIR, f"venn_analysis_{config['name']}_local_local_cot_remote.json")
        json_data = {
            'dataset': config['display_name'],
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
    
    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Generate Venn diagrams for all datasets."""
    print("="*80)
    print("Generating Venn Diagrams: S1 (Local), S2 (Local+CoT), S3 (Remote)")
    print("="*80)
    
    if not HAS_VENN:
        print("ERROR: matplotlib-venn is required. Install with: pip install matplotlib-venn")
        return
    
    generator = VennDiagramGenerator(DB_PATH)
    
    all_analyses = {}
    
    for config in DATASET_CONFIGS:
        print(f"\nProcessing {config['display_name']}...")
        analysis = generator.generate_venn_diagram(config)
        all_analyses[config['name']] = analysis
    
    generator.close()
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    total_s1_minus_s2 = sum(len(a['key_sets']['s1_minus_s2']) for a in all_analyses.values())
    total_s2_minus_s3 = sum(len(a['key_sets']['s2_minus_s3']) for a in all_analyses.values())
    
    print(f"\nTotal degradation cases (S1 \\ S2) across all datasets: {total_s1_minus_s2}")
    print(f"Total CoT advantage cases (S2 \\ S3) across all datasets: {total_s2_minus_s3}")
    
    print(f"\nPer-dataset breakdown:")
    for config in DATASET_CONFIGS:
        name = config['name']
        if name in all_analyses:
            analysis = all_analyses[name]
            s1_minus_s2 = len(analysis['key_sets']['s1_minus_s2'])
            s2_minus_s3 = len(analysis['key_sets']['s2_minus_s3'])
            print(f"  {config['display_name']}:")
            print(f"    S1 \\ S2: {s1_minus_s2}")
            print(f"    S2 \\ S3: {s2_minus_s3}")
    
    print(f"\nAll diagrams saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

