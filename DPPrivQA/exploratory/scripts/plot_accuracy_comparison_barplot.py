#!/usr/bin/env python3
"""
Generate Accuracy Comparison Barplot
=====================================

Creates a grouped barplot comparing accuracy across three scenarios:
- Local (S1)
- Local+CoT (S2)
- Remote (S3)

For all 4 MMLU datasets.
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
VENN_ANALYSIS_DIR = os.path.join(PROJECT_ROOT, "exp-results", "venn_diagrams")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "exploratory", "plots")

# Dataset configuration
DATASETS = [
    "professional_law",
    "professional_medicine",
    "clinical_knowledge",
    "college_medicine"
]

DATASET_DISPLAY_NAMES = {
    "professional_law": "Professional Law",
    "professional_medicine": "Professional Medicine",
    "clinical_knowledge": "Clinical Knowledge",
    "college_medicine": "College Medicine"
}


def load_accuracy_data():
    """Load accuracy data from JSON files."""
    data = {}
    
    for dataset in DATASETS:
        json_file = os.path.join(VENN_ANALYSIS_DIR, f"venn_analysis_{dataset}.json")
        
        if not os.path.exists(json_file):
            print(f"Warning: {json_file} not found, skipping {dataset}")
            continue
        
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        stats = json_data.get('statistics', {})
        data[dataset] = {
            'local': stats.get('s1_accuracy', 0),
            'local_cot': stats.get('s2_accuracy', 0),
            'remote': stats.get('s3_accuracy', 0)
        }
    
    return data


def create_barplot(data):
    """Create grouped barplot."""
    # Prepare data
    datasets = list(data.keys())
    dataset_labels = [DATASET_DISPLAY_NAMES.get(d, d.replace('_', ' ').title()) for d in datasets]
    
    local_acc = [data[d]['local'] for d in datasets]
    local_cot_acc = [data[d]['local_cot'] for d in datasets]
    remote_acc = [data[d]['remote'] for d in datasets]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set bar width and positions
    x = np.arange(len(dataset_labels))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, local_acc, width, label='Local (S1)', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x, local_cot_acc, width, label='Local+CoT (S2)', color='#4ECDC4', alpha=0.8)
    bars3 = ax.bar(x + width, remote_acc, width, label='Remote (S3)', color='#45B7D1', alpha=0.8)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # Customize plot
    ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Comparison: Local vs Local+CoT vs Remote\nAcross MMLU Datasets', 
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, "accuracy_comparison_local_vs_local_cot_vs_remote_barplot.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved barplot to: {output_file}")
    
    plt.close()


def main():
    """Main function."""
    print("="*80)
    print("Generating Accuracy Comparison Barplot")
    print("="*80)
    
    # Load data
    print("\nLoading accuracy data...")
    data = load_accuracy_data()
    
    if not data:
        print("ERROR: No data loaded. Check if JSON files exist in exp-results/venn_diagrams/")
        return
    
    print(f"Loaded data for {len(data)} datasets")
    
    # Create barplot
    print("\nCreating barplot...")
    create_barplot(data)
    
    print("\nDone!")


if __name__ == "__main__":
    main()


