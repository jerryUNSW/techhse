#!/usr/bin/env python3
"""
Create visualizations for NER-based PII privacy evaluation results
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_ner_results(file_path):
    """Load NER-based privacy evaluation results from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_privacy_comparison_plot(results, output_dir):
    """Create a bar chart comparing mean privacy levels."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Phrase DP', 'InferDPT']
    privacy_levels = [results['results']['phrase_dp']['mean_privacy_level'], 
                     results['results']['inferdpt']['mean_privacy_level']]
    colors = ['#2E8B57', '#DC143C']  # Sea Green and Crimson
    
    bars = ax.bar(methods, privacy_levels, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, level in zip(bars, privacy_levels):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{level:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Mean Privacy Protection Level', fontsize=14, fontweight='bold')
    ax.set_title('NER-Based PII Privacy Protection Comparison\nPhrase DP vs InferDPT (100 Questions)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.0)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Add privacy level annotations
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Excellent (≥0.8)')
    ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Good (≥0.6)')
    ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.7, label='Poor (<0.6)')
    
    # Add difference annotation
    diff = privacy_levels[1] - privacy_levels[0]
    improvement = results['comparison']['improvement_percentage']
    ax.annotate(f'InferDPT is {improvement:.1f}% better\nDifference: {diff:.3f}', 
                xy=(0.5, 0.5), xytext=(0.5, 0.3),
                ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'ner_privacy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_entity_protection_plot(results, output_dir):
    """Create a heatmap showing protection levels by entity type."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Extract entity data
    phrase_dp_entities = results['results']['phrase_dp']['entity_analysis']
    inferdpt_entities = results['results']['inferdpt']['entity_analysis']
    
    # Get all entity types
    entity_types = list(phrase_dp_entities.keys())
    
    # Create data for heatmap
    phrase_dp_levels = [phrase_dp_entities[entity]['protection_level'] for entity in entity_types]
    inferdpt_levels = [inferdpt_entities[entity]['protection_level'] for entity in entity_types]
    
    # Create heatmap data
    heatmap_data = np.array([phrase_dp_levels, inferdpt_levels])
    
    # Plot heatmap
    im = ax1.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax1.set_xticks(range(len(entity_types)))
    ax1.set_xticklabels(entity_types, rotation=45, ha='right')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Phrase DP', 'InferDPT'])
    ax1.set_title('PII Protection Levels by Entity Type', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('Protection Level', fontsize=12)
    
    # Add text annotations
    for i in range(2):
        for j in range(len(entity_types)):
            text = ax1.text(j, i, f'{heatmap_data[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=9)
    
    # Create bar chart comparison
    x = np.arange(len(entity_types))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, phrase_dp_levels, width, label='Phrase DP', color='#2E8B57', alpha=0.8)
    bars2 = ax2.bar(x + width/2, inferdpt_levels, width, label='InferDPT', color='#DC143C', alpha=0.8)
    
    ax2.set_xlabel('Entity Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Protection Level', fontsize=12, fontweight='bold')
    ax2.set_title('Entity Protection Level Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(entity_types, rotation=45, ha='right')
    ax2.legend()
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal lines for reference
    ax2.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Excellent (≥0.8)')
    ax2.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='Good (≥0.6)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ner_entity_protection.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_privacy_distribution_plot(results, output_dir):
    """Create histograms showing privacy level distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract privacy levels
    phrase_dp_levels = results['results']['phrase_dp']['privacy_levels']
    inferdpt_levels = results['results']['inferdpt']['privacy_levels']
    
    # Create histograms
    ax1.hist(phrase_dp_levels, bins=20, alpha=0.7, color='#2E8B57', edgecolor='black')
    ax1.axvline(np.mean(phrase_dp_levels), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(phrase_dp_levels):.3f}')
    ax1.set_title('Phrase DP\nPrivacy Level Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Privacy Protection Level', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    
    ax2.hist(inferdpt_levels, bins=20, alpha=0.7, color='#DC143C', edgecolor='black')
    ax2.axvline(np.mean(inferdpt_levels), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(inferdpt_levels):.3f}')
    ax2.set_title('InferDPT\nPrivacy Level Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Privacy Protection Level', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    
    plt.suptitle('Privacy Protection Level Distributions (100 Questions)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'ner_privacy_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_entity_count_analysis(results, output_dir):
    """Create plots showing entity counts and protection effectiveness."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract entity data
    phrase_dp_entities = results['results']['phrase_dp']['entity_analysis']
    inferdpt_entities = results['results']['inferdpt']['entity_analysis']
    
    entity_types = list(phrase_dp_entities.keys())
    
    # 1. Original entity counts
    original_counts = [phrase_dp_entities[entity]['original_count'] for entity in entity_types]
    
    bars1 = ax1.bar(entity_types, original_counts, color='lightblue', alpha=0.7, edgecolor='black')
    ax1.set_title('Original PII Entity Counts', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars1, original_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + max(original_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. Preserved entity counts comparison
    phrase_dp_preserved = [phrase_dp_entities[entity]['preserved_count'] for entity in entity_types]
    inferdpt_preserved = [inferdpt_entities[entity]['preserved_count'] for entity in entity_types]
    
    x = np.arange(len(entity_types))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, phrase_dp_preserved, width, label='Phrase DP', color='#2E8B57', alpha=0.8)
    bars3 = ax2.bar(x + width/2, inferdpt_preserved, width, label='InferDPT', color='#DC143C', alpha=0.8)
    
    ax2.set_title('Preserved PII Entity Counts', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(entity_types, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Protection level comparison
    phrase_dp_protection = [phrase_dp_entities[entity]['protection_level'] for entity in entity_types]
    inferdpt_protection = [inferdpt_entities[entity]['protection_level'] for entity in entity_types]
    
    bars4 = ax3.bar(x - width/2, phrase_dp_protection, width, label='Phrase DP', color='#2E8B57', alpha=0.8)
    bars5 = ax3.bar(x + width/2, inferdpt_protection, width, label='InferDPT', color='#DC143C', alpha=0.8)
    
    ax3.set_title('Protection Level by Entity Type', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Protection Level', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(entity_types, rotation=45, ha='right')
    ax3.legend()
    ax3.set_ylim(0, 1.0)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add reference lines
    ax3.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
    
    # 4. Protection effectiveness (1 - preserved/original)
    phrase_dp_effectiveness = [1 - (phrase_dp_entities[entity]['preserved_count'] / phrase_dp_entities[entity]['original_count']) 
                              for entity in entity_types]
    inferdpt_effectiveness = [1 - (inferdpt_entities[entity]['preserved_count'] / inferdpt_entities[entity]['original_count']) 
                             for entity in entity_types]
    
    bars6 = ax4.bar(x - width/2, phrase_dp_effectiveness, width, label='Phrase DP', color='#2E8B57', alpha=0.8)
    bars7 = ax4.bar(x + width/2, inferdpt_effectiveness, width, label='InferDPT', color='#DC143C', alpha=0.8)
    
    ax4.set_title('Protection Effectiveness (1 - Preserved/Original)', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Effectiveness', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(entity_types, rotation=45, ha='right')
    ax4.legend()
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('NER-Based PII Privacy Analysis (100 Questions)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'ner_entity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_comprehensive_summary_plot(results, output_dir):
    """Create a comprehensive summary plot with multiple metrics."""
    fig = plt.figure(figsize=(16, 10))
    
    # Create a 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Mean Privacy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    methods = ['Phrase DP', 'InferDPT']
    privacy_levels = [results['results']['phrase_dp']['mean_privacy_level'], 
                     results['results']['inferdpt']['mean_privacy_level']]
    colors = ['#2E8B57', '#DC143C']
    
    bars = ax1.bar(methods, privacy_levels, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, level in zip(bars, privacy_levels):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{level:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Mean Privacy Level', fontsize=12, fontweight='bold')
    ax1.set_title('Mean Privacy Protection', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 2. Privacy Level Statistics
    ax2 = fig.add_subplot(gs[0, 1])
    phrase_dp_levels = results['results']['phrase_dp']['privacy_levels']
    inferdpt_levels = results['results']['inferdpt']['privacy_levels']
    
    stats_data = [
        [np.mean(phrase_dp_levels), np.std(phrase_dp_levels), np.min(phrase_dp_levels), np.max(phrase_dp_levels)],
        [np.mean(inferdpt_levels), np.std(inferdpt_levels), np.min(inferdpt_levels), np.max(inferdpt_levels)]
    ]
    
    stats_labels = ['Mean', 'Std Dev', 'Min', 'Max']
    x = np.arange(len(stats_labels))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, stats_data[0], width, label='Phrase DP', color='#2E8B57', alpha=0.8)
    bars2 = ax2.bar(x + width/2, stats_data[1], width, label='InferDPT', color='#DC143C', alpha=0.8)
    
    ax2.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax2.set_title('Privacy Level Statistics', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(stats_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. High Protection Questions
    ax3 = fig.add_subplot(gs[1, 0])
    phrase_dp_high = sum(1 for level in phrase_dp_levels if level >= 0.8)
    inferdpt_high = sum(1 for level in inferdpt_levels if level >= 0.8)
    
    high_protection = [phrase_dp_high, inferdpt_high]
    bars = ax3.bar(methods, high_protection, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, count in zip(bars, high_protection):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax3.set_ylabel('Number of Questions', fontsize=12, fontweight='bold')
    ax3.set_title('High Protection Questions (≥0.8)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 4. Entity Protection Summary
    ax4 = fig.add_subplot(gs[1, 1])
    phrase_dp_entities = results['results']['phrase_dp']['entity_analysis']
    inferdpt_entities = results['results']['inferdpt']['entity_analysis']
    
    # Calculate average protection by entity type
    entity_types = list(phrase_dp_entities.keys())
    phrase_dp_avg = np.mean([phrase_dp_entities[entity]['protection_level'] for entity in entity_types])
    inferdpt_avg = np.mean([inferdpt_entities[entity]['protection_level'] for entity in entity_types])
    
    avg_protection = [phrase_dp_avg, inferdpt_avg]
    bars = ax4.bar(methods, avg_protection, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, avg in zip(bars, avg_protection):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{avg:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax4.set_ylabel('Average Protection Level', fontsize=12, fontweight='bold')
    ax4.set_title('Average Entity Protection', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add main title
    fig.suptitle('NER-Based PII Privacy Evaluation Summary\nPhrase DP vs InferDPT (100 Questions)', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    plt.savefig(output_dir / 'ner_comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Load results
    results_file = 'ner_pii_privacy_results.json'
    results = load_ner_results(results_file)
    
    # Create output directory
    output_dir = Path('plots/privacy-evaluations')
    output_dir.mkdir(exist_ok=True)
    
    print("Creating NER-based privacy evaluation visualization plots...")
    
    # Create all plots
    create_privacy_comparison_plot(results, output_dir)
    print("✓ Created privacy comparison plot")
    
    create_entity_protection_plot(results, output_dir)
    print("✓ Created entity protection plot")
    
    create_privacy_distribution_plot(results, output_dir)
    print("✓ Created privacy distribution plot")
    
    create_entity_count_analysis(results, output_dir)
    print("✓ Created entity count analysis plot")
    
    create_comprehensive_summary_plot(results, output_dir)
    print("✓ Created comprehensive summary plot")
    
    print(f"\nAll plots saved to: {output_dir.absolute()}")
    print("\nGenerated plots:")
    print("- ner_privacy_comparison.png: Bar chart comparing mean privacy levels")
    print("- ner_entity_protection.png: Heatmap and bar chart of entity protection levels")
    print("- ner_privacy_distributions.png: Histograms of privacy level distributions")
    print("- ner_entity_analysis.png: 2x2 grid with entity count and protection analysis")
    print("- ner_comprehensive_summary.png: 2x2 grid with multiple privacy metrics")

if __name__ == "__main__":
    main()
