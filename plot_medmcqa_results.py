#!/usr/bin/env python3
"""
MedMCQA Results Visualization Script
====================================

Creates three plots similar to Figure 1 based on MedMCQA experiment results:
1. Accuracy vs Epsilon for Privacy Mechanisms
2. Privacy-Utility Trade-off Analysis
3. Mechanism Comparison Bar Chart

Author: Tech4HSE Team
Date: 2025-01-27
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import glob
import pandas as pd
from datetime import datetime

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_medmcqa_results():
    """Load results from all MedMCQA experiment files."""
    results_dir = Path("QA-results/medmcqa")
    result_files = list(results_dir.glob("medmcqa_results_*_100q_eps*.json"))
    
    if not result_files:
        raise FileNotFoundError("No MedMCQA result files found")
    
    data = {}
    for file_path in result_files:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            epsilon = file_data['experiment_info']['epsilon']
            data[epsilon] = file_data
    
    return data

def create_accuracy_vs_epsilon_plot(data):
    """Create Plot 1: Accuracy vs Epsilon for Privacy Mechanisms."""
    epsilons = sorted(data.keys())
    
    # Extract data
    mechanisms = {
        'Old PhraseDP': [],
        'InferDPT': [],
        'SANTEXT+': []
    }
    
    for epsilon in epsilons:
        results = data[epsilon]['results']
        total = results['total_questions']
        
        mechanisms['Old PhraseDP'].append(results['old_phrase_dp_local_cot_correct'] / total * 100)
        mechanisms['InferDPT'].append(results['inferdpt_local_cot_correct'] / total * 100)
        mechanisms['SANTEXT+'].append(results['santext_local_cot_correct'] / total * 100)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    for mechanism, accuracies in mechanisms.items():
        plt.plot(epsilons, accuracies, marker='o', linewidth=2, markersize=8, label=mechanism)
    
    # Add baseline and reference lines
    baseline_acc = data[epsilons[0]]['results']['local_alone_correct'] / data[epsilons[0]]['results']['total_questions'] * 100
    non_private_acc = data[epsilons[0]]['results']['non_private_cot_correct'] / data[epsilons[0]]['results']['total_questions'] * 100
    remote_acc = data[epsilons[0]]['results']['purely_remote_correct'] / data[epsilons[0]]['results']['total_questions'] * 100
    
    plt.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7, label='Local Baseline')
    plt.axhline(y=non_private_acc, color='green', linestyle='--', alpha=0.7, label='Non-Private CoT')
    plt.axhline(y=remote_acc, color='red', linestyle='--', alpha=0.7, label='Remote Model')
    
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('MedMCQA: Privacy Mechanism Performance vs Epsilon', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(epsilons)
    plt.ylim(0, 100)
    
    # Add annotations
    plt.text(0.02, 0.98, f'Dataset: MedMCQA (100 questions)\nModel: Llama 3.1 8B', 
             transform=plt.gca().transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('medmcqa_accuracy_vs_epsilon.png', dpi=300, bbox_inches='tight')
    plt.savefig('medmcqa_accuracy_vs_epsilon.pdf', bbox_inches='tight')
    plt.show()

def create_privacy_utility_tradeoff_plot(data):
    """Create Plot 2: Privacy-Utility Trade-off Analysis."""
    epsilons = sorted(data.keys())
    
    # Calculate privacy cost (difference from remote model)
    remote_acc = data[epsilons[0]]['results']['purely_remote_correct'] / data[epsilons[0]]['results']['total_questions'] * 100
    
    mechanisms = {
        'Old PhraseDP': [],
        'InferDPT': [],
        'SANTEXT+': []
    }
    
    privacy_costs = []
    
    for epsilon in epsilons:
        results = data[epsilon]['results']
        total = results['total_questions']
        
        # Calculate privacy cost for each mechanism
        phrasedp_cost = remote_acc - (results['old_phrase_dp_local_cot_correct'] / total * 100)
        inferdpt_cost = remote_acc - (results['inferdpt_local_cot_correct'] / total * 100)
        santext_cost = remote_acc - (results['santext_local_cot_correct'] / total * 100)
        
        mechanisms['Old PhraseDP'].append(phrasedp_cost)
        mechanisms['InferDPT'].append(inferdpt_cost)
        mechanisms['SANTEXT+'].append(santext_cost)
        
        privacy_costs.append([phrasedp_cost, inferdpt_cost, santext_cost])
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Privacy Cost vs Epsilon
    for i, (mechanism, costs) in enumerate(mechanisms.items()):
        ax1.plot(epsilons, costs, marker='o', linewidth=2, markersize=8, label=mechanism)
    
    ax1.set_xlabel('Epsilon (ε)', fontsize=12)
    ax1.set_ylabel('Privacy Cost (Accuracy Loss %)', fontsize=12)
    ax1.set_title('Privacy Cost vs Epsilon', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epsilons)
    
    # Plot 2: Privacy-Utility Scatter
    colors = ['red', 'blue', 'green']
    mechanism_keys = {
        'Old PhraseDP': 'old_phrase_dp_local_cot_correct',
        'InferDPT': 'inferdpt_local_cot_correct', 
        'SANTEXT+': 'santext_local_cot_correct'
    }
    
    for i, mechanism in enumerate(['Old PhraseDP', 'InferDPT', 'SANTEXT+']):
        costs = mechanisms[mechanism]
        accuracies = [data[eps]['results'][mechanism_keys[mechanism]] / 
                     data[eps]['results']['total_questions'] * 100 for eps in epsilons]
        
        ax2.scatter(costs, accuracies, c=colors[i], s=100, alpha=0.7, label=mechanism)
        
        # Add epsilon labels
        for j, eps in enumerate(epsilons):
            ax2.annotate(f'ε={eps}', (costs[j], accuracies[j]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Privacy Cost (Accuracy Loss %)', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Privacy-Utility Trade-off', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('medmcqa_privacy_utility_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.savefig('medmcqa_privacy_utility_tradeoff.pdf', bbox_inches='tight')
    plt.show()

def create_mechanism_comparison_plot(data):
    """Create Plot 3: Individual Bar Plots per Epsilon."""
    epsilons = sorted(data.keys())
    
    # Create subplots for each epsilon
    fig, axes = plt.subplots(1, len(epsilons), figsize=(6*len(epsilons), 8))
    if len(epsilons) == 1:
        axes = [axes]
    
    mechanisms = ['Old PhraseDP', 'InferDPT', 'SANTEXT+']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, epsilon in enumerate(epsilons):
        ax = axes[i]
        results = data[epsilon]['results']
        total = results['total_questions']
        
        # Calculate accuracies for privacy mechanisms
        accuracies = [
            results['old_phrase_dp_local_cot_correct'] / total * 100,
            results['inferdpt_local_cot_correct'] / total * 100,
            results['santext_local_cot_correct'] / total * 100
        ]
        
        # Create bar plot
        bars = ax.bar(mechanisms, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Add baseline and reference lines
        baseline_acc = results['local_alone_correct'] / total * 100
        non_private_acc = results['non_private_cot_correct'] / total * 100
        remote_acc = results['purely_remote_correct'] / total * 100
        
        ax.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Local Baseline')
        ax.axhline(y=non_private_acc, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Non-Private CoT')
        ax.axhline(y=remote_acc, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Remote Model')
        
        ax.set_xlabel('Privacy Mechanisms', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'MedMCQA Results (ε = {epsilon})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)
        
        # Rotate x-axis labels for better readability
        ax.tick_params(axis='x', rotation=45)
        
        # Add summary statistics for this epsilon
        summary_text = f"""
        ε = {epsilon}
        Local: {baseline_acc:.1f}%
        Non-Private: {non_private_acc:.1f}%
        Remote: {remote_acc:.1f}%
        """
        
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=8)
    
    # Add overall title
    fig.suptitle('MedMCQA: Privacy Mechanism Performance by Epsilon', fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig('medmcqa_mechanism_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('medmcqa_mechanism_comparison.pdf', bbox_inches='tight')
    plt.show()

def create_summary_table(data):
    """Create a summary table of results."""
    epsilons = sorted(data.keys())
    
    # Create summary data
    summary_data = []
    
    for epsilon in epsilons:
        results = data[epsilon]['results']
        total = results['total_questions']
        
        row = {
            'Epsilon': epsilon,
            'Local Baseline': f"{results['local_alone_correct']/total*100:.1f}%",
            'Non-Private CoT': f"{results['non_private_cot_correct']/total*100:.1f}%",
            'Old PhraseDP': f"{results['old_phrase_dp_local_cot_correct']/total*100:.1f}%",
            'InferDPT': f"{results['inferdpt_local_cot_correct']/total*100:.1f}%",
            'SANTEXT+': f"{results['santext_local_cot_correct']/total*100:.1f}%",
            'Remote Model': f"{results['purely_remote_correct']/total*100:.1f}%"
        }
        summary_data.append(row)
    
    # Create DataFrame and save
    df = pd.DataFrame(summary_data)
    print("\n=== MedMCQA Results Summary ===")
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv('medmcqa_results_summary.csv', index=False)
    print(f"\nSummary saved to: medmcqa_results_summary.csv")
    
    return df

def main():
    """Main function to generate all plots."""
    print("Loading MedMCQA results...")
    data = load_medmcqa_results()
    
    print("Creating summary table...")
    summary_df = create_summary_table(data)
    
    print("Creating Plot 1: Accuracy vs Epsilon...")
    create_accuracy_vs_epsilon_plot(data)
    
    print("Creating Plot 2: Privacy-Utility Trade-off...")
    create_privacy_utility_tradeoff_plot(data)
    
    print("Creating Plot 3: Mechanism Comparison...")
    create_mechanism_comparison_plot(data)
    
    print("\n✅ All plots generated successfully!")
    print("Generated files:")
    print("- medmcqa_accuracy_vs_epsilon.png & .pdf")
    print("- medmcqa_privacy_utility_tradeoff.png & .pdf") 
    print("- medmcqa_mechanism_comparison.png & .pdf")
    print("- medmcqa_results_summary.csv")

if __name__ == "__main__":
    main()
