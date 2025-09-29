#!/usr/bin/env python3
"""
MedMCQA Results Visualization Script
====================================

Creates plots similar to Figure 2 (PII protection plots) but for MedMCQA results.
Generates:
1. Accuracy vs Epsilon line plot
2. Individual bar plots per epsilon (1.0, 2.0, 3.0)
3. Privacy-utility trade-off analysis

Author: Tech4HSE Team
Date: 2025-01-27
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from glob import glob

# Set up directories
RESULTS_DIR = "/home/yizhang/tech4HSE/QA-results/medmcqa"
PLOTS_DIR = "/home/yizhang/tech4HSE/plots/medmcqa"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Global font sizing for high-readability figures (~20pt)
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

def load_medmcqa_results():
    """Load MedMCQA results from JSON files."""
    result_files = list(Path(RESULTS_DIR).glob("medmcqa_results_*_100q_eps*.json"))
    
    if not result_files:
        raise FileNotFoundError("No MedMCQA result files found")
    
    data = {}
    for file_path in result_files:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            epsilon = file_data['experiment_info']['epsilon']
            data[epsilon] = file_data
    
    return data

def create_accuracy_vs_epsilon_plot(data, output_path):
    """Create accuracy vs epsilon line plot similar to PII protection plots."""
    epsilons = sorted(data.keys())
    
    # Extract data for privacy mechanisms
    mechanisms = {
        'PhraseDP (Old)': [],
        'InferDPT': [],
        'SANTEXT+': []
    }
    
    for epsilon in epsilons:
        results = data[epsilon]['results']
        total = results['total_questions']
        
        mechanisms['PhraseDP (Old)'].append(results['old_phrase_dp_local_cot_correct'] / total * 100)
        mechanisms['InferDPT'].append(results['inferdpt_local_cot_correct'] / total * 100)
        mechanisms['SANTEXT+'].append(results['santext_local_cot_correct'] / total * 100)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = {
        'PhraseDP (Old)': '#1f77b4',
        'InferDPT': '#ff7f0e', 
        'SANTEXT+': '#2ca02c'
    }
    
    for mechanism, accuracies in mechanisms.items():
        ax.plot(epsilons, accuracies, marker='o', linewidth=3, markersize=8, 
                label=mechanism, color=colors[mechanism])
    
    # Add baseline and reference lines
    baseline_acc = data[epsilons[0]]['results']['local_alone_correct'] / data[epsilons[0]]['results']['total_questions'] * 100
    non_private_acc = data[epsilons[0]]['results']['non_private_cot_correct'] / data[epsilons[0]]['results']['total_questions'] * 100
    remote_acc = data[epsilons[0]]['results']['purely_remote_correct'] / data[epsilons[0]]['results']['total_questions'] * 100
    
    ax.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Local Baseline')
    ax.axhline(y=non_private_acc, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Non-Private CoT')
    ax.axhline(y=remote_acc, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Remote Model')
    
    ax.set_xlabel('Epsilon (ε)', fontsize=20)
    ax.set_ylabel('Accuracy (%)', fontsize=20)
    ax.set_title('MedMCQA: Privacy Mechanism Performance vs Epsilon', fontsize=22, fontweight='bold')
    ax.legend(fontsize=18)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epsilons)
    ax.set_ylim(0, 100)
    
    # Add annotations
    ax.text(0.02, 0.98, f'Dataset: MedMCQA (100 questions)\nModel: Llama 3.1 8B', 
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=16)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

def create_individual_bar_plots(data, output_prefix):
    """Create individual bar plots for each epsilon value."""
    epsilons = sorted(data.keys())
    
    for epsilon in epsilons:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        results = data[epsilon]['results']
        total = results['total_questions']
        
        # Calculate accuracies for privacy mechanisms
        mechanisms = ['PhraseDP (Old)', 'InferDPT', 'SANTEXT+']
        accuracies = [
            results['old_phrase_dp_local_cot_correct'] / total * 100,
            results['inferdpt_local_cot_correct'] / total * 100,
            results['santext_local_cot_correct'] / total * 100
        ]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Create bar plot
        bars = ax.bar(mechanisms, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=16, fontweight='bold')
        
        # Add baseline and reference lines
        baseline_acc = results['local_alone_correct'] / total * 100
        non_private_acc = results['non_private_cot_correct'] / total * 100
        remote_acc = results['purely_remote_correct'] / total * 100
        
        ax.axhline(y=baseline_acc, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Local Baseline')
        ax.axhline(y=non_private_acc, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Non-Private CoT')
        ax.axhline(y=remote_acc, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Remote Model')
        
        ax.set_xlabel('Privacy Mechanisms', fontsize=20)
        ax.set_ylabel('Accuracy (%)', fontsize=20)
        ax.set_title(f'MedMCQA Results (ε = {epsilon})', fontsize=22, fontweight='bold')
        ax.legend(fontsize=16)
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
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=14)
        
        plt.tight_layout()
        output_path_png = f"{output_prefix}_eps_{str(epsilon).replace('.', '_')}.png"
        output_path_pdf = f"{output_prefix}_eps_{str(epsilon).replace('.', '_')}.pdf"
        plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_path_pdf, bbox_inches='tight')
        plt.close()

def create_privacy_utility_plot(data, output_path):
    """Create privacy-utility trade-off plot."""
    epsilons = sorted(data.keys())
    
    # Calculate privacy cost (difference from remote model)
    remote_acc = data[epsilons[0]]['results']['purely_remote_correct'] / data[epsilons[0]]['results']['total_questions'] * 100
    
    mechanisms = {
        'PhraseDP (Old)': [],
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
        
        mechanisms['PhraseDP (Old)'].append(phrasedp_cost)
        mechanisms['InferDPT'].append(inferdpt_cost)
        mechanisms['SANTEXT+'].append(santext_cost)
        
        privacy_costs.append([phrasedp_cost, inferdpt_cost, santext_cost])
    
    # Create subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    colors = {
        'PhraseDP (Old)': '#1f77b4',
        'InferDPT': '#ff7f0e',
        'SANTEXT+': '#2ca02c'
    }
    
    # Plot 1: Privacy Cost vs Epsilon
    for mechanism, costs in mechanisms.items():
        ax1.plot(epsilons, costs, marker='o', linewidth=3, markersize=8, 
                label=mechanism, color=colors[mechanism])
    
    ax1.set_xlabel('Epsilon (ε)', fontsize=20)
    ax1.set_ylabel('Privacy Cost (Accuracy Loss %)', fontsize=20)
    ax1.set_title('Privacy Cost vs Epsilon', fontsize=22, fontweight='bold')
    ax1.legend(fontsize=18)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epsilons)
    
    # Plot 2: Privacy-Utility Scatter
    mechanism_keys = {
        'PhraseDP (Old)': 'old_phrase_dp_local_cot_correct',
        'InferDPT': 'inferdpt_local_cot_correct', 
        'SANTEXT+': 'santext_local_cot_correct'
    }
    
    for i, mechanism in enumerate(['PhraseDP (Old)', 'InferDPT', 'SANTEXT+']):
        costs = mechanisms[mechanism]
        accuracies = [data[eps]['results'][mechanism_keys[mechanism]] / 
                     data[eps]['results']['total_questions'] * 100 for eps in epsilons]
        
        ax2.scatter(costs, accuracies, c=colors[mechanism], s=150, alpha=0.7, label=mechanism)
        
        # Add epsilon labels
        for j, eps in enumerate(epsilons):
            ax2.annotate(f'ε={eps}', (costs[j], accuracies[j]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=14)
    
    ax2.set_xlabel('Privacy Cost (Accuracy Loss %)', fontsize=20)
    ax2.set_ylabel('Accuracy (%)', fontsize=20)
    ax2.set_title('Privacy-Utility Trade-off', fontsize=22, fontweight='bold')
    ax2.legend(fontsize=18)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # Also save as PDF
    pdf_path = output_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()

def remove_old_plots():
    """Remove old plot files."""
    patterns = [
        os.path.join(PLOTS_DIR, 'medmcqa_accuracy_vs_epsilon_*.png'),
        os.path.join(PLOTS_DIR, 'medmcqa_bar_plots_*_eps_*.png'),
        os.path.join(PLOTS_DIR, 'medmcqa_privacy_utility_*.png'),
    ]
    removed = []
    for pat in patterns:
        for p in glob(pat):
            try:
                os.remove(p)
                removed.append(p)
            except Exception:
                pass
    return removed

def main():
    """Main function to generate all MedMCQA plots."""
    print("Loading MedMCQA results...")
    data = load_medmcqa_results()
    
    print("Removing old plots...")
    removed = remove_old_plots()
    if removed:
        print(f"Removed {len(removed)} old plot files")
    
    # Generate timestamp for new plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print("Creating accuracy vs epsilon plot...")
    accuracy_path = os.path.join(PLOTS_DIR, f'medmcqa_accuracy_vs_epsilon_{timestamp}.png')
    create_accuracy_vs_epsilon_plot(data, accuracy_path)
    
    print("Creating individual bar plots...")
    bar_prefix = os.path.join(PLOTS_DIR, f'medmcqa_bar_plots_{timestamp}')
    create_individual_bar_plots(data, bar_prefix)
    
    print("Creating privacy-utility trade-off plot...")
    privacy_path = os.path.join(PLOTS_DIR, f'medmcqa_privacy_utility_{timestamp}.png')
    create_privacy_utility_plot(data, privacy_path)
    
    print("\n✅ All MedMCQA plots generated successfully!")
    print(f"Generated files:")
    print(f"- {accuracy_path}")
    print(f"- {bar_prefix}_eps_1_0.png")
    print(f"- {bar_prefix}_eps_2_0.png") 
    print(f"- {bar_prefix}_eps_3_0.png")
    print(f"- {privacy_path}")

if __name__ == "__main__":
    main()
