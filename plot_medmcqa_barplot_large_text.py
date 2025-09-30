#!/usr/bin/env python3
"""
Update Fig 1 (MedMCQA epsilon bar plots) with larger text and increased height.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import shutil
import glob
from pathlib import Path

def get_medmcqa_results_by_epsilon():
    """Get MedMCQA results organized by epsilon from JSON files."""
    results = {}
    
    # Find the main 500q results files
    json_files = {
        1.0: 'experiment_results/QA-results/medmcqa/medmcqa_results_local_meta-llama_Meta-Llama-3.1-8B-Instruct_remote_deepseek_chat_500q_eps1.0_20250929_153118.json',
        2.0: 'experiment_results/QA-results/medmcqa/medmcqa_results_local_meta-llama_Meta-Llama-3.1-8B-Instruct_remote_deepseek_chat_500q_eps2.0_20250929_153118.json',
        3.0: 'experiment_results/QA-results/medmcqa/medmcqa_results_local_meta-llama_Meta-Llama-3.1-8B-Instruct_remote_deepseek_chat_500q_eps3.0_20250929_153118.json'
    }
    
    for epsilon in [1.0, 2.0, 3.0]:
        json_file = json_files[epsilon]
        if not Path(json_file).exists():
            print(f"Warning: {json_file} not found, skipping epsilon {epsilon}")
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results[epsilon] = {}
        total_questions = data['results']['total_questions']
        
        # Extract accuracies for each mechanism
        mechanisms = {
            'Local': data['results']['local_alone_correct'],
            'CoT': data['results']['non_private_cot_correct'],
            'PhraseDP': data['results']['old_phrase_dp_local_cot_correct'],
            'InferDPT': data['results']['inferdpt_local_cot_correct'],
            'SANTEXT+': data['results']['santext_local_cot_correct'],
            'Remote': data['results']['purely_remote_correct']
        }
        
        # Convert to percentages
        for mechanism, correct in mechanisms.items():
            accuracy = (correct / total_questions * 100) if total_questions > 0 else 0
            results[epsilon][mechanism] = accuracy
    
    return results

def create_larger_text_height_plots():
    """Create three separate plots with larger text and increased height."""
    results = get_medmcqa_results_by_epsilon()
    
    # Define the desired order from left to right
    desired_order = [
        'Local',
        'InferDPT', 
        'SANTEXT+',
        'PhraseDP',
        'CoT',
        'Remote'
    ]
    
    # Color palette
    colors = [
        '#FFE55C',  # Bright light yellow
        '#D2691E',  # Muted earthy orange
        '#90EE90',  # Medium desaturated green
        '#4169E1',  # Vibrant medium blue
        '#191970',  # Deep dark blue
        '#4169E1'   # Vibrant medium blue
    ]
    
    # Apply sample plot settings using plt_settings function
    def plt_settings():
        plt.style.use('default')
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams["legend.framealpha"] = 0
        plt.rcParams["legend.handletextpad"] = 0.1
        plt.rcParams["legend.columnspacing"] = 0.2
        # for the varying plots 
        plt.rcParams["figure.figsize"] = (6,5)
        plt.rcParams['pdf.fonttype'] = 42
    
    plt_settings()
    
    # Create separate plots for each epsilon
    for epsilon in [1.0, 2.0, 3.0]:
        fig, ax = plt.subplots(figsize=(6, 5))  # Square-like figure size from sample plot
        
        # Get mechanisms and accuracies in the desired order
        mechanisms = []
        accuracies = []
        
        for mech in desired_order:
            if mech in results[epsilon]:
                mechanisms.append(mech)
                accuracies.append(results[epsilon][mech])
        
        # Create bars with optimal width and styling
        bars = ax.bar(range(len(mechanisms)), accuracies, 
                     color=colors[:len(mechanisms)],
                     edgecolor='black',
                     linewidth=0.5,  # Thinner borders like sample plot
                     alpha=0.9,
                     zorder=3,
                     width=0.6)  # Optimal bar width from sample plot
        
        # Remove 3D effects for cleaner look like sample plot
        
        # Customize plot with sample plot font sizes
        ax.set_title(f'MedMCQA Accuracy - Epsilon = {epsilon}', 
                    fontsize=20, fontweight='bold', pad=20,
                    color='black')  # Sample plot title size
        ax.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold', color='black')  # Sample plot y-label size
        ax.set_ylim(0, 100)
        
        # Set x-axis labels with rotation and sample plot font
        ax.set_xticks(range(len(mechanisms)))
        ax.set_xticklabels(mechanisms, rotation=45, ha='right', fontsize=18, fontweight='bold', color='black')  # Sample plot x-labels
        
        # Enhanced grid with subtle styling
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=1.0, color='gray')
        ax.set_axisbelow(True)
        
        # Add value labels on top of bars with sample plot font
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                   f'{acc:.1f}%', 
                   ha='center', va='bottom', 
                   fontsize=15, fontweight='bold',  # Sample plot value labels
                   color='black')
        
        # Clean appearance with no background color
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3)  # Thicker spines
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['left'].set_color('black')
        ax.spines['bottom'].set_color('black')
        
        # No background color - clean white
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Set tick colors to black with sample plot font
        ax.tick_params(colors='black', which='both', labelsize=20)  # Sample plot tick labels
        
        # Add arrows to axes
        ax.annotate('', xy=(1, 0), xytext=(0, 0), 
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.annotate('', xy=(0, 1), xytext=(0, 0), 
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        
        plt.tight_layout()
        
        # Save individual plots
        filename_pdf = f'medmcqa_epsilon_{epsilon}.pdf'
        
        plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white')
        
        # Copy to overleaf folder
        shutil.copy(filename_pdf, 'overleaf-folder/plots/')
        
        print(f"Updated and copied {filename_pdf} with larger text and increased height to overleaf-folder/plots/")
        
        plt.close()

if __name__ == "__main__":
    print("Updating MedMCQA plots with larger text and increased height...")
    create_larger_text_height_plots()
    print("All MedMCQA plots updated successfully with larger text and height!")
