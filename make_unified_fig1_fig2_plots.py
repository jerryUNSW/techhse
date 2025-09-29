#!/usr/bin/env python3
"""
Unified script to make both Figure 1 (MedQA) and Figure 2 (MedMCQA) plots with identical styling.
Adopts the style from make_medmcqa_fig1_plots.py for consistency.
"""

import sqlite3
import json
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
from pathlib import Path

def get_medqa_results_by_epsilon():
    """Get MedQA results organized by epsilon from SQLite database."""
    conn = sqlite3.connect('tech4hse_results.db')
    cursor = conn.cursor()
    
    results = {}
    
    for epsilon in [1.0, 2.0, 3.0]:
        cursor.execute('''
            SELECT mechanism, local_answer
            FROM medqa_results 
            WHERE epsilon = ? AND question_id = -1
            ORDER BY mechanism
        ''', (epsilon,))
        
        mechanisms_data = cursor.fetchall()
        results[epsilon] = {}
        
        for mechanism, local_answer in mechanisms_data:
            # Extract accuracy from "Summary: X/500 = Y.Y%"
            accuracy_str = local_answer.split('=')[1].strip().replace('%', '')
            accuracy = float(accuracy_str)
            
            # Clean up mechanism names for plotting with custom naming
            clean_name = mechanism.replace('Private Local Model + CoT (', '').replace(')', '')
            if 'Purely Local Model' in mechanism:
                clean_name = 'Purely Local'
            elif 'Non-Private Local Model' in mechanism:
                clean_name = 'Non-Private + CoT'
            elif 'Purely Remote Model' in mechanism:
                clean_name = 'Purely Remote'
            elif 'Old Phrase DP' in mechanism:
                clean_name = 'PhraseDP'
            
            results[epsilon][clean_name] = accuracy
    
    conn.close()
    return results

def get_medmcqa_results_by_epsilon():
    """Get MedMCQA results organized by epsilon from JSON files."""
    results_dir = Path("QA-results/medmcqa")
    result_files = list(results_dir.glob("medmcqa_results_*_100q_eps*.json"))
    
    if not result_files:
        raise FileNotFoundError("No MedMCQA result files found")
    
    results = {}
    
    for file_path in result_files:
        with open(file_path, 'r') as f:
            file_data = json.load(f)
            epsilon = file_data['experiment_info']['epsilon']
            results[epsilon] = file_data
    
    return results

def create_plot_with_unified_style(ax, mechanisms, accuracies, colors, epsilon, dataset_name):
    """Create a plot with unified styling for both MedQA and MedMCQA."""
    
    # Create bars with 3D effects
    bars = ax.bar(range(len(mechanisms)), accuracies, 
                 color=colors[:len(mechanisms)],
                 edgecolor='white',
                 linewidth=4,  # Thick borders
                 alpha=0.9,
                 zorder=3,
                 width=0.8)  # Bar width
    
    # Add gradient effect to bars for depth
    for i, (bar, color) in enumerate(zip(bars, colors[:len(mechanisms)])):
        # Create gradient effect by varying alpha
        gradient = np.linspace(0.6, 1.0, 50)
        for j, alpha in enumerate(gradient):
            height = bar.get_height() * (j / 50)
            ax.bar(i, height, 
                  color=color, 
                  alpha=alpha * 0.4,
                  width=0.8,
                  zorder=1)
    
    # Add subtle shadows behind bars for 3D effect
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.bar(i + 0.05, acc, 
              color='black', 
              alpha=0.2,
              width=0.8,
              zorder=1)
    
    # Set ALL text elements to much larger sizes
    ax.set_title(f'{dataset_name} Privacy Mechanisms - Epsilon = {epsilon}', 
                fontsize=26, fontweight='bold', pad=40)  # 26pt title
    ax.set_ylabel('Accuracy (%)', fontsize=24, fontweight='bold')  # 24pt y-label
    ax.set_ylim(0, 100)
    
    # Set x-axis labels with rotation and larger font
    ax.set_xticks(range(len(mechanisms)))
    ax.set_xticklabels(mechanisms, rotation=45, ha='right', fontsize=24, fontweight='bold')  # 24pt x-labels
    
    # Enhanced grid with subtle styling
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=1.2, color='gray')
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars with larger font
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
               f'{acc:.1f}%', 
               ha='center', va='bottom', 
               fontsize=22, fontweight='bold',  # 22pt value labels
               color='black')
    
    # Clean appearance with no background color
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(4)  # Thicker spines
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # No background color - clean white
    ax.set_facecolor('white')
    
    # Set tick labels to larger size
    ax.tick_params(colors='black', which='both', labelsize=22)  # 22pt tick labels
    
    # Ensure y-axis tick labels are large
    for label in ax.get_yticklabels():
        label.set_fontsize(22)
    
    # Ensure x-axis tick labels are large
    for label in ax.get_xticklabels():
        label.set_fontsize(24)

def create_medqa_plots():
    """Create MedQA Figure 1 plots with unified styling."""
    results = get_medqa_results_by_epsilon()
    
    # Define the desired order from left to right
    desired_order = [
        'Purely Local',
        'InferDPT',
        'SANTEXT+',
        'PhraseDP',
        'Non-Private + CoT',
        'Purely Remote'
    ]
    
    # Color palette matching the order
    colors = [
        '#FFE55C',  # Bright light yellow for Purely Local
        '#D2691E',  # Muted earthy orange for InferDPT
        '#90EE90',  # Medium desaturated green for SANTEXT+
        '#4169E1',  # Vibrant medium blue for PhraseDP
        '#191970',  # Deep dark blue for Non-Private + CoT
        '#4169E1'   # Vibrant medium blue for Purely Remote
    ]
    
    # Set global font size to 24pt for base
    plt.rcParams.update({'font.size': 24})
    
    # Create separate plots for each epsilon
    for epsilon in [1.0, 2.0, 3.0]:
        if epsilon not in results:
            print(f"Warning: No MedQA results found for epsilon {epsilon}")
            continue
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get mechanisms and accuracies in the desired order
        mechanisms = []
        accuracies = []
        
        for mech in desired_order:
            if mech in results[epsilon]:
                mechanisms.append(mech)
                accuracies.append(results[epsilon][mech])
        
        # Create plot with unified styling
        create_plot_with_unified_style(ax, mechanisms, accuracies, colors, epsilon, "MedQA")
        
        plt.tight_layout()
        
        # Save individual plots
        filename_pdf = f'medqa_epsilon_{epsilon}.pdf'
        
        plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white')
        
        # Copy to overleaf folder
        os.makedirs('overleaf-folder/plots', exist_ok=True)
        shutil.copy(filename_pdf, 'overleaf-folder/plots/')
        
        print(f"Created and copied {filename_pdf} with unified styling to overleaf-folder/plots/")
        
        plt.close()
    
    # Reset font size
    plt.rcParams.update({'font.size': 10})

def create_medmcqa_plots():
    """Create MedMCQA Figure 2 plots with unified styling."""
    results = get_medmcqa_results_by_epsilon()
    
    # Define the desired order from left to right (all scenarios)
    desired_order = [
        'Purely Local',
        'InferDPT',
        'SANTEXT+',
        'PhraseDP (Old)',
        'Non-Private + CoT',
        'Purely Remote'
    ]
    
    # Color palette for all scenarios (matching the new order)
    colors = [
        '#FFE55C',  # Bright light yellow for Purely Local
        '#D2691E',  # Muted earthy orange for InferDPT
        '#90EE90',  # Medium desaturated green for SANTEXT+
        '#4169E1',  # Vibrant medium blue for PhraseDP
        '#191970',  # Deep dark blue for Non-Private + CoT
        '#4169E1'   # Vibrant medium blue for Purely Remote
    ]
    
    # Set global font size to 24pt for base
    plt.rcParams.update({'font.size': 24})
    
    # Create separate plots for each epsilon
    for epsilon in [1.0, 2.0, 3.0]:
        if epsilon not in results:
            print(f"Warning: No MedMCQA results found for epsilon {epsilon}")
            continue
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get mechanisms and accuracies in the desired order
        mechanisms = []
        accuracies = []
        
        # Extract data from results
        result_data = results[epsilon]['results']
        total_questions = result_data['total_questions']
        
        # Calculate accuracies for all scenarios
        mechanism_data = {
            'Purely Local': result_data['local_alone_correct'] / total_questions * 100,
            'Non-Private + CoT': result_data['non_private_cot_correct'] / total_questions * 100,
            'PhraseDP (Old)': result_data['old_phrase_dp_local_cot_correct'] / total_questions * 100,
            'InferDPT': result_data['inferdpt_local_cot_correct'] / total_questions * 100,
            'SANTEXT+': result_data['santext_local_cot_correct'] / total_questions * 100,
            'Purely Remote': result_data['purely_remote_correct'] / total_questions * 100
        }
        
        for mech in desired_order:
            mechanisms.append(mech)
            accuracies.append(mechanism_data[mech])
        
        # Create plot with unified styling
        create_plot_with_unified_style(ax, mechanisms, accuracies, colors, epsilon, "MedMCQA")
        
        plt.tight_layout()
        
        # Save individual plots
        filename_pdf = f'medmcqa_epsilon_{epsilon}.pdf'
        
        plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white')
        
        # Copy to overleaf folder
        os.makedirs('overleaf-folder/plots', exist_ok=True)
        shutil.copy(filename_pdf, 'overleaf-folder/plots/')
        
        print(f"Created and copied {filename_pdf} with unified styling to overleaf-folder/plots/")
        
        plt.close()
    
    # Reset font size
    plt.rcParams.update({'font.size': 10})

def main():
    """Create both Figure 1 (MedQA) and Figure 2 (MedMCQA) plots with unified styling."""
    print("Creating unified Figure 1 (MedQA) and Figure 2 (MedMCQA) plots with identical styling...")
    
    # Create MedQA plots (Figure 1)
    print("\n=== Creating MedQA Figure 1 plots ===")
    create_medqa_plots()
    
    # Create MedMCQA plots (Figure 2)
    print("\n=== Creating MedMCQA Figure 2 plots ===")
    create_medmcqa_plots()
    
    print("\nAll plots created successfully with unified styling!")

if __name__ == "__main__":
    main()
