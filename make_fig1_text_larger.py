#!/usr/bin/env python3
"""
Make Fig 1 (MedQA epsilon bar plots) text even larger - 24-26pt for better visibility.
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import shutil

def get_medqa_results_by_epsilon():
    """Get MedQA results organized by epsilon."""
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
                clean_name = 'PhraseDP'  # Rename as requested
            
            results[epsilon][clean_name] = accuracy
    
    conn.close()
    return results

def create_larger_text_plots():
    """Create three separate plots with much larger text (24-26pt)."""
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
    
    # Color palette
    colors = [
        '#FFE55C',  # Bright light yellow
        '#D2691E',  # Muted earthy orange
        '#90EE90',  # Medium desaturated green
        '#4169E1',  # Vibrant medium blue
        '#191970',  # Deep dark blue
        '#4169E1'   # Vibrant medium blue
    ]
    
    # Set global font size to 24pt for base
    plt.rcParams.update({'font.size': 24})
    
    # Create separate plots for each epsilon
    for epsilon in [1.0, 2.0, 3.0]:
        fig, ax = plt.subplots(figsize=(16, 10))  # Even larger figure for better proportions
        
        # Get mechanisms and accuracies in the desired order
        mechanisms = []
        accuracies = []
        
        for mech in desired_order:
            if mech in results[epsilon]:
                mechanisms.append(mech)
                accuracies.append(results[epsilon][mech])
        
        # Create bars with 3D effects
        bars = ax.bar(range(len(mechanisms)), accuracies, 
                     color=colors[:len(mechanisms)],
                     edgecolor='white',
                     linewidth=4,  # Even thicker borders
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
        ax.set_title(f'MedQA UME Accuracy - Epsilon = {epsilon}', 
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
        fig.patch.set_facecolor('white')
        
        # Set tick labels to larger size
        ax.tick_params(colors='black', which='both', labelsize=22)  # 22pt tick labels
        
        # Ensure y-axis tick labels are large
        for label in ax.get_yticklabels():
            label.set_fontsize(22)
        
        # Ensure x-axis tick labels are large
        for label in ax.get_xticklabels():
            label.set_fontsize(24)
        
        plt.tight_layout()
        
        # Save individual plots
        filename_pdf = f'medqa_epsilon_{epsilon}.pdf'
        
        plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white')
        
        # Copy to overleaf folder
        shutil.copy(filename_pdf, 'overleaf-folder/plots/')
        
        print(f"Updated and copied {filename_pdf} with much larger text (22-26pt) to overleaf-folder/plots/")
        
        plt.close()
    
    # Reset font size
    plt.rcParams.update({'font.size': 10})

if __name__ == "__main__":
    print("Making Fig 1 plots with much larger text (22-26pt)...")
    create_larger_text_plots()
    print("All Fig 1 plots updated successfully with much larger text!")

