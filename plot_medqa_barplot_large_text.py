#!/usr/bin/env python3
"""
Update Fig 1 (MedQA epsilon bar plots) with larger text and increased height.
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
            if 'Purely Local Model' in mechanism:
                clean_name = 'Local'
            elif 'Non-Private Local Model' in mechanism:
                clean_name = 'Non-Private + CoT'
            elif 'Purely Remote Model' in mechanism:
                clean_name = 'Remote'
            elif 'Old Phrase DP' in mechanism:
                clean_name = 'PhraseDP'
            elif 'InferDPT' in mechanism:
                clean_name = 'InferDPT'
            elif 'SANTEXT+' in mechanism:
                clean_name = 'SANTEXT+'
            else:
                clean_name = mechanism  # Fallback
            
            results[epsilon][clean_name] = accuracy
    
    conn.close()
    return results

def create_larger_text_height_plots():
    """Create three separate plots with larger text and increased height."""
    results = get_medqa_results_by_epsilon()
    
    # Define the desired order from left to right
    desired_order = [
        'Local',
        'InferDPT', 
        'SANTEXT+',
        'PhraseDP',
        'Non-Private + CoT',
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
        ax.set_title(f'MedQA UME Accuracy - Epsilon = {epsilon}', 
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
        
        plt.tight_layout()
        
        # Save individual plots
        filename_pdf = f'medqa_epsilon_{epsilon}.pdf'
        
        plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white')
        
        # Copy to overleaf folder
        shutil.copy(filename_pdf, 'overleaf-folder/plots/')
        
        print(f"Updated and copied {filename_pdf} with larger text and increased height to overleaf-folder/plots/")
        
        plt.close()

if __name__ == "__main__":
    print("Updating Fig 1 plots with larger text and increased height...")
    create_larger_text_height_plots()
    print("All Fig 1 plots updated successfully with larger text and height!")
