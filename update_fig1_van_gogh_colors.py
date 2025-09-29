#!/usr/bin/env python3
"""
Update Fig 1 (MedQA epsilon bar plots) with Van Gogh irises-inspired color palette.
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

def create_van_gogh_style_plots():
    """Create three separate plots with Van Gogh irises-inspired color palette."""
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
    
    # Van Gogh irises-inspired color palette
    van_gogh_colors = [
        '#FFD700',  # Bright Yellow (background)
        '#D2691E',  # Muted Orange/Ochre (vase)
        '#32CD32',  # Bright Green (leaves)
        '#4169E1',  # Medium Blue (iris petals)
        '#191970',  # Dark Blue/Indigo (iris petals)
        '#8A2BE2'   # Purple (additional iris color)
    ]
    
    # Create separate plots for each epsilon
    for epsilon in [1.0, 2.0, 3.0]:
        fig, ax = plt.subplots(figsize=(12, 7))  # Larger figure for better proportions
        
        # Get mechanisms and accuracies in the desired order
        mechanisms = []
        accuracies = []
        
        for mech in desired_order:
            if mech in results[epsilon]:
                mechanisms.append(mech)
                accuracies.append(results[epsilon][mech])
        
        # Create bars with Van Gogh-inspired colors and 3D effects
        bars = ax.bar(range(len(mechanisms)), accuracies, 
                     color=van_gogh_colors[:len(mechanisms)],
                     edgecolor='white',
                     linewidth=2,
                     alpha=0.9,
                     zorder=3)
        
        # Add gradient effect to bars for depth
        for i, (bar, color) in enumerate(zip(bars, van_gogh_colors[:len(mechanisms)])):
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
            ax.bar(i + 0.03, acc, 
                  color='black', 
                  alpha=0.2,
                  width=0.8,
                  zorder=1)
        
        # Customize plot with larger fonts and artistic styling
        ax.set_title(f'MedQA UME Accuracy - Epsilon = {epsilon}', 
                    fontsize=22, fontweight='bold', pad=20,
                    color='#2F4F4F')  # Dark slate gray for elegance
        ax.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold', color='#2F4F4F')
        ax.set_ylim(0, 100)
        
        # Set x-axis labels with rotation and larger font
        ax.set_xticks(range(len(mechanisms)))
        ax.set_xticklabels(mechanisms, rotation=45, ha='right', fontsize=18, fontweight='bold', color='#2F4F4F')
        
        # Enhanced grid with subtle styling
        ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.8, color='gray')
        ax.set_axisbelow(True)
        
        # Add value labels on top of bars with artistic styling
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', 
                   ha='center', va='bottom', 
                   fontsize=14, fontweight='bold',
                   color='#2F4F4F')
        
        # Enhance the overall appearance with artistic touches
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_color('#2F4F4F')
        ax.spines['bottom'].set_color('#2F4F4F')
        
        # Set background color with warm undertone
        ax.set_facecolor('#FFF8DC')  # Cornsilk background
        fig.patch.set_facecolor('white')
        
        # Set tick colors
        ax.tick_params(colors='#2F4F4F', which='both')
        
        plt.tight_layout()
        
        # Save individual plots
        filename_pdf = f'medqa_epsilon_{epsilon}.pdf'
        
        plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white')
        
        # Copy to overleaf folder
        shutil.copy(filename_pdf, 'overleaf-folder/plots/')
        
        print(f"Updated and copied {filename_pdf} with Van Gogh-inspired colors to overleaf-folder/plots/")
        
        plt.close()

if __name__ == "__main__":
    print("Updating Fig 1 plots with Van Gogh irises-inspired color palette...")
    create_van_gogh_style_plots()
    print("All Fig 1 plots updated successfully with artistic Van Gogh styling!")

