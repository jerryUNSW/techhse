#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_hse_bench_results_by_epsilon():
    """Get HSE-bench results organized by epsilon using correct data from analysis report."""
    results = {}
    
    # Use correct data from analysis report (the JSON files are corrupted)
    # Baseline performance (epsilon-independent)
    baseline_results = {
        'Local': 80.0,  # 8/10
        'Local + CoT': 90.0,  # 9/10  
        'Remote': 90.0  # 9/10
    }
    
    # Privacy mechanisms performance by epsilon
    epsilon_results = {
        1.0: {
            'PhraseDP': 90.0,  # 9/10
            'InferDPT': 80.0,  # 8/10
            'SANTEXT+': 70.0   # 7/10
        },
        2.0: {
            'PhraseDP': 90.0,  # 9/10
            'InferDPT': 70.0,  # 7/10
            'SANTEXT+': 80.0   # 8/10
        },
        3.0: {
            'PhraseDP': 90.0,  # 9/10
            'InferDPT': 80.0,  # 8/10
            'SANTEXT+': 80.0   # 8/10
        }
    }
    
    for epsilon in [1.0, 2.0, 3.0]:
        results[epsilon] = {}
        
        # Add baseline results (same for all epsilon)
        for mechanism, accuracy in baseline_results.items():
            results[epsilon][mechanism] = accuracy
            
        # Add privacy mechanism results for this epsilon
        for mechanism, accuracy in epsilon_results[epsilon].items():
            results[epsilon][mechanism] = accuracy
    
    return results

def create_larger_text_height_plots():
    """Create three separate plots with larger text and increased height."""
    results = get_hse_bench_results_by_epsilon()
    
    # Define the desired order from left to right
    desired_order = [
        'Local',
        'InferDPT', 
        'SANTEXT+',
        'PhraseDP',
        'Local + CoT',
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
        ax.set_title(f'HSE-bench Accuracy - Epsilon = {epsilon}', 
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
        
        # No background color - clean white
        ax.set_facecolor('white')
        fig.patch.set_facecolor('white')
        
        # Set tick colors to black with sample plot font
        ax.tick_params(colors='black', which='both', labelsize=20)  # Sample plot tick labels
        
        # Bring x-axis to front so bars appear to sit on it
        ax.set_axisbelow(False)
        ax.spines['bottom'].set_zorder(10)
        ax.spines['left'].set_zorder(10)
        
        # Clean styling like sample plot - no arrows needed
        
        plt.tight_layout()
        
        # Save individual plots
        filename_pdf = f'hse_bench_epsilon_{epsilon}.pdf'
        plt.savefig(filename_pdf, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Copy to overleaf folder
        import shutil
        overleaf_path = 'overleaf-folder/plots/'
        Path(overleaf_path).mkdir(parents=True, exist_ok=True)
        shutil.copy2(filename_pdf, overleaf_path + filename_pdf)
        
        print(f"Updated and copied {filename_pdf} with larger text and increased height to overleaf-folder/plots/")
    
    print("All HSE-bench plots updated successfully with larger text and height!")

if __name__ == "__main__":
    print("Updating HSE-bench plots with larger text and increased height...")
    create_larger_text_height_plots()
