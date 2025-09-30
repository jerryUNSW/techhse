#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_hse_bench_results_by_epsilon():
    """Get HSE-bench results organized by epsilon from JSON files."""
    results = {}
    
    # Find the main 10q results files
    json_files = {
        1.0: 'experiment_results/QA-results/hse-bench/hse_bench_enhanced_results_local_meta-llama_Meta-Llama-3.1-8B-Instruct_remote_gpt4o_mini_10q_eps1.0_20250929_220742.json',
        2.0: 'experiment_results/QA-results/hse-bench/hse_bench_enhanced_results_local_meta-llama_Meta-Llama-3.1-8B-Instruct_remote_gpt4o_mini_10q_eps2.0_20250929_220839.json',
        3.0: 'experiment_results/QA-results/hse-bench/hse_bench_enhanced_results_local_meta-llama_Meta-Llama-3.1-8B-Instruct_remote_gpt4o_mini_10q_eps3.0_20250929_220926.json'
    }
    
    for epsilon in [1.0, 2.0, 3.0]:
        json_file = json_files[epsilon]
        if not Path(json_file).exists():
            print(f"Warning: {json_file} not found, skipping epsilon {epsilon}")
            continue
            
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        results[epsilon] = {}
        total_questions = data['num_samples']
        
        # Extract accuracies for each mechanism
        shared_results = data['summary_results']['shared_results']
        epsilon_results = data['summary_results']['epsilon_results'][str(epsilon)]
        
        mechanisms = {
            'Local': shared_results['local_alone_correct'],
            'Local + CoT': shared_results['non_private_cot_correct'],
            'PhraseDP': epsilon_results['old_phrase_dp_local_cot_correct'],
            'InferDPT': epsilon_results['inferdpt_local_cot_correct'],
            'SANTEXT+': epsilon_results['santext_local_cot_correct'],
            'Remote': shared_results['purely_remote_correct']
        }
        
        # Convert to percentages
        for mechanism, correct in mechanisms.items():
            accuracy = (correct / total_questions * 100) if total_questions > 0 else 0
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
        '#FF6B6B',  # Vibrant coral red
        '#4ECDC4',  # Soft teal
        '#45B7D1',  # Sky blue
        '#96CEB4'   # Mint green
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
