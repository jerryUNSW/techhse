#!/usr/bin/env python3
"""
Update the three epsilon bar plots with pastel color palette and increase font sizes.
Also generate a PDF version of the overall protection plot for Fig 2.

FIGURE REFERENCE DOCUMENTATION:
- Fig 1: MedQA UME accuracy plots (three separate bar charts for epsilon 1.0, 2.0, 3.0)
- Fig 2: Overall PII protection rate vs epsilon line plot (5 mechanisms: PhraseDP, InferDPT, SANTEXT+, CusText+, CluSanT)
- Fig 3: Per-epsilon PII protection radar charts (5 mechanisms, 5 dimensions: overall, emails, phones, addresses, names)
- Fig 4: Scalability plots (accuracy and processing time vs number of questions)

Current files in overleaf-folder/plots/:
- medqa_epsilon_1.0.pdf, medqa_epsilon_2.0.pdf, medqa_epsilon_3.0.pdf (Fig 1)
- overall_protection_vs_epsilon_all_mechanisms.png (Fig 2 - needs PDF version)
- protection_radar_5mech_20250927_eps_1_0.pdf, eps_2_0.pdf, eps_3_0.pdf (Fig 3)
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import shutil
import json

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

def create_pastel_epsilon_plots():
    """Create three separate plots with pastel color palette."""
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
    
    # Pastel color palette from user
    pastel_colors = [
        '#FFECD2',  # Light creamy peach
        '#FFD79F',  # Pale soft yellow
        '#C4D6E7',  # Light muted sky blue
        '#DCEAFF',  # Very pale blue
        '#F9B288',  # Soft light coral
        '#FFDECA'   # Light warm peach
    ]
    
    # Create separate plots for each epsilon
    for epsilon in [1.0, 2.0, 3.0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get mechanisms and accuracies in the desired order
        mechanisms = []
        accuracies = []
        
        for mech in desired_order:
            if mech in results[epsilon]:
                mechanisms.append(mech)
                accuracies.append(results[epsilon][mech])
        
        # Create bars with pastel colors
        bars = ax.bar(range(len(mechanisms)), accuracies, color=pastel_colors[:len(mechanisms)])
        
        # Customize plot with larger fonts
        ax.set_title(f'MedQA UME Accuracy - Epsilon = {epsilon}', fontsize=20, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=20)
        ax.set_ylim(0, 100)
        
        # Set x-axis labels with rotation and larger font
        ax.set_xticks(range(len(mechanisms)))
        ax.set_xticklabels(mechanisms, rotation=45, ha='right', fontsize=20)
        
        # Add subtle grid
        ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save individual plots
        filename_pdf = f'medqa_epsilon_{epsilon}.pdf'
        
        plt.savefig(filename_pdf, bbox_inches='tight')
        
        # Copy to overleaf folder
        shutil.copy(filename_pdf, 'overleaf-folder/plots/')
        
        print(f"Updated and copied {filename_pdf} with pastel colors to overleaf-folder/plots/")
        
        plt.close()

def create_overall_protection_pdf():
    """Create a PDF version of the overall protection vs epsilon plot for Fig 2."""
    
    # Load comprehensive results
    results_file = 'experiment_results/ppi-protection/comprehensive_ppi_protection_results_20250927_164033_backup.json'
    with open(results_file, 'r') as f:
        comprehensive_results = json.load(f)
    
    # Load latest CluSanT results
    clusant_file = 'experiment_results/ppi-protection/pii_protection_results_20250929_071204.json'
    with open(clusant_file, 'r') as f:
        clusant_results = json.load(f)
    
    # Merge results
    merged_results = comprehensive_results.copy()
    if 'CluSanT' in clusant_results:
        merged_results['CluSanT'] = clusant_results['CluSanT']
    
    # Define mechanisms and colors
    mechanisms = ["PhraseDP", "InferDPT", "SANTEXT+", "CusText+", "CluSanT"]
    epsilons = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    colors = {
        "PhraseDP": "#1f77b4",   # blue
        "InferDPT": "#17becf",   # cyan
        "SANTEXT+": "#2ca02c",   # green
        "CusText+": "#9467bd",   # purple
        "CluSanT": "#e377c2",    # magenta
    }
    
    linestyles = {
        "PhraseDP": "-",
        "InferDPT": "--",
        "SANTEXT+": "-.",
        "CusText+": ":",
        "CluSanT": (0, (3, 1, 1, 1)),
    }
    
    markers = {
        "PhraseDP": "o",
        "InferDPT": "s",
        "SANTEXT+": "^",
        "CusText+": "D",
        "CluSanT": "P",
    }
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    for mech in mechanisms:
        if mech not in merged_results:
            continue
        
        # Extract protection rates for this mechanism across epsilons
        y_values = []
        for e in epsilons:
            protection_rate = merged_results[mech].get(str(e), {}).get('overall', 0.0)
            y_values.append(protection_rate)
        
        # Plot the line
        plt.plot(
            epsilons,
            y_values,
            marker=markers.get(mech, 'o'),
            linestyle=linestyles.get(mech, '-'),
            linewidth=3.2,
            label=mech,
            color=colors.get(mech),
            markersize=7,
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
            zorder=3,
        )
    
    # Customize plot with larger fonts
    plt.title('Overall PII Protection Rate vs Epsilon (5 mechanisms)', fontsize=20, fontweight='bold')
    plt.xlabel('Epsilon (ε)', fontsize=18)
    plt.ylabel('Protection Rate', fontsize=18)
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.legend(frameon=True, fontsize=16)
    
    # Set tick font sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.tight_layout()
    
    # Save as PDF
    pdf_filename = 'overall_protection_vs_epsilon_all_mechanisms.pdf'
    plt.savefig(pdf_filename, bbox_inches='tight')
    
    # Copy to overleaf folder
    shutil.copy(pdf_filename, 'overleaf-folder/plots/')
    
    print(f"Created and copied {pdf_filename} with larger fonts to overleaf-folder/plots/")
    
    plt.close()

if __name__ == "__main__":
    print("Updating plots with pastel colors and larger fonts...")
    
    # Update bar plots with pastel colors
    create_pastel_epsilon_plots()
    
    # Create PDF version of overall protection plot
    create_overall_protection_pdf()
    
    print("All plots updated successfully!")
