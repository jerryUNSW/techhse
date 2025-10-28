#!/usr/bin/env python3

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def get_medqa_epsilon_trend():
    """Get MedQA results organized by epsilon for trend plotting."""
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
            
            # Clean up mechanism names for plotting
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

def plot_line(a, data, color, linestyle, marker, label):
    """Plot line function from sample plot."""
    plt.plot(a, data, color=color, linestyle=linestyle, markersize=15, 
             markeredgewidth=1.5, markerfacecolor='none', marker=marker, label=label)

def plt_settings():
    """Plot settings from sample plot."""
    plt.style.use('default')
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams["legend.framealpha"] = 0
    plt.rcParams["legend.handletextpad"] = 0.1
    plt.rcParams["legend.columnspacing"] = 0.2
    plt.rcParams["figure.figsize"] = (10, 6)  # Optimal size from sample plot
    plt.rcParams['pdf.fonttype'] = 42

def create_epsilon_trend_plot():
    """Create epsilon trend plot for MedQA results."""
    print("Creating MedQA epsilon trend plot...")
    
    # Get results
    results = get_medqa_epsilon_trend()
    
    # Epsilon values (matching sample plot)
    epsilons = [1.0, 2.0, 3.0]
    
    # Mechanism order and styling (matching sample plot approach)
    mechanisms = ['Purely Local', 'Non-Private + CoT', 'PhraseDP', 'InferDPT', 'SANTEXT+', 'Purely Remote']
    colors = ['#cc3333', '#e95814', '#236133', '#4169E1', '#191970', '#808080']  # Grey for Remote baseline
    markers = ['^', 'o', 's', 's', 's', 's']
    linestyles = ['-', '-', '-', '-', '-', '-']
    
    plt_settings()
    
    # Plot lines for each mechanism
    for i, mechanism in enumerate(mechanisms):
        if mechanism in results[1.0]:  # Check if mechanism exists
            accuracies = []
            for eps in epsilons:
                if mechanism in results[eps]:
                    accuracies.append(results[eps][mechanism])
                else:
                    accuracies.append(0)  # Default if missing
            
            plot_line(epsilons, accuracies, colors[i], linestyles[i], markers[i], mechanism)
    
    # Customize plot
    plt.xticks(epsilons, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel(r"$\epsilon$", fontsize=20)
    plt.ylabel("Accuracy (%)", fontsize=20)
    plt.ylim(0, 100)
    
    # Legend
    plt.legend(fontsize=15, ncol=2, loc="upper right")
    plt.rcParams["legend.columnspacing"] = 0.3
    plt.subplots_adjust(top=0.9, left=0.15, bottom=0.15)
    
    # Save plot
    plt.savefig("medqa_epsilon_trend.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Copy to overleaf folder
    overleaf_path = Path("overleaf-folder/plots/medqa_epsilon_trend.pdf")
    overleaf_path.parent.mkdir(parents=True, exist_ok=True)
    
    import shutil
    shutil.copy2("medqa_epsilon_trend.pdf", overleaf_path)
    
    print(f"MedQA epsilon trend plot saved as medqa_epsilon_trend.pdf")
    print(f"Copied to {overleaf_path}")

if __name__ == "__main__":
    create_epsilon_trend_plot()

