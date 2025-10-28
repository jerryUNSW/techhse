#!/usr/bin/env python3
"""
HSE-bench Local Model Task Type Performance Bar Plot
====================================================

Create grouped bar plots showing local model performance across different
task types for three HSE-bench categories: Court Case, Regulation, and Safety Exam.

Author: Tech4HSE Team
Date: 2025-10-01
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import shutil

def get_hse_bench_local_task_results():
    """Get HSE-bench local model results by category and task type."""
    conn = sqlite3.connect('tech4hse_results.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            category,
            rule_recall_correct, rule_recall_total,
            rule_application_correct, rule_application_total,
            issue_spotting_correct, issue_spotting_total,
            rule_conclusion_correct, rule_conclusion_total
        FROM hse_bench_local_results
        ORDER BY 
            CASE category
                WHEN 'court_case' THEN 1
                WHEN 'regulation' THEN 2
                WHEN 'safety_exam' THEN 3
            END
    ''')
    
    results = {}
    for row in cursor.fetchall():
        category = row[0]
        
        # Calculate accuracies
        recall_acc = (row[1] / row[2] * 100) if row[2] > 0 else 0
        application_acc = (row[3] / row[4] * 100) if row[4] > 0 else 0
        spotting_acc = (row[5] / row[6] * 100) if row[6] > 0 else 0
        conclusion_acc = (row[7] / row[8] * 100) if row[8] > 0 else 0
        
        # Format category name
        if category == 'court_case':
            display_name = 'Court Case'
        elif category == 'regulation':
            display_name = 'Regulation'
        elif category == 'safety_exam':
            display_name = 'Safety Exam'
        else:
            display_name = category
        
        results[display_name] = {
            'Rule Recall': recall_acc,
            'Rule Application': application_acc,
            'Issue Spotting': spotting_acc,
            'Rule Conclusion': conclusion_acc
        }
    
    conn.close()
    return results

def create_hse_bench_task_type_plot():
    """Create grouped bar plot for HSE-bench task types."""
    # Get data
    results = get_hse_bench_local_task_results()
    
    # Categories and task types
    categories = ['Court Case', 'Regulation', 'Safety Exam']
    task_types = ['Rule Recall', 'Rule Application', 'Issue Spotting', 'Rule Conclusion']
    
    # Color palette - consistent with our style
    colors = [
        '#4169E1',  # Blue - Rule Recall
        '#FF6B6B',  # Red - Rule Application
        '#90EE90',  # Green - Issue Spotting
        '#FFE55C',  # Yellow - Rule Conclusion
    ]
    
    # Apply plot settings
    def plt_settings():
        plt.style.use('default')
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams["legend.framealpha"] = 0
        plt.rcParams["legend.handletextpad"] = 0.1
        plt.rcParams["legend.columnspacing"] = 0.2
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams['pdf.fonttype'] = 42
    
    plt_settings()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up bar positions
    x = np.arange(len(categories))
    width = 0.2  # Width of each bar
    
    # Plot bars for each task type
    for i, task_type in enumerate(task_types):
        accuracies = [results[cat][task_type] for cat in categories]
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, accuracies, width, 
                     label=task_type,
                     color=colors[i],
                     edgecolor='black',
                     linewidth=0.5,
                     alpha=0.9,
                     zorder=3)
        
        # Add value labels on top of bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1.5,
                   f'{acc:.1f}%',
                   ha='center', va='bottom',
                   fontsize=11, fontweight='bold',
                   color='black')
    
    # Customize plot
    ax.set_title('HSE-bench Local Model Performance by Task Type', 
                fontsize=20, fontweight='bold', pad=20,
                color='black')
    ax.set_ylabel('Accuracy (%)', fontsize=22, fontweight='bold', color='black')
    ax.set_ylim(50, 110)
    
    # Set x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=20, fontweight='bold', color='black')
    
    # Grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=1.0, color='gray')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', fontsize=16, frameon=False, ncol=4)
    
    # Clean appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # Background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Tick colors
    ax.tick_params(colors='black', which='both', labelsize=14)
    
    # Bring axes to front
    ax.set_axisbelow(False)
    ax.spines['bottom'].set_zorder(10)
    ax.spines['left'].set_zorder(10)
    
    plt.tight_layout()
    
    # Save plot
    filename_pdf = 'hse_bench_local_task_types.pdf'
    plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white')
    
    # Copy to overleaf folder
    shutil.copy(filename_pdf, 'overleaf-folder/plots/')
    
    print(f"Created and copied {filename_pdf} to overleaf-folder/plots/")
    
    # Print statistics
    print("\nHSE-bench Local Model Task Type Performance:")
    for category in categories:
        print(f"\n{category}:")
        for task_type in task_types:
            print(f"  {task_type}: {results[category][task_type]:.1f}%")
    
    plt.close()

if __name__ == "__main__":
    print("Creating HSE-bench local model task type performance plot...")
    create_hse_bench_task_type_plot()
    print("Plot created successfully!")

