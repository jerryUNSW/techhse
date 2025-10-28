#!/usr/bin/env python3
"""
MedQA UME Medical Improvement Bar Plots - Combined Old and New Results
====================================================================

Create bar plots showing MedQA UME results with medical improvement data.
Combines old 500 questions (from medqa_detailed_results) with new 500 questions 
(from medqa_results) for comprehensive analysis.

Author: Tech4HSE Team
Date: 2025-01-30
"""

import sqlite3
import matplotlib.pyplot as plt
import numpy as np
import shutil

def get_combined_medqa_results():
    """Get combined MedQA results from both old and new experiments."""
    conn = sqlite3.connect('tech4hse_results.db')
    cursor = conn.cursor()
    
    results = {}
    
    for epsilon in [3.0]:  # Focus on epsilon 3.0 for now
        # Get old results from medqa_detailed_results (first 500 questions)
        cursor.execute('''
            SELECT mechanism, 
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_old,
                   COUNT(*) as total_old
            FROM medqa_detailed_results 
            WHERE epsilon = ?
            GROUP BY mechanism
        ''', (epsilon,))
        
        old_results = {}
        for mechanism, correct, total in cursor.fetchall():
            accuracy = (correct / total * 100) if total > 0 else 0
            
            # Clean up mechanism names
            if 'Purely Local Model' in mechanism:
                clean_name = 'Local'
            elif 'Non-Private Local Model' in mechanism:
                clean_name = 'Local + CoT'
            elif 'Purely Remote Model' in mechanism:
                clean_name = 'Remote'
            elif 'Old Phrase DP' in mechanism:
                clean_name = 'PhraseDP'
            elif 'InferDPT' in mechanism:
                clean_name = 'InferDPT'
            elif 'SANTEXT+' in mechanism:
                clean_name = 'SANTEXT+'
            else:
                clean_name = mechanism
            
            old_results[clean_name] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
        
        # Get new results from medqa_results (next 500 questions)
        cursor.execute('''
            SELECT mechanism, 
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_new,
                   COUNT(*) as total_new
            FROM medqa_results 
            WHERE experiment_id = 7 AND epsilon = ?
            GROUP BY mechanism
        ''', (epsilon,))
        
        new_results = {}
        for mechanism, correct, total in cursor.fetchall():
            accuracy = (correct / total * 100) if total > 0 else 0
            
            # Clean up mechanism names
            if 'Local Model' in mechanism:
                clean_name = 'Local'
            elif 'Local + CoT' in mechanism:
                clean_name = 'Local + CoT'
            elif 'PhraseDP + CoT' in mechanism:
                clean_name = 'PhraseDP + CoT'
            else:
                clean_name = mechanism
            
            new_results[clean_name] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            }
        
        # Combine old and new results
        combined_results = {}
        
        # Get all mechanisms that appear in either old or new results
        all_mechanisms = set(old_results.keys()) | set(new_results.keys())
        
        for mechanism in all_mechanisms:
            old_data = old_results.get(mechanism, {'correct': 0, 'total': 0, 'accuracy': 0})
            new_data = new_results.get(mechanism, {'correct': 0, 'total': 0, 'accuracy': 0})
            
            # Combine the counts
            total_correct = old_data['correct'] + new_data['correct']
            total_questions = old_data['total'] + new_data['total']
            combined_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
            
            combined_results[mechanism] = {
                'correct': total_correct,
                'total': total_questions,
                'accuracy': combined_accuracy,
                'old_correct': old_data['correct'],
                'old_total': old_data['total'],
                'new_correct': new_data['correct'],
                'new_total': new_data['total']
            }
        
        results[epsilon] = combined_results
    
    conn.close()
    return results

def get_medical_improvement_results():
    """Get medical improvement results from database."""
    try:
        conn = sqlite3.connect('tech4hse_results.db')
        cursor = conn.cursor()
        
        # Get improvement rates by epsilon
        cursor.execute("""
            SELECT 
                epsilon,
                COUNT(*) as total_questions,
                SUM(CASE WHEN improvement = 1 THEN 1 ELSE 0 END) as improvements,
                ROUND(AVG(CASE WHEN improvement = 1 THEN 1.0 ELSE 0.0 END) * 100, 1) as improvement_rate
            FROM medical_improvement_results 
            GROUP BY epsilon 
            ORDER BY epsilon
        """)
        
        results = {}
        for row in cursor.fetchall():
            epsilon, total, improvements, rate = row
            results[epsilon] = {
                'total_questions': total,
                'improvements': improvements,
                'improvement_rate': rate
            }
        
        conn.close()
        return results
        
    except Exception as e:
        print(f"Error getting medical improvement results: {e}")
        return {}

def calculate_medical_mode_accuracy(original_accuracy, improvement_rate):
    """Calculate medical mode accuracy based on original accuracy and improvement rate."""
    wrong_questions = 100 - original_accuracy
    fixed_questions = (improvement_rate / 100) * wrong_questions
    new_accuracy = original_accuracy + fixed_questions
    return min(new_accuracy, 100.0)  # Cap at 100%

def create_combined_medical_improvement_plot():
    """Create combined plot showing medical improvement results."""
    # Get combined results
    combined_results = get_combined_medqa_results()
    
    # Get medical improvement data
    improvement_data = get_medical_improvement_results()
    
    # Define the desired order from left to right
    desired_order = [
        'Local',
        'InferDPT', 
        'SANTEXT+',
        'PhraseDP',
        'PhraseDP+',
        'Local + CoT',
        'Remote'
    ]
    
    # Color palette
    colors = [
        '#FFE55C',  # Bright light yellow - Local
        '#D2691E',  # Muted earthy orange - InferDPT
        '#90EE90',  # Medium desaturated green - SANTEXT+
        '#4169E1',  # Vibrant medium blue - PhraseDP
        '#FF6B6B',  # Bright red - PhraseDP+ (medical mode)
        '#191970',  # Deep dark blue - Local + CoT
        '#808080'   # Grey - Remote (non-private baseline)
    ]
    
    # Apply plot settings
    def plt_settings():
        plt.style.use('default')
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams["legend.framealpha"] = 0
        plt.rcParams["legend.handletextpad"] = 0.1
        plt.rcParams["legend.columnspacing"] = 0.2
        plt.rcParams["figure.figsize"] = (7,5)
        plt.rcParams['pdf.fonttype'] = 42
    
    plt_settings()
    
    # Create plot for epsilon 3.0
    epsilon = 3.0
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Get mechanisms and accuracies in the desired order
    mechanisms = []
    accuracies = []
    
    if epsilon in combined_results:
        for mech in desired_order:
            if mech == 'PhraseDP+':
                # Calculate medical mode accuracy
                if 'PhraseDP' in combined_results[epsilon]:
                    original_phrasedp = combined_results[epsilon]['PhraseDP']['accuracy']
                    if epsilon in improvement_data:
                        improvement_rate = improvement_data[epsilon]['improvement_rate']
                        medical_accuracy = calculate_medical_mode_accuracy(original_phrasedp, improvement_rate)
                        mechanisms.append(mech)
                        accuracies.append(medical_accuracy)
                    else:
                        # Fallback to original if no improvement data
                        mechanisms.append(mech)
                        accuracies.append(original_phrasedp)
                else:
                    continue
            elif mech in combined_results[epsilon]:
                mechanisms.append(mech)
                accuracies.append(combined_results[epsilon][mech]['accuracy'])
    
    # Create bars
    bars = ax.bar(range(len(mechanisms)), accuracies, 
                 color=colors[:len(mechanisms)],
                 edgecolor='black',
                 linewidth=0.5,
                 alpha=0.9,
                 zorder=3,
                 width=0.6)
    
    # Customize plot
    ax.set_title(f'MedQA UME Medical Improvement - Epsilon = {epsilon}\\n(Combined: Old 500 + New 500 Questions)', 
                fontsize=18, fontweight='bold', pad=20,
                color='black')
    ax.set_ylabel('Accuracy (%)', fontsize=20, fontweight='bold', color='black')
    ax.set_ylim(0, 100)
    
    # Set x-axis labels
    ax.set_xticks(range(len(mechanisms)))
    ax.set_xticklabels(mechanisms, rotation=45, ha='right', fontsize=18, fontweight='bold', color='black')
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=1.0, color='gray')
    ax.set_axisbelow(True)
    
    # Add value labels on top of bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
               f'{acc:.1f}%', 
               ha='center', va='bottom', 
               fontsize=15, fontweight='bold',
               color='black')
    
    # Clean appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(3)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    # No background color
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Set tick colors
    ax.tick_params(colors='black', which='both', labelsize=20)
    
    # Bring x-axis to front
    ax.set_axisbelow(False)
    ax.spines['bottom'].set_zorder(10)
    ax.spines['left'].set_zorder(10)
    
    plt.tight_layout()
    
    # Save plot
    filename_pdf = f'medqa_medical_improvement_epsilon_{epsilon}.pdf'
    
    plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white')
    
    # Copy to overleaf folder
    shutil.copy(filename_pdf, 'overleaf-folder/plots/')
    
    print(f"Created and copied {filename_pdf} with combined old+new results to overleaf-folder/plots/")
    
    # Print combined statistics
    if epsilon in combined_results:
        print(f"\\nCombined Results for Epsilon {epsilon} (Old 500 + New 500):")
        for mech, data in combined_results[epsilon].items():
            print(f"  {mech}: {data['correct']}/{data['total']} = {data['accuracy']:.1f}%")
            print(f"    Old: {data['old_correct']}/{data['old_total']}, New: {data['new_correct']}/{data['new_total']}")
    
    # Print improvement statistics
    if epsilon in improvement_data:
        data = improvement_data[epsilon]
        print(f"\\nMedical Improvement: {data['improvements']}/{data['total_questions']} questions improved ({data['improvement_rate']}%)")
    
    plt.close()

if __name__ == "__main__":
    print("Creating combined MedQA UME medical improvement plot...")
    create_combined_medical_improvement_plot()
    print("Combined MedQA UME medical improvement plot created successfully!")
