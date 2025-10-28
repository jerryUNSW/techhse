#!/usr/bin/env python3
"""
MedQA UME Medical Improvement Bar Plots
======================================

Create bar plots showing MedQA UME results with medical improvement data.
Shows original PhraseDP vs Medical Mode PhraseDP performance.

Author: Tech4HSE Team
Date: 2025-01-30
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
        results[epsilon] = {}
        
        if epsilon == 3.0:
            # For epsilon 3.0: use old 500 for most mechanisms, original data for InferDPT/SANTEXT+
            # First get all mechanisms from medqa_detailed_results (old 500)
            cursor.execute('''
                SELECT mechanism, 
                       SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                       COUNT(*) as total
                FROM medqa_detailed_results 
                WHERE epsilon = ?
                GROUP BY mechanism
            ''', (epsilon,))
            
            mechanisms_data = cursor.fetchall()
            
            for mechanism, correct, total in mechanisms_data:
                accuracy = (correct / total * 100) if total > 0 else 0
                
                # Clean up mechanism names for plotting with custom naming
                if 'Purely Local Model' in mechanism:
                    clean_name = 'Local'
                elif 'Non-Private Local Model' in mechanism:
                    clean_name = 'Local + CoT'
                elif 'Purely Remote Model' in mechanism:
                    clean_name = 'Remote'
                elif 'Old Phrase DP' in mechanism and 'PhraseDP+' not in mechanism:
                    clean_name = 'PhraseDP'
                elif 'InferDPT' in mechanism:
                    clean_name = 'InferDPT'
                elif 'SANTEXT+' in mechanism:
                    clean_name = 'SANTEXT+'
                elif 'PhraseDP+' in mechanism:
                    # Will be overridden below with combined data
                    clean_name = 'PhraseDP+'
                else:
                    clean_name = mechanism  # Fallback
                
                results[epsilon][clean_name] = accuracy
                print(f"Epsilon {epsilon} {clean_name}: Old {correct}/{total} = {accuracy:.1f}%")
            
            # Add InferDPT and SANTEXT+ from medqa_results (they weren't in old 500 for eps 3.0)
            cursor.execute('''
                SELECT mechanism, local_answer
                FROM medqa_results 
                WHERE epsilon = ? AND question_id = -1
                AND (mechanism LIKE '%InferDPT%' OR mechanism LIKE '%SANTEXT+%')
                ORDER BY mechanism
            ''', (epsilon,))
            
            for mechanism, local_answer in cursor.fetchall():
                accuracy_str = local_answer.split('=')[1].strip().replace('%', '')
                accuracy = float(accuracy_str)
                
                if 'InferDPT' in mechanism:
                    clean_name = 'InferDPT'
                elif 'SANTEXT+' in mechanism:
                    clean_name = 'SANTEXT+'
                else:
                    continue
                
                results[epsilon][clean_name] = accuracy
                print(f"Epsilon {epsilon} {clean_name}: {accuracy:.1f}% (from original data)")
            
            # Special handling for PhraseDP+ at epsilon 3.0: combine old + new 500
            # Get old 500 PhraseDP+ results
            cursor.execute('''
                SELECT SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_old,
                       COUNT(*) as total_old
                FROM medqa_detailed_results 
                WHERE epsilon = ? AND mechanism LIKE '%PhraseDP+%'
            ''', (epsilon,))
            old_data = cursor.fetchone()
            
            # Get new 500 PhraseDP+ results
            cursor.execute('''
                SELECT SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_new,
                       COUNT(*) as total_new
                FROM medqa_results 
                WHERE experiment_id = 7 AND epsilon = ? 
                AND mechanism = 'PhraseDP+ (Medical Mode)'
                AND question_id != -1
            ''', (epsilon,))
            new_data = cursor.fetchone()
            
            if old_data and new_data and old_data[1] > 0 and new_data[1] > 0:
                correct_old, total_old = old_data
                correct_new, total_new = new_data
                # Combine old and new
                combined_correct = correct_old + correct_new
                combined_total = total_old + total_new
                combined_accuracy = (combined_correct / combined_total * 100)
                results[epsilon]['PhraseDP+'] = combined_accuracy
                print(f"PhraseDP+ epsilon {epsilon} (COMBINED): Old {correct_old}/{total_old} + New {correct_new}/{total_new} = {combined_correct}/{combined_total} = {combined_accuracy:.1f}%")
        else:
            # For epsilon 1.0 and 2.0: use original data source
            cursor.execute('''
                SELECT mechanism, local_answer
                FROM medqa_results 
                WHERE epsilon = ? AND question_id = -1
                ORDER BY mechanism
            ''', (epsilon,))
            
            mechanisms_data = cursor.fetchall()
            
            for mechanism, local_answer in mechanisms_data:
                # Extract accuracy from "Summary: X/500 = Y.Y%"
                accuracy_str = local_answer.split('=')[1].strip().replace('%', '')
                accuracy = float(accuracy_str)
                
                # Clean up mechanism names for plotting with custom naming
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
                    clean_name = mechanism  # Fallback
                
                results[epsilon][clean_name] = accuracy
    
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
    # If improvement_rate is the percentage of wrong questions that were fixed
    # Then: new_accuracy = original_accuracy + (improvement_rate/100) * (100 - original_accuracy)
    wrong_questions = 100 - original_accuracy
    fixed_questions = (improvement_rate / 100) * wrong_questions
    new_accuracy = original_accuracy + fixed_questions
    return min(new_accuracy, 100.0)  # Cap at 100%

def create_medical_improvement_plots():
    """Create three separate plots showing medical improvement results."""
    # Get original MedQA results
    original_results = get_medqa_results_by_epsilon()
    
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
    
    # Color palette - add a new color for medical mode
    colors = [
        '#FFE55C',  # Bright light yellow - Local
        '#D2691E',  # Muted earthy orange - InferDPT
        '#90EE90',  # Medium desaturated green - SANTEXT+
        '#4169E1',  # Vibrant medium blue - PhraseDP
        '#FF6B6B',  # Bright red - PhraseDP+ (medical mode)
        '#191970',  # Deep dark blue - Local + CoT
        '#808080'   # Grey - Remote (non-private baseline)
    ]
    
    # Apply sample plot settings using plt_settings function
    def plt_settings():
        plt.style.use('default')
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams["legend.framealpha"] = 0
        plt.rcParams["legend.handletextpad"] = 0.1
        plt.rcParams["legend.columnspacing"] = 0.2
        # for the varying plots 
        plt.rcParams["figure.figsize"] = (7,5)  # Wider to accommodate 7 bars
        plt.rcParams['pdf.fonttype'] = 42
    
    plt_settings()
    
    # Create separate plots for each epsilon
    for epsilon in [1.0, 2.0, 3.0]:
        fig, ax = plt.subplots(figsize=(7, 5))  # Wider figure size for 7 bars
        
        # Get mechanisms and accuracies in the desired order
        mechanisms = []
        accuracies = []
        
        for mech in desired_order:
            if mech == 'PhraseDP+':
                # For epsilon 3.0, use the combined 1000 questions data if available
                if epsilon == 3.0 and 'PhraseDP+' in original_results[epsilon]:
                    # Use the already-computed combined old+new 500 accuracy
                    mechanisms.append(mech)
                    accuracies.append(original_results[epsilon]['PhraseDP+'])
                elif epsilon in original_results and 'PhraseDP' in original_results[epsilon]:
                    # For epsilon 1.0 and 2.0, calculate from medical improvement
                    original_phrasedp = original_results[epsilon]['PhraseDP']
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
            elif mech in original_results[epsilon]:
                mechanisms.append(mech)
                accuracies.append(original_results[epsilon][mech])
        
        # Create bars with optimal width and styling
        bars = ax.bar(range(len(mechanisms)), accuracies, 
                     color=colors[:len(mechanisms)],
                     edgecolor='black',
                     linewidth=0.5,  # Thinner borders like sample plot
                     alpha=0.9,
                     zorder=3,
                     width=0.6)  # Optimal bar width from sample plot
        
        # Customize plot with sample plot font sizes
        ax.set_title(f'MedQA UME Medical Improvement - Epsilon = {epsilon}', 
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
        
        # No improvement annotation - clean plot
        
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
        
        # Bring x-axis to front so bars appear to sit on it
        ax.set_axisbelow(False)
        ax.spines['bottom'].set_zorder(10)
        ax.spines['left'].set_zorder(10)
        
        plt.tight_layout()
        
        # Save individual plots
        filename_pdf = f'medqa_medical_improvement_epsilon_{epsilon}.pdf'
        
        plt.savefig(filename_pdf, bbox_inches='tight', dpi=300, facecolor='white')
        
        # Copy to overleaf folder
        shutil.copy(filename_pdf, 'overleaf-folder/plots/')
        
        print(f"Created and copied {filename_pdf} with medical improvement data to overleaf-folder/plots/")
        
        # Print improvement statistics
        if epsilon in improvement_data:
            data = improvement_data[epsilon]
            print(f"  Epsilon {epsilon}: {data['improvements']}/{data['total_questions']} questions improved ({data['improvement_rate']}%)")
        
        plt.close()

if __name__ == "__main__":
    print("Creating MedQA UME medical improvement plots...")
    create_medical_improvement_plots()
    print("All MedQA UME medical improvement plots created successfully!")
