#!/usr/bin/env python3
"""
Privacy Evaluation Visualization
===============================

This script creates comprehensive plots to document the privacy evaluation results
for Phrase DP vs InferDPT methods, showing the privacy-utility trade-off.

Author: Tech4HSE Team
Date: 2025-08-26
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.patches import Rectangle
import seaborn as sns

# Set style for better-looking plots
plt.style.use('default')
sns.set_palette("husl")

class PrivacyVisualizer:
    """Visualizer for privacy evaluation results."""
    
    def __init__(self):
        # Load the privacy evaluation results
        with open('quick_privacy_results.json', 'r') as f:
            self.privacy_data = json.load(f)
        
        # Load the performance data from our earlier analysis
        self.performance_data = {
            'phrase_dp': {'accuracy': 83.91, 'privacy': 0.295},
            'inferdpt': {'accuracy': 71.20, 'privacy': 0.884}
        }
        
        # Colors for the methods
        self.colors = {
            'phrase_dp': '#FF6B6B',  # Red for Phrase DP
            'inferdpt': '#4ECDC4'    # Teal for InferDPT
        }
        
        # Method names for display
        self.method_names = {
            'phrase_dp': 'Phrase DP',
            'inferdpt': 'InferDPT'
        }
    
    def create_comprehensive_plot(self):
        """Create a comprehensive plot showing all privacy evaluation results."""
        fig = plt.figure(figsize=(20, 16))
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Main Privacy-Utility Trade-off Plot
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_privacy_utility_tradeoff(ax1)
        
        # 2. Individual Attack Comparison
        ax2 = fig.add_subplot(gs[0, 2])
        self.plot_attack_comparison(ax2)
        
        # 3. Privacy Scores Breakdown
        ax3 = fig.add_subplot(gs[1, :])
        self.plot_privacy_breakdown(ax3)
        
        # 4. Method Comparison Radar Chart
        ax4 = fig.add_subplot(gs[2, :])
        self.plot_radar_comparison(ax4)
        
        # Add title
        fig.suptitle('Privacy Evaluation Results: Phrase DP vs InferDPT (ε=1)\n'
                    'MedQA Dataset - 10 Questions Analysis', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        # Add epsilon note
        fig.text(0.02, 0.02, 'Note: All evaluations performed with ε=1 (privacy parameter)', 
                fontsize=12, style='italic', color='gray')
        
        plt.tight_layout()
        plt.savefig('plots/privacy_evaluation_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comprehensive plot saved as: plots/privacy_evaluation_comprehensive.png")
    
    def plot_privacy_utility_tradeoff(self, ax):
        """Plot the privacy-utility trade-off."""
        methods = ['phrase_dp', 'inferdpt']
        accuracies = [self.performance_data[m]['accuracy'] for m in methods]
        privacy_scores = [self.performance_data[m]['privacy'] for m in methods]
        
        # Create scatter plot
        for i, method in enumerate(methods):
            ax.scatter(privacy_scores[i], accuracies[i], 
                      s=300, c=self.colors[method], alpha=0.8, 
                      label=self.method_names[method], edgecolors='black', linewidth=2)
            
            # Add method labels
            ax.annotate(self.method_names[method], 
                       (privacy_scores[i], accuracies[i]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Add trade-off arrows and annotations
        ax.annotate('Better Privacy\n(Lower Recovery)', 
                   xy=(0.5, 75), xytext=(0.3, 85),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'),
                   fontsize=10, ha='center')
        
        ax.annotate('Better Utility\n(Higher Accuracy)', 
                   xy=(0.3, 85), xytext=(0.1, 90),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray'),
                   fontsize=10, ha='center')
        
        # Add privacy zones
        ax.axvspan(0, 0.3, alpha=0.1, color='red', label='Poor Privacy')
        ax.axvspan(0.3, 0.6, alpha=0.1, color='orange', label='Moderate Privacy')
        ax.axvspan(0.6, 1.0, alpha=0.1, color='green', label='Good Privacy')
        
        ax.set_xlabel('Privacy Protection Score\n(Higher = Better Privacy)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('Privacy-Utility Trade-off Analysis\n(ε=1)', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(65, 95)
    
    def plot_attack_comparison(self, ax):
        """Plot comparison of individual attack methods."""
        attacks = ['BERT Inference', 'Embedding Inversion', 'GPT Inference']
        phrase_dp_scores = [
            self.privacy_data['privacy_results']['phrase_dp']['bert_inference_attack'],
            self.privacy_data['privacy_results']['phrase_dp']['embedding_inversion_attack'],
            self.privacy_data['privacy_results']['phrase_dp']['gpt_inference_attack']
        ]
        inferdpt_scores = [
            self.privacy_data['privacy_results']['inferdpt']['bert_inference_attack'],
            self.privacy_data['privacy_results']['inferdpt']['embedding_inversion_attack'],
            self.privacy_data['privacy_results']['inferdpt']['gpt_inference_attack']
        ]
        
        x = np.arange(len(attacks))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, phrase_dp_scores, width, label='Phrase DP', 
                      color=self.colors['phrase_dp'], alpha=0.8)
        bars2 = ax.bar(x + width/2, inferdpt_scores, width, label='InferDPT', 
                      color=self.colors['inferdpt'], alpha=0.8)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Attack Methods', fontsize=12, fontweight='bold')
        ax.set_ylabel('Privacy Protection Score', fontsize=12, fontweight='bold')
        ax.set_title('Individual Attack Comparison\n(Higher = Better Privacy)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attacks, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add improvement annotations
        for i, (pd, id) in enumerate(zip(phrase_dp_scores, inferdpt_scores)):
            improvement = ((id - pd) / pd) * 100 if pd > 0 else 0
            ax.annotate(f'+{improvement:.0f}%', 
                       xy=(i, max(pd, id) + 0.05), ha='center', va='bottom',
                       fontsize=10, fontweight='bold', color='darkgreen')
    
    def plot_privacy_breakdown(self, ax):
        """Plot detailed privacy breakdown."""
        methods = ['Phrase DP', 'InferDPT']
        attacks = ['BERT Inference', 'Embedding Inversion', 'GPT Inference']
        
        # Extract data
        phrase_dp_data = [
            self.privacy_data['privacy_results']['phrase_dp']['bert_inference_attack'],
            self.privacy_data['privacy_results']['phrase_dp']['embedding_inversion_attack'],
            self.privacy_data['privacy_results']['phrase_dp']['gpt_inference_attack']
        ]
        inferdpt_data = [
            self.privacy_data['privacy_results']['inferdpt']['bert_inference_attack'],
            self.privacy_data['privacy_results']['inferdpt']['embedding_inversion_attack'],
            self.privacy_data['privacy_results']['inferdpt']['gpt_inference_attack']
        ]
        
        # Create stacked bar chart
        x = np.arange(len(attacks))
        width = 0.8
        
        bars1 = ax.bar(x, phrase_dp_data, width, label='Phrase DP', 
                      color=self.colors['phrase_dp'], alpha=0.7)
        bars2 = ax.bar(x, inferdpt_data, width, bottom=phrase_dp_data, label='InferDPT', 
                      color=self.colors['inferdpt'], alpha=0.7)
        
        # Add value labels
        for i, (pd, id) in enumerate(zip(phrase_dp_data, inferdpt_data)):
            ax.text(i, pd/2, f'{pd:.2f}', ha='center', va='center', fontweight='bold', color='white')
            ax.text(i, pd + id/2, f'{id:.2f}', ha='center', va='center', fontweight='bold', color='white')
        
        ax.set_xlabel('Attack Methods', fontsize=14, fontweight='bold')
        ax.set_ylabel('Privacy Protection Score', fontsize=14, fontweight='bold')
        ax.set_title('Privacy Protection Breakdown by Attack Method\n(ε=1)', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attacks)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add privacy level annotations
        ax.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Poor Privacy Threshold')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='Moderate Privacy Threshold')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Good Privacy Threshold')
    
    def plot_radar_comparison(self, ax):
        """Plot radar chart comparison of all metrics."""
        # Define metrics
        metrics = ['Accuracy', 'BERT Privacy', 'Embedding Privacy', 'GPT Privacy', 'Overall Privacy']
        
        # Extract data
        phrase_dp_values = [
            self.performance_data['phrase_dp']['accuracy'] / 100,  # Normalize accuracy
            self.privacy_data['privacy_results']['phrase_dp']['bert_inference_attack'],
            self.privacy_data['privacy_results']['phrase_dp']['embedding_inversion_attack'],
            self.privacy_data['privacy_results']['phrase_dp']['gpt_inference_attack'],
            self.privacy_data['overall_scores']['phrase_dp']
        ]
        
        inferdpt_values = [
            self.performance_data['inferdpt']['accuracy'] / 100,  # Normalize accuracy
            self.privacy_data['privacy_results']['inferdpt']['bert_inference_attack'],
            self.privacy_data['privacy_results']['inferdpt']['embedding_inversion_attack'],
            self.privacy_data['privacy_results']['inferdpt']['gpt_inference_attack'],
            self.privacy_data['overall_scores']['inferdpt']
        ]
        
        # Number of variables
        N = len(metrics)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add the first value to the end to close the plot
        phrase_dp_values += phrase_dp_values[:1]
        inferdpt_values += inferdpt_values[:1]
        
        # Create the plot
        ax.plot(angles, phrase_dp_values, 'o-', linewidth=2, label='Phrase DP', 
               color=self.colors['phrase_dp'])
        ax.fill(angles, phrase_dp_values, alpha=0.25, color=self.colors['phrase_dp'])
        
        ax.plot(angles, inferdpt_values, 'o-', linewidth=2, label='InferDPT', 
               color=self.colors['inferdpt'])
        ax.fill(angles, inferdpt_values, alpha=0.25, color=self.colors['inferdpt'])
        
        # Set the labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Set title
        ax.set_title('Comprehensive Method Comparison\n(ε=1)', fontsize=16, fontweight='bold')
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        
        # Add grid
        ax.grid(True)
    
    def create_summary_plot(self):
        """Create a simple summary plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Privacy-Utility Trade-off
        methods = ['Phrase DP', 'InferDPT']
        accuracies = [83.91, 71.20]
        privacy_scores = [0.295, 0.884]
        
        scatter = ax1.scatter(privacy_scores, accuracies, 
                            s=400, c=['#FF6B6B', '#4ECDC4'], alpha=0.8, 
                            edgecolors='black', linewidth=2)
        
        # Add labels
        for i, method in enumerate(methods):
            ax1.annotate(method, (privacy_scores[i], accuracies[i]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Privacy Protection Score (Higher = Better)', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax1.set_title('Privacy-Utility Trade-off (ε=1)', fontsize=16, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(65, 95)
        
        # Plot 2: Attack Comparison
        attacks = ['BERT', 'Embedding', 'GPT']
        phrase_dp = [0.237, 0.336, 0.312]
        inferdpt = [0.993, 0.704, 0.955]
        
        x = np.arange(len(attacks))
        width = 0.35
        
        ax2.bar(x - width/2, phrase_dp, width, label='Phrase DP', color='#FF6B6B', alpha=0.8)
        ax2.bar(x + width/2, inferdpt, width, label='InferDPT', color='#4ECDC4', alpha=0.8)
        
        ax2.set_xlabel('Attack Methods', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Privacy Protection Score', fontsize=14, fontweight='bold')
        ax2.set_title('Privacy Protection by Attack Method (ε=1)', fontsize=16, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(attacks)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add improvement percentages
        for i, (pd, id) in enumerate(zip(phrase_dp, inferdpt)):
            improvement = ((id - pd) / pd) * 100 if pd > 0 else 0
            ax2.annotate(f'+{improvement:.0f}%', 
                        xy=(i, max(pd, id) + 0.05), ha='center', va='bottom',
                        fontsize=12, fontweight='bold', color='darkgreen')
        
        plt.tight_layout()
        plt.savefig('plots/privacy_evaluation_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Summary plot saved as: plots/privacy_evaluation_summary.png")

def main():
    """Main function to create visualizations."""
    visualizer = PrivacyVisualizer()
    
    # Create comprehensive plot
    print("Creating comprehensive privacy evaluation plot...")
    visualizer.create_comprehensive_plot()
    
    # Create summary plot
    print("Creating summary privacy evaluation plot...")
    visualizer.create_summary_plot()
    
    print("\nVisualization complete!")
    print("Files created:")
    print("- plots/privacy_evaluation_comprehensive.png")
    print("- plots/privacy_evaluation_summary.png")

if __name__ == "__main__":
    main()
