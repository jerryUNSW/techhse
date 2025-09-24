#!/usr/bin/env python3
"""
Create a summary visualization showing the key insights from the epsilon comparison analysis.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_epsilon_results(filename):
    """Load the epsilon comparison results from JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def create_summary_visualization():
    """Create a comprehensive summary visualization."""
    print("📊 Creating summary visualization...")
    
    # Load results
    results = load_epsilon_results('epsilon_comparison_results_20250920_144014.json')
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Epsilon Comparison Analysis: Old vs New Phrase DP Methods', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Extract data
    epsilon_values = results['epsilon_values']
    questions = results['questions']
    
    # Collect all data
    old_similarities = []
    new_similarities = []
    old_ranges = []
    new_ranges = []
    question_names = []
    epsilons = []
    
    for q_idx, question_data in enumerate(questions):
        question_text = question_data['question_text']
        question_names.append(f"Q{q_idx+1}: {question_text[:30]}...")
        
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                epsilon = eps_test['epsilon']
                
                old_sim = eps_test['old_method']['similarity_to_original']
                new_sim = eps_test['new_method']['similarity_to_original']
                
                old_candidates = eps_test['old_method']['candidate_similarities']
                new_candidates = eps_test['new_method']['candidate_similarities']
                old_range = max(old_candidates) - min(old_candidates) if old_candidates else 0
                new_range = max(new_candidates) - min(new_candidates) if new_candidates else 0
                
                old_similarities.append(old_sim)
                new_similarities.append(new_sim)
                old_ranges.append(old_range)
                new_ranges.append(new_range)
                epsilons.append(epsilon)
    
    # Plot 1: Overall Similarity Comparison
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(old_similarities, new_similarities, alpha=0.7, s=60, c=epsilons, cmap='viridis')
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal Performance')
    ax1.set_xlabel('Old Method Similarity')
    ax1.set_ylabel('New Method Similarity')
    ax1.set_title('Selected Similarity Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(old_similarities, new_similarities)[0, 1]
    ax1.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=ax1.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 2: Overall Range Comparison
    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(old_ranges, new_ranges, alpha=0.7, s=60, c=epsilons, cmap='viridis')
    ax2.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Equal Performance')
    ax2.set_xlabel('Old Method Range')
    ax2.set_ylabel('New Method Range')
    ax2.set_title('Candidate Diversity Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add improvement statistics
    range_improvement = np.mean(new_ranges) - np.mean(old_ranges)
    ax2.text(0.05, 0.95, f'Improvement: {range_improvement:+.3f}\n({range_improvement/np.mean(old_ranges)*100:+.1f}%)', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: Epsilon Sensitivity Analysis
    ax3 = plt.subplot(3, 3, 3)
    
    # Calculate mean similarities for each epsilon
    old_eps_means = []
    new_eps_means = []
    for eps in epsilon_values:
        old_eps_sims = [sim for i, sim in enumerate(old_similarities) if epsilons[i] == eps]
        new_eps_sims = [sim for i, sim in enumerate(new_similarities) if epsilons[i] == eps]
        old_eps_means.append(np.mean(old_eps_sims))
        new_eps_means.append(np.mean(new_eps_sims))
    
    ax3.plot(epsilon_values, old_eps_means, 'ro-', label='Old Method', linewidth=2, markersize=8)
    ax3.plot(epsilon_values, new_eps_means, 'bo-', label='New Method', linewidth=2, markersize=8)
    ax3.set_xlabel('Epsilon Value')
    ax3.set_ylabel('Mean Selected Similarity')
    ax3.set_title('Epsilon Sensitivity Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution Comparison
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(old_similarities, bins=15, alpha=0.6, color='red', label='Old Method', density=True)
    ax4.hist(new_similarities, bins=15, alpha=0.6, color='blue', label='New Method', density=True)
    ax4.set_xlabel('Selected Similarity')
    ax4.set_ylabel('Density')
    ax4.set_title('Similarity Distribution Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Range Distribution Comparison
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(old_ranges, bins=15, alpha=0.6, color='red', label='Old Method', density=True)
    ax5.hist(new_ranges, bins=15, alpha=0.6, color='blue', label='New Method', density=True)
    ax5.set_xlabel('Candidate Range')
    ax5.set_ylabel('Density')
    ax5.set_title('Diversity Distribution Comparison')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Per-Question Performance
    ax6 = plt.subplot(3, 3, 6)
    
    # Calculate per-question statistics
    question_stats = []
    for q_idx, question_data in enumerate(questions):
        question_text = question_data['question_text']
        q_old_ranges = []
        q_new_ranges = []
        
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                old_candidates = eps_test['old_method']['candidate_similarities']
                new_candidates = eps_test['new_method']['candidate_similarities']
                old_range = max(old_candidates) - min(old_candidates) if old_candidates else 0
                new_range = max(new_candidates) - min(new_candidates) if new_candidates else 0
                q_old_ranges.append(old_range)
                q_new_ranges.append(new_range)
        
        question_stats.append({
            'question': f"Q{q_idx+1}",
            'old_mean': np.mean(q_old_ranges),
            'new_mean': np.mean(q_new_ranges),
            'improvement': np.mean(q_new_ranges) - np.mean(q_old_ranges)
        })
    
    questions_list = [q['question'] for q in question_stats]
    old_means = [q['old_mean'] for q in question_stats]
    new_means = [q['new_mean'] for q in question_stats]
    
    x = np.arange(len(questions_list))
    width = 0.35
    
    ax6.bar(x - width/2, old_means, width, label='Old Method', color='red', alpha=0.7)
    ax6.bar(x + width/2, new_means, width, label='New Method', color='blue', alpha=0.7)
    ax6.set_xlabel('Questions')
    ax6.set_ylabel('Mean Candidate Range')
    ax6.set_title('Per-Question Diversity Performance')
    ax6.set_xticks(x)
    ax6.set_xticklabels(questions_list, rotation=45, ha='right')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Improvement Analysis
    ax7 = plt.subplot(3, 3, 7)
    improvements = [q['improvement'] for q in question_stats]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    
    bars = ax7.bar(questions_list, improvements, color=colors, alpha=0.7)
    ax7.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax7.set_xlabel('Questions')
    ax7.set_ylabel('Range Improvement')
    ax7.set_title('Per-Question Improvement')
    ax7.set_xticklabels(questions_list, rotation=45, ha='right')
    ax7.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax7.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                f'{imp:+.3f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    # Plot 8: Epsilon Correlation Analysis
    ax8 = plt.subplot(3, 3, 8)
    
    # Calculate correlations for each method
    old_correlations = []
    new_correlations = []
    
    for q_idx, question_data in enumerate(questions):
        q_epsilons = []
        q_old_sims = []
        q_new_sims = []
        
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                epsilon = eps_test['epsilon']
                old_sim = eps_test['old_method']['similarity_to_original']
                new_sim = eps_test['new_method']['similarity_to_original']
                
                q_epsilons.append(epsilon)
                q_old_sims.append(old_sim)
                q_new_sims.append(new_sim)
        
        if len(q_epsilons) > 1:
            old_corr = np.corrcoef(q_epsilons, q_old_sims)[0, 1]
            new_corr = np.corrcoef(q_epsilons, q_new_sims)[0, 1]
            old_correlations.append(old_corr)
            new_correlations.append(new_corr)
    
    x = np.arange(len(questions_list))
    ax8.bar(x - width/2, old_correlations, width, label='Old Method', color='red', alpha=0.7)
    ax8.bar(x + width/2, new_correlations, width, label='New Method', color='blue', alpha=0.7)
    ax8.set_xlabel('Questions')
    ax8.set_ylabel('Epsilon-Similarity Correlation')
    ax8.set_title('Epsilon Sensitivity by Question')
    ax8.set_xticks(x)
    ax8.set_xticklabels(questions_list, rotation=45, ha='right')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    
    # Calculate summary statistics
    old_mean_sim = np.mean(old_similarities)
    new_mean_sim = np.mean(new_similarities)
    old_mean_range = np.mean(old_ranges)
    new_mean_range = np.mean(new_ranges)
    
    old_corr = np.corrcoef(epsilons, old_similarities)[0, 1]
    new_corr = np.corrcoef(epsilons, new_similarities)[0, 1]
    
    summary_text = f"""
SUMMARY STATISTICS

Selected Similarity:
• Old Method: {old_mean_sim:.3f} ± {np.std(old_similarities):.3f}
• New Method: {new_mean_sim:.3f} ± {np.std(new_similarities):.3f}
• Change: {new_mean_sim - old_mean_sim:+.3f}

Candidate Diversity:
• Old Method: {old_mean_range:.3f} ± {np.std(old_ranges):.3f}
• New Method: {new_mean_range:.3f} ± {np.std(new_ranges):.3f}
• Improvement: {new_mean_range - old_mean_range:+.3f} ({((new_mean_range - old_mean_range)/old_mean_range)*100:+.1f}%)

Epsilon Sensitivity:
• Old Method: {old_corr:.3f}
• New Method: {new_corr:.3f}
• Improvement: {new_corr - old_corr:+.3f}

OVERALL ASSESSMENT:
{'✅ SIGNIFICANT IMPROVEMENT' if (new_mean_range - old_mean_range) > 0.05 and (new_corr - old_corr) > 0.1 else '⚠️ MODEST IMPROVEMENT' if (new_mean_range - old_mean_range) > 0.02 else '❌ MINIMAL IMPROVEMENT'}
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('epsilon_comparison_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Summary visualization created: epsilon_comparison_summary.png")

if __name__ == "__main__":
    create_summary_visualization()

