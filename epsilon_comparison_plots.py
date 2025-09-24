import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
import seaborn as sns

def exponential_mechanism_probabilities(utilities, epsilon):
    """Calculate exponential mechanism probabilities"""
    delta_u = 9  # Sensitivity = max - min = 10 - 1
    scaled_utilities = epsilon * utilities / (2 * delta_u)
    probabilities = softmax(scaled_utilities)
    return probabilities

def generate_normal_utilities(n_candidates=50, mean=5.5, std=1.5, min_val=1, max_val=10):
    """Generate normally distributed utilities"""
    np.random.seed(42)
    utilities = np.random.normal(mean, std, n_candidates)
    utilities = np.clip(utilities, min_val, max_val)
    return utilities

def create_comprehensive_plots():
    """Create comprehensive plots for epsilon = 1, 2, 3"""
    
    # Generate utilities
    utilities = generate_normal_utilities()
    epsilon_values = [1, 2, 3]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Utility distribution
    ax1 = plt.subplot(3, 4, 1)
    plt.hist(utilities, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Utility Distribution\n(Normal Distribution)', fontsize=12, fontweight='bold')
    plt.xlabel('Utility Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    plt.text(0.02, 0.98, f'Mean: {utilities.mean():.2f}\nStd: {utilities.std():.2f}\nRange: [{utilities.min():.2f}, {utilities.max():.2f}]', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Probability distributions for each epsilon
    for i, epsilon in enumerate(epsilon_values):
        ax = plt.subplot(3, 4, 2 + i)
        
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        sorted_indices = np.argsort(utilities)[::-1]  # Sort by utility (descending)
        sorted_probabilities = probabilities[sorted_indices]
        sorted_utilities = utilities[sorted_indices]
        
        # Create bar plot
        bars = plt.bar(range(len(sorted_probabilities)), sorted_probabilities, 
                      color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        plt.title(f'ε = {epsilon}\nSampling Probabilities', fontsize=12, fontweight='bold')
        plt.xlabel('Candidates (sorted by utility)')
        plt.ylabel('Sampling Probability')
        plt.grid(True, alpha=0.3)
        
        # Add utility value annotations for top 5 and bottom 5
        for j in range(0, len(sorted_utilities), 10):
            if j < len(sorted_utilities):
                plt.text(j, sorted_probabilities[j] + 0.0005, 
                        f'{sorted_utilities[j]:.1f}', 
                        ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Add statistics
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        concentration = 1 - entropy / np.log(len(utilities))
        top5_prob = np.sum(probabilities[sorted_indices[:5]])
        
        plt.text(0.02, 0.98, f'Max P: {max_prob:.4f}\nConcentration: {concentration:.4f}\nTop 5: {top5_prob:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Utility vs Probability scatter plots
    ax5 = plt.subplot(3, 4, 5)
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        plt.scatter(utilities, probabilities, label=f'ε = {epsilon}', 
                   alpha=0.7, s=60, color=colors[i])
    
    plt.xlabel('Utility Value')
    plt.ylabel('Sampling Probability')
    plt.title('Utility vs Probability\nRelationship', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Cumulative probability distributions
    ax6 = plt.subplot(3, 4, 6)
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        sorted_indices = np.argsort(probabilities)[::-1]
        cumulative_probs = np.cumsum(probabilities[sorted_indices])
        
        plt.plot(range(len(cumulative_probs)), cumulative_probs, 
                label=f'ε = {epsilon}', linewidth=2, color=colors[i])
    
    plt.xlabel('Number of Candidates (sorted by probability)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Probability\nDistribution', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='80%')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50%')
    
    # 5. Probability ratios analysis
    ax7 = plt.subplot(3, 4, 7)
    utility_diffs = np.linspace(0, 5, 100)
    
    for i, epsilon in enumerate(epsilon_values):
        ratios = np.exp(epsilon * utility_diffs / (2 * 9))
        plt.plot(utility_diffs, ratios, label=f'ε = {epsilon}', 
                linewidth=3, color=colors[i])
    
    plt.xlabel('Utility Difference')
    plt.ylabel('Probability Ratio (P(high)/P(low))')
    plt.title('Probability Ratio vs\nUtility Difference', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 6. Concentration comparison
    ax8 = plt.subplot(3, 4, 8)
    concentrations = []
    top5_probs = []
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        concentration = 1 - entropy / np.log(len(utilities))
        concentrations.append(concentration)
        
        sorted_indices = np.argsort(probabilities)[::-1]
        top5_prob = np.sum(probabilities[sorted_indices[:5]])
        top5_probs.append(top5_prob)
    
    x = np.arange(len(epsilon_values))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, concentrations, width, label='Concentration Index', 
                   color='lightblue', alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x + width/2, top5_probs, width, label='Top 5 Probability', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    
    plt.xlabel('ε (Epsilon)')
    plt.ylabel('Value')
    plt.title('Concentration vs ε\nComparison', fontsize=12, fontweight='bold')
    plt.xticks(x, epsilon_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 7. Probability distribution comparison (overlay)
    ax9 = plt.subplot(3, 4, 9)
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probabilities = probabilities[sorted_indices]
        
        plt.plot(range(len(sorted_probabilities)), sorted_probabilities, 
                label=f'ε = {epsilon}', linewidth=2, color=colors[i], marker='o', markersize=3)
    
    plt.xlabel('Rank (by probability)')
    plt.ylabel('Probability')
    plt.title('Probability Distribution\nComparison (Log Scale)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 8. Heatmap of probability differences
    ax10 = plt.subplot(3, 4, 10)
    
    # Create probability matrix
    prob_matrix = []
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        prob_matrix.append(probabilities)
    
    prob_matrix = np.array(prob_matrix)
    
    # Create heatmap
    im = plt.imshow(prob_matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im, label='Probability')
    plt.title('Probability Heatmap\n(ε vs Candidates)', fontsize=12, fontweight='bold')
    plt.xlabel('Candidate Index')
    plt.ylabel('ε Value')
    plt.yticks(range(len(epsilon_values)), epsilon_values)
    
    # 9. Statistical summary table
    ax11 = plt.subplot(3, 4, 11)
    ax11.axis('off')
    
    # Create summary statistics
    summary_data = []
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        concentration = 1 - entropy / np.log(len(utilities))
        sorted_indices = np.argsort(probabilities)[::-1]
        top5_prob = np.sum(probabilities[sorted_indices[:5]])
        top10_prob = np.sum(probabilities[sorted_indices[:10]])
        
        summary_data.append([
            f'{max_prob:.4f}',
            f'{min_prob:.4f}',
            f'{concentration:.4f}',
            f'{top5_prob:.3f}',
            f'{top10_prob:.3f}'
        ])
    
    # Create table
    table_data = [['Metric', 'ε=1', 'ε=2', 'ε=3']]
    metrics = ['Max Probability', 'Min Probability', 'Concentration', 'Top 5 Prob', 'Top 10 Prob']
    
    for i, metric in enumerate(metrics):
        row = [metric]
        for j in range(len(epsilon_values)):
            row.append(summary_data[j][i])
        table_data.append(row)
    
    table = plt.table(cellText=table_data[1:], colLabels=table_data[0],
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code the table
    for i in range(1, len(table_data)):
        for j in range(1, len(table_data[0])):
            if i == 1:  # Max probability row
                table[(i, j)].set_facecolor('#ffcccc')
            elif i == 2:  # Min probability row
                table[(i, j)].set_facecolor('#ccffcc')
            elif i == 3:  # Concentration row
                table[(i, j)].set_facecolor('#ccccff')
    
    plt.title('Statistical Summary', fontsize=12, fontweight='bold')
    
    # 10. Probability distribution shapes (normalized)
    ax12 = plt.subplot(3, 4, 12)
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        # Normalize probabilities to show relative differences
        normalized_probs = probabilities / np.mean(probabilities)
        sorted_indices = np.argsort(utilities)[::-1]
        sorted_normalized = normalized_probs[sorted_indices]
        
        plt.plot(range(len(sorted_normalized)), sorted_normalized, 
                label=f'ε = {epsilon}', linewidth=2, color=colors[i], marker='s', markersize=2)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Uniform (1.0)')
    plt.xlabel('Candidates (sorted by utility)')
    plt.ylabel('Normalized Probability')
    plt.title('Normalized Probability\nDistributions', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/epsilon_comparison_comprehensive.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("=" * 80)
    print("EPSILON COMPARISON ANALYSIS (ε = 1, 2, 3)")
    print("=" * 80)
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        concentration = 1 - entropy / np.log(len(utilities))
        sorted_indices = np.argsort(probabilities)[::-1]
        top5_prob = np.sum(probabilities[sorted_indices[:5]])
        top10_prob = np.sum(probabilities[sorted_indices[:10]])
        
        print(f"\nε = {epsilon}:")
        print(f"  Probability range: [{min_prob:.6f}, {max_prob:.6f}]")
        print(f"  Ratio (max/min): {max_prob/min_prob:.2f}")
        print(f"  Concentration index: {concentration:.4f}")
        print(f"  Top 5 candidates probability: {top5_prob:.3f}")
        print(f"  Top 10 candidates probability: {top10_prob:.3f}")
        
        # Find highest utility candidate
        max_utility_idx = np.argmax(utilities)
        max_utility_prob = probabilities[max_utility_idx]
        print(f"  Highest utility candidate probability: {max_utility_prob:.4f}")

if __name__ == "__main__":
    create_comprehensive_plots()
