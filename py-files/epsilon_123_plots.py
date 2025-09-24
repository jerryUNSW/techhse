import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

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

def create_epsilon_plots():
    """Create focused plots for epsilon = 1, 2, 3"""
    
    # Generate utilities
    utilities = generate_normal_utilities()
    epsilon_values = [1, 2, 3]
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Exponential Mechanism Analysis: ε = 1, 2, 3', fontsize=16, fontweight='bold')
    
    # Plot 1: Utility Distribution
    ax1 = axes[0, 0]
    ax1.hist(utilities, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_title('Utility Distribution (Normal)', fontweight='bold')
    ax1.set_xlabel('Utility Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    ax1.text(0.02, 0.98, f'Mean: {utilities.mean():.2f}\nStd: {utilities.std():.2f}\nRange: [{utilities.min():.2f}, {utilities.max():.2f}]', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plots 2-4: Probability distributions for each epsilon
    plot_positions = [(0, 1), (0, 2), (1, 0)]
    for i, epsilon in enumerate(epsilon_values):
        row, col = plot_positions[i]
        ax = axes[row, col]
        
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        sorted_indices = np.argsort(utilities)[::-1]  # Sort by utility (descending)
        sorted_probabilities = probabilities[sorted_indices]
        sorted_utilities = utilities[sorted_indices]
        
        # Create bar plot
        bars = ax.bar(range(len(sorted_probabilities)), sorted_probabilities, 
                     color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'ε = {epsilon} - Sampling Probabilities', fontweight='bold')
        ax.set_xlabel('Candidates (sorted by utility)')
        ax.set_ylabel('Sampling Probability')
        ax.grid(True, alpha=0.3)
        
        # Add utility value annotations for every 10th candidate
        for j in range(0, len(sorted_utilities), 10):
            if j < len(sorted_utilities):
                ax.text(j, sorted_probabilities[j] + 0.0005, 
                       f'{sorted_utilities[j]:.1f}', 
                       ha='center', va='bottom', fontsize=8, rotation=90)
        
        # Add statistics
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        concentration = 1 - entropy / np.log(len(utilities))
        top5_prob = np.sum(probabilities[sorted_indices[:5]])
        
        ax.text(0.02, 0.98, f'Max P: {max_prob:.4f}\nConcentration: {concentration:.4f}\nTop 5: {top5_prob:.3f}', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Utility vs Probability relationship
    ax5 = axes[1, 1]
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        ax5.scatter(utilities, probabilities, label=f'ε = {epsilon}', 
                   alpha=0.7, s=80, color=colors[i], edgecolors='black', linewidth=0.5)
    
    ax5.set_xlabel('Utility Value')
    ax5.set_ylabel('Sampling Probability')
    ax5.set_title('Utility vs Probability Relationship', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add trend lines
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        z = np.polyfit(utilities, probabilities, 1)
        p = np.poly1d(z)
        ax5.plot(utilities, p(utilities), color=colors[i], linestyle='--', alpha=0.8)
    
    # Plot 5: Probability distribution comparison
    ax6 = axes[1, 2]
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        sorted_indices = np.argsort(probabilities)[::-1]
        sorted_probabilities = probabilities[sorted_indices]
        
        ax6.plot(range(len(sorted_probabilities)), sorted_probabilities, 
                label=f'ε = {epsilon}', linewidth=3, color=colors[i], marker='o', markersize=4)
    
    ax6.set_xlabel('Rank (by probability)')
    ax6.set_ylabel('Probability')
    ax6.set_title('Probability Distribution Comparison', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_yscale('log')
    
    # Note: Cumulative probability plot moved to additional plots section
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/epsilon_123_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional focused plots
    create_additional_plots(utilities, epsilon_values, colors)

def create_additional_plots(utilities, epsilon_values, colors):
    """Create additional focused plots"""
    
    # Plot 1: Probability ratios
    plt.figure(figsize=(12, 8))
    
    utility_diffs = np.linspace(0, 5, 100)
    
    for i, epsilon in enumerate(epsilon_values):
        ratios = np.exp(epsilon * utility_diffs / (2 * 9))
        plt.plot(utility_diffs, ratios, label=f'ε = {epsilon}', 
                linewidth=4, color=colors[i])
    
    plt.xlabel('Utility Difference', fontsize=12)
    plt.ylabel('Probability Ratio (P(high)/P(low))', fontsize=12)
    plt.title('Probability Ratio vs Utility Difference\n(How ε amplifies utility differences)', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Add annotations
    for i, epsilon in enumerate(epsilon_values):
        ratio_at_2 = np.exp(epsilon * 2 / (2 * 9))
        plt.annotate(f'ε={epsilon}: {ratio_at_2:.2f}x', 
                    xy=(2, ratio_at_2), xytext=(2.5, ratio_at_2 * 1.5),
                    arrowprops=dict(arrowstyle='->', color=colors[i], lw=2),
                    fontsize=10, color=colors[i], fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/probability_ratios_123.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Cumulative probability comparison
    plt.figure(figsize=(12, 8))
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        sorted_indices = np.argsort(probabilities)[::-1]
        cumulative_probs = np.cumsum(probabilities[sorted_indices])
        
        plt.plot(range(len(cumulative_probs)), cumulative_probs, 
                label=f'ε = {epsilon}', linewidth=3, color=colors[i])
    
    plt.xlabel('Number of Candidates (sorted by probability)', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Probability Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80%')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50%')
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/cumulative_probability_123.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Concentration comparison
    plt.figure(figsize=(10, 6))
    
    concentrations = []
    top5_probs = []
    max_probs = []
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Concentration
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        concentration = 1 - entropy / np.log(len(utilities))
        concentrations.append(concentration)
        
        # Top 5 probability
        sorted_indices = np.argsort(probabilities)[::-1]
        top5_prob = np.sum(probabilities[sorted_indices[:5]])
        top5_probs.append(top5_prob)
        
        # Max probability
        max_probs.append(np.max(probabilities))
    
    x = np.arange(len(epsilon_values))
    width = 0.25
    
    bars1 = plt.bar(x - width, concentrations, width, label='Concentration Index', 
                   color='lightblue', alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x, top5_probs, width, label='Top 5 Probability', 
                   color='lightcoral', alpha=0.8, edgecolor='black')
    bars3 = plt.bar(x + width, max_probs, width, label='Max Probability', 
                   color='lightgreen', alpha=0.8, edgecolor='black')
    
    plt.xlabel('ε (Epsilon)', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('Concentration and Probability Measures vs ε', fontsize=14, fontweight='bold')
    plt.xticks(x, epsilon_values)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/concentration_comparison_123.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_epsilon_plots()
    
    # Print summary
    utilities = generate_normal_utilities()
    epsilon_values = [1, 2, 3]
    
    print("\n" + "=" * 70)
    print("EPSILON COMPARISON SUMMARY (ε = 1, 2, 3)")
    print("=" * 70)
    
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
        
        # Calculate some key ratios
        if epsilon > 1:
            prev_epsilon = epsilon - 1
            prev_probabilities = exponential_mechanism_probabilities(utilities, prev_epsilon)
            prev_max_prob = np.max(prev_probabilities)
            ratio_improvement = max_prob / prev_max_prob
            print(f"  Improvement over ε={prev_epsilon}: {ratio_improvement:.2f}x")
