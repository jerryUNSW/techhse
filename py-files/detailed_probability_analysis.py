import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy import stats
import seaborn as sns

def exponential_mechanism_probabilities(utilities, epsilon):
    """Calculate exponential mechanism probabilities"""
    delta_u = 9  # Sensitivity = max - min = 10 - 1
    scaled_utilities = epsilon * utilities / (2 * delta_u)
    probabilities = softmax(scaled_utilities)
    return probabilities

def analyze_probability_distribution_shape():
    """Detailed analysis of probability distribution shape"""
    np.random.seed(42)
    
    # Generate utilities with different distributions
    n_candidates = 50
    
    # Normal distribution (as specified)
    utilities_normal = np.clip(np.random.normal(5.5, 1.5, n_candidates), 1, 10)
    
    # Uniform distribution for comparison
    utilities_uniform = np.random.uniform(1, 10, n_candidates)
    
    # Bimodal distribution for comparison
    utilities_bimodal = np.concatenate([
        np.clip(np.random.normal(3, 0.5, n_candidates//2), 1, 5),
        np.clip(np.random.normal(7, 0.5, n_candidates//2), 5, 10)
    ])
    
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Utility distributions
    ax1 = plt.subplot(4, 3, 1)
    plt.hist(utilities_normal, bins=15, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Normal Utility Distribution')
    plt.xlabel('Utility Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(4, 3, 2)
    plt.hist(utilities_uniform, bins=15, alpha=0.7, color='green', edgecolor='black')
    plt.title('Uniform Utility Distribution')
    plt.xlabel('Utility Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    ax3 = plt.subplot(4, 3, 3)
    plt.hist(utilities_bimodal, bins=15, alpha=0.7, color='red', edgecolor='black')
    plt.title('Bimodal Utility Distribution')
    plt.xlabel('Utility Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Probability distributions for normal utilities
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, epsilon in enumerate(epsilon_values):
        ax = plt.subplot(4, 3, 4 + i)
        
        probabilities = exponential_mechanism_probabilities(utilities_normal, epsilon)
        sorted_indices = np.argsort(utilities_normal)[::-1]
        sorted_probabilities = probabilities[sorted_indices]
        
        plt.bar(range(len(sorted_probabilities)), sorted_probabilities, 
                color=colors[i], alpha=0.7, edgecolor='black')
        plt.title(f'Normal Utilities: ε = {epsilon}')
        plt.xlabel('Candidates (sorted by utility)')
        plt.ylabel('Probability')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(n_candidates)
        concentration = 1 - entropy / max_entropy
        
        plt.text(0.02, 0.98, f'Max P: {max_prob:.4f}\nConcentration: {concentration:.4f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/detailed_probability_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print("=" * 80)
    print("DETAILED PROBABILITY DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    print("\n1. NORMAL UTILITY DISTRIBUTION ANALYSIS:")
    print("-" * 50)
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities_normal, epsilon)
        
        # Basic statistics
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        
        # Concentration measures
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(n_candidates)
        concentration = 1 - entropy / max_entropy
        
        # Gini coefficient (inequality measure)
        sorted_probs = np.sort(probabilities)
        n = len(sorted_probs)
        cumsum = np.cumsum(sorted_probs)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        
        # Top-k probabilities
        sorted_indices = np.argsort(probabilities)[::-1]
        top5_prob = np.sum(probabilities[sorted_indices[:5]])
        top10_prob = np.sum(probabilities[sorted_indices[:10]])
        
        print(f"\nε = {epsilon}:")
        print(f"  Probability range: [{min_prob:.6f}, {max_prob:.6f}]")
        print(f"  Mean probability: {mean_prob:.6f} (expected: {1/n_candidates:.6f})")
        print(f"  Std deviation: {std_prob:.6f}")
        print(f"  Concentration index: {concentration:.4f}")
        print(f"  Gini coefficient: {gini:.4f}")
        print(f"  Top 5 candidates: {top5_prob:.4f} (vs uniform: {5/n_candidates:.4f})")
        print(f"  Top 10 candidates: {top10_prob:.4f} (vs uniform: {10/n_candidates:.4f})")
        
        # Probability distribution shape
        if concentration < 0.1:
            shape_desc = "Nearly uniform"
        elif concentration < 0.3:
            shape_desc = "Slightly concentrated"
        elif concentration < 0.6:
            shape_desc = "Moderately concentrated"
        else:
            shape_desc = "Highly concentrated"
        
        print(f"  Distribution shape: {shape_desc}")
    
    # Compare with other distributions
    print("\n\n2. COMPARISON WITH OTHER UTILITY DISTRIBUTIONS:")
    print("-" * 50)
    
    distributions = {
        'Normal': utilities_normal,
        'Uniform': utilities_uniform,
        'Bimodal': utilities_bimodal
    }
    
    epsilon = 1.0  # Use medium epsilon for comparison
    
    print(f"\nComparison at ε = {epsilon}:")
    print(f"{'Distribution':<12} {'Max P':<10} {'Concentration':<12} {'Top5 P':<10}")
    print("-" * 50)
    
    for name, utilities in distributions.items():
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        max_prob = np.max(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        concentration = 1 - entropy / np.log(n_candidates)
        sorted_indices = np.argsort(probabilities)[::-1]
        top5_prob = np.sum(probabilities[sorted_indices[:5]])
        
        print(f"{name:<12} {max_prob:<10.4f} {concentration:<12.4f} {top5_prob:<10.4f}")
    
    # Theoretical insights
    print("\n\n3. THEORETICAL INSIGHTS:")
    print("-" * 50)
    
    print("\nFor Normal Utility Distribution with Exponential Mechanism:")
    print("• The probability distribution is approximately log-normal shaped")
    print("• Most candidates cluster around the mean utility with similar probabilities")
    print("• High-utility outliers get disproportionately higher probabilities")
    print("• The 'tail' of the probability distribution is heavy and skewed")
    print("• As ε increases, the skewness becomes more pronounced")
    
    print("\nKey Characteristics:")
    print("• Probability ratios: P(high_utility) / P(low_utility) = exp(ε × utility_diff / (2×Δu))")
    print("• For ε=1, a utility difference of 2 units creates a probability ratio of ~1.12")
    print("• For ε=5, the same difference creates a ratio of ~1.72")
    print("• The mechanism amplifies small utility differences exponentially")
    
    # Create probability ratio analysis
    plt.figure(figsize=(12, 8))
    
    # Calculate probability ratios for different utility differences
    utility_diffs = np.linspace(0, 5, 100)
    
    for epsilon in [0.5, 1.0, 2.0, 5.0]:
        ratios = np.exp(epsilon * utility_diffs / (2 * 9))
        plt.plot(utility_diffs, ratios, label=f'ε = {epsilon}', linewidth=2)
    
    plt.xlabel('Utility Difference')
    plt.ylabel('Probability Ratio (P(high) / P(low))')
    plt.title('Probability Ratio vs Utility Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig('/home/yizhang/tech4HSE/probability_ratio_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    analyze_probability_distribution_shape()
