import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

def exponential_mechanism_probabilities(utilities, epsilon):
    """
    Calculate sampling probabilities for exponential mechanism
    
    P(i) = exp(ε * u_i / (2 * Δu)) / Σ_j exp(ε * u_j / (2 * Δu))
    """
    delta_u = 9  # Sensitivity of utility function (max - min = 10 - 1)
    scaled_utilities = epsilon * utilities / (2 * delta_u)
    probabilities = softmax(scaled_utilities)
    return probabilities

def generate_normal_utilities(n_candidates=50, mean=5.5, std=1.5, min_val=1, max_val=10):
    """Generate normally distributed utilities clipped to [min_val, max_val]"""
    np.random.seed(42)  # For reproducible results
    utilities = np.random.normal(mean, std, n_candidates)
    utilities = np.clip(utilities, min_val, max_val)
    return utilities

def main():
    # Generate 50 candidates with normal utility distribution
    utilities = generate_normal_utilities()
    
    # Different epsilon values
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Exponential Mechanism: Sampling Probability Distribution', fontsize=16, fontweight='bold')
    
    # Plot utility distribution
    ax_util = axes[0, 0]
    ax_util.hist(utilities, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    ax_util.set_title('Utility Distribution (Normal)')
    ax_util.set_xlabel('Utility Value')
    ax_util.set_ylabel('Frequency')
    ax_util.grid(True, alpha=0.3)
    
    # Sort utilities for better visualization
    sorted_indices = np.argsort(utilities)[::-1]  # Descending order
    sorted_utilities = utilities[sorted_indices]
    
    # Plot probability distributions for different epsilon values
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for i, epsilon in enumerate(epsilon_values):
        row, col = positions[i]
        ax = axes[row, col]
        
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        sorted_probabilities = probabilities[sorted_indices]
        
        bars = ax.bar(range(len(sorted_utilities)), sorted_probabilities, 
                     color=colors[i], alpha=0.7, edgecolor='black')
        
        ax.set_title(f'ε = {epsilon}')
        ax.set_xlabel('Candidates (sorted by utility)')
        ax.set_ylabel('Sampling Probability')
        ax.grid(True, alpha=0.3)
        
        # Add utility value annotations for first subplot
        if i == 0:
            for j in range(0, len(sorted_utilities), 10):
                ax.text(j, sorted_probabilities[j] + 0.001, 
                       f'{sorted_utilities[j]:.2f}', 
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/exponential_mechanism_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analysis results
    print("=" * 60)
    print("Exponential Mechanism Analysis Results")
    print("=" * 60)
    
    print(f"Number of candidates: {len(utilities)}")
    print(f"Utility range: [{utilities.min():.2f}, {utilities.max():.2f}]")
    print(f"Utility mean: {utilities.mean():.2f}")
    print(f"Utility std: {utilities.std():.2f}")
    print()
    
    # Analyze behavior for each epsilon value
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        max_utility_idx = np.argmax(utilities)
        max_utility_prob = probabilities[max_utility_idx]
        
        # Calculate concentration (entropy-based)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(utilities))
        concentration = 1 - entropy / max_entropy
        
        print(f"ε = {epsilon}:")
        print(f"  Highest utility: {utilities[max_utility_idx]:.2f}")
        print(f"  Probability of selecting highest utility: {max_utility_prob:.4f}")
        print(f"  Distribution concentration: {concentration:.4f} (0=uniform, 1=concentrated)")
        print(f"  Top 5 candidates cumulative probability: {np.sum(probabilities[sorted_indices[:5]]):.4f}")
        print()
    
    # Utility vs Probability relationship
    plt.figure(figsize=(12, 8))
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        plt.scatter(utilities, probabilities, label=f'ε = {epsilon}', 
                   alpha=0.7, s=50, color=colors[i])
    
    plt.xlabel('Utility Value')
    plt.ylabel('Sampling Probability')
    plt.title('Utility vs Sampling Probability (Different ε values)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/yizhang/tech4HSE/utility_vs_probability.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Theoretical analysis
    print("=" * 60)
    print("Theoretical Analysis")
    print("=" * 60)
    
    print("Exponential Mechanism Formula:")
    print("P(i) = exp(ε * u_i / (2 * Δu)) / Σ_j exp(ε * u_j / (2 * Δu))")
    print()
    
    print("Key Properties:")
    print("1. Higher ε → more concentrated on high-utility candidates")
    print("2. Lower ε → more uniform distribution")
    print("3. Utility differences are amplified by ε")
    print("4. As ε → 0, all candidates have equal probability")
    print("5. As ε → ∞, only highest utility candidate has non-zero probability")
    print()
    
    print("Behavior with Normal Utility Distribution:")
    print("- Most candidates have utilities near the mean")
    print("- Few candidates have significantly higher/lower utilities")
    print("- Exponential mechanism amplifies these differences, especially with high ε")
    print("- Medium ε values balance privacy and utility")

if __name__ == "__main__":
    main()
