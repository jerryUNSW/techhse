import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy import stats

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

def create_epsilon_05_to_3_analysis():
    """Create analysis for epsilon from 0.5 to 3"""
    
    # Generate utilities
    utilities = generate_normal_utilities()
    n_candidates = len(utilities)
    
    # Epsilon values from 0.5 to 3
    epsilon_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Create main figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Exponential Mechanism Analysis: ε = 0.5 to 3.0', fontsize=16, fontweight='bold')
    
    # Plot 1: Utility distribution
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
    plot_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    
    for i, epsilon in enumerate(epsilon_values[:5]):  # Only show first 5 in main plot
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
    
    # Plot 6: Probability vs Utility relationship
    ax6 = axes[0, 1] if len(epsilon_values) <= 5 else axes[1, 2]
    
    # Move this to the last subplot if we have 6 epsilons
    if len(epsilon_values) == 6:
        ax6 = axes[1, 2]
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        ax6.scatter(utilities, probabilities, label=f'ε = {epsilon}', 
                   alpha=0.7, s=60, color=colors[i], edgecolors='black', linewidth=0.5)
    
    ax6.set_xlabel('Utility Value')
    ax6.set_ylabel('Sampling Probability')
    ax6.set_title('Utility vs Probability Relationship', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add trend lines
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        z = np.polyfit(utilities, probabilities, 1)
        p = np.poly1d(z)
        ax6.plot(utilities, p(utilities), color=colors[i], linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/epsilon_05_to_3_main.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional analysis plots
    create_additional_plots(utilities, epsilon_values, colors)

def create_additional_plots(utilities, epsilon_values, colors):
    """Create additional analysis plots"""
    
    n_candidates = len(utilities)
    
    # Plot 1: Uniformity measures
    plt.figure(figsize=(15, 10))
    
    # Calculate uniformity measures
    entropies = []
    concentrations = []
    kl_divergences = []
    max_min_ratios = []
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(n_candidates)
        concentration = 1 - entropy / max_entropy
        
        # Calculate KL divergence from uniform
        uniform_probs = np.full(n_candidates, 1.0/n_candidates)
        kl_divergence = np.sum(probabilities * np.log(probabilities / uniform_probs + 1e-10))
        
        # Max/Min ratio
        max_min_ratio = np.max(probabilities) / np.min(probabilities)
        
        entropies.append(entropy)
        concentrations.append(concentration)
        kl_divergences.append(kl_divergence)
        max_min_ratios.append(max_min_ratio)
    
    # Plot entropy
    plt.subplot(2, 3, 1)
    plt.plot(epsilon_values, entropies, 'o-', linewidth=3, markersize=8, color='blue')
    plt.axhline(y=np.log(n_candidates), color='red', linestyle='--', alpha=0.7, 
               label=f'Max Entropy: {np.log(n_candidates):.4f}')
    plt.xlabel('ε (Epsilon)')
    plt.ylabel('Entropy')
    plt.title('Entropy vs ε', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot concentration
    plt.subplot(2, 3, 2)
    plt.plot(epsilon_values, concentrations, 'o-', linewidth=3, markersize=8, color='green')
    plt.xlabel('ε (Epsilon)')
    plt.ylabel('Concentration Index')
    plt.title('Concentration vs ε', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot KL divergence
    plt.subplot(2, 3, 3)
    plt.plot(epsilon_values, kl_divergences, 'o-', linewidth=3, markersize=8, color='red')
    plt.xlabel('ε (Epsilon)')
    plt.ylabel('KL Divergence from Uniform')
    plt.title('KL Divergence vs ε', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Plot max/min ratio
    plt.subplot(2, 3, 4)
    plt.plot(epsilon_values, max_min_ratios, 'o-', linewidth=3, markersize=8, color='purple')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, 
               label='Perfect Uniformity (Ratio = 1)')
    plt.xlabel('ε (Epsilon)')
    plt.ylabel('Max/Min Probability Ratio')
    plt.title('Probability Ratio vs ε', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot probability density comparison
    plt.subplot(2, 3, 5)
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        plt.hist(probabilities, bins=15, alpha=0.3, color=colors[i], 
                density=True, label=f'ε = {epsilon}')
    
    plt.xlabel('Sampling Probability')
    plt.ylabel('Density')
    plt.title('Probability Density Comparison', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot cumulative probability
    plt.subplot(2, 3, 6)
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        sorted_indices = np.argsort(probabilities)[::-1]
        cumulative_probs = np.cumsum(probabilities[sorted_indices])
        
        plt.plot(range(len(cumulative_probs)), cumulative_probs, 
                label=f'ε = {epsilon}', linewidth=3, color=colors[i])
    
    plt.xlabel('Number of Candidates (sorted by probability)')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Probability Distribution', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80%')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='50%')
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/epsilon_05_to_3_measures.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Probability distributions overlay
    plt.figure(figsize=(12, 8))
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Create histogram
        plt.hist(probabilities, bins=20, alpha=0.3, color=colors[i], 
                density=True, label=f'ε = {epsilon}')
        
        # Add KDE smooth curve
        kde = stats.gaussian_kde(probabilities)
        x_smooth = np.linspace(probabilities.min(), probabilities.max(), 200)
        kde_values = kde(x_smooth)
        plt.plot(x_smooth, kde_values, color=colors[i], linewidth=3, 
                label=f'ε = {epsilon} (KDE)')
    
    # Add uniform distribution reference
    uniform_prob = 1.0 / n_candidates
    plt.axvline(uniform_prob, color='black', linestyle='--', linewidth=3, 
               alpha=0.8, label=f'Uniform: {uniform_prob:.4f}')
    
    plt.xlabel('Sampling Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Probability Density Distributions (ε = 0.5 to 3.0)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/epsilon_05_to_3_overlay.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print("=" * 80)
    print("EPSILON ANALYSIS: ε = 0.5 to 3.0")
    print("=" * 80)
    
    print(f"Number of candidates: {n_candidates}")
    print(f"Uniform probability (each candidate): {1.0/n_candidates:.6f}")
    print(f"Max entropy (uniform distribution): {np.log(n_candidates):.4f}")
    print()
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Calculate uniformity measures
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(n_candidates)
        concentration = 1 - entropy / max_entropy
        
        # KL divergence from uniform
        uniform_probs = np.full(n_candidates, 1.0/n_candidates)
        kl_divergence = np.sum(probabilities * np.log(probabilities / uniform_probs + 1e-10))
        
        # Probability statistics
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        ratio = max_prob / min_prob
        std_prob = np.std(probabilities)
        
        # Top candidates analysis
        sorted_indices = np.argsort(probabilities)[::-1]
        top5_prob = np.sum(probabilities[sorted_indices[:5]])
        top10_prob = np.sum(probabilities[sorted_indices[:10]])
        
        # Highest utility candidate
        max_utility_idx = np.argmax(utilities)
        max_utility_prob = probabilities[max_utility_idx]
        
        print(f"ε = {epsilon}:")
        print(f"  Entropy: {entropy:.6f} ({entropy/max_entropy*100:.1f}% of max)")
        print(f"  Concentration: {concentration:.6f}")
        print(f"  KL divergence: {kl_divergence:.6f}")
        print(f"  Probability range: [{min_prob:.6f}, {max_prob:.6f}]")
        print(f"  Max/Min ratio: {ratio:.2f}")
        print(f"  Top 5 probability: {top5_prob:.3f}")
        print(f"  Top 10 probability: {top10_prob:.3f}")
        print(f"  Highest utility prob: {max_utility_prob:.4f}")
        
        # Uniformity assessment
        if concentration < 0.01:
            uniformity_desc = "Nearly uniform"
        elif concentration < 0.02:
            uniformity_desc = "Very close to uniform"
        elif concentration < 0.05:
            uniformity_desc = "Close to uniform"
        elif concentration < 0.1:
            uniformity_desc = "Moderately concentrated"
        else:
            uniformity_desc = "Highly concentrated"
        
        print(f"  Uniformity: {uniformity_desc}")
        print()

if __name__ == "__main__":
    create_epsilon_05_to_3_analysis()
