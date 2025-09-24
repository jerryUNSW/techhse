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

def create_correct_epsilon_analysis():
    """Create correct analysis showing epsilon vs uniformity"""
    
    # Generate utilities
    utilities = generate_normal_utilities()
    n_candidates = len(utilities)
    
    # Test a wider range of epsilon values including very small ones
    epsilon_values = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(epsilon_values)))
    
    # Create figure
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Epsilon vs Uniformity: Lower ε → More Uniform Distribution', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Utility distribution
    ax1 = axes[0, 0]
    ax1.hist(utilities, bins=15, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_title('Utility Distribution (Normal)', fontweight='bold')
    ax1.set_xlabel('Utility Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Probability distributions for different epsilon values
    for i, epsilon in enumerate(epsilon_values):
        row = i // 4
        col = (i % 4) + 1 if row == 0 else i % 4
        
        if row < 2 and col < 4:  # Only plot if within subplot bounds
            ax = axes[row, col]
            
            probabilities = exponential_mechanism_probabilities(utilities, epsilon)
            
            # Create histogram of probabilities
            n, bins, patches = ax.hist(probabilities, bins=15, alpha=0.7, 
                                      color=colors[i], edgecolor='black', density=True)
            
            # Add uniform distribution reference line
            uniform_prob = 1.0 / n_candidates
            ax.axvline(uniform_prob, color='red', linestyle='--', linewidth=2, 
                      alpha=0.8, label=f'Uniform: {uniform_prob:.4f}')
            
            ax.set_title(f'ε = {epsilon}', fontweight='bold')
            ax.set_xlabel('Sampling Probability')
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Calculate and display uniformity measures
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
            max_entropy = np.log(n_candidates)
            concentration = 1 - entropy / max_entropy
            
            # Calculate KL divergence from uniform
            uniform_probs = np.full(n_candidates, 1.0/n_candidates)
            kl_divergence = np.sum(probabilities * np.log(probabilities / uniform_probs + 1e-10))
            
            ax.text(0.02, 0.98, f'Concentration: {concentration:.4f}\nKL Div: {kl_divergence:.6f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/correct_epsilon_uniformity.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional analysis plots
    create_uniformity_analysis(utilities, epsilon_values)

def create_uniformity_analysis(utilities, epsilon_values):
    """Create detailed uniformity analysis"""
    
    n_candidates = len(utilities)
    
    # Plot 1: Entropy vs Epsilon
    plt.figure(figsize=(12, 8))
    
    entropies = []
    concentrations = []
    kl_divergences = []
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(n_candidates)
        concentration = 1 - entropy / max_entropy
        
        # Calculate KL divergence from uniform
        uniform_probs = np.full(n_candidates, 1.0/n_candidates)
        kl_divergence = np.sum(probabilities * np.log(probabilities / uniform_probs + 1e-10))
        
        entropies.append(entropy)
        concentrations.append(concentration)
        kl_divergences.append(kl_divergence)
    
    # Plot entropy
    plt.subplot(2, 2, 1)
    plt.plot(epsilon_values, entropies, 'o-', linewidth=2, markersize=6, color='blue')
    plt.axhline(y=max_entropy, color='red', linestyle='--', alpha=0.7, 
               label=f'Max Entropy: {max_entropy:.4f}')
    plt.xlabel('ε (Epsilon)')
    plt.ylabel('Entropy')
    plt.title('Entropy vs ε\n(Higher = More Uniform)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xscale('log')
    
    # Plot concentration
    plt.subplot(2, 2, 2)
    plt.plot(epsilon_values, concentrations, 'o-', linewidth=2, markersize=6, color='green')
    plt.xlabel('ε (Epsilon)')
    plt.ylabel('Concentration Index')
    plt.title('Concentration vs ε\n(Lower = More Uniform)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # Plot KL divergence
    plt.subplot(2, 2, 3)
    plt.plot(epsilon_values, kl_divergences, 'o-', linewidth=2, markersize=6, color='red')
    plt.xlabel('ε (Epsilon)')
    plt.ylabel('KL Divergence from Uniform')
    plt.title('KL Divergence vs ε\n(Lower = More Uniform)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    
    # Plot probability range
    plt.subplot(2, 2, 4)
    max_probs = []
    min_probs = []
    ratios = []
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        ratio = max_prob / min_prob
        
        max_probs.append(max_prob)
        min_probs.append(min_prob)
        ratios.append(ratio)
    
    plt.plot(epsilon_values, ratios, 'o-', linewidth=2, markersize=6, color='purple')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.7, 
               label='Perfect Uniformity (Ratio = 1)')
    plt.xlabel('ε (Epsilon)')
    plt.ylabel('Max/Min Probability Ratio')
    plt.title('Probability Ratio vs ε\n(Lower = More Uniform)', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/uniformity_measures.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Probability distributions overlay
    plt.figure(figsize=(14, 10))
    
    # Select key epsilon values for overlay
    key_epsilons = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    colors = ['blue', 'green', 'orange', 'red', 'purple', 'brown']
    
    for i, epsilon in enumerate(key_epsilons):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Create histogram
        n, bins, patches = plt.hist(probabilities, bins=20, alpha=0.3, 
                                   color=colors[i], density=True, 
                                   label=f'ε = {epsilon}')
        
        # Add KDE smooth curve
        kde = stats.gaussian_kde(probabilities)
        x_smooth = np.linspace(probabilities.min(), probabilities.max(), 200)
        kde_values = kde(x_smooth)
        plt.plot(x_smooth, kde_values, color=colors[i], linewidth=3, 
                label=f'ε = {epsilon} (KDE)')
    
    # Add uniform distribution reference
    uniform_prob = 1.0 / n_candidates
    plt.axvline(uniform_prob, color='black', linestyle='--', linewidth=3, 
               alpha=0.8, label=f'Perfect Uniform: {uniform_prob:.4f}')
    
    plt.xlabel('Sampling Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Probability Density Distributions\nLower ε → Closer to Uniform Distribution', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/probability_distributions_overlay.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed analysis
    print("=" * 80)
    print("CORRECT EPSILON ANALYSIS: Lower ε → More Uniform Distribution")
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
        
        print(f"ε = {epsilon}:")
        print(f"  Entropy: {entropy:.6f} / {max_entropy:.6f} ({entropy/max_entropy*100:.1f}% of max)")
        print(f"  Concentration: {concentration:.6f} (0=uniform, 1=concentrated)")
        print(f"  KL divergence from uniform: {kl_divergence:.6f}")
        print(f"  Probability range: [{min_prob:.6f}, {max_prob:.6f}]")
        print(f"  Max/Min ratio: {ratio:.2f}")
        print(f"  Std deviation: {std_prob:.6f}")
        
        # Uniformity assessment
        if concentration < 0.01:
            uniformity_desc = "Nearly uniform"
        elif concentration < 0.05:
            uniformity_desc = "Very close to uniform"
        elif concentration < 0.1:
            uniformity_desc = "Close to uniform"
        elif concentration < 0.2:
            uniformity_desc = "Moderately concentrated"
        else:
            uniformity_desc = "Highly concentrated"
        
        print(f"  Uniformity: {uniformity_desc}")
        print()

if __name__ == "__main__":
    create_correct_epsilon_analysis()
