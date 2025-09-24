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

def generate_normal_utilities(n_candidates=50, mean=5.5, std=1.5, min_val=1, max_val=10):
    """Generate normally distributed utilities"""
    np.random.seed(42)
    utilities = np.random.normal(mean, std, n_candidates)
    utilities = np.clip(utilities, min_val, max_val)
    return utilities

def create_probability_density_plots():
    """Create probability density function plots for epsilon = 1, 2, 3"""
    
    # Generate utilities
    utilities = generate_normal_utilities()
    epsilon_values = [1, 2, 3]
    colors = ['#2E86AB', '#A23B72', '#F18F01']  # Blue, Purple, Orange
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Probability Density Functions: ε = 1, 2, 3', fontsize=16, fontweight='bold')
    
    # Plot 1: Utility distribution (histogram)
    ax1 = axes[0, 0]
    n, bins, patches = ax1.hist(utilities, bins=15, alpha=0.7, color='lightblue', 
                               edgecolor='black', density=True)
    ax1.set_title('Utility Distribution (PDF)', fontweight='bold')
    ax1.set_xlabel('Utility Value')
    ax1.set_ylabel('Probability Density')
    ax1.grid(True, alpha=0.3)
    
    # Add normal distribution overlay
    x = np.linspace(utilities.min(), utilities.max(), 100)
    normal_pdf = stats.norm.pdf(x, utilities.mean(), utilities.std())
    ax1.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal PDF')
    ax1.legend()
    
    # Add statistics
    ax1.text(0.02, 0.98, f'Mean: {utilities.mean():.2f}\nStd: {utilities.std():.2f}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plots 2-4: Probability density for each epsilon
    for i, epsilon in enumerate(epsilon_values):
        row, col = [(0, 1), (0, 2), (1, 0)][i]
        ax = axes[row, col]
        
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Create histogram of probabilities
        n, bins, patches = ax.hist(probabilities, bins=20, alpha=0.7, 
                                  color=colors[i], edgecolor='black', density=True)
        
        ax.set_title(f'ε = {epsilon} - Probability Density', fontweight='bold')
        ax.set_xlabel('Sampling Probability')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        max_prob = np.max(probabilities)
        min_prob = np.min(probabilities)
        mean_prob = np.mean(probabilities)
        std_prob = np.std(probabilities)
        
        ax.text(0.02, 0.98, f'Mean: {mean_prob:.4f}\nStd: {std_prob:.4f}\nRange: [{min_prob:.4f}, {max_prob:.4f}]', 
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add vertical lines for key statistics
        ax.axvline(mean_prob, color='red', linestyle='--', alpha=0.7, label='Mean')
        ax.axvline(max_prob, color='green', linestyle='--', alpha=0.7, label='Max')
        ax.legend()
    
    # Plot 5: Probability density comparison (overlay)
    ax5 = axes[1, 1]
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Create histogram
        n, bins, patches = ax5.hist(probabilities, bins=20, alpha=0.3, 
                                   color=colors[i], density=True, 
                                   label=f'ε = {epsilon}')
        
        # Add smooth curve using KDE
        kde = stats.gaussian_kde(probabilities)
        x_smooth = np.linspace(probabilities.min(), probabilities.max(), 100)
        kde_values = kde(x_smooth)
        ax5.plot(x_smooth, kde_values, color=colors[i], linewidth=3, 
                label=f'ε = {epsilon} (KDE)')
    
    ax5.set_xlabel('Sampling Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Probability Density Comparison', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Probability vs Utility density scatter
    ax6 = axes[1, 2]
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Create density scatter plot
        ax6.scatter(utilities, probabilities, label=f'ε = {epsilon}', 
                   alpha=0.6, s=60, color=colors[i], edgecolors='black', linewidth=0.5)
    
    ax6.set_xlabel('Utility Value')
    ax6.set_ylabel('Sampling Probability')
    ax6.set_title('Probability vs Utility Density', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add trend lines
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        z = np.polyfit(utilities, probabilities, 1)
        p = np.poly1d(z)
        ax6.plot(utilities, p(utilities), color=colors[i], linestyle='--', alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/probability_density_functions.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional density plots
    create_additional_density_plots(utilities, epsilon_values, colors)

def create_additional_density_plots(utilities, epsilon_values, colors):
    """Create additional probability density plots"""
    
    # Plot 1: Log-scale probability density
    plt.figure(figsize=(12, 8))
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Create histogram on log scale
        plt.hist(probabilities, bins=25, alpha=0.3, color=colors[i], 
                density=True, label=f'ε = {epsilon}')
        
        # Add smooth KDE curve
        kde = stats.gaussian_kde(probabilities)
        x_smooth = np.linspace(probabilities.min(), probabilities.max(), 200)
        kde_values = kde(x_smooth)
        plt.plot(x_smooth, kde_values, color=colors[i], linewidth=3, 
                label=f'ε = {epsilon} (KDE)')
    
    plt.xlabel('Sampling Probability', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Probability Density Functions (Log Scale)', fontsize=14, fontweight='bold')
    plt.yscale('log')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/probability_density_log_scale.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Probability density by utility bins
    plt.figure(figsize=(14, 8))
    
    # Create utility bins
    utility_bins = np.linspace(utilities.min(), utilities.max(), 6)
    bin_centers = (utility_bins[:-1] + utility_bins[1:]) / 2
    
    for i, epsilon in enumerate(epsilon_values):
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        # Calculate mean probability for each utility bin
        mean_probs_by_bin = []
        for j in range(len(utility_bins) - 1):
            mask = (utilities >= utility_bins[j]) & (utilities < utility_bins[j + 1])
            if np.any(mask):
                mean_prob = np.mean(probabilities[mask])
                mean_probs_by_bin.append(mean_prob)
            else:
                mean_probs_by_bin.append(0)
        
        plt.plot(bin_centers, mean_probs_by_bin, 'o-', linewidth=3, 
                markersize=8, color=colors[i], label=f'ε = {epsilon}')
    
    plt.xlabel('Utility Value (Bin Centers)', fontsize=12)
    plt.ylabel('Mean Sampling Probability', fontsize=12)
    plt.title('Mean Probability by Utility Bins', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/probability_by_utility_bins.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Probability distribution statistics
    plt.figure(figsize=(12, 8))
    
    # Calculate statistics for each epsilon
    stats_data = {
        'Mean': [],
        'Std': [],
        'Skewness': [],
        'Kurtosis': [],
        'Max/Min Ratio': []
    }
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        stats_data['Mean'].append(np.mean(probabilities))
        stats_data['Std'].append(np.std(probabilities))
        stats_data['Skewness'].append(stats.skew(probabilities))
        stats_data['Kurtosis'].append(stats.kurtosis(probabilities))
        stats_data['Max/Min Ratio'].append(np.max(probabilities) / np.min(probabilities))
    
    # Create subplots for each statistic
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Probability Distribution Statistics', fontsize=16, fontweight='bold')
    
    stat_names = list(stats_data.keys())
    for i, stat_name in enumerate(stat_names):
        row, col = i // 3, i % 3
        ax = axes[row, col]
        
        bars = ax.bar(epsilon_values, stats_data[stat_name], 
                     color=[colors[j] for j in range(len(epsilon_values))],
                     alpha=0.7, edgecolor='black')
        
        ax.set_title(f'{stat_name}', fontweight='bold')
        ax.set_xlabel('ε (Epsilon)')
        ax.set_ylabel(stat_name)
        ax.set_xticks(epsilon_values)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, stats_data[stat_name]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Remove empty subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/yizhang/tech4HSE/probability_statistics.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    create_probability_density_plots()
    
    # Print density analysis summary
    utilities = generate_normal_utilities()
    epsilon_values = [1, 2, 3]
    
    print("\n" + "=" * 80)
    print("PROBABILITY DENSITY FUNCTION ANALYSIS")
    print("=" * 80)
    
    for epsilon in epsilon_values:
        probabilities = exponential_mechanism_probabilities(utilities, epsilon)
        
        print(f"\nε = {epsilon}:")
        print(f"  Mean probability: {np.mean(probabilities):.6f}")
        print(f"  Std probability: {np.std(probabilities):.6f}")
        print(f"  Skewness: {stats.skew(probabilities):.4f}")
        print(f"  Kurtosis: {stats.kurtosis(probabilities):.4f}")
        print(f"  Max/Min ratio: {np.max(probabilities)/np.min(probabilities):.2f}")
        
        # Distribution shape description
        skewness = stats.skew(probabilities)
        if abs(skewness) < 0.5:
            shape_desc = "Approximately symmetric"
        elif skewness > 0.5:
            shape_desc = "Right-skewed (positive skew)"
        else:
            shape_desc = "Left-skewed (negative skew)"
        
        print(f"  Distribution shape: {shape_desc}")
        
        # Percentiles
        percentiles = np.percentile(probabilities, [25, 50, 75, 90, 95, 99])
        print(f"  Percentiles: 25%={percentiles[0]:.4f}, 50%={percentiles[1]:.4f}, "
              f"75%={percentiles[2]:.4f}, 90%={percentiles[3]:.4f}, "
              f"95%={percentiles[4]:.4f}, 99%={percentiles[5]:.4f}")
