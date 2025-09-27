#!/usr/bin/env python3
"""
Generate line plots showing PPI protection vs epsilon for all mechanisms
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math

def load_results(file_path):
    """Load the PPI protection results"""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_epsilon_data(results):
    """Extract protection rates by epsilon for each mechanism"""
    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    data = {}
    
    for mechanism in mechanisms:
        if mechanism in results and results[mechanism]:
            data[mechanism] = {
                'overall': [],
                'emails': [],
                'phones': [],
                'addresses': [],
                'names': []
            }
            
            for eps in epsilon_values:
                eps_str = str(eps)
                if eps_str in results[mechanism]:
                    data[mechanism]['overall'].append(results[mechanism][eps_str].get('overall', 0))
                    data[mechanism]['emails'].append(results[mechanism][eps_str].get('emails', 0))
                    data[mechanism]['phones'].append(results[mechanism][eps_str].get('phones', 0))
                    data[mechanism]['addresses'].append(results[mechanism][eps_str].get('addresses', 0))
                    data[mechanism]['names'].append(results[mechanism][eps_str].get('names', 0))
                else:
                    # Fill with zeros if epsilon not found
                    data[mechanism]['overall'].append(0)
                    data[mechanism]['emails'].append(0)
                    data[mechanism]['phones'].append(0)
                    data[mechanism]['addresses'].append(0)
                    data[mechanism]['names'].append(0)
    
    return data, epsilon_values

def create_line_plots(data, epsilon_values, output_dir):
    """Create line plots for each PII type"""
    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    colors = ['#4169E1', '#228B22', '#FF6347', '#8B008B', '#DC143C']  # Blue, Green, Tomato, Dark Magenta, Crimson (matching image style)
    markers = ['o', 's', '^', 'v', 'D']
    linestyles = ['-', '--', '-.', ':', '-']
    
    # Create overall protection plot
    plt.figure(figsize=(14, 10))
    
    for i, mechanism in enumerate(mechanisms):
        if mechanism in data:
            plt.plot(epsilon_values, data[mechanism]['overall'], 
                    color=colors[i], marker=markers[i], linestyle=linestyles[i],
                    linewidth=3.5, markersize=12, label=mechanism)
    
    plt.xlabel('Epsilon (ε)', fontsize=18, fontweight='bold')
    plt.ylabel('Overall Protection Rate', fontsize=18, fontweight='bold')
    plt.title('PPI Protection Rate vs Epsilon\n(All Mechanisms)', fontsize=22, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linewidth=1.5)
    plt.legend(fontsize=16, loc='best', framealpha=0.9)
    plt.ylim(0, 1.05)
    plt.xlim(0.8, 3.2)
    
    # Add percentage labels on y-axis with larger font
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.tick_params(axis='both', which='major', labelsize=16)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'overall_protection_vs_epsilon_all_mechanisms.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual PII type plots
    pii_types = ['emails', 'phones', 'addresses', 'names']
    pii_labels = ['Email Protection', 'Phone Protection', 'Address Protection', 'Name Protection']
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for idx, (pii_type, pii_label) in enumerate(zip(pii_types, pii_labels)):
        ax = axes[idx]
        
        for i, mechanism in enumerate(mechanisms):
            if mechanism in data:
                ax.plot(epsilon_values, data[mechanism][pii_type], 
                       color=colors[i], marker=markers[i], linestyle=linestyles[i],
                       linewidth=3.5, markersize=12, label=mechanism)
        
        ax.set_xlabel('Epsilon (ε)', fontsize=16, fontweight='bold')
        ax.set_ylabel(pii_label, fontsize=16, fontweight='bold')
        ax.set_title(pii_label, fontsize=18, fontweight='bold')
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.legend(fontsize=14, framealpha=0.9)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0.8, 3.2)
        
        # Add percentage labels on y-axis with larger font
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'individual_pii_protection_vs_epsilon.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_radar_plots(data, epsilon_values, output_dir):
    """Create radar plots for each epsilon value"""
    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    colors = ['#4169E1', '#228B22', '#FF6347', '#8B008B', '#DC143C']  # Blue, Green, Tomato, Dark Magenta, Crimson (matching image style)
    linestyles = ['-', '--', '-.', ':', '-']  # Different line styles for better distinction
    
    # PII dimensions for radar plot (including overall as first dimension)
    pii_types = ['overall', 'emails', 'phones', 'addresses', 'names']
    pii_labels = ['Overall', 'Emails', 'Phones', 'Addresses', 'Names']
    
    # Number of variables for radar plot
    N = len(pii_types)
    
    # Calculate angles for each axis - start with Overall at the top (12 o'clock position)
    angles = [n / float(N) * 2 * math.pi - math.pi/2 for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create radar plot for each epsilon
    for eps in epsilon_values:
        eps_str = str(eps)
        
        fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(projection='polar'))
        
        for i, mechanism in enumerate(mechanisms):
            if mechanism in data:
                # Get values for this epsilon
                values = []
                eps_idx = epsilon_values.index(eps)
                for pii_type in pii_types:
                    if eps_idx < len(data[mechanism][pii_type]):
                        values.append(data[mechanism][pii_type][eps_idx])
                    else:
                        values.append(0)
                
                
                # Complete the circle
                values += values[:1]
                
                # Plot
                ax.plot(angles, values, 'o', linewidth=2.0, linestyle=linestyles[i],
                       label=mechanism, color=colors[i], markersize=10)
                ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(pii_labels, fontsize=20, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=18)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        
        # Move dimension labels further from circle - set them outside the plot area
        ax.tick_params(axis='x', pad=50)  # Increase padding for dimension labels
        
        # Add title
        plt.title(f'PPI Protection Radar Plot\nEpsilon = {eps}', 
                 size=24, fontweight='bold', pad=40)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(1.6, 1.0), fontsize=18, framealpha=0.9)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(output_dir / f'protection_radar_eps_{eps}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created radar plot for epsilon {eps}")


def main():
    # Setup paths
    results_file = '/home/yizhang/tech4HSE/pii_protection_results_20250927_220805.json'
    output_dir = Path('/home/yizhang/tech4HSE/experiment_results/ppi-protection')
    output_dir.mkdir(exist_ok=True)
    
    print("Loading PPI protection results...")
    results = load_results(results_file)
    
    print("Extracting epsilon data...")
    data, epsilon_values = extract_epsilon_data(results)
    
    print("Creating line plots...")
    create_line_plots(data, epsilon_values, output_dir)
    
    print("Creating radar plots...")
    create_radar_plots(data, epsilon_values, output_dir)
    
    print(f"✅ All plots saved to: {output_dir}")
    print("Generated files:")
    print("- overall_protection_vs_epsilon_all_mechanisms.png")
    print("- individual_pii_protection_vs_epsilon.png")
    print("- protection_radar_eps_1.0.png")
    print("- protection_radar_eps_1.5.png")
    print("- protection_radar_eps_2.0.png")
    print("- protection_radar_eps_2.5.png")
    print("- protection_radar_eps_3.0.png")
    
    # Print summary statistics
    print("\n📊 Protection Rate Summary:")
    print("-" * 50)
    for mechanism in ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']:
        if mechanism in data:
            overall_rates = data[mechanism]['overall']
            max_protection = max(overall_rates)
            avg_protection = np.mean(overall_rates)
            print(f"{mechanism:12}: Max={max_protection:.1%}, Avg={avg_protection:.1%}")

if __name__ == "__main__":
    main()
