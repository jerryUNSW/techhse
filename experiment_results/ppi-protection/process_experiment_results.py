#!/usr/bin/env python3
"""
Process Experiment Results
Automatically generates plots from completed experiment results
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Global font sizing for high-readability figures
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})

def find_latest_results():
    """Find the latest experiment results file."""
    pattern = "pii_protection_results_*.json"
    files = glob.glob(pattern)

    if files:
        return max(files, key=os.path.getctime)
    return None

def load_results(results_file):
    """Load experiment results."""
    with open(results_file, 'r') as f:
        return json.load(f)

def create_comprehensive_plots(results, output_dir="/home/yizhang/tech4HSE/experiment_results/ppi-protection"):
    """Create comprehensive plots from experiment results."""

    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    pii_types = ['emails', 'phones', 'addresses', 'names']

    # Colors for each mechanism
    colors = {
        "PhraseDP": "#1f77b4",   # blue
        "InferDPT": "#ff7f0e",   # orange
        "SANTEXT+": "#2ca02c",   # green
        "CusText+": "#d62728",   # red
        "CluSanT": "#9467bd"     # purple
    }

    # Line styles
    linestyles = {
        "PhraseDP": "-",
        "InferDPT": "--",
        "SANTEXT+": "-.",
        "CusText+": ":",
        "CluSanT": (0, (3, 1, 1, 1))
    }

    # Markers
    markers = {
        "PhraseDP": "o",
        "InferDPT": "s",
        "SANTEXT+": "^",
        "CusText+": "D",
        "CluSanT": "P"
    }

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Overall Protection vs Epsilon Plot
    plt.figure(figsize=(12, 8))

    for mechanism in mechanisms:
        if mechanism in results:
            overall_rates = []
            for eps in epsilon_values:
                if str(eps) in results[mechanism]:
                    overall_rates.append(results[mechanism][str(eps)]['overall'])
                else:
                    overall_rates.append(0.0)

            plt.plot(epsilon_values, overall_rates,
                    color=colors[mechanism], linestyle=linestyles[mechanism],
                    marker=markers[mechanism], linewidth=3, markersize=8,
                    label=mechanism)

    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Overall Protection Rate')
    plt.title('PII Protection Rate vs Epsilon - Comprehensive Experiment\n(100 Samples, 5 Mechanisms)', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0.8, 3.2)
    plt.ylim(-0.05, 1.05)

    # Add value annotations
    for mechanism in mechanisms:
        if mechanism in results:
            for eps in epsilon_values:
                if str(eps) in results[mechanism]:
                    rate = results[mechanism][str(eps)]['overall']
                    plt.annotate(f'{rate:.2f}', (eps, rate),
                               textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    plt.tight_layout()
    overall_plot_path = os.path.join(output_dir, f'comprehensive_overall_protection_{timestamp}.png')
    plt.savefig(overall_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved overall protection plot: {overall_plot_path}")
    plt.close()

    # 2. Individual PII Type Protection Plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    pii_labels = {
        'emails': 'Email Protection',
        'phones': 'Phone Protection',
        'addresses': 'Address Protection',
        'names': 'Name Protection'
    }

    for i, pii_type in enumerate(pii_types):
        ax = axes[i]

        for mechanism in mechanisms:
            if mechanism in results:
                rates = []
                for eps in epsilon_values:
                    if str(eps) in results[mechanism]:
                        rates.append(results[mechanism][str(eps)][pii_type])
                    else:
                        rates.append(0.0)

                ax.plot(epsilon_values, rates,
                       color=colors[mechanism], linestyle=linestyles[mechanism],
                       marker=markers[mechanism], linewidth=2, markersize=6,
                       label=mechanism)

        ax.set_xlabel('Epsilon (ε)')
        ax.set_ylabel('Protection Rate')
        ax.set_title(pii_labels[pii_type], fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.8, 3.2)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle('PII Protection by Type vs Epsilon - Comprehensive Experiment', fontsize=16, fontweight='bold')
    plt.tight_layout()

    pii_types_plot_path = os.path.join(output_dir, f'comprehensive_pii_types_protection_{timestamp}.png')
    plt.savefig(pii_types_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved PII types protection plot: {pii_types_plot_path}")
    plt.close()

    # 3. Radar Plots for Each Epsilon
    for eps in epsilon_values:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Calculate angles for each PII type
        angles = np.linspace(0, 2 * np.pi, len(pii_types), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for mechanism in mechanisms:
            if mechanism in results and str(eps) in results[mechanism]:
                values = []
                for pii_type in pii_types:
                    values.append(results[mechanism][str(eps)][pii_type])
                values += values[:1]  # Complete the circle

                ax.plot(angles, values, color=colors[mechanism], linewidth=2,
                       linestyle=linestyles[mechanism], marker=markers[mechanism],
                       markersize=6, label=mechanism)
                ax.fill(angles, values, color=colors[mechanism], alpha=0.25)

        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([pii_labels[pii_type] for pii_type in pii_types], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.title(f'PII Protection Radar Chart - Epsilon {eps}\nComprehensive Experiment (100 Samples)',
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

        plt.tight_layout()
        radar_plot_path = os.path.join(output_dir, f'comprehensive_radar_eps_{eps}_{timestamp}.png')
        plt.savefig(radar_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved radar plot for epsilon {eps}: {radar_plot_path}")
        plt.close()

    return [overall_plot_path, pii_types_plot_path] + [os.path.join(output_dir, f'comprehensive_radar_eps_{eps}_{timestamp}.png') for eps in epsilon_values]

def print_summary_report(results):
    """Print a comprehensive summary report."""
    print("\n" + "="*80)
    print("COMPREHENSIVE PII PROTECTION EXPERIMENT RESULTS")
    print("="*80)
    print("100 Samples | 5 Mechanisms | 5 Epsilon Values | 4 PII Types")
    print("="*80)

    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]

    for mechanism in mechanisms:
        if mechanism not in results:
            continue

        print(f"\n{mechanism}:")
        print("  Eps  | Overall | Names  | Emails | Phones | Addrs")
        print("  -----|---------|--------|--------|--------|-------")

        for eps in epsilon_values:
            eps_str = str(eps)
            if eps_str in results[mechanism]:
                data = results[mechanism][eps_str]
                print(f"  {eps:3.1f}  |  {data['overall']:5.3f}  | {data['names']:5.3f}  | {data['emails']:5.3f}  | {data['phones']:5.3f}  | {data['addresses']:5.3f}")

    # Find best performing mechanism overall
    print("\n" + "="*80)
    print("PERFORMANCE RANKING (Average Overall Protection)")
    print("="*80)

    mechanism_averages = {}
    for mechanism in mechanisms:
        if mechanism in results:
            overall_rates = []
            for eps in epsilon_values:
                if str(eps) in results[mechanism]:
                    overall_rates.append(results[mechanism][str(eps)]['overall'])

            if overall_rates:
                mechanism_averages[mechanism] = np.mean(overall_rates)

    ranked_mechanisms = sorted(mechanism_averages.items(), key=lambda x: x[1], reverse=True)

    for i, (mechanism, avg_rate) in enumerate(ranked_mechanisms, 1):
        print(f"{i}. {mechanism}: {avg_rate:.3f} average protection rate")

def main():
    """Main function to process experiment results and generate plots."""
    print("🔍 Looking for latest experiment results...")

    results_file = find_latest_results()
    if not results_file:
        print("❌ No experiment results found!")
        print("   Make sure pii_protection_experiment.py has completed successfully")
        return False

    print(f"✅ Found results: {os.path.basename(results_file)}")

    # Load results
    results = load_results(results_file)
    print(f"✅ Loaded results for {len(results)} mechanisms")

    # Print summary report
    print_summary_report(results)

    # Generate plots
    print(f"\n📊 Generating comprehensive plots...")
    plot_files = create_comprehensive_plots(results)

    print(f"\n🎉 Processing complete!")
    print(f"📁 Results file: {results_file}")
    print(f"🖼️  Generated {len(plot_files)} plot files")

    return True, results_file, plot_files

if __name__ == "__main__":
    main()