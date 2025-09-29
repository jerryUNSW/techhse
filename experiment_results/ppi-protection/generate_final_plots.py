#!/usr/bin/env python3
"""
Generate Final PPI Protection Plots using Recomputed Results

Uses the recomputed_ppi_protection_results.json file which contains
accurate protection rates computed directly from BIO label comparison.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Global font sizing for high-readability figures
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})

def load_recomputed_results():
    """Load the comprehensive protection results."""
    with open('comprehensive_ppi_protection_results_20250927_164033_backup.json', 'r') as f:
        return json.load(f)

def create_overall_protection_plot(results):
    """Create line plot showing protection rate vs epsilon for all mechanisms."""

    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]

    # Colors and styling
    colors = {
        "PhraseDP": "#1f77b4",   # blue
        "InferDPT": "#17becf",   # cyan
        "SANTEXT+": "#2ca02c",   # green
        "CusText+": "#9467bd",   # purple
        "CluSanT": "#e377c2",   # magenta
    }
    linestyles = {
        "PhraseDP": "-",
        "InferDPT": "--",
        "SANTEXT+": "-.",
        "CusText+": ":",
        "CluSanT": (0, (3, 1, 1, 1)),
    }
    markers = {
        "PhraseDP": "o",
        "InferDPT": "s",
        "SANTEXT+": "^",
        "CusText+": "D",
        "CluSanT": "P",
    }

    plt.figure(figsize=(12, 8))

    # Plot lines for each mechanism
    for mech in mechanisms:
        if mech not in results:
            continue
        y = [results[mech].get(str(e), {}).get('overall', 0.0) for e in epsilon_values]
        plt.plot(
            epsilon_values,
            y,
            marker=markers.get(mech, 'o'),
            linestyle=linestyles.get(mech, '-'),
            linewidth=3.2,
            label=mech,
            color=colors.get(mech),
            markersize=8,
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
            zorder=3,
        )

    plt.title('Overall PII Protection Rate vs Epsilon (Recomputed Results)', fontweight='bold')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Protection Rate')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3, zorder=0)
    plt.legend(frameon=True, loc='lower left')
    plt.tight_layout()

    # Save plot in both PNG and PDF formats
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_png = f"final_overall_protection_vs_epsilon_{timestamp}.png"
    output_file_pdf = f"final_overall_protection_vs_epsilon_{timestamp}.pdf"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved overall protection plot: {output_file_png} and {output_file_pdf}")
    plt.close()

def create_pii_type_plots(results):
    """Create individual PII type protection plots."""

    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    pii_types = ['emails', 'phones', 'addresses', 'names']

    colors = {
        "PhraseDP": "#1f77b4",
        "InferDPT": "#17becf",
        "SANTEXT+": "#2ca02c",
        "CusText+": "#9467bd",
        "CluSanT": "#e377c2",
    }

    linestyles = {
        "PhraseDP": "-",
        "InferDPT": "--",
        "SANTEXT+": "-.",
        "CusText+": ":",
        "CluSanT": (0, (3, 1, 1, 1)),
    }

    markers = {
        "PhraseDP": "o",
        "InferDPT": "s",
        "SANTEXT+": "^",
        "CusText+": "D",
        "CluSanT": "P",
    }

    # Create subplot for each PII type
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

        for j, mechanism in enumerate(mechanisms):
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

        ax.set_xlabel('Epsilon (ε)', fontsize=12)
        ax.set_ylabel('Protection Rate', fontsize=12)
        ax.set_title(pii_labels[pii_type], fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.8, 3.2)
        ax.set_ylim(-0.05, 1.05)

    plt.suptitle('PII Protection by Type vs Epsilon (Recomputed Results)', fontsize=16, fontweight='bold')
    plt.tight_layout()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file_png = f"final_pii_protection_by_type_{timestamp}.png"
    output_file_pdf = f"final_pii_protection_by_type_{timestamp}.pdf"
    plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_file_pdf, bbox_inches='tight')
    print(f"✅ Saved PII type protection plots: {output_file_png} and {output_file_pdf}")
    plt.close()

def create_radar_plots(results):
    """Create radar plots for each epsilon value."""

    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    pii_types = ['emails', 'phones', 'addresses', 'names']

    colors = {
        'PhraseDP': '#1f77b4',
        'InferDPT': '#17becf',
        'SANTEXT+': '#2ca02c',
        'CusText+': '#9467bd',
        'CluSanT': '#e377c2'
    }

    linestyles = {
        'PhraseDP': '-',
        'InferDPT': '--',
        'SANTEXT+': '-.',
        'CusText+': ':',
        'CluSanT': (0, (3, 1, 1, 1))
    }

    markers = {
        'PhraseDP': 'o',
        'InferDPT': 's',
        'SANTEXT+': '^',
        'CusText+': 'D',
        'CluSanT': 'P'
    }

    pii_labels = {
        'emails': 'Email Protection',
        'phones': 'Phone Protection',
        'addresses': 'Address Protection',
        'names': 'Name Protection'
    }

    for eps in epsilon_values:
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Calculate angles for each PII type
        angles = np.linspace(0, 2 * np.pi, len(pii_types), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for i, mechanism in enumerate(mechanisms):
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

        plt.title(f'PII Protection Radar Chart - Epsilon {eps} (Recomputed Results)',
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

        plt.tight_layout()

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file_png = f"final_protection_radar_eps_{eps}_{timestamp}.png"
        output_file_pdf = f"final_protection_radar_eps_{eps}_{timestamp}.pdf"
        plt.savefig(output_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(output_file_pdf, bbox_inches='tight')
        print(f"✅ Saved radar plot for epsilon {eps}: {output_file_png} and {output_file_pdf}")
        plt.close()

def main():
    """Main function to generate all final plots."""
    print("=== Generating Final PII Protection Plots ===")
    print()

    # Load recomputed results
    print("Loading recomputed protection results...")
    results = load_recomputed_results()
    print("✅ Recomputed results loaded successfully")
    print()

    # Create plots
    print("Creating overall protection vs epsilon plot...")
    create_overall_protection_plot(results)
    print()

    print("Creating individual PII type protection plots...")
    create_pii_type_plots(results)
    print()

    print("Creating radar plots for each epsilon value...")
    create_radar_plots(results)
    print()

    print("🎉 All final plots generated successfully!")

if __name__ == "__main__":
    main()