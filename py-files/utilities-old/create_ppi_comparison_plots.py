import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

# Data from Experiment 1 (rows 0-9) for existing mechanisms
base_data = {
    "PhraseDP": {
        1.0: {"overall": 0.950, "emails": 0.900, "phones": 1.000, "addresses": 1.000, "names": 0.900},
        1.5: {"overall": 1.000, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 1.000},
        2.0: {"overall": 1.000, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 1.000},
        2.5: {"overall": 0.975, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 0.900},
        3.0: {"overall": 1.000, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 1.000}
    },
    "InferDPT": {
        1.0: {"overall": 1.000, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 1.000},
        1.5: {"overall": 1.000, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 1.000},
        2.0: {"overall": 1.000, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 1.000},
        2.5: {"overall": 1.000, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 1.000},
        3.0: {"overall": 1.000, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 1.000}
    },
    "SANTEXT+": {
        1.0: {"overall": 0.842, "emails": 1.000, "phones": 1.000, "addresses": 1.000, "names": 0.500},
        1.5: {"overall": 0.842, "emails": 1.000, "phones": 1.000, "addresses": 0.900, "names": 0.600},
        2.0: {"overall": 0.867, "emails": 1.000, "phones": 1.000, "addresses": 0.900, "names": 0.600},
        2.5: {"overall": 0.817, "emails": 1.000, "phones": 1.000, "addresses": 0.900, "names": 0.500},
        3.0: {"overall": 0.792, "emails": 1.000, "phones": 1.000, "addresses": 0.800, "names": 0.500}
    }
}

epsilons = [1.0, 1.5, 2.0, 2.5, 3.0]
mechanisms = ["PhraseDP", "InferDPT", "SANTEXT+"]
colors = {"PhraseDP": "#1f77b4", "InferDPT": "#2ca02c", "SANTEXT+": "#d62728"}
pii_types = ["overall", "emails", "phones", "addresses", "names"]
pii_labels = ["Overall", "Emails", "Phones", "Addresses", "Names"]

# Try to load latest CluSanT results and merge
results_dir = "/home/yizhang/tech4HSE/results"
clusant_prefix = "clusant_ppi_protection_"
latest_clusant = None
if os.path.isdir(results_dir):
    files = [f for f in os.listdir(results_dir) if f.startswith(clusant_prefix) and f.endswith(".json")]
    if files:
        latest_clusant = max(files)  # lexicographically ok due to timestamp

combined_data = dict(base_data)
if latest_clusant:
    try:
        with open(os.path.join(results_dir, latest_clusant), "r") as f:
            cl_data = json.load(f)
        # Normalize CluSanT into the same schema
        cl_mech = cl_data.get("CluSanT", {})
        clusant_entry = {}
        for eps in epsilons:
            eps_key = str(eps)
            # values might be stored with float keys; check both
            val = cl_mech.get(eps, None)
            if val is None:
                val = cl_mech.get(eps_key, None)
            if isinstance(val, dict):
                clusant_entry[eps] = {
                    "overall": float(val.get("overall", 0.0)),
                    "emails": float(val.get("emails", 0.0)),
                    "phones": float(val.get("phones", 0.0)),
                    "addresses": float(val.get("addresses", 0.0)),
                    "names": float(val.get("names", 0.0)),
                }
        if clusant_entry:
            combined_data["CluSanT"] = clusant_entry
            mechanisms.append("CluSanT")
            colors["CluSanT"] = "#9467bd"
            print(f"Loaded CluSanT results from {latest_clusant}")
        else:
            print("CluSanT data present but empty for requested epsilons; skipping.")
    except Exception as e:
        print(f"Failed to load CluSanT results: {e}")
else:
    print("No CluSanT results found; proceeding without it.")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = "/home/yizhang/tech4HSE/plots/ppi"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# 1. Overall Performance Comparison
plt.figure(figsize=(10, 6))
for mech in mechanisms:
    overall_scores = [combined_data[mech][eps]["overall"] for eps in epsilons if eps in combined_data[mech]]
    plt.plot(epsilons[:len(overall_scores)], overall_scores, marker='o', linewidth=3,
             label=mech, color=colors.get(mech, None), markersize=8)

plt.title('Overall PII Protection Performance vs Epsilon\n(Experiment 1: 10 samples)', fontsize=14, fontweight='bold')
plt.xlabel('Epsilon (ε)', fontsize=12)
plt.ylabel('Protection Rate', fontsize=12)
plt.ylim(0.7, 1.05)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11, frameon=False)
plt.tight_layout()
plt.savefig(f"{output_dir}/overall_performance_comparison_{timestamp}.png", dpi=150, bbox_inches='tight')
plt.close()

# 2. PII Type-Specific Performance (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, (pii_type, pii_label) in enumerate(zip(pii_types[1:], pii_labels[1:])):
    ax = axes[i]
    for mech in mechanisms:
        scores = [combined_data[mech][eps][pii_type] for eps in epsilons if eps in combined_data[mech]]
        ax.plot(epsilons[:len(scores)], scores, marker='o', linewidth=2.5,
                label=mech, color=colors.get(mech, None), markersize=6)
    ax.set_title(f'{pii_label} Protection', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epsilon (ε)', fontsize=10)
    ax.set_ylabel('Protection Rate', fontsize=10)
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, frameon=False)

plt.suptitle('PII Type-Specific Protection Performance\n(Experiment 1: 10 samples)',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{output_dir}/pii_type_performance_{timestamp}.png", dpi=150, bbox_inches='tight')
plt.close()

# 3. Mechanism Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 8))

# Calculate average performance across all epsilons
avg_performance = {}
for mech in mechanisms:
    avg_performance[mech] = {
        pii_type: np.mean([combined_data[mech][eps][pii_type] for eps in epsilons if eps in combined_data[mech]])
        for pii_type in pii_types
    }

x = np.arange(len(pii_labels))
width = 0.8 / max(1, len(mechanisms))

for i, mech in enumerate(mechanisms):
    scores = [avg_performance[mech][pii_type] for pii_type in pii_types]
    ax.bar(x + i*width, scores, width, label=mech, color=colors.get(mech, None), alpha=0.8)

ax.set_xlabel('PII Types', fontsize=12)
ax.set_ylabel('Average Protection Rate', fontsize=12)
ax.set_title('Average PII Protection Performance by Mechanism\n(Experiment 1: 10 samples, averaged across all epsilons)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x + (len(mechanisms)-1)*width/2)
ax.set_xticklabels(pii_labels)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11, frameon=False)
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, mech in enumerate(mechanisms):
    scores = [avg_performance[mech][pii_type] for pii_type in pii_types]
    for j, score in enumerate(scores):
        ax.text(j + i*width, score + 0.01, f'{score:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{output_dir}/mechanism_comparison_bars_{timestamp}.png", dpi=150, bbox_inches='tight')
plt.close()

# 4. Epsilon Sensitivity Heatmap
fig, ax = plt.subplots(figsize=(10, 6))

# Create matrix for heatmap
heatmap_data = []
for mech in mechanisms:
    row = [combined_data[mech][eps]["overall"] for eps in epsilons if eps in combined_data[mech]]
    # pad to full length if needed
    if len(row) < len(epsilons):
        row = row + [np.nan] * (len(epsilons) - len(row))
    heatmap_data.append(row)

heatmap_data = np.array(heatmap_data)

im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0.7, vmax=1.0)

# Set ticks and labels
ax.set_xticks(range(len(epsilons)))
ax.set_xticklabels([f'ε={eps}' for eps in epsilons])
ax.set_yticks(range(len(mechanisms)))
ax.set_yticklabels(mechanisms)

# Add text annotations
for i in range(len(mechanisms)):
    for j in range(len(epsilons)):
        val = heatmap_data[i, j]
        if not np.isnan(val):
            ax.text(j, i, f'{val:.3f}',
                    ha="center", va="center", color="black", fontweight='bold')

ax.set_title('Overall Protection Rate Heatmap\n(Experiment 1: 10 samples)',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Epsilon Values', fontsize=12)
ax.set_ylabel('Mechanisms', fontsize=12)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Protection Rate', fontsize=12)

plt.tight_layout()
plt.savefig(f"{output_dir}/epsilon_sensitivity_heatmap_{timestamp}.png", dpi=150, bbox_inches='tight')
plt.close()

# 5. Performance Ranking Radar Chart
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

# Calculate average scores for radar chart
angles = np.linspace(0, 2 * np.pi, len(pii_types), endpoint=False).tolist()
angles += angles[:1]  # Complete the circle

for mech in mechanisms:
    values = [avg_performance[mech][pii_type] for pii_type in pii_types]
    values += values[:1]  # Complete the circle
    ax.plot(angles, values, 'o-', linewidth=2, label=mech, color=colors.get(mech, None))
    ax.fill(angles, values, alpha=0.1, color=colors.get(mech, None))

ax.set_xticks(angles[:-1])
ax.set_xticklabels(pii_labels)
ax.set_ylim(0, 1)
ax.set_title('PII Protection Performance Radar\n(Experiment 1: 10 samples, averaged across all epsilons)',
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
ax.grid(True)

plt.tight_layout()
plt.savefig(f"{output_dir}/performance_radar_{timestamp}.png", dpi=150, bbox_inches='tight')
plt.close()

# 5b. Per-epsilon Radar Charts
angles = np.linspace(0, 2 * np.pi, len(pii_types), endpoint=False).tolist()
angles += angles[:1]
for eps in epsilons:
    # only plot mechanisms that have this epsilon
    mech_for_eps = [m for m in mechanisms if eps in combined_data.get(m, {})]
    if not mech_for_eps:
        continue
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    for mech in mech_for_eps:
        values = [combined_data[mech][eps][pii_type] for pii_type in pii_types]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=mech, color=colors.get(mech, None))
        ax.fill(angles, values, alpha=0.1, color=colors.get(mech, None))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(pii_labels)
    ax.set_ylim(0, 1)
    ax.set_title(f'PII Protection Radar by Mechanism\n(ε={eps}, 10 samples)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_radar_eps_{str(eps).replace('.', '_')}_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()

# 6. Summary Statistics Table
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('tight')
ax.axis('off')

# Create summary table
table_data = []
headers = ['Mechanism', 'Avg Overall', 'Avg Names', 'Min Overall', 'Max Overall', 'Epsilon Range']

for mech in mechanisms:
    available_eps = [eps for eps in epsilons if eps in combined_data[mech]]
    overall_scores = [combined_data[mech][eps]["overall"] for eps in available_eps]
    name_scores = [combined_data[mech][eps]["names"] for eps in available_eps]
    row = [
        mech,
        f'{np.mean(overall_scores):.3f}' if overall_scores else 'n/a',
        f'{np.mean(name_scores):.3f}' if name_scores else 'n/a',
        f'{np.min(overall_scores):.3f}' if overall_scores else 'n/a',
        f'{np.max(overall_scores):.3f}' if overall_scores else 'n/a',
        f'{(np.max(overall_scores) - np.min(overall_scores)):.3f}' if len(overall_scores) > 1 else 'n/a'
    ]
    table_data.append(row)

table = ax.table(cellText=table_data, colLabels=headers,
                cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 2)

# Color code the Avg Overall column where numeric
for i in range(len(mechanisms)):
    try:
        val = float(table_data[i][1])
    except Exception:
        continue
    cell = table[(i+1, 1)]
    if val >= 0.99:
        cell.set_facecolor('#d4edda')  # Light green
    elif val >= 0.9:
        cell.set_facecolor('#fff3cd')  # Light yellow
    else:
        cell.set_facecolor('#f8d7da')  # Light red

plt.tight_layout()
plt.savefig(f"{output_dir}/performance_summary_table_{timestamp}.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"Generated 6 comprehensive plots in {output_dir}:")
print(f"1. overall_performance_comparison_{timestamp}.png")
print(f"2. pii_type_performance_{timestamp}.png")
print(f"3. mechanism_comparison_bars_{timestamp}.png")
print(f"4. epsilon_sensitivity_heatmap_{timestamp}.png")
print(f"5. performance_radar_{timestamp}.png")
print(f"6. performance_summary_table_{timestamp}.png")
for eps in epsilons:
    print(f"- performance_radar_eps_{str(eps).replace('.', '_')}_{timestamp}.png")
