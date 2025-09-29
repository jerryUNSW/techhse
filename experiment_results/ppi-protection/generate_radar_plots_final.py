#!/usr/bin/env python3
"""
Generate radar plots from comprehensive PPI protection results
Using Plotly for interactive 5-dimension radar plots
FINAL VERSION: CluSanT brought to front with subtle visibility improvement
"""

import os
import json
import plotly.graph_objects as go
import plotly.express as px

def load_comprehensive_results():
    """Load the comprehensive PPI protection results"""
    results_file = 'comprehensive_ppi_protection_results_20250927_164033_backup.json'
    with open(results_file, 'r') as f:
        return json.load(f)

def load_latest_clusant_results():
    """Load the latest CluSanT results."""
    filename = 'pii_protection_results_20250929_071204.json'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def merge_results(comprehensive_results, clusant_results):
    """Merge comprehensive results with latest CluSanT results."""
    merged = comprehensive_results.copy()
    
    if 'CluSanT' in clusant_results:
        merged['CluSanT'] = clusant_results['CluSanT']
        print("✅ Updated CluSanT results with latest data")
    
    return merged

def radar_plots_per_epsilon(merged, epsilons, out_prefix):
    """Generate radar plots for each epsilon value using Plotly - FINAL VERSION"""
    # Draw other mechanisms first, then CluSanT last (on top)
    other_mechs = ["InferDPT", "PhraseDP", "SANTEXT+", "CusText+"]
    clusant_mech = ["CluSanT"]
    
    # Reorder to put Overall at top, then arrange others symmetrically
    metrics = ['overall', 'emails', 'phones', 'addresses', 'names']
    labels = ['Overall', 'Emails', 'Phones', 'Addresses', 'Names']
    
    colors = {
        "PhraseDP": "blue",
        "InferDPT": "cyan", 
        "SANTEXT+": "green",
        "CusText+": "purple",
        "CluSanT": "red",  # Regular red for CluSanT
    }
    
    # Distinct line styles for each mechanism
    line_styles = {
        "PhraseDP": "solid",
        "InferDPT": "dash", 
        "SANTEXT+": "dot",
        "CusText+": "dashdot",
        "CluSanT": "longdash",  # Use longdash for CluSanT
    }
    
    # Border colors for outlines (darker versions)
    border_colors = {
        "PhraseDP": "darkblue",
        "InferDPT": "darkcyan", 
        "SANTEXT+": "darkgreen",
        "CusText+": "darkmagenta",
        "CluSanT": "darkred",
    }
    
    for e in epsilons:
        fig = go.Figure()
        
        # First, draw other mechanisms with original styling
        for mech in other_mechs:
            if mech not in merged:
                continue
            
            # Extract values for this mechanism and epsilon
            vals = [merged[mech].get(str(e), {}).get(m, 0.0) for m in metrics]
            
            # Convert to percentages (0-100)
            vals_percent = [v * 100 for v in vals]
            
            # Add trace for this mechanism with original styling
            opacity = 0.4  # Original opacity for other mechanisms
            
            fig.add_trace(go.Scatterpolar(
                r=vals_percent,
                theta=labels,
                fill='toself',
                name=mech,
                fillcolor=colors.get(mech),
                opacity=opacity,
                line=dict(width=0),  # Remove outlines
                mode='lines',  # Ensure it's treated as lines only
                showlegend=True,
                legendgroup=mech
            ))
        
        # Now draw CluSanT last (on top) with subtle visibility improvement
        for mech in clusant_mech:
            if mech not in merged:
                continue
            
            # Extract values for this mechanism and epsilon
            vals = [merged[mech].get(str(e), {}).get(m, 0.0) for m in metrics]
            
            # Convert to percentages (0-100)
            vals_percent = [v * 100 for v in vals]
            
            # Add trace for CluSanT with subtle visibility improvement
            # Slightly higher opacity than original 0.2, but not too prominent
            opacity = 0.25  # Just slightly higher than original 0.2
            
            fig.add_trace(go.Scatterpolar(
                r=vals_percent,
                theta=labels,
                fill='toself',
                name=mech,
                fillcolor=colors.get(mech),
                opacity=opacity,
                line=dict(width=0),  # Remove outlines - keep original style
                mode='lines',  # Ensure it's treated as lines only
                showlegend=True,
                legendgroup=mech
            ))
        
        # Update layout - keeping original styling
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickmode='linear',
                    tick0=0,
                    dtick=20,
                    tickfont=dict(size=14)
                ),
                angularaxis=dict(
                    tickfont=dict(size=16),
                    rotation=90,  # Start from top (12 o'clock position)
                    direction='counterclockwise'
                )
            ),
            showlegend=True,
            title=dict(
                text=f'PII Protection Rate by Mechanism (ε={e})',
                x=0.5,
                font=dict(size=20)
            ),
            width=800,
            height=800,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Save the plot
        output_file = f"{out_prefix}_eps_{e}.png"
        fig.write_image(output_file, scale=2)
        print(f"✅ Saved final radar plot for epsilon {e}: {output_file}")
        
        # Also save as PDF and SVG for high-quality formats
        pdf_file = f"{out_prefix}_eps_{e}.pdf"
        svg_file = f"{out_prefix}_eps_{e}.svg"
        fig.write_image(pdf_file, scale=2)
        fig.write_image(svg_file, scale=2)

def main():
    """Main function to generate final radar plots with subtle CluSanT visibility"""
    print("=== Generating Final Radar Plots with Subtle CluSanT Visibility ===")
    print()
    
    # Load results
    print("Loading comprehensive results...")
    comprehensive_results = load_comprehensive_results()
    print("✅ Comprehensive results loaded")
    
    print("Loading latest CluSanT results...")
    clusant_results = load_latest_clusant_results()
    print("✅ Latest CluSanT results loaded")
    
    # Merge results
    print("Merging results...")
    merged_results = merge_results(comprehensive_results, clusant_results)
    print("✅ Results merged successfully")
    print()
    
    # Generate radar plots
    epsilons = [1.0, 1.5, 2.0, 2.5, 3.0]
    out_prefix = "protection_radar_5mech_final_20250929"
    
    print("Creating final radar plots for each epsilon value...")
    radar_plots_per_epsilon(merged_results, epsilons, out_prefix)
    print()
    
    print("🎉 All final radar plots generated successfully!")
    print("CluSanT is drawn last (on top) with subtle visibility improvement (0.25 opacity vs 0.2 original).")

if __name__ == "__main__":
    main()
