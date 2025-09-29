#!/usr/bin/env python3
"""
Generate radar plots from comprehensive PPI protection results
Using Plotly for interactive 5-dimension radar plots
"""

import os
import json
import plotly.graph_objects as go
import plotly.express as px

def load_comprehensive_results():
    """Load the comprehensive PPI protection results"""
    results_file = 'pii_protection_results_20250927_220805.json'
    with open(results_file, 'r') as f:
        return json.load(f)

def radar_plots_per_epsilon(merged, epsilons, out_prefix):
    """Generate radar plots for each epsilon value using Plotly"""
    mechs = ["InferDPT", "PhraseDP", "SANTEXT+", "CusText+", "CluSanT"]  # InferDPT first for legend top position, CluSanT last for visual front
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
        
        for mech in mechs:
            if mech not in merged:
                continue
            
            # Extract values for this mechanism and epsilon
            vals = [merged[mech].get(str(e), {}).get(m, 0.0) for m in metrics]
            
            # Convert to percentages (0-100)
            vals_percent = [v * 100 for v in vals]
            
            # Add trace for this mechanism with clean fill only
            # Lower opacity for CluSanT to make it more faded
            opacity = 0.2 if mech == "CluSanT" else 0.4
            
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
        
        # Update layout
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
                text=f"PII Protection Radar Plot (ε={e})",
                font=dict(size=20)
            ),
            font=dict(size=14),
            height=600,
            width=800,
            legend=dict(
                font=dict(size=14),
                x=1.1,
                y=1.0,
                traceorder="normal",  # Normal order to keep InferDPT at top
                itemsizing="constant"
            )
        )
        
        # Save as PDF for publication quality
        pdf_path = f"{out_prefix}_eps_{str(e).replace('.', '_')}.pdf"
        
        fig.write_image(pdf_path, width=800, height=600)
        
        print(f"Created radar plot: {pdf_path}")

def main():
    # Load comprehensive results
    print("Loading comprehensive PPI protection results...")
    results = load_comprehensive_results()
    
    # Generate radar plots for each epsilon
    epsilons = [1.0, 1.5, 2.0, 2.5, 3.0]
    output_dir = "/home/yizhang/tech4HSE/experiment_results/ppi-protection"
    timestamp = "20250927"
    
    radar_prefix = os.path.join(output_dir, f'protection_radar_5mech_{timestamp}')
    
    print("Generating radar plots...")
    radar_plots_per_epsilon(results, epsilons, radar_prefix)
    
    print("✅ Radar plots generated successfully!")
    print("Generated files:")
    for e in epsilons:
        print(f"- {radar_prefix}_eps_{str(e).replace('.', '_')}.pdf")

if __name__ == "__main__":
    main()
