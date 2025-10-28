#!/usr/bin/env python3
"""
Generate radar plots from comprehensive PPI protection results
Using Plotly for interactive 5-dimension radar plots
FINAL VERSION: CluSanT brought to front with subtle visibility improvement
DATA SOURCE: SQLite database (tech4hse_results.db)
"""

import os
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for PDF generation
from pdf2image import convert_from_path
from PIL import Image
import subprocess

def autocrop_pdf(pdf_path, margin=10, manual_crop=None):
    """
    Auto-crop PDF by converting to image, detecting content bounds, and saving back.
    
    Args:
        pdf_path: Path to PDF file
        margin: Margin to add around auto-detected content (pixels)
        manual_crop: Dict with crop amounts in pixels: {'left': 0, 'right': 0, 'top': 0, 'bottom': 0}
                    If provided, overrides auto-detection
    """
    try:
        # Convert PDF to image (high DPI for quality)
        images = convert_from_path(pdf_path, dpi=300)
        if not images:
            print(f"⚠️  Could not convert {pdf_path} to image")
            return
        
        # Get the first (and only) page
        img = images[0]
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        width, height = img.size
        
        if manual_crop:
            # Manual cropping mode
            left = manual_crop.get('left', 0)
            top = manual_crop.get('top', 0)
            right = width - manual_crop.get('right', 0)
            bottom = height - manual_crop.get('bottom', 0)
            
            print(f"✂️  Manual crop: left={left}, top={top}, right={right}, bottom={bottom}")
        else:
            # Auto-detect content bounds
            gray = img.convert('L')
            bbox = gray.getbbox()
            
            if bbox:
                # Add margin
                left, upper, right, lower = bbox
                left = max(0, left - margin)
                top = max(0, upper - margin)
                right = min(width, right + margin)
                bottom = min(height, lower + margin)
            else:
                print(f"⚠️  Could not detect content bounds for {pdf_path}")
                return
        
        # Crop the image
        cropped = img.crop((left, top, right, bottom))
        
        # Save as PDF
        cropped.save(pdf_path, 'PDF', resolution=300.0)
        print(f"✂️  Cropped: {pdf_path} (size: {cropped.size[0]}x{cropped.size[1]})")
        
    except Exception as e:
        print(f"⚠️  Error cropping {pdf_path}: {e}")

def load_results_from_db(db_path='../../tech4hse_results.db'):
    """Load PPI protection results directly from SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query all protection results
    cursor.execute("""
        SELECT mechanism, epsilon, 
               overall_protection, email_protection, phone_protection,
               address_protection, name_protection
        FROM pii_protection_results
        ORDER BY mechanism, epsilon
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    # Structure results in the same format as JSON files
    results = {}
    
    for row in rows:
        mechanism, epsilon, overall, email, phone, address, name = row
        
        if mechanism not in results:
            results[mechanism] = {}
        
        results[mechanism][str(epsilon)] = {
            'overall': overall,
            'emails': email,
            'phones': phone,
            'addresses': address,
            'names': name
        }
    
    print(f"✅ Loaded data for {len(results)} mechanisms from database")
    for mech in results:
        epsilon_count = len(results[mech])
        print(f"   - {mech}: {epsilon_count} epsilon values")
    
    return results

def radar_plots_per_epsilon(merged, epsilons, out_prefix):
    """Generate radar plots for each epsilon value using Plotly - FINAL VERSION"""
    # Draw other mechanisms first, then CluSanT last (on top)
    other_mechs = ["PhraseDP", "InferDPT", "SANTEXT+", "CusText+"]
    clusant_mech = ["CluSanT"]
    
    # Reorder to put Overall at top, then arrange others symmetrically
    metrics = ['overall', 'emails', 'phones', 'addresses', 'names']
    labels = ['Overall', 'Email', 'Phone', 'Address', 'Name']
    
    # Use same colors as line plots
    colors = {
        "PhraseDP": "#FFD700",     # Yellow
        "InferDPT": "#2563EB",     # Royal blue
        "SANTEXT+": "#059669",     # Emerald
        "CusText+": "#7C3AED",     # Violet
        "CluSanT": "#EA580C",      # Orange
    }
    
    # Distinct line styles for each mechanism
    line_styles = {
        "PhraseDP": "solid",
        "InferDPT": "dash", 
        "SANTEXT+": "dot",
        "CusText+": "dashdot",
        "CluSanT": "longdash",
    }
    
    for e in epsilons:
        fig = go.Figure()
        
        # Draw all mechanisms with lines only (no fill)
        all_mechs = ["PhraseDP", "InferDPT", "SANTEXT+", "CusText+", "CluSanT"]
        
        for mech in all_mechs:
            if mech not in merged:
                continue
            
            # Extract values for this mechanism and epsilon
            vals = [merged[mech].get(str(e), {}).get(m, 0.0) for m in metrics]
            
            # Convert to percentages (0-100)
            vals_percent = [v * 100 for v in vals]
            
            # Close the polygon by adding first value to the end
            vals_percent_closed = vals_percent + [vals_percent[0]]
            labels_closed = labels + [labels[0]]
            
            # Add fill layer first (very transparent)
            # Make PhraseDP (yellow) less transparent for better visibility
            fill_opacity = 0.15 if mech == "PhraseDP" else 0.08
            fig.add_trace(go.Scatterpolar(
                r=vals_percent_closed,
                theta=labels_closed,
                fill='toself',
                fillcolor=colors.get(mech),
                opacity=fill_opacity,
                line=dict(width=0),
                showlegend=False,
                legendgroup=mech,
                hoverinfo='skip'
            ))
            
            # Add lines and markers on top (fully opaque)
            fig.add_trace(go.Scatterpolar(
                r=vals_percent_closed,
                theta=labels_closed,
                name=mech,
                line=dict(
                    color=colors.get(mech),
                    width=3,
                    dash=line_styles.get(mech)
                ),
                marker=dict(
                    color=colors.get(mech),
                    size=10,
                    symbol='circle',
                    line=dict(
                        color=colors.get(mech),
                        width=1
                    )
                ),
                mode='lines+markers',
                showlegend=True,
                legendgroup=mech
            ))
        
        # Update layout - no title, larger text and legend in 3 columns at top
        fig.update_layout(
            polar=dict(
                bgcolor='white',
                domain=dict(x=[0.0, 1.0], y=[0.02, 0.92]),  # Fill horizontal space, leave space at top for labels
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickmode='linear',
                    tick0=0,
                    dtick=20,
                    tickfont=dict(size=24, family='Arial', color='black'),
                    angle=18,  # Position radial labels along the line to Name (opposite direction)
                    gridcolor='lightgray',
                    gridwidth=1,
                    showline=False
                ),
                angularaxis=dict(
                    tickfont=dict(size=34, family='Arial Black', color='black'),
                    rotation=90,  # Start from top (12 o'clock position)
                    direction='counterclockwise',
                    gridcolor='lightgray',
                    gridwidth=1,
                    linecolor='lightgray',
                    linewidth=2,
                    ticklen=20,  # Make tick marks longer to push labels further out
                    tickcolor='lightgray'
                )
            ),
            showlegend=True,
            legend=dict(
                font=dict(size=32, family='Arial Black', color='black'),
                orientation='h',
                yanchor='bottom',
                y=-0.45,
                xanchor='center',
                x=0.5,
                bgcolor='rgba(255,255,255,0)',
                bordercolor='rgba(0,0,0,0)',
                borderwidth=0,
                itemsizing='constant',
                tracegroupgap=15,
                itemwidth=30
            ),
            width=550,
            height=900,
            margin=dict(l=0, r=0, t=40, b=5),
            autosize=False
        )
        
        # Save as PDF only
        pdf_file = f"radar_{out_prefix}_eps_{e}.pdf"
        fig.write_image(pdf_file, scale=2)
        print(f"✅ Saved final radar plot for epsilon {e}: {pdf_file}")
        
        # Manual crop to remove white space (crop more from left/right sides)
        # Adjust these values in pixels to control how much to crop from each side
        manual_crop_values = {
            'left': 400,    # Crop 400 pixels from left
            'right': 400,   # Crop 400 pixels from right
            'top': 50,      # Reduce top cropping for legend space
            'bottom': 0     # No bottom cropping to preserve legend space
        }
        autocrop_pdf(pdf_file, manual_crop=manual_crop_values)

def create_individual_pii_plots(results, epsilons, output_prefix):
    """Create 5 separate plots showing overall and individual PII protection rates vs epsilon using matplotlib."""
    
    # Set matplotlib rcParams following sample-plot.py style
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams["legend.framealpha"] = 0
    plt.rcParams["figure.figsize"] = (6, 5)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams["legend.handletextpad"] = 0.1
    plt.rcParams["legend.columnspacing"] = 0.3
    
    # PII types to plot (including overall)
    pii_types = ['overall', 'emails', 'phones', 'addresses', 'names']
    pii_labels = [
        'Overall PII Protection',
        'Email Protection', 
        'Phone Protection',
        'Address Protection',
        'Name Protection'
    ]
    
    # Mechanisms - order to match matplotlib plots
    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    
    # Use same colors as radar plots for consistency
    colors = {
        "PhraseDP": "#DC2626",     # Crimson red
        "InferDPT": "#2563EB",     # Royal blue
        "SANTEXT+": "#059669",     # Emerald
        "CusText+": "#7C3AED",     # Violet
        "CluSanT": "#EA580C",      # Orange
    }
    
    # Marker styles for each mechanism
    markers = {
        "PhraseDP": "o",     # Circle
        "InferDPT": "s",     # Square
        "SANTEXT+": "^",     # Triangle up
        "CusText+": "D",     # Diamond
        "CluSanT": "v",      # Triangle down
    }
    
    # Create a separate plot for each PII type
    for pii_type, label in zip(pii_types, pii_labels):
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(6, 5))
        
        # Plot each mechanism
        for mech in mechanisms:
            if mech not in results:
                continue
            
            # Extract protection rates for this PII type across epsilons
            protection_rates = []
            for eps in epsilons:
                rate = results[mech].get(str(eps), {}).get(pii_type, 0.0)
                protection_rates.append(rate * 100)  # Convert to percentage
            
            # Plot line with matplotlib style
            ax.plot(epsilons, protection_rates, 
                   color=colors[mech], 
                   linestyle='-', 
                   marker=markers[mech],
                   markersize=12,
                   markeredgewidth=2,
                   markerfacecolor='none',
                   linewidth=2,
                   label=mech)
        
        # Set labels and styling
        ax.set_xlabel(r"Privacy Budget ($\epsilon$)", fontsize=24, fontweight='bold')
        ax.set_ylabel("Protection Rate", fontsize=24, fontweight='bold')
        ax.set_title(label, fontsize=24, fontweight='bold', pad=15)
        
        # Set tick font sizes
        ax.tick_params(axis='both', labelsize=22)
        
        # Set y-axis range with more space at top for legend
        ax.set_ylim(0, 150)
        
        # Remove grid
        ax.grid(False)
        
        # Set axis linewidth
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
        
        # Add legend with 3 columns at the top with minimal spacing
        ax.legend(fontsize=14, loc='upper center', framealpha=0, ncol=3, columnspacing=0.1)
        
        plt.tight_layout()
        
        # Save as PDF only with "line" prefix
        pdf_file = f"line_{output_prefix}_{pii_type}.pdf"
        plt.savefig(pdf_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Saved {label} line plot: {pdf_file}")

def main():
    """Main function to generate final radar plots with subtle CluSanT visibility"""
    print("=== Generating Final Radar Plots with Subtle CluSanT Visibility ===")
    print("DATA SOURCE: SQLite database (tech4hse_results.db)")
    print()
    
    # Load results from database
    print("Loading PPI protection results from database...")
    merged_results = load_results_from_db()
    print()
    
    # Generate radar plots
    epsilons = [1.0, 1.5, 2.0, 2.5, 3.0]
    out_prefix = "pii_protection"
    
    print("Creating final radar plots for each epsilon value...")
    radar_plots_per_epsilon(merged_results, epsilons, out_prefix)
    print()
    
    print("Creating individual PII protection vs epsilon plots...")
    create_individual_pii_plots(merged_results, epsilons, out_prefix)
    print()
    
    print("🎉 All final plots generated successfully!")
    print("   - Output format: PDF only")
    print("   - Radar plots: 5 plots (one per epsilon value)")
    print("   - Line plots: 5 separate plots (overall, emails, phones, addresses, names)")
    print("   - CluSanT is drawn last (on top) with subtle visibility improvement (0.25 opacity)")
    print("   - Data source: SQLite database")

if __name__ == "__main__":
    main()
