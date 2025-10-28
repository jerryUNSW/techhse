# Figure Reference Documentation

This document provides a clear reference for all figures used in the Overleaf document to avoid confusion in future work.

## Figure Overview

- **Fig 1**: MedQA UME accuracy plots (three separate bar charts for epsilon 1.0, 2.0, 3.0)
- **Fig 2**: Overall PII protection rate vs epsilon line plot (5 mechanisms: PhraseDP, InferDPT, SANTEXT+, CusText+, CluSanT)
- **Fig 3**: Per-epsilon PII protection radar charts (5 mechanisms, 5 dimensions: overall, emails, phones, addresses, names)
- **Fig 4**: Scalability plots (accuracy and processing time vs number of questions)

## Current Files in overleaf-folder/plots/

### Fig 1 - MedQA UME Accuracy Plots
- `medqa_epsilon_1.0.pdf` - Bar chart for epsilon 1.0
- `medqa_epsilon_2.0.pdf` - Bar chart for epsilon 2.0  
- `medqa_epsilon_3.0.pdf` - Bar chart for epsilon 3.0

### Fig 2 - Overall PII Protection vs Epsilon
- `overall_protection_vs_epsilon_all_mechanisms.png` - Line plot showing protection rates across epsilon values
- **Note**: Currently PNG format, needs PDF version for better quality

### Fig 3 - Per-Epsilon Radar Charts
- `protection_radar_5mech_20250927_eps_1_0.pdf` - Radar chart for epsilon 1.0
- `protection_radar_5mech_20250927_eps_2_0.pdf` - Radar chart for epsilon 2.0
- `protection_radar_5mech_20250927_eps_3_0.pdf` - Radar chart for epsilon 3.0

### Fig 4 - Scalability Plots
- (To be documented when created)

## LaTeX References

In `overleaf-folder/4experiment.tex`:
- Fig 1: `\ref{fig:medqa_accuracy}` - MedQA accuracy plots
- Fig 2: `\ref{fig:ppi_overall_vs_eps}` - Overall PII protection vs epsilon
- Fig 3: `\ref{fig:ppi_radar_eps}` - Per-epsilon radar charts
- Fig 4: `\ref{fig:scalability}` - Scalability plots

## Notes for Future Updates

1. **Font Sizes**: All plots should use 20pt fonts for titles and labels for consistency
2. **File Formats**: Prefer PDF over PNG for better quality in publications
3. **Color Schemes**: Use consistent color palettes across related figures
4. **Radar Charts**: Only show epsilon values 1.0, 2.0, 3.0 (not 1.5, 2.5)
5. **Bar Charts**: Use pastel color palette for MedQA accuracy plots

## Last Updated
- Created: 2025-01-27
- Purpose: Document figure references to avoid confusion in future work






