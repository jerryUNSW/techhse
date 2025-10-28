# Experiment Data and Plotting Scripts Organization

## Overview
This document organizes the experiment data files and plotting scripts for the three key experiment types in the Tech4HSE project.

---

## 1. PPI Protection Experiments

### 📊 **Data Files (Input for Plotting)**
```
Primary Data Sources:
├── experiment_results/ppi-protection/
│   ├── pii_protection_results_20250927_220805.json (1.9MB - Main dataset)
│   ├── pii_protection_results_20250929_071204.json (41KB - Latest)
│   └── pii_protection_results_20250929_070453.json (142KB - Intermediate)
├── pii_protection_results_row_by_row_20250925_072822.json (1.1MB - Row-by-row data)
└── ppi-protection-exp.txt (14MB - Raw experiment output)
```

**Data Structure:**
- **Epsilon values**: 1.0, 1.5, 2.0, 2.5, 3.0
- **Mechanisms**: PhraseDP, InferDPT, SANTEXT+, CusText+, CluSanT
- **PII Types**: overall, emails, phones, addresses, names
- **Format**: JSON with nested structure by mechanism → epsilon → PII type

### 🎨 **Plotting Scripts**
```
Main Plotting Scripts:
├── create_ppi_comparison_plots.py (Hardcoded data comparison)
├── generate_epsilon_line_plots.py (Line plots + radar plots)
├── generate_unified_ppi_plots.py (Unified plotting with all mechanisms)
└── experiment_results/ppi-protection/
    ├── generate_final_plots.py (Final publication plots)
    ├── generate_radar_plots.py (Radar plots only)
    └── generate_radar_plots_final.py (Enhanced radar plots)
```

**Generated Plots:**
- **Line plots**: `overall_protection_vs_epsilon_5mech_*.png`
- **Radar plots**: `protection_radar_5mech_*_eps_*.png` (5 plots, one per epsilon)
- **Individual PII**: `individual_pii_protection_vs_epsilon.png`

---

## 2. MedQA-UME Experiments

### 📊 **Data Files (Input for Plotting)**
```
Primary Data Sources:
├── tech4hse_results.db (2.2MB - SQLite database)
├── experiment_results/
│   ├── complete_medqa_results.json (2.8KB - Summary results)
│   └── medqa_detailed_results.json (1.1KB - Detailed results)
└── test-medqa-usmle-4-options-results.txt (9.3MB - Raw experiment output)
```

**Data Structure:**
- **Epsilon values**: 1.0, 2.0, 3.0
- **Mechanisms**: Purely Local, InferDPT, SANTEXT+, PhraseDP, Non-Private + CoT, Purely Remote
- **Sample size**: 500 questions
- **Format**: SQLite database with `medqa_results` table

### 🎨 **Plotting Scripts**
```
Main Plotting Scripts:
├── update_plots_pastel_colors.py (Pastel color palette)
├── update_fig1_van_gogh_colors.py (Van Gogh-inspired colors)
├── make_fig1_text_larger.py (Large text formatting)
├── update_fig1_larger_text_height.py (Increased height)
└── py-files/analyze_medqa_performance.py (Performance analysis)
```

**Generated Plots:**
- **Bar plots**: `medqa_epsilon_1.0.pdf`, `medqa_epsilon_2.0.pdf`, `medqa_epsilon_3.0.pdf`
- **Accuracy comparison**: Shows mechanism performance across epsilon values
- **Publication ready**: PDF format for overleaf integration

---

## 3. MedMCQA Experiments

### 📊 **Data Files (Input for Plotting)**
```
Primary Data Sources:
├── experiment_results/QA-results/medmcqa/ (25 JSON files)
│   ├── medmcqa_results_local_meta-llama_*_500q_eps1.0_*.json (Main dataset)
│   ├── medmcqa_results_local_meta-llama_*_500q_eps2.0_*.json
│   ├── medmcqa_results_local_meta-llama_*_500q_eps3.0_*.json
│   └── medmcqa_results_meta-llama_*_1q_eps*.json (Single question tests)
└── plots/medmcqa/ (Generated plots directory)
```

**Data Structure:**
- **Epsilon values**: 1.0, 2.0, 3.0
- **Mechanisms**: PhraseDP, InferDPT, SANTEXT+
- **Sample sizes**: 1q, 100q, 500q
- **Format**: JSON with experiment results and metadata

### 🎨 **Plotting Scripts**
```
Main Plotting Scripts:
├── py-files/plot_epsilon_trend_balanced.py (Epsilon trend analysis)
├── py-files/analyze_medqa_performance.py (Performance analysis)
└── test-medmcqa.py (Experiment script + results generation)
```

**Generated Plots:**
- **Accuracy vs epsilon**: `medmcqa_accuracy_vs_epsilon_*.png`
- **Bar plots per epsilon**: `medmcqa_bar_plots_*_eps_1.0.png`, `eps_2.0.png`, `eps_3.0.png`
- **Privacy-utility tradeoff**: `medmcqa_privacy_utility_*.png`
- **Mechanism comparison**: Shows PhraseDP, InferDPT, SANTEXT+ performance

---

## 📋 **Quick Reference Guide**

### **For PPI Protection Plots:**
```bash
# Use this data file:
experiment_results/ppi-protection/pii_protection_results_20250927_220805.json

# Run this script:
python generate_epsilon_line_plots.py
```

### **For MedQA-UME Plots:**
```bash
# Use this data source:
tech4hse_results.db (SQLite database)

# Run this script:
python update_plots_pastel_colors.py
```

### **For MedMCQA Plots:**
```bash
# Use these data files:
experiment_results/QA-results/medmcqa/medmcqa_results_local_*_500q_eps*.json

# Run this script:
python py-files/plot_epsilon_trend_balanced.py
```

---

## 🎯 **Data File Priorities**

### **Most Recent/Complete Data:**
1. **PPI Protection**: `pii_protection_results_20250927_220805.json` (1.9MB)
2. **MedQA-UME**: `tech4hse_results.db` (2.2MB SQLite)
3. **MedMCQA**: `medmcqa_results_local_*_500q_eps*.json` (500 question experiments)

### **Plotting Script Priorities:**
1. **PPI Protection**: `generate_epsilon_line_plots.py` (most comprehensive)
2. **MedQA-UME**: `update_plots_pastel_colors.py` (publication ready)
3. **MedMCQA**: `py-files/plot_epsilon_trend_balanced.py` (trend analysis)

---

## 📁 **File Organization Summary**

```
Experiment Data Files:
├── PPI Protection: 4 JSON files + 1 TXT file
├── MedQA-UME: 1 SQLite DB + 2 JSON files + 1 TXT file  
└── MedMCQA: 25 JSON files in organized directory

Plotting Scripts:
├── PPI Protection: 6 Python scripts
├── MedQA-UME: 5 Python scripts
└── MedMCQA: 3 Python scripts

Generated Plots:
├── PPI Protection: 6 plot types (line + radar)
├── MedQA-UME: 3 bar charts (one per epsilon)
└── MedMCQA: 4 plot types (accuracy + bar + privacy-utility)
```

---

*Last updated: 2025-01-30*
*Total data files: ~35 files*
*Total plotting scripts: ~14 scripts*

