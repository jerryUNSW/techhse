# Plots Directory

This directory contains all visualization plots generated for the Tech4HSE project.

## 📊 **Privacy Evaluation Plots**

### **privacy_evaluation_comprehensive.png**
- **Created**: August 26, 2025
- **Description**: Multi-panel comprehensive privacy evaluation visualization
- **Contents**:
  - Privacy-Utility Trade-off Analysis (ε=1)
  - Individual Attack Comparison (BERT, Embedding, GPT)
  - Privacy Protection Breakdown with thresholds
  - Radar Chart Method Comparison
- **Methods**: Phrase DP vs InferDPT
- **Dataset**: MedQA (10 questions)

### **privacy_evaluation_summary.png**
- **Created**: August 26, 2025
- **Description**: Simplified two-panel privacy evaluation summary
- **Contents**:
  - Privacy-Utility Trade-off scatter plot
  - Attack Method Comparison bar chart
- **Methods**: Phrase DP vs InferDPT
- **Dataset**: MedQA (10 questions)

## 📈 **Similarity Analysis Plots**

### **similarity_distributions.png**
- **Created**: August 25, 2025
- **Description**: Distribution plots of similarity metrics
- **Contents**: Histograms and density plots of similarity scores

### **similarity_summary.png**
- **Created**: August 25, 2025
- **Description**: Summary statistics of similarity analysis
- **Contents**: Box plots and summary statistics

## 🔧 **Usage**

All plots are generated using Python with matplotlib and seaborn. The main visualization script is `privacy_visualization.py` in the root directory.

## 📋 **File Sizes**

- `privacy_evaluation_comprehensive.png`: ~880 KB
- `privacy_evaluation_summary.png`: ~230 KB
- `similarity_distributions.png`: ~635 KB
- `similarity_summary.png`: ~223 KB

## 🎯 **Key Insights**

The privacy evaluation plots demonstrate:
1. **Clear privacy-utility trade-off** between methods
2. **InferDPT's superior privacy protection** (199.3% improvement)
3. **Consistent performance** across different attack methods
4. **Impact of ε=1** privacy parameter on protection levels

---

*Last updated: August 26, 2025*
