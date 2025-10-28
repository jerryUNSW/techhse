---
title: "Tech4HSE: Privacy-Preserving Multi-Hop Question Answering"
subtitle: "Comprehensive Framework with Differential Privacy"
author: "Tech4HSE Team"
date: "August 26, 2025"
theme: "white"
transition: "slide"
revealjs-url: "https://cdn.jsdelivr.net/npm/reveal.js@4.3.1"
css: |
  <style>
    .reveal {
      font-size: 0.45em;
    }
    .reveal h1 {
      font-size: 1.4em;
    }
    .reveal h2 {
      font-size: 1.1em;
    }
    .reveal h3 {
      font-size: 0.9em;
    }
    .reveal p {
      font-size: 0.55em;
    }
    .reveal li {
      font-size: 0.5em;
    }
    .reveal table {
      font-size: 0.45em;
    }
    .reveal img {
      max-width: 90%;
      max-height: 60vh;
    }
    .small-text {
      font-size: 0.4em;
    }
    .plot-explanation {
      font-size: 0.42em;
      color: #666;
      font-style: italic;
    }
  </style>
---

# Tech4HSE: Privacy-Preserving Multi-Hop QA
## Comprehensive Framework with Differential Privacy

**Project**: Multi-hop QA with local + remote models  
**Privacy**: Phrase-level differential privacy  
**Domain**: Medical question answering

---

## Project Objective

Implement privacy-preserving multi-hop question answering combining local and remote language models

---

## Key Features

- **Multi-Scenario Experiments**: Compare different inference strategies
- **Privacy-Preserving Inference**: Phrase-level differential privacy
- **Local + Remote Integration**: Combine local models with remote CoT assistance
- **Medical QA Support**: Specialized for medical datasets
- **Flexible Model Support**: Multiple LLM providers

---

## Dataset 1: MedQA-USMLE-4-options

- **Type**: Medical multiple choice questions
- **Format**: Clinical vignettes with patient scenarios
- **Size**: 11,451 examples (10,178 train + 1,273 test)
- **Source**: USMLE examination style questions

---

## Dataset 2: MedMCQA

- **Type**: Medical multiple choice questions
- **Format**: 21 medical subjects, AIIMS & NEET PG exams
- **Size**: 193,155 examples (182,822 train + 4,183 valid + 6,150 test)
- **Source**: Indian medical entrance exams

---

## Dataset 3: EMRQA-MSQUAD

- **Type**: Extractive QA from medical records
- **Format**: Medical records + questions + extractive answers
- **Size**: 163,695 examples (130,956 train + 32,739 valid)
- **Source**: Real medical records and clinical notes

---

## Experiment Scenarios Overview

1. **Scenario 1**: Purely Local Model (baseline)
2. **Scenario 2**: Local + Remote CoT (non-private)
3. **Scenario 2.5**: Local + Local CoT
4. **Scenario 3.1**: Local + Private CoT (Phrase DP)
5. **Scenario 3.2**: Local + Private CoT (InferDPT)
6. **Scenario 4**: Purely Remote Model

---

## Privacy Mechanisms

### **Phrase DP**
- Semantic similarity-based phrase replacement
- Preserves semantic coherence

### **InferDPT**
- Token-level perturbation framework
- Higher privacy, lower utility

---

## MedQA Experiment Results (50 Questions)

| **Scenario** | **Accuracy** | **Correct/Total** | **Performance** |
|--------------|-------------|-------------------|-----------------|
| **1. Purely Local Model** | **74.00%** | 37/50 | Baseline |
| **2. Non-Private Local + Remote CoT** | **90.00%** | 45/50 | ⭐ **Best** |
| **3.1. Private Local + CoT (Phrase DP)** | **78.00%** | 39/50 | ⬆️ Above baseline |
| **3.2. Private Local + CoT (InferDPT)** | **72.00%** | 36/50 | ⬇️ Below baseline |
| **4. Purely Remote Model** | **90.00%** | 45/50 | ⭐ **Best** |

---

## Privacy-Utility Trade-off

![Privacy-Utility Trade-off](plots/privacy_evaluation_summary.png)

<div class="plot-explanation">
**Left**: Privacy-Utility Trade-off | **Right**: Attack Method Comparison
</div>

---

## Key Results Summary

- **Phrase DP**: 83.91% accuracy, 0.295 privacy
- **InferDPT**: 71.20% accuracy, 0.884 privacy
- **Trade-off**: Better privacy = lower accuracy

---

## Attack Performance Results

| Attack | Phrase DP | InferDPT | Improvement |
|--------|-----------|----------|-------------|
| **BERT** | 0.237 | 0.993 | **+318%** |
| **Embedding** | 0.336 | 0.704 | **+109%** |
| **GPT** | 0.312 | 0.955 | **+209%** |

---

## Overall Privacy Scores

- **InferDPT**: 0.884 (Excellent)
- **Phrase DP**: 0.295 (Poor)

---

## Comprehensive Analysis

![Comprehensive Analysis](plots/privacy_evaluation_comprehensive.png)

<div class="plot-explanation">
**Multi-panel**: Trade-off, Attack Comparison, Privacy Breakdown, Radar Chart
</div>

---

## Four-Panel Analysis

1. **Privacy-Utility Trade-off** (top-left)
2. **Individual Attack Comparison** (top-right)
3. **Privacy Protection Breakdown** (middle)
4. **Radar Chart Method Comparison** (bottom)

---

## Similarity Analysis

![Similarity Distributions](plots/similarity_distributions.png)

<div class="plot-explanation">
**Distribution plots**: Similarity scores between original and perturbed questions
</div>

---

## Summary Statistics

![Similarity Summary](plots/similarity_summary.png)

<div class="plot-explanation">
**Box plots**: Central tendency and spread of privacy scores
</div>

---

## Performance Gap 1: Remote vs Local

- **Remote models**: 90.00% accuracy
- **Local models**: 74.00% accuracy
- **Gap**: 16% difference
- **Implication**: Remote models have superior reasoning capabilities

---

## Performance Gap 2: Phrase DP vs InferDPT

- **Phrase DP**: 78.00% accuracy (semantic-preserving)
- **InferDPT**: 72.00% accuracy (token-level perturbation)
- **Gap**: 6% difference
- **Implication**: Semantic coherence preservation matters

---

## Phrase DP (Scenario 3.1)

- **Approach**: Semantic similarity-based phrase replacement
- **Privacy**: Moderate (0.295)
- **Utility**: Good (78.00%)
- **Semantic Coherence**: Preserved

---

## InferDPT (Scenario 3.2)

- **Approach**: Token-level perturbation
- **Privacy**: Excellent (0.884)
- **Utility**: Poor (72.00%)
- **Semantic Coherence**: Disrupted

---

## Privacy Protection Zones

- **Red (0-0.3)**: Poor Privacy
- **Orange (0.3-0.6)**: Moderate Privacy  
- **Green (0.6-1.0)**: Good Privacy

---

## Method Performance by Zone

- **Phrase DP**: Consistently in poor zone
- **InferDPT**: Consistently in good zone

---

## Key Insight 1: Clear Trade-off

- **Phrase DP**: Prioritizes utility over privacy
- **InferDPT**: Prioritizes privacy over utility

---

## Key Insight 2: InferDPT Superiority

- **199.3% overall improvement** in privacy
- **Consistent superiority** across all attacks

---

## Key Insight 3: Attack Effectiveness

- **BERT**: Most challenging for both
- **GPT**: InferDPT shows strongest protection

---

## Core Components

- **`main_qa.py`**: Main QA system with privacy features
- **`multi_hop_experiment.py`**: Multi-hop experiment framework
- **`dp_sanitizer.py`**: Differential privacy implementation
- **`utils.py`**: LLM interaction utilities

---

## Privacy Features

- **Phrase-Level DP**: Semantic similarity-based replacement
- **Configurable Epsilon**: Privacy budget control
- **Exponential Mechanism**: Privacy-preserving selection

---

## Local Models (via Nebius API)

- Microsoft Phi-4
- Google Gemma models
- Qwen models
- Mistral models
- Meta Llama models

---

## Remote Models

- OpenAI GPT-4o
- DeepSeek models
- Other OpenAI-compatible APIs

---

## Method Selection Guidelines

- **High privacy needed**: Choose InferDPT
- **High accuracy needed**: Choose Phrase DP

---

## Key Findings Summary

- **InferDPT dramatically outperforms** Phrase DP in privacy
- **ε=1 provides good balance** for most applications
- **Clear guidelines** for method selection

---

## Future Work: Extensions

- **Larger dataset evaluation**
- **Different ε values**
- **Additional attack methods**
- **Real-world deployment**

---

## Future Work: Research Directions

- **Hybrid approaches**
- **Adaptive privacy**
- **Domain-specific optimizations**

---

## Thank You!

**Contact**: Tech4HSE Team  
**Repository**: [GitHub Link]

---

## Technical Setup

- **Dataset**: MedQA-USMLE-4-options (50 questions)
- **Privacy Parameter**: ε=1
- **Attack Methods**: BERT, Embedding, GPT
- **Tools**: Python, Matplotlib, Sentence Transformers, OpenAI API
