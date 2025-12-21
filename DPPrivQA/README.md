# DPPrivQA

Privacy-Preserving CoT-induced QA Framework using PhraseDP (Phrase-level Differential Privacy)

## Overview

DPPrivQA is a framework for privacy-preserving question answering that combines local and remote language models with phrase-level differential privacy techniques. The framework supports multiple datasets and privacy mechanisms.

## Datasets

- MedQA-USMLE
- MedMCQA
- HSE-benchmark
- MMLU Professional Law (1,534 questions total)
- MMLU Professional Medicine (272 questions)
- MMLU Clinical Knowledge (265 questions)
- MMLU College Medicine (173 questions)

## Experiment Progress

### Local Model Tests (Epsilon-Independent)

**Completed Tests:**

1. **MMLU Professional Law**
   - Questions tested: 100/1,534
   - Accuracy: 48.0% (48/100 correct)
   - Experiment ID: 7
   - Completed: 2025-12-18 00:42:14

2. **MMLU Professional Medicine**
   - Questions tested: 272/272 (100%)
   - Accuracy: 79.8% (217/272 correct)
   - Experiment ID: 8
   - Completed: 2025-12-18 00:51:19

3. **MMLU Clinical Knowledge**
   - Questions tested: 265/265 (100%)
   - Accuracy: 73.2% (194/265 correct)
   - Experiment ID: 9
   - Completed: 2025-12-18 01:08:39

4. **MMLU College Medicine**
   - Questions tested: 173/173 (100%)
   - Accuracy: 65.9% (114/173 correct)
   - Experiment ID: 10
   - Completed: 2025-12-18 01:09:01

All results are stored in `exp-results/results.db` with detailed logs in `exp-results/logs/`.

## QA Test Scenarios

### Epsilon-Independent Tests
1. **Local** - Local model only (baseline)
2. **Local + CoT** - Local model with non-private CoT from remote
3. **Remote** - Remote model only (baseline)

### Epsilon-Dependent Tests (epsilon = 1.0, 2.0, 3.0)
4. **DPPrivQA** - Local → PhraseDP sanitization → Remote CoT → Local answer

## CoT Improvement Attempt: Why Option 1 Performed Worse

We attempted to improve Chain of Thought (CoT) generation by modifying the prompt to avoid refusals from the remote model. However, **Option 1** (no refusal approach) performed worse than the baseline. This section documents our analysis of why this occurred.

### CoT Content Difference

**Baseline (with refusal):**
- Often includes direct conclusions like "Greatest risk: Hypertension" or "Most appropriate: Add folinic acid"
- Provides actionable key points despite initial refusal

**Option 1 (no refusal):**
- More structured analysis like "1) Relevant concepts... 2) Factors... 3) Analytical approach..."
- Focuses on process-oriented guidance

**Impact:** The local model may extract answers better from direct hints than from structured frameworks.

### Information Density

**Baseline CoT:**
- Answer-oriented despite refusal
- Provides actionable key points
- More information-dense for answer extraction

**Option 1 CoT:**
- Process-oriented
- Focuses on "how to think" rather than "what to conclude"
- Less actionable for the local model

**Impact:** Less actionable information for the local model to extract answers from.

### Prompt Mismatch

**Baseline:**
- Asks for "step-by-step reasoning" → refusal → "concise answer and key points" (more direct)
- The refusal format leads to more direct, answer-oriented summaries

**Option 1:**
- Asks for "analytical guidance" → structured analysis (less direct)
- Avoids refusals but produces less direct guidance

**Impact:** The refusal pattern may be more helpful than structured guidance because it leads to concise, answer-oriented summaries.

### Key Insight

**The refusal pattern is not the problem.** The "concise summaries" that GPT-5 provides after refusal can be more effective because they:
- Include direct conclusions/hints
- Are answer-oriented
- Provide actionable key points

**Option 1's structured guidance avoids refusals but may be less effective because it:**
- Focuses on process over conclusions
- Is more educational than actionable
- Requires more inference from the local model

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:
- `OPENAI_API_KEY` - For remote LLM access
- `NEBIUS_API` - For local model access

Edit `config.yaml` to configure models, mechanisms, and experiment settings.

## Running Experiments

```bash
# Run MedQA experiment
python experiments/test_medqa.py

# Run MMLU Professional Law experiment
python experiments/test_mmlu_professional_law.py
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=dpprivqa --cov-report=html
```

## Project Structure

- `dpprivqa/` - Main package with mechanisms, datasets, QA framework, and database
- `experiments/` - Experiment scripts for each dataset
- `exp-results/` - Experiment results database and logs
- `tests/` - Unit and integration tests

