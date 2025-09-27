# MedQA USMLE Experiment Summary

## Overview
Two comprehensive experiments were conducted on the MedQA USMLE dataset with 500 questions each, testing various privacy-preserving mechanisms at epsilon=1.

## File Locations
- **Test1 (Comprehensive):** `MedQA-UME_epsilon1_comprehensive_mechanisms.txt`
- **Test2 (Three-Scenario):** `MedQA-UME_epsilon1_three_scenario_test.txt`

## Mechanisms Tested in Each Experiment

### Old Test1 (Comprehensive) — 7 mechanisms:
1. **Purely Local Model** (meta-llama/Meta-Llama-3.1-8B-Instruct) — 59.80%
2. **Non-Private Local Model + Remote CoT** — 83.00%
3. **Old PhraseDP - Single API Call** — 74.40%
4. **Old PhraseDP + Batch Perturbed Options** — 0.00% (buggy)
5. **InferDPT + Batch Perturbed Options** — 58.80%
6. **SANTEXT+ + Batch Perturbed Options** — 56.60%
7. **Purely Remote Model** (gpt-5-chat-latest) — 89.60%

### Old Test2 (Three-Scenario) — 3 mechanisms:
1. **Old PhraseDP + Batch Perturbed Options (FIXED)** — 67.00%
2. **InferDPT WITHOUT Batch Options** — 47.40%
3. **SANTEXT+ WITHOUT Batch Options** — 53.20%

## Key Differences

### Test1 (Comprehensive)
- **7 mechanisms** including baseline comparisons
- **Batch options** used for privacy mechanisms
- **Buggy PhraseDP** (0% accuracy due to implementation error)
- **Includes baselines:** Local Only, Local + Remote CoT, Remote Only

### Test2 (Three-Scenario)
- **3 mechanisms** focused on specific scenarios
- **No batch options** for privacy mechanisms
- **Fixed PhraseDP** (67% accuracy after bug fix)
- **Focused comparison** of privacy mechanisms only

## Key Findings

### Performance Rankings (Combined Results)
1. **Remote Only (Baseline):** 89.60% 🥇
2. **Local + Remote CoT (Baseline):** 83.00% 🥈
3. **Old PhraseDP:** 74.40% 🥉
4. **Old PhraseDP + Batch (Fixed):** 67.00%
5. **Local Only (Baseline):** 59.80%
6. **SANTEXT+:** 53.20%
7. **InferDPT:** 47.40%

### Privacy Mechanism Insights
- **Old PhraseDP** performs best among privacy mechanisms (74.40%)
- **Batch options** show mixed results (67% fixed vs 0% buggy)
- **SANTEXT+** outperforms **InferDPT** in both scenarios
- **Privacy overhead** varies significantly across mechanisms

### Bug Fix Impact
- **Old PhraseDP + Batch** improved from 0% to 67% after fixing implementation error
- Demonstrates importance of proper implementation for privacy mechanisms

## Experimental Setup
- **Dataset:** MedQA USMLE (500 questions each)
- **Privacy Parameter:** Epsilon = 1.0
- **Local Model:** meta-llama/Meta-Llama-3.1-8B-Instruct
- **Remote Model:** gpt-5-chat-latest
- **Date:** September 2025

## Files Generated
- `MedQA-UME_epsilon1_comprehensive_mechanisms.txt` (13.4 MB)
- `MedQA-UME_epsilon1_three_scenario_test.txt` (9.9 MB)
- `privacy_mechanisms_comparison.png` (bar plot visualization)

---
*Generated: September 27, 2025*
*Tech4HSE Privacy-Preserving Medical QA Project*
