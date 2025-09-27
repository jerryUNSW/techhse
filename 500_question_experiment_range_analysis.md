# 500-Question Experiment Range Analysis

## Key Finding

**The 500-question experiment used the first 500 questions from the MedQA dataset (Dataset indices 0-499).**

## Evidence from Code Analysis

### **Dataset Selection Pattern**
```python
# From medmcqa_experiment.py line 351
sample_questions = dataset.select(range(min(num_samples, len(dataset))))
```

This code pattern shows that:
- **`range(min(num_samples, len(dataset)))`** selects the first `num_samples` questions
- **For 500 questions**: `range(500)` = indices 0, 1, 2, ..., 499
- **Dataset indices**: 0-499 (first 500 questions)

### **Comparison with 76-Question Experiment**

| Experiment | Dataset Indices | Questions | Range |
|------------|----------------|-----------|-------|
| **500-question experiment** | 0-499 | 1-500 | First 500 questions |
| **76-question experiment** | 60-135 | 1-76 | Questions 61-136 |

## Key Differences

### **1. Question Set Overlap**
- **500-question experiment**: Questions 1-500 (Dataset indices 0-499)
- **76-question experiment**: Questions 1-76 (Dataset indices 60-135)
- **Overlap**: Questions 61-76 (Dataset indices 60-75) - **16 questions overlap**

### **2. Question Difficulty Distribution**
- **500-question experiment**: First 500 questions (may include easier questions)
- **76-question experiment**: Questions 61-136 (may include more difficult questions)
- **Impact**: Different question difficulty could affect performance

### **3. Implementation Differences**
- **500-question experiment**: Old PhraseDP implementation (simple, conservative)
- **76-question experiment**: New PhraseDP implementation (10-band diversity, refill)

## Performance Impact Analysis

### **Why PhraseDP + CoT Performed Better in 500-Question Experiment**

1. **Old Implementation**: Conservative perturbations, easier for local model
2. **Question Set**: First 500 questions (potentially easier)
3. **No Quota Issues**: Remote CoT generation working properly
4. **Result**: PhraseDP + CoT = 83.80% (better than purely local 76.80%)

### **Why PhraseDP + CoT Performed Worse in 76-Question Experiment**

1. **New Implementation**: Aggressive perturbations, harder for local model
2. **Question Set**: Questions 61-136 (potentially more difficult)
3. **Quota Issues**: Remote CoT generation failed for questions 18-100
4. **Result**: PhraseDP + CoT = 35.29% (worse than purely local 64.71%)

## Conclusion

**The 500-question experiment used Dataset indices 0-499 (first 500 questions), while the 76-question experiment used Dataset indices 60-135 (questions 61-136).**

**The performance discrepancy is due to:**
1. **Implementation changes** (old vs new PhraseDP)
2. **Question set differences** (first 500 vs questions 61-136)
3. **Quota issues** affecting remote CoT generation

**The 16 overlapping questions (Dataset indices 60-75) could provide a direct comparison between old and new PhraseDP implementations on the same questions.**

---
*Analysis of the tested range in the 500-question experiment*
*Date: 2025-01-27*
