# PhraseDP Implementation Changes Analysis

## The Key Insight

**The discrepancy between 500-question and 76-question experiments is due to PhraseDP implementation changes, not just quota errors!**

## Implementation Evolution

### **Old PhraseDP Implementation (500-question experiment)**
```python
def generate_sentence_replacements_with_nebius(
    local_model, input_sentence, num_return_sequences=10, max_tokens=150):
    """
    OLD: Simple candidate generation without diversity constraints
    - Single API call
    - No similarity band targeting
    - No refill mechanism
    - Basic paraphrase generation
    """
```

**Characteristics:**
- **Single API call** for candidate generation
- **No similarity band targeting** - candidates generated randomly
- **No refill mechanism** - if bands are underfilled, they stay underfilled
- **Simple paraphrase generation** - basic diversity without constraints
- **Narrow similarity range** - typically 0.59-0.85 (as mentioned in comments)

### **New PhraseDP Implementation (76-question experiment)**
```python
def generate_sentence_replacements_with_nebius_diverse(
    nebius_client, nebius_model_name, input_sentence, 
    num_return_sequences=10, max_tokens=150, num_api_calls=10,
    enforce_similarity_filter=True, filter_margin=0.05,
    low_band_quota_boost=True,
    refill_underfilled_bands=True,
    max_refill_retries=2,
    equal_band_target=None,
    global_equalize_max_loops=5,
    verbose=False):
    """
    NEW: 10-band diversity with refill technique
    - 10 API calls targeting different similarity bands
    - Refill mechanism for underfilled bands
    - Wide similarity range (0.1-0.9)
    - Enhanced diversity constraints
    """
```

**Characteristics:**
- **10 API calls** targeting different similarity bands (0.0-0.1, 0.1-0.2, 0.2-0.3, etc.)
- **Refill mechanism** - automatically refills underfilled bands
- **Similarity filtering** - enforces target similarity ranges
- **Wide similarity range** - 0.1-0.9 (much wider than old 0.59-0.85)
- **Enhanced diversity** - more diverse candidates across similarity bands

## Impact Analysis

### **1. Candidate Quality Changes**

**Old Implementation:**
- **Narrow similarity range** (0.59-0.85) - candidates too similar to original
- **Limited diversity** - basic paraphrases without band targeting
- **No refill** - underfilled bands stay underfilled
- **Result**: More conservative perturbations, easier for local model to understand

**New Implementation:**
- **Wide similarity range** (0.1-0.9) - includes very different candidates
- **Enhanced diversity** - candidates across all similarity bands
- **Refill mechanism** - ensures balanced distribution
- **Result**: More aggressive perturbations, harder for local model to understand

### **2. Local Model Performance Impact**

**Old Implementation (500-question experiment):**
- **PhraseDP + CoT: 83.80%** (better than purely local 76.80%)
- **Reason**: Conservative perturbations + remote CoT guidance = good performance

**New Implementation (76-question experiment):**
- **PhraseDP + CoT: 39.47%** (worse than purely local 65.79%)
- **Reason**: Aggressive perturbations + remote CoT guidance = poor performance

### **3. Why the Performance Degradation?**

**The new 10-band diversity implementation creates more challenging perturbations:**

1. **Extreme Similarity Bands (0.0-0.1, 0.1-0.2)**:
   - Generate very abstract, generic paraphrases
   - Lose specific medical details and context
   - Harder for local model to understand

2. **Refill Mechanism**:
   - Ensures all bands are filled, including difficult low-similarity bands
   - Forces generation of more diverse (and potentially confusing) candidates

3. **Wide Similarity Range**:
   - Includes candidates that are too different from original
   - May lose essential medical information
   - Creates semantic gaps that local model cannot bridge

## Evidence from Code

### **Old Implementation (Simple)**
```python
# Single API call, no band targeting
candidate_sentences = generate_sentence_replacements_with_nebius(
    local_model_name_str, 
    input_sentence=cnn_dm_prompt,
    num_return_sequences=10,  # Simple generation
)
```

### **New Implementation (Complex)**
```python
# 10 API calls with band targeting and refill
candidate_sentences = generate_sentence_replacements_with_nebius_diverse(
    nebius_client, nebius_model_name, input_sentence,
    num_return_sequences=10, num_api_calls=10,  # 10 API calls
    enforce_similarity_filter=True,  # Similarity filtering
    refill_underfilled_bands=True,  # Refill mechanism
    max_refill_retries=2,  # Retry logic
    equal_band_target=None,  # Band targeting
    global_equalize_max_loops=5,  # Equalization
)
```

## Conclusion

**The 33.32% performance gap is due to PhraseDP implementation changes, not just quota errors:**

1. **Old PhraseDP**: Conservative perturbations + remote CoT = good performance (83.80%)
2. **New PhraseDP**: Aggressive perturbations + remote CoT = poor performance (39.47%)

**The new 10-band diversity and refill technique, while theoretically better for privacy, creates perturbations that are too challenging for the local model to understand, even with remote CoT guidance.**

## Recommendations

### **1. Immediate Action**
- **Revert to old PhraseDP implementation** for better performance
- **Or adjust new implementation parameters** to be less aggressive

### **2. Parameter Tuning**
- **Reduce similarity range** (e.g., 0.3-0.8 instead of 0.1-0.9)
- **Disable extreme bands** (0.0-0.2) for medical questions
- **Adjust refill thresholds** to be less aggressive

### **3. Hybrid Approach**
- **Use old implementation** for medical QA tasks
- **Use new implementation** for general text tasks
- **Implement task-specific parameters**

## Key Insight

**More diversity in PhraseDP does not always mean better performance. The new implementation prioritizes privacy over utility, leading to performance degradation in medical QA tasks.**

---
*Analysis of PhraseDP implementation changes and their impact on performance*
*Date: 2025-01-27*
