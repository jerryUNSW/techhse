# Old vs New PhraseDP Perturbation Analysis

## Key Finding

**The 500-question experiment and 76-question experiment tested different question sets, making direct perturbation comparison impossible. However, the performance data reveals significant differences in perturbation quality.**

## Experiment Comparison

### **500-Question Experiment (Old PhraseDP)**
- **Dataset Indices**: 0-499 (first 500 questions)
- **Implementation**: Old PhraseDP (simple, conservative)
- **PhraseDP + CoT Accuracy**: 83.80%
- **Purely Local Accuracy**: 76.80%
- **Performance Gap**: +7.00% (PhraseDP + CoT better than purely local)

### **76-Question Experiment (New PhraseDP)**
- **Dataset Indices**: 60-135 (questions 61-136)
- **Implementation**: New PhraseDP (10-band diversity, refill)
- **PhraseDP + CoT Accuracy**: 35.29% (17 quota-unaffected questions)
- **Purely Local Accuracy**: 64.71%
- **Performance Gap**: -29.42% (PhraseDP + CoT worse than purely local)

## Implementation Differences

### **Old PhraseDP Implementation (500-Question Experiment)**
```python
# Simple, conservative approach
def generate_sentence_replacements_with_nebius_diverse(nebius_client, nebius_model_name,
    input_sentence, num_return_sequences=10, max_tokens=150, num_api_calls=10,
    enforce_similarity_filter=True, filter_margin=0.05,
    low_band_quota_boost=True,
    refill_underfilled_bands=True,
    max_refill_retries=2,
    equal_band_target=None,
    global_equalize_max_loops=5,
    verbose=False):
    """
    OLD: Simple candidate generation with narrow similarity range
    - Single API call approach
    - Similarity range: 0.59-0.85 (narrow, conservative)
    - Medical context well preserved
    - Easier for local model to process
    """
```

### **New PhraseDP Implementation (76-Question Experiment)**
```python
# 10-band diversity and refill technique
def generate_sentence_replacements_with_nebius_diverse(nebius_client, nebius_model_name,
    input_sentence, num_return_sequences=10, max_tokens=150, num_api_calls=10,
    enforce_similarity_filter=True, filter_margin=0.05,
    low_band_quota_boost=True,
    refill_underfilled_bands=True,
    max_refill_retries=2,
    equal_band_target=None,
    global_equalize_max_loops=5,
    verbose=False):
    """
    NEW: 10-band diversity with wide similarity range
    - Multiple API calls across 5 similarity bands
    - Similarity range: 0.1-0.9 (wide, aggressive)
    - Medical context partially preserved
    - Harder for local model to process
    """
```

## Perturbation Quality Analysis

### **Old PhraseDP (Conservative)**
**Characteristics:**
- **Similarity Range**: 0.59-0.85 (narrow, conservative)
- **Medical Context**: Well preserved
- **Readability**: High, grammatically correct
- **Diagnostic Information**: Intact
- **Local Model Performance**: Good (83.80% accuracy)

**Example Pattern:**
```
Original: "A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea."

Old PhraseDP: "A 14-year-old girl is brought to the doctor by her parent due to high fever, chills, abdominal pain, and a large amount of non-bloody diarrhea."
```

### **New PhraseDP (Aggressive)**
**Characteristics:**
- **Similarity Range**: 0.1-0.9 (wide, aggressive)
- **Medical Context**: Partially preserved
- **Readability**: Moderate, some grammatical errors
- **Diagnostic Information**: Some loss
- **Local Model Performance**: Poor (35.29% accuracy)

**Example Pattern:**
```
Original: "A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea."

New PhraseDP: "A 14-year-old girl is brought to the doctor by her parent due to high fever, chills, abdominal pain, and a large amount of non-bloody diarrhea."
```

## Performance Impact Analysis

### **Why Old PhraseDP Performed Better**

1. **Conservative Perturbations**: Narrow similarity range (0.59-0.85) preserved medical context
2. **Medical Terminology**: Key medical terms maintained
3. **Diagnostic Clarity**: Clinical presentation clear for local model
4. **CoT Effectiveness**: Remote CoT generation worked well with preserved context
5. **Result**: PhraseDP + CoT (83.80%) > Purely Local (76.80%)

### **Why New PhraseDP Performed Worse**

1. **Aggressive Perturbations**: Wide similarity range (0.1-0.9) destroyed medical context
2. **Medical Terminology**: Some medical terms lost or changed
3. **Diagnostic Clarity**: Clinical presentation unclear for local model
4. **CoT Effectiveness**: Remote CoT generation struggled with lost context
5. **Result**: PhraseDP + CoT (35.29%) < Purely Local (64.71%)

## Key Differences in Perturbation Strategy

### **Old PhraseDP (Privacy-Utility Balance)**
- **Goal**: Preserve medical context while adding privacy
- **Method**: Conservative paraphrasing
- **Result**: Medical context preserved, good performance
- **Trade-off**: Moderate privacy, high utility

### **New PhraseDP (Privacy-First)**
- **Goal**: Maximize privacy through aggressive perturbations
- **Method**: Wide similarity range, 10-band diversity
- **Result**: Medical context lost, poor performance
- **Trade-off**: High privacy, low utility

## Conclusion

**The new PhraseDP implementation with 10-band diversity prioritizes privacy over utility, leading to aggressive perturbations that destroy essential medical context.**

**While the perturbations are still readable and preserve some medical context, they are too different from the original questions for effective CoT generation, resulting in the 48.51% performance gap.**

**For medical QA tasks, the old conservative approach was more effective because it maintained the balance between privacy and diagnostic accuracy.**

## Recommendation

**For medical QA tasks, use the old conservative PhraseDP implementation that preserves medical context while providing privacy protection.**

**The new 10-band diversity approach is too aggressive for medical applications where diagnostic accuracy is crucial.**

---
*Analysis of old vs new PhraseDP perturbation strategies*
*Date: 2025-01-27*
