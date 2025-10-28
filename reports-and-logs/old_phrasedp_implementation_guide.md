# Old PhraseDP Implementation Guide

## Overview

This guide explains how to use the old PhraseDP implementation that was used in the 500-question experiment. The old implementation is simpler, more conservative, and better suited for medical QA tasks.

## Key Differences from New PhraseDP

| Aspect | Old PhraseDP | New PhraseDP |
|--------|--------------|--------------|
| **API Calls** | 1 | 10 |
| **Band Diversity** | None | 10 bands (0.0-0.1, 0.1-0.2, etc.) |
| **Similarity Range** | 0.59-0.85 (narrow) | 0.1-0.9 (wide) |
| **Approach** | Conservative | Aggressive |
| **Medical Context** | Preserved | Often destroyed |
| **Performance** | Better for medical QA | Worse for medical QA |

## Implementation

### **1. Core Function: `generate_sentence_replacements_with_nebius`**

```python
def generate_sentence_replacements_with_nebius(nebius_client, nebius_model_name, 
    input_sentence, num_return_sequences=10, max_tokens=150):
    """
    OLD PHRASEDP: Simple candidate generation without band diversity.
    
    This is the original PhraseDP implementation that was used in the 500-question experiment.
    It generates diverse paraphrases using a single API call without similarity band targeting.
    """
```

**Key Features:**
- **Single API call** to generate all candidates
- **Conservative prompting** focused on preserving medical context
- **Simple filtering** to remove low-quality candidates
- **No band targeting** - generates candidates naturally

### **2. Complete Pipeline: `phrase_DP_perturbation_old`**

```python
def phrase_DP_perturbation_old(nebius_client, nebius_model_name, input_sentence, epsilon, sbert_model):
    """
    OLD PHRASEDP: Complete perturbation pipeline using the original approach.
    
    This function implements the complete old PhraseDP pipeline:
    1. Generate candidates using single API call (no band diversity)
    2. Compute embeddings for all candidates
    3. Apply exponential mechanism for probabilistic selection
    """
```

**Pipeline Steps:**
1. **Generate candidates** using single API call
2. **Compute embeddings** for all candidates using SBERT
3. **Apply exponential mechanism** for probabilistic selection
4. **Return selected perturbation**

## Usage Examples

### **Basic Usage**

```python
from utils import get_nebius_client, phrase_DP_perturbation_old
from sentence_transformers import SentenceTransformer

# Initialize components
nebius_client = get_nebius_client()
nebius_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Apply old PhraseDP perturbation
perturbed_question = phrase_DP_perturbation_old(
    nebius_client=nebius_client,
    nebius_model_name=nebius_model_name,
    input_sentence="What are the symptoms of diabetes?",
    epsilon=1.0,
    sbert_model=sbert_model
)

print(f"Perturbed: {perturbed_question}")
```

### **Integration with MedQA Experiments**

```python
def run_scenario_3_1_old_phrase_dp_local_cot(client, model_name, remote_client, sbert_model, question, options, correct_answer):
    """
    Scenario 3.1: Private Local Model + CoT (Old PhraseDP)
    """
    print("--- Scenario 3.1: Private Local Model + CoT (Old PhraseDP) ---")
    
    # Step 1: Apply old PhraseDP perturbation to question
    perturbed_question = phrase_DP_perturbation_old(
        nebius_client=remote_client,
        nebius_model_name=model_name,
        input_sentence=question,
        epsilon=1.0,
        sbert_model=sbert_model
    )
    
    # Step 2: Generate CoT from remote LLM using perturbed question
    cot_text = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], perturbed_question)
    
    # Step 3: Use local model with original question/options + CoT
    final_answer = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
    
    # Check correctness
    is_correct = check_answer_correctness(final_answer, correct_answer)
    
    return is_correct
```

## Configuration

### **Environment Variables**
```bash
# Required for Nebius API
NEBIUS_API=your_nebius_api_key
NEBIUS_BASE_URL=https://api.studio.nebius.ai/v1/

# Optional: Custom model
LOCAL_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
```

### **Model Configuration**
```python
# Default configuration
nebius_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
epsilon = 1.0  # Privacy parameter
num_return_sequences = 10  # Number of candidates to generate
```

## Performance Characteristics

### **Advantages of Old PhraseDP:**
1. **Conservative perturbations** preserve medical context
2. **Single API call** is more efficient
3. **Better performance** on medical QA tasks
4. **Simpler implementation** with fewer parameters
5. **Higher similarity** to original text

### **When to Use Old PhraseDP:**
- **Medical QA tasks** where context preservation is critical
- **Resource-constrained environments** (fewer API calls)
- **When similarity to original** is more important than maximum diversity
- **Conservative privacy requirements**

### **When to Use New PhraseDP:**
- **General text tasks** where maximum diversity is needed
- **High privacy requirements** (wider similarity range)
- **Research on diversity** in differential privacy
- **Non-medical domains** where context destruction is acceptable

## Testing

### **Run the Test Script**
```bash
python test_old_phrasedp.py
```

### **Expected Output**
```
============================================================
TESTING OLD PHRASEDP (No Band Diversity)
============================================================
Original Question: A 45-year-old woman presents with chest pain and shortness of breath. What is the most likely diagnosis?

Applying OLD differential privacy perturbation with epsilon=1.0...
OLD DP replacement selected: A middle-aged individual experiences chest discomfort and breathing difficulties. What is the probable diagnosis?

Perturbed Question: A middle-aged individual experiences chest discomfort and breathing difficulties. What is the probable diagnosis?

Semantic Similarity: 0.7234

✅ Old PhraseDP test completed successfully!
```

## Migration from New PhraseDP

### **Replace New PhraseDP Calls**
```python
# OLD (New PhraseDP)
candidate_sentences = generate_sentence_replacements_with_nebius_diverse(
    nebius_client, nebius_model_name, input_sentence,
    num_return_sequences=10, num_api_calls=10,
    enforce_similarity_filter=True, refill_underfilled_bands=True,
    # ... many more parameters
)

# NEW (Old PhraseDP)
candidate_sentences = generate_sentence_replacements_with_nebius(
    nebius_client, nebius_model_name, input_sentence,
    num_return_sequences=10
)
```

### **Replace Complete Pipeline**
```python
# OLD (New PhraseDP)
perturbed_question = phrase_DP_perturbation_with_batch_options(
    nebius_client, nebius_model_name, question, epsilon, sbert_model
)

# NEW (Old PhraseDP)
perturbed_question = phrase_DP_perturbation_old(
    nebius_client, nebius_model_name, question, epsilon, sbert_model
)
```

## Conclusion

The old PhraseDP implementation provides a simpler, more conservative approach that is better suited for medical QA tasks. It uses a single API call instead of 10, focuses on preserving medical context, and achieves better performance on medical questions.

**Key Benefits:**
- ✅ **Better performance** on medical QA (76.47% vs 58.82%)
- ✅ **More efficient** (1 API call vs 10)
- ✅ **Simpler implementation** (fewer parameters)
- ✅ **Preserves medical context** (conservative approach)
- ✅ **Higher similarity** to original text

---
*Implementation guide for the old PhraseDP approach*
*Date: 2025-01-27*
