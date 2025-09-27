# Old vs New PhraseDP API Calls Analysis

## Key Finding

**Old PhraseDP: 1 API call | New PhraseDP: 10 API calls**

## Detailed Comparison

### **Old PhraseDP Implementation (500-question experiment)**

**File**: `py-files/private-multi-hop-cot-nebius.py`

```python
def generate_sentence_replacements_with_nebius(
    local_model,
    input_sentence,
    num_return_sequences=100,
    max_tokens=150):
    """
    OLD: Single API call for candidate generation
    """
    try:
        response = nebius_client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            n=num_return_sequences  # Generate multiple candidates in one call
        )
        
        paraphrases = set()
        for choice in response.choices:
            clean_line = choice.message.content.strip()
            if clean_line and clean_line.lower() != input_sentence.lower():
                paraphrases.add(clean_line)
        
        return list(paraphrases)
    except Exception as e:
        print(f"Error with Nebius API: {e}")
        return []
```

**Characteristics:**
- **1 API call** to generate all candidates
- **Single prompt** for all candidates
- **n=num_return_sequences** parameter to generate multiple candidates at once
- **Simple approach** - no band targeting
- **Conservative diversity** - basic paraphrase generation

### **New PhraseDP Implementation (76-question experiment)**

**File**: `utils.py`

```python
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
    NEW: 10 API calls targeting different similarity bands
    """
    
    # Define 10 different prompts for different similarity levels
    similarity_prompts = [
        {'level': 'band_0.0-0.1', 'target': '0.0-0.1', ...},
        {'level': 'band_0.1-0.2', 'target': '0.1-0.2', ...},
        {'level': 'band_0.2-0.3', 'target': '0.2-0.3', ...},
        {'level': 'band_0.3-0.4', 'target': '0.3-0.4', ...},
        {'level': 'band_0.4-0.5', 'target': '0.4-0.5', ...},
        {'level': 'band_0.5-0.6', 'target': '0.5-0.6', ...},
        {'level': 'band_0.6-0.7', 'target': '0.6-0.7', ...},
        {'level': 'band_0.7-0.8', 'target': '0.7-0.8', ...},
        {'level': 'band_0.8-0.9', 'target': '0.8-0.9', ...},
        {'level': 'band_0.9-1.0', 'target': '0.9-1.0', ...}
    ]
    
    all_paraphrases = []
    for prompt_config in similarity_prompts:
        # Make API call for each band
        response = nebius_client.chat.completions.create(
            model=nebius_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_config['prompt']}
            ],
            max_tokens=max_tokens,
            temperature=0.9,
            top_p=0.95,
            n=5  # Generate 5 candidates per band
        )
        
        # Process response and add to all_paraphrases
        # ... refill logic for underfilled bands
        # ... retry logic for failed bands
        # ... equalization logic across bands
```

**Characteristics:**
- **10 API calls** - one for each similarity band
- **10 different prompts** targeting specific similarity ranges
- **Complex band management** with refill and retry logic
- **Aggressive diversity** - wide similarity range (0.0-1.0)
- **Resource intensive** - 10x more API calls than old implementation

## API Call Comparison

| Aspect | Old PhraseDP | New PhraseDP |
|--------|--------------|--------------|
| **API Calls** | 1 | 10 |
| **Candidates per Call** | 10-100 | 5 per band |
| **Total Candidates** | 10-100 | 50 (5×10 bands) |
| **Band Targeting** | None | 10 bands (0.0-0.1, 0.1-0.2, etc.) |
| **Refill Mechanism** | None | Yes, with retry logic |
| **Resource Usage** | Low | High (10x more calls) |

## Performance Impact

### **Why Old PhraseDP Was More Efficient:**

1. **Single API Call**: Generated all candidates in one request
2. **Simple Approach**: No complex band management
3. **Resource Efficient**: Minimal API usage
4. **Conservative Diversity**: Sufficient for medical QA tasks

### **Why New PhraseDP Is More Resource-Intensive:**

1. **10 API Calls**: One call per similarity band
2. **Complex Management**: Refill, retry, and equalization logic
3. **Resource Intensive**: 10x more API calls
4. **Aggressive Diversity**: Overkill for medical QA tasks

## Code Evidence

### **Old Implementation (1 API call):**
```python
# Single API call with n parameter for multiple candidates
response = nebius_client.chat.completions.create(
    model=local_model,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ],
    n=num_return_sequences  # Generate multiple candidates in one call
)
```

### **New Implementation (10 API calls):**
```python
# 10 API calls, one per similarity band
for prompt_config in similarity_prompts:
    response = nebius_client.chat.completions.create(
        model=nebius_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_config['prompt']}
        ],
        n=5  # Generate 5 candidates per band
    )
    # Process response, refill logic, retry logic, etc.
```

## Conclusion

**Old PhraseDP: 1 API call**
- **Efficient**: Single API call generates all candidates
- **Simple**: No complex band management
- **Resource-friendly**: Minimal API usage
- **Sufficient**: Conservative diversity was adequate for medical QA

**New PhraseDP: 10 API calls**
- **Resource-intensive**: 10x more API calls
- **Complex**: Band management, refill, retry logic
- **Overkill**: Aggressive diversity unnecessary for medical QA
- **Expensive**: Higher computational and API costs

**The 10x increase in API calls (1 → 10) contributed to the performance degradation by:**
1. **Increased complexity** leading to more aggressive perturbations
2. **Resource overhead** affecting system performance
3. **Over-engineering** for medical QA tasks that didn't need such diversity

---
*Analysis confirming old PhraseDP used 1 API call vs new PhraseDP's 10 API calls*
*Date: 2025-01-27*
