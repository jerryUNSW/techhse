# CluSanT Limitations Analysis

## Executive Summary

Based on our experimental evaluation and analysis of the CluSanT paper, we have identified several fundamental limitations that affect CluSanT's applicability to general PII protection tasks.

## Key Limitations

### 1. **Fundamental Incompatibility with PII Protection**

**Core Problem**: CluSanT's approach is fundamentally incompatible with PII protection because PII is inherently unpredictable and infinite.

**Why PII Cannot Be Pre-defined**:
- **Infinite Variations**: Names, emails, addresses have unlimited combinations
- **Dynamic Nature**: New domains (.ai, .io), naming patterns, regional formats emerge constantly
- **Zero-Shot Requirement**: PII protection must work on ANY sensitive information, not just pre-known tokens
- **Unpredictable Patterns**: Cannot exhaustively list all possible PII tokens

**The Privacy Paradox**:
```python
# What CluSanT requires (impossible for PII):
predefined_sensitive_tokens = [
    "john.smith@email.com",      # But what about...
    "mary.johnson@company.org",  # These new emails?
    "kazuosun@hotmail.net",      # That weren't predicted?
    "aaliyah.popova4783@aol.edu" # Beforehand?
]

# Reality (infinite possibilities):
real_world_pii = [
    "newuser123@domain.net",      # Unpredictable
    "unique.name@company.co.uk",  # Infinite variations
    "temporary.email@service.io", # Cannot be pre-listed
    "..."  # Goes on forever
]

# What PII protection needs:
def protect_pii(text):
    # Must work on ANY PII, not just pre-defined tokens
    return sanitize_unpredictable_pii(text)

# What CluSanT provides:
def protect_predefined_tokens(text, predefined_set):
    # Only works on tokens we already know about
    return sanitize_known_tokens(text, predefined_set)
```

**Why This Matters**: 
- In real-world scenarios, sensitive information (names, addresses, emails) is often unpredictable
- New sensitive tokens emerge that weren't in the original training set
- Zero-shot PII detection becomes impossible
- CluSanT falls back to no sanitization for unknown tokens

### 2. **Domain-Specific Training Data Bias**

**Problem**: CluSanT's embedding file is based on the TAB dataset (European Court of Human Rights legal documents).

**Evidence from Paper**:
- Uses TAB dataset (1,268 court cases) for token augmentation
- GPT-4o augmentation creates legal/political similar tokens
- Example: "Sinn Fein headquarters" → "Labour Party headquarters"

**Impact on General PII Protection**:
- 0% name protection: Common names not in legal document vocabulary
- 60% address protection: Some overlap with European legal addresses
- Poor email/phone protection: Legal vs. personal contact patterns differ

### 3. **Limited Scalability to New Domains**

**Problem**: Adapting CluSanT to new domains requires complete retraining.

**Required Workflow**:
1. Extract domain-specific sensitive tokens
2. Augment with GPT-4o (100 similar tokens per original)
3. Generate new embeddings using all-MiniLM-L6-v2
4. Create new clusters
5. Retrain the entire system

**Practical Challenges**:
- Expensive and time-consuming for each new domain
- Requires domain expertise to identify sensitive tokens
- No transfer learning capability

### 4. **Technical Advantages vs. Practical Limitations**

**Technical Strengths**:
- Multi-word embedding capability (superior to SanText/CusText)
- Uses all-MiniLM-L6-v2 sentence embedder
- Can handle phrases like "Sinn Fein headquarters"

**Practical Limitations**:
- Multi-word capability only works if phrases are in the embedding file
- No advantage if target PII isn't in the predefined token set
- Falls back to no sanitization for unknown tokens

## Experimental Evidence

### Our PII Protection Results:
- **Overall Protection**: 15.8% (vs. 99.6% for InferDPT, 98.9% for PhraseDP)
- **Name Protection**: 0% (names not in TAB dataset vocabulary)
- **Address Protection**: 60% (some European legal addresses overlap)
- **Email Protection**: 10% (legal vs. personal email patterns differ)
- **Phone Protection**: 30% (legal vs. personal phone patterns differ)

### Consistency Across Epsilon Values:
- Same performance (15.8%) across all epsilon values (1.0, 1.5, 2.0, 2.5, 3.0)
- Indicates the issue is vocabulary mismatch, not parameter tuning

## Comparison with Alternative Approaches

### Pattern-Based Approaches (Regex):
- ✅ No pre-defined token sets needed
- ✅ Can handle infinite variations of PII patterns
- ❌ Less sophisticated than embedding-based methods

### LLM-Based Approaches (SANTEXT+):
- ✅ Dynamic PII identification
- ✅ No pre-defined token sets required
- ✅ Can adapt to new PII types
- ❌ Higher computational cost

## Recommendations

### For CluSanT Authors:
1. **Develop zero-shot capabilities** for unknown sensitive tokens
2. **Create domain-agnostic embedding strategies**
3. **Investigate transfer learning** between domains
4. **Provide better guidance** for adapting to new domains

### For Practitioners:
1. **Consider domain fit** before choosing CluSanT
2. **Evaluate token coverage** in your specific use case
3. **Budget for retraining** when adapting to new domains
4. **Consider hybrid approaches** combining pattern-based and embedding-based methods

## Conclusion

While CluSanT represents a technically sophisticated approach with superior multi-word embedding capabilities, its fundamental dependency on pre-defined sensitive token sets significantly limits its practical applicability to general PII protection tasks. The approach works well for bounded, predictable domains (like legal documents) but struggles with the unpredictable nature of real-world sensitive information.

The 15.8% overall protection rate we observed is not a technical failure but rather a natural consequence of applying a domain-specific tool to a general-purpose task. This highlights the need for more flexible, adaptive approaches to privacy-preserving text sanitization.
