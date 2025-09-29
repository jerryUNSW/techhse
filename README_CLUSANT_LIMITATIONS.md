# CluSanT Limitations

## Key Findings

Our experimental evaluation of CluSanT reveals several fundamental limitations that affect its applicability to general PII protection:

### 1. **Fundamental Incompatibility with PII Protection**
- **Core Problem**: PII is inherently unpredictable and infinite - cannot be pre-defined
- **Infinite Variations**: Names, emails, addresses have unlimited combinations
- **Dynamic Nature**: New domains, naming patterns, regional formats emerge constantly
- **Zero-Shot Requirement**: PII protection must work on ANY sensitive information, not just pre-known tokens
- **The Privacy Paradox**: To protect sensitive info, CluSanT must first know what's sensitive

### 2. **Domain-Specific Training Bias**
- Embedding file based on TAB dataset (European legal documents)
- Poor performance on general PII: 15.8% overall protection
- Name protection: 0% (names not in legal vocabulary)
- Address protection: 60% (some European legal addresses overlap)

### 3. **Limited Scalability**
- Complete retraining required for new domains
- Expensive GPT-4o augmentation process (100 similar tokens per original)
- No transfer learning capability

## Experimental Results

| Mechanism | Overall Protection | Email | Phone | Address | Names |
|-----------|------------------|-------|-------|---------|-------|
| CluSanT   | 15.8%           | 10%   | 30%   | 60%     | 0%    |
| InferDPT  | 99.6%           | 100%  | 100%  | 99%     | 99.6% |
| PhraseDP  | 98.9%           | 100%  | 100%  | 99%     | 97.4% |
| SANTEXT+  | 83.9%           | 100%  | 100%  | 89.4%   | 62.4% |

## Conclusion

CluSanT works well for bounded, predictable domains (legal documents) but struggles with the unpredictable nature of real-world sensitive information. Consider domain fit and token coverage before choosing CluSanT for general PII protection tasks.
