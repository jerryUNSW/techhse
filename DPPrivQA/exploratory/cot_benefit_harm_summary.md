# Summary: When CoT Helps vs Hurts

## Key Statistics

- **Total Degradation Cases (S1 \ S2)**: 52 (Local ✓ but Local+CoT ✗)
- **Total CoT-Helpful Cases (S2 \ S1)**: 133 (Local ✗ but Local+CoT ✓)
- **Ratio**: ~2.6x more helpful cases than harmful cases

## Critical Differences

| Characteristic | Degradation Cases | CoT-Helpful Cases | Difference |
|----------------|-------------------|-------------------|------------|
| **Question Length** | 58.1 words | 76.1 words | **+18.0 words** |
| **Complexity Score** | 67.6 | 85.9 | **+18.3** |
| **Requires Reasoning** | 38.5% | 44.4% | +5.9% |
| **Has Scenario** | 15.4% | 12.8% | -2.6% |
| **Factual Questions** | 40.4% | 52.6% | +12.2% |
| **Comparison Questions** | 30.8% | 40.6% | +9.8% |
| **Negative Questions** | 19.2% | 36.1% | **+16.9%** |

## Key Findings

### 1. **Question Complexity Matters**
- CoT-helpful cases are **significantly longer** (76.1 vs 58.1 words)
- CoT-helpful cases have **higher complexity scores** (85.9 vs 67.6)
- **Pattern**: More complex questions benefit more from CoT

### 2. **Question Type Patterns**

**Degradation Cases Tend To Be:**
- Shorter questions (median: 21 words)
- Simpler structure
- Often factual recall questions
- Examples: "Which of the following is true of...", "With an increasing number of sprints the..."

**CoT-Helpful Cases Tend To Be:**
- Longer questions (median: 55 words)
- More complex reasoning required
- Often comparison/negative questions ("most appropriate", "NOT", "false")
- Examples: "Which of the following is NOT...", "A patient with... most likely..."

### 3. **Domain-Specific Patterns**

**Professional Law:**
- Degradation: 7 cases (avg complexity: 156.9)
- CoT-helpful: 31 cases (avg complexity: 158.6)
- Both are highly complex, but CoT helps more often

**Professional Medicine:**
- Degradation: 12 cases (avg complexity: 140.6)
- CoT-helpful: 38 cases (avg complexity: 121.4)
- Interesting: Degradation cases are MORE complex but still fail

**Clinical Knowledge:**
- Degradation: 23 cases (avg complexity: 34.9) - **LOWEST**
- CoT-helpful: 38 cases (avg complexity: 45.8)
- **Key insight**: Simple questions degrade more often

**College Medicine:**
- Degradation: 10 cases (avg complexity: 34.9)
- CoT-helpful: 26 cases (avg complexity: 45.8)
- Similar pattern to Clinical Knowledge

## Recommendations

### Questions That Benefit from CoT:

1. **Longer, complex questions** (>60 words, complexity score >80)
2. **Comparison questions** ("most appropriate", "best", "least")
3. **Negative questions** ("NOT", "false", "except")
4. **Multi-step reasoning questions** requiring domain knowledge
5. **Scenario-based questions** with multiple factors to evaluate

### Questions Better Answered Locally:

1. **Short, simple questions** (<30 words, complexity score <50)
2. **Straightforward factual recall** questions
3. **Questions where local model has strong domain knowledge**
4. **Simple definition questions**
5. **Questions with clear, unambiguous answers**

## Critical Insight

**The refusal pattern is NOT the problem.** 

Even though GPT-5 refuses to provide "step-by-step reasoning," the "concise summaries" it provides are often MORE effective than structured analytical guidance because they:
- Include direct conclusions/hints
- Are answer-oriented
- Provide actionable key points

**Simple questions degrade more often** because:
- Local model already knows the answer
- CoT adds noise/confusion
- CoT may overthink simple questions

**Complex questions benefit more** because:
- Local model lacks domain knowledge
- CoT provides needed context
- CoT helps navigate complex scenarios

