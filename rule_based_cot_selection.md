# Rule-Based CoT Selection System for PhraseDP

**Date:** January 2025  
**Purpose:** Optimize when to use CoT vs. local-only in PhraseDP+QA pipeline  
**Based on:** Analysis of cases where Local succeeds but PhraseDP+CoT fails

---

## Overview

This document defines rule-based heuristics to decide when the local model should skip using the induced CoT and fall back to local-only answering. The rules are based on experimental evidence from degradation cases analysis.

**Key Insight:** CoT helps when Local needs guidance, but hurts when:
- Local can already answer correctly (easy questions)
- Perturbation is too aggressive (removes critical information)
- CoT quality is poor (brief, ambiguous, or generic)

---

## Rule Configuration

Each rule can be individually enabled/disabled and threshold values can be adjusted for testing.

```python
# Rule configuration for testing
RULE_CONFIG = {
    'rule_1_perturbed_length': {
        'enabled': True,
        'threshold': 200,  # characters
    },
    'rule_2_length_reduction': {
        'enabled': True,
        'threshold': 0.60,  # 60% reduction
    },
    'rule_3_cot_quality': {
        'enabled': True,
        'min_cot_length': 100,  # characters
        'max_ambiguity_indicators': 2,
        'min_length_with_restriction': 200,  # characters
    },
    'rule_4_local_confidence': {
        'enabled': True,
        'high_confidence_threshold': 0.9,  # 0-1 scale
        'token_probability_threshold': 0.8,  # 0-1 scale
    },
    'rule_5_original_length': {
        'enabled': True,
        'threshold': 1500,  # characters
    },
}
```

---

## Rule 1: Perturbed Question Length Check

**Priority:** High (Primary Rule)  
**Type:** Pre-CoT Generation  
**Evidence:** Very short perturbed questions fail 3.2x more often

### Description

Skip CoT if the perturbed question is too short after perturbation. This indicates aggressive perturbation has removed too much critical information.

### Evidence

| Perturbed Length | Failure Rate | Avg Reduction |
|------------------|--------------|---------------|
| <200 chars | **35.97%** ⚠️ | 70.9% |
| 200-299 chars | 21.46% | 59.2% |
| 700-799 chars | **11.32%** | 26.7% |

**Finding:** Very short perturbed questions (<200 chars) fail 3.2x more often than medium-length questions (700-799 chars).

### Implementation

```python
def should_skip_cot_by_perturbed_length(
    perturbed_question: str,
    threshold: int = 200
) -> bool:
    """
    Rule 1: Skip CoT if perturbed question is too short.
    
    Args:
        perturbed_question: Question text after PhraseDP perturbation
        threshold: Minimum acceptable perturbed question length (default: 200 chars)
    
    Returns:
        True if CoT should be skipped, False otherwise
    """
    return len(perturbed_question.strip()) < threshold
```

### Rationale

- Very short perturbed questions lose critical medical details
- Insufficient context for effective CoT generation
- Local model may struggle with incomplete information
- Better to rely on local model's direct knowledge than poor CoT

### Testing

To test this rule independently:
- Set `RULE_CONFIG['rule_1_perturbed_length']['enabled'] = True`
- Set all other rules to `enabled = False`
- Adjust `threshold` to test sensitivity (e.g., 150, 200, 250, 300)

---

## Rule 2: Aggressive Perturbation Check (Length Reduction)

**Priority:** High  
**Type:** Pre-CoT Generation  
**Evidence:** Failed cases have larger length reduction than successful cases

### Description

Skip CoT if the perturbation removes too much content (high percentage reduction), indicating aggressive perturbation that destroys critical information.

### Evidence

| Case Type | Avg Original | Avg Perturbed | Length Reduction |
|-----------|--------------|---------------|------------------|
| **Failed** | 764 chars | 378 chars | **-50.5%** ⚠️ |
| **Successful** | 709 chars | 405 chars | **-42.9%** |

**Finding:** Failed cases have 7.6 percentage points more reduction than successful cases.

### Implementation

```python
def should_skip_cot_by_length_reduction(
    original_question: str,
    perturbed_question: str,
    threshold: float = 0.60  # 60% reduction
) -> bool:
    """
    Rule 2: Skip CoT if length reduction is too aggressive.
    
    Args:
        original_question: Original unperturbed question text
        perturbed_question: Question text after PhraseDP perturbation
        threshold: Maximum acceptable length reduction (default: 0.60 = 60%)
    
    Returns:
        True if CoT should be skipped, False otherwise
    """
    original_len = len(original_question.strip())
    if original_len == 0:
        return False
    
    perturbed_len = len(perturbed_question.strip())
    length_reduction = 1.0 - (perturbed_len / original_len)
    
    return length_reduction > threshold
```

### Rationale

- Aggressive perturbation (>60% reduction) removes critical medical information
- Context needed for reasoning is lost
- Specific symptoms, findings, and qualifiers may be removed
- Perturbed question becomes too vague for effective CoT

### Testing

To test this rule independently:
- Set `RULE_CONFIG['rule_2_length_reduction']['enabled'] = True`
- Set all other rules to `enabled = False`
- Adjust `threshold` to test sensitivity (e.g., 0.50, 0.60, 0.70)

---

## Rule 3: CoT Quality Indicators

**Priority:** Medium  
**Type:** Post-CoT Generation  
**Evidence:** Degradation cases have brief, ambiguous, or generic CoT

### Description

Skip CoT if quality indicators suggest the CoT will be harmful rather than helpful. This includes brief CoT, ambiguous CoT (mentions multiple options), or generic CoT with restriction phrases.

### Evidence

**Degradation Cases (Local ✓, Local+CoT ✗):**
- Average CoT length: 80 words
- **ALL 8 valid CoT cases mention BOTH correct and wrong answers**
- Explicitly states "concise" or "brief" answers
- Provides only 2-3 bullet points after restriction phrases

**Success Cases (Local ✗, Local+CoT ✓):**
- Average CoT length: 69 words (but well-structured)
- Explicitly guides to correct answer (22.6% of cases)
- Provides substantial detailed reasoning (average ~69 words, up to 188 words)

### Implementation

```python
def should_skip_cot_by_quality(
    cot_text: str,
    min_cot_length: int = 100,
    max_ambiguity_indicators: int = 2,
    min_length_with_restriction: int = 200
) -> bool:
    """
    Rule 3: Skip CoT if quality indicators suggest it will be harmful.
    
    Args:
        cot_text: Generated CoT text
        min_cot_length: Minimum acceptable CoT length in characters
        max_ambiguity_indicators: Maximum allowed mentions of multiple options
        min_length_with_restriction: Minimum length required if restriction phrases present
    
    Returns:
        True if CoT should be skipped, False otherwise
    """
    cot_clean = cot_text.strip()
    
    # Check if CoT is too short (brief/generic)
    if len(cot_clean) < min_cot_length:
        return True
    
    # Check for ambiguity indicators (mentions multiple answer options)
    ambiguity_keywords = [
        'option A', 'option B', 'option C', 'option D',
        'could be A', 'could be B', 'could be C', 'could be D',
        'either A or', 'either B or', 'either C or', 'either D or'
    ]
    cot_lower = cot_clean.lower()
    ambiguity_count = sum(1 for keyword in ambiguity_keywords 
                         if keyword in cot_lower)
    
    if ambiguity_count > max_ambiguity_indicators:
        return True
    
    # Check for restriction phrases without substantial reasoning
    restriction_phrases = [
        "i can't share",
        "i can't provide",
        "i'm sorry, but i can't",
        "concise answer",
        "brief answer"
    ]
    has_restriction = any(phrase in cot_lower for phrase in restriction_phrases)
    
    if has_restriction and len(cot_clean) < min_length_with_restriction:
        # Has restriction phrase but very brief → skip
        return True
    
    return False
```

### Rationale

- Brief CoT (<100 chars) provides minimal reasoning, not helpful
- CoT mentioning multiple options creates ambiguity and confusion
- Generic/restricted CoT without detailed reasoning doesn't guide to answer
- Degradation cases show this pattern consistently

### Testing

To test this rule independently:
- Set `RULE_CONFIG['rule_3_cot_quality']['enabled'] = True`
- Set all other rules to `enabled = False`
- Adjust `min_cot_length`, `max_ambiguity_indicators`, or `min_length_with_restriction`

---

## Rule 4: Local Model Confidence Check

**Priority:** Medium  
**Type:** Pre-CoT Generation (if local answer available)  
**Evidence:** Degradation cases show Local gets 100% correct without CoT

### Description

Skip CoT if the local model is highly confident in its answer. Easy questions that Local can answer correctly may not benefit from CoT, and CoT may add noise.

### Evidence

**Degradation Cases (Local ✓, Local+CoT ✗):**
- Local model accuracy: **100% (14/14)** ✓
- Remote model accuracy: 78.6% (11/14)
- **These are questions that Local can answer correctly on its own**

**Success Cases (Local ✗, Local+CoT ✓):**
- Local model accuracy: 0% (0/20) ✗
- Remote model accuracy: 95.0% (19/20)
- **These are questions where Local needs CoT guidance**

### Implementation

```python
def should_skip_cot_by_local_confidence(
    local_answer: str = None,
    local_confidence_score: float = None,
    local_token_probabilities: Dict[str, float] = None,
    high_confidence_threshold: float = 0.9,
    token_probability_threshold: float = 0.8
) -> bool:
    """
    Rule 4: Skip CoT if local model is highly confident.
    
    Args:
        local_answer: Local model's answer (optional, for logging)
        local_confidence_score: Local model's confidence score (0-1 scale)
        local_token_probabilities: Token probabilities from local model
        high_confidence_threshold: Threshold for high confidence (default: 0.9)
        token_probability_threshold: Threshold for token probability (default: 0.8)
    
    Returns:
        True if CoT should be skipped, False otherwise
    """
    # If confidence score is available and very high, skip CoT
    if local_confidence_score is not None:
        if local_confidence_score > high_confidence_threshold:
            return True
    
    # If token probabilities show clear winner, skip CoT
    if local_token_probabilities:
        max_prob = max(local_token_probabilities.values())
        if max_prob > token_probability_threshold:
            return True
    
    return False
```

### Rationale

- Local model gets 100% correct on degradation cases without CoT
- These are relatively straightforward questions for the local model
- Adding CoT to easy questions can introduce noise and confusion
- CoT helps when Local needs guidance, not when Local already knows

### Testing

To test this rule independently:
- Set `RULE_CONFIG['rule_4_local_confidence']['enabled'] = True`
- Set all other rules to `enabled = False`
- Requires local model to provide confidence scores or token probabilities
- Adjust thresholds to test sensitivity

### Note

This rule requires the local model to provide confidence information. If not available, this rule will always return `False` (don't skip).

---

## Rule 5: Original Question Length Check

**Priority:** Low  
**Type:** Pre-CoT Generation  
**Evidence:** Very long questions have higher CoT generation failure rates

### Description

Skip CoT if the original question is very long. Even though CoT generation failures are fixed, very long questions may still produce lower quality CoT due to complexity.

### Evidence

| Question Length | Local+CoT Failure Rate | PhraseDP+ Failure Rate |
|-----------------|------------------------|------------------------|
| <500 chars | 5.77% | 13.40% |
| 500-999 chars | 9.90% | 15.11% |
| 1000-1499 chars | 19.12% | 25.58% |
| **1500+ chars** | **50.00%** ⚠️ | **37.50%** ⚠️ |

### Implementation

```python
def should_skip_cot_by_original_length(
    original_question: str,
    threshold: int = 1500
) -> bool:
    """
    Rule 5: Skip CoT if original question is very long.
    
    Args:
        original_question: Original unperturbed question text
        threshold: Maximum acceptable original question length (default: 1500 chars)
    
    Returns:
        True if CoT should be skipped, False otherwise
    """
    return len(original_question.strip()) > threshold
```

### Rationale

- Very long questions are complex and may produce lower quality CoT
- Even if CoT generation succeeds, quality may be degraded
- Local model may handle long questions better with direct reasoning
- May be less relevant now that CoT generation failures are fixed

### Testing

To test this rule independently:
- Set `RULE_CONFIG['rule_5_original_length']['enabled'] = True`
- Set all other rules to `enabled = False`
- Adjust `threshold` to test sensitivity (e.g., 1200, 1500, 2000)

### Note

This rule may be less important now that CoT generation failures are fixed. Consider testing to see if it still provides value.

---

## Complete Decision Function

```python
from typing import Dict, Optional, Tuple

def should_use_cot(
    original_question: str,
    perturbed_question: str,
    cot_text: Optional[str] = None,
    local_answer: Optional[str] = None,
    local_confidence: Optional[float] = None,
    local_token_probabilities: Optional[Dict[str, float]] = None,
    rule_config: Dict = None
) -> Tuple[bool, str]:
    """
    Main decision function: Should we use CoT or fall back to local-only?
    
    Args:
        original_question: Original unperturbed question text
        perturbed_question: Perturbed question text after PhraseDP
        cot_text: Generated CoT text (if already generated, None otherwise)
        local_answer: Local model's answer (if already generated)
        local_confidence: Local model's confidence score (if available)
        local_token_probabilities: Token probabilities from local model (if available)
        rule_config: Rule configuration dictionary (uses default if None)
    
    Returns:
        Tuple of (should_use_cot: bool, reason: str)
        - should_use_cot: True if CoT should be used, False if should skip CoT
        - reason: Explanation of why CoT was skipped (empty if using CoT)
    """
    if rule_config is None:
        rule_config = RULE_CONFIG
    
    # Pre-CoT Generation Rules (Run Before Generating CoT)
    
    # Rule 5: Check original question length
    if rule_config['rule_5_original_length']['enabled']:
        if should_skip_cot_by_original_length(
            original_question,
            threshold=rule_config['rule_5_original_length']['threshold']
        ):
            return False, "Rule 5: Original question too long (>{} chars)".format(
                rule_config['rule_5_original_length']['threshold']
            )
    
    # Rule 1: Check perturbed question length
    if rule_config['rule_1_perturbed_length']['enabled']:
        if should_skip_cot_by_perturbed_length(
            perturbed_question,
            threshold=rule_config['rule_1_perturbed_length']['threshold']
        ):
            return False, "Rule 1: Perturbed question too short (<{} chars)".format(
                rule_config['rule_1_perturbed_length']['threshold']
            )
    
    # Rule 2: Check length reduction
    if rule_config['rule_2_length_reduction']['enabled']:
        if should_skip_cot_by_length_reduction(
            original_question,
            perturbed_question,
            threshold=rule_config['rule_2_length_reduction']['threshold']
        ):
            reduction = (1.0 - len(perturbed_question.strip()) / len(original_question.strip())) * 100
            return False, "Rule 2: Length reduction too aggressive ({:.1f}% > {:.0f}%)".format(
                reduction, rule_config['rule_2_length_reduction']['threshold'] * 100
            )
    
    # Rule 4: Check local model confidence
    if rule_config['rule_4_local_confidence']['enabled']:
        if local_answer and should_skip_cot_by_local_confidence(
            local_answer,
            local_confidence,
            local_token_probabilities,
            high_confidence_threshold=rule_config['rule_4_local_confidence']['high_confidence_threshold'],
            token_probability_threshold=rule_config['rule_4_local_confidence']['token_probability_threshold']
        ):
            return False, "Rule 4: Local model confidence too high"
    
    # Post-CoT Generation Rule (Run After Generating CoT)
    
    # Rule 3: Check CoT quality
    if rule_config['rule_3_cot_quality']['enabled'] and cot_text:
        if should_skip_cot_by_quality(
            cot_text,
            min_cot_length=rule_config['rule_3_cot_quality']['min_cot_length'],
            max_ambiguity_indicators=rule_config['rule_3_cot_quality']['max_ambiguity_indicators'],
            min_length_with_restriction=rule_config['rule_3_cot_quality']['min_length_with_restriction']
        ):
            return False, "Rule 3: CoT quality indicators suggest skipping"
    
    # All checks passed, use CoT
    return True, ""
```

---

## Usage Example

```python
# Example usage in PhraseDP+QA pipeline

# Step 1: Generate perturbed question
perturbed_question = phrase_dp_perturbation(original_question, epsilon)

# Step 2: Pre-CoT checks (before generating CoT)
should_use, reason = should_use_cot(
    original_question=original_question,
    perturbed_question=perturbed_question,
    rule_config=RULE_CONFIG
)

if not should_use:
    # Skip CoT, use local-only
    print(f"Skipping CoT: {reason}")
    final_answer = local_model_answer(perturbed_question)
    return final_answer

# Step 3: Generate CoT (only if pre-checks passed)
cot_text = generate_cot(perturbed_question)

# Step 4: Post-CoT quality check
should_use, reason = should_use_cot(
    original_question=original_question,
    perturbed_question=perturbed_question,
    cot_text=cot_text,
    rule_config=RULE_CONFIG
)

if not should_use:
    # CoT quality is poor, skip it
    print(f"Skipping CoT: {reason}")
    final_answer = local_model_answer(perturbed_question)
    return final_answer

# Step 5: Use CoT (all checks passed)
final_answer = local_model_answer_with_cot(perturbed_question, cot_text)
return final_answer
```

---

## Testing Strategy

### Individual Rule Testing

To test each rule independently:

1. **Enable only one rule at a time:**
```python
# Test Rule 1 only
test_config = {
    'rule_1_perturbed_length': {'enabled': True, 'threshold': 200},
    'rule_2_length_reduction': {'enabled': False},
    'rule_3_cot_quality': {'enabled': False},
    'rule_4_local_confidence': {'enabled': False},
    'rule_5_original_length': {'enabled': False},
}
```

2. **Run experiments with each rule enabled**
3. **Compare performance:**
   - Accuracy improvement vs. baseline
   - Number of CoT skips
   - Cases where skipping CoT helped vs. hurt

### Threshold Sensitivity Testing

For each rule with a threshold:
- Test multiple threshold values
- Plot accuracy vs. threshold
- Identify optimal threshold

### Combined Rules Testing

1. **Start with all rules enabled**
2. **Disable rules one by one** to see individual contribution
3. **Combine rules in different orders** to see priority effects

---

## Expected Impact

Based on the evidence:

| Rule | Expected Impact | Confidence |
|------|----------------|------------|
| **Rule 1: Perturbed Length** | High - Addresses 3.2x failure rate difference | High |
| **Rule 2: Length Reduction** | High - Addresses 7.6pp reduction difference | High |
| **Rule 3: CoT Quality** | Medium - Addresses 57% of degradation cases | Medium |
| **Rule 4: Local Confidence** | Medium - Addresses 100% local accuracy cases | Medium (if confidence available) |
| **Rule 5: Original Length** | Low - May be less relevant with fixed failures | Low |

**Combined Expected Impact:**
- Reduce degradation cases (Local ✓, PhraseDP+CoT ✗)
- Maintain or improve overall accuracy
- Better balance between privacy and utility

---

## References

- `degradation_cases_analysis.md` - Analysis of 14 cases where Local ✓ but Local+CoT ✗
- `phrasedp_cot_failure_analysis.md` - PhraseDP CoT failure rates by length
- `phrasedp_cot_failure_root_cause.md` - Root cause analysis of CoT failures
- `detailed_investigation_report.md` - Net improvement analysis
- `comparison_custext_clusant_vs_phrasedp.md` - Comparison with other methods

