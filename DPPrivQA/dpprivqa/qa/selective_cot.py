"""
Selective CoT decision logic based on question characteristics.
"""

from typing import Dict, Tuple


def should_use_cot(question: str, options: Dict[str, str]) -> Tuple[bool, str]:
    """
    Decide whether to use CoT based on question characteristics.
    
    Rules based on cot_benefit_harm_summary.md:
    - Use CoT for: longer/complex questions, comparison, negative, reasoning, scenario-based
    - Use local only for: short/simple, factual recall, simple definitions
    
    Args:
        question: Question text
        options: Options dictionary
    
    Returns:
        (use_cot: bool, reason: str)
    """
    # Analyze question
    question_words = question.split()
    question_length = len(question_words)
    
    option_lengths = [len(v.split()) for v in options.values()]
    avg_option_length = sum(option_lengths) / len(option_lengths) if option_lengths else 0
    
    # Simplified complexity score (question_length + avg_option_length)
    complexity_score = question_length + avg_option_length
    
    question_lower = question.lower()
    
    # ===== "Better Answered Locally" Criteria =====
    
    # 1. Short, simple questions (<30 words, complexity <50)
    if question_length < 30 and complexity_score < 50:
        return False, "short_simple"
    
    # 2. Straightforward factual recall questions
    factual_keywords = ['what is', 'which of the following', 'what are', 'define', 'identify']
    is_factual = any(kw in question_lower for kw in factual_keywords)
    if is_factual and question_length < 40 and complexity_score < 60:
        return False, "factual_recall"
    
    # 3. Simple definition questions (very short factual)
    if is_factual and question_length < 20:
        return False, "simple_definition"
    
    # ===== "Benefit from CoT" Criteria =====
    
    # 1. Longer, complex questions (>60 words OR complexity >80)
    if question_length > 60 or complexity_score > 80:
        return True, "long_complex"
    
    # 2. Comparison questions
    comparison_keywords = ['most appropriate', 'most likely', 'best', 'least', 'greatest', 'most common']
    is_comparison = any(kw in question_lower for kw in comparison_keywords)
    if is_comparison:
        return True, "comparison"
    
    # 3. Negative questions
    is_negative = 'not' in question_lower or 'false' in question_lower or 'except' in question_lower
    if is_negative:
        return True, "negative"
    
    # 4. Requires reasoning
    reasoning_keywords = ['why', 'how', 'explain', 'analyze', 'compare', 'evaluate', 'determine']
    requires_reasoning = any(kw in question_lower for kw in reasoning_keywords)
    if requires_reasoning:
        return True, "reasoning"
    
    # 5. Scenario-based questions (patient, suspect, plaintiff, etc.)
    scenario_keywords = ['patient', 'suspect', 'plaintiff', 'defendant', 'witness']
    has_scenario = any(kw in question_lower for kw in scenario_keywords)
    if has_scenario and question_length > 40:
        return True, "scenario_based"
    
    # ===== Default Decision =====
    
    # Medium complexity: use CoT
    if complexity_score >= 50:
        return True, "medium_complexity"
    
    # Very simple: local only
    return False, "simple"


def should_use_cot_clinical_knowledge(question: str, options: Dict[str, str]) -> Tuple[bool, str]:
    """
    Dataset-specific decision rules for Clinical Knowledge - CONSERVATIVE VERSION.
    
    Key insight: The difference between degradation and CoT-helpful is very small.
    Only use CoT when we have STRONG signals (negative questions).
    Default to local-only to avoid degradation cases.
    
    Analysis:
    - Degradation: 10.7 words avg, 16.6 complexity, 4.3% negative
    - CoT-helpful: 13.6 words avg, 18.5 complexity, 18.4% negative (4x more!)
    - Strongest signal: Negative questions (18.4% vs 4.3%)
    - Everything else overlaps too much to be reliable
    
    Args:
        question: Question text
        options: Options dictionary
    
    Returns:
        (use_cot: bool, reason: str)
    """
    question_words = question.split()
    question_length = len(question_words)
    
    option_lengths = [len(v.split()) for v in options.values()]
    avg_option_length = sum(option_lengths) / len(option_lengths) if option_lengths else 0
    complexity_score = question_length + avg_option_length
    
    question_lower = question.lower()
    
    # ===== USE CoT ONLY FOR STRONG SIGNALS =====
    
    # 1. Negative questions - STRONGEST SIGNAL (18.4% vs 4.3%)
    is_negative = 'not' in question_lower or 'false' in question_lower or 'except' in question_lower
    if is_negative:
        return True, "negative_question"
    
    # 2. Comparison questions - but only if longer (avoid short degradation cases)
    comparison_keywords = ['most appropriate', 'most likely', 'best', 'least', 'greatest']
    is_comparison = any(kw in question_lower for kw in comparison_keywords)
    if is_comparison and question_length > 12:
        return True, "comparison_question"
    
    # 3. Requires reasoning - but only if longer
    reasoning_keywords = ['why', 'how', 'explain', 'analyze', 'evaluate']
    requires_reasoning = any(kw in question_lower for kw in reasoning_keywords)
    if requires_reasoning and question_length > 12:
        return True, "reasoning_required"
    
    # ===== DEFAULT: LOCAL ONLY =====
    # The overlap between degradation and CoT-helpful is too large
    # Better to miss some CoT benefits than to trigger degradation cases
    
    return False, "default_local_only"

