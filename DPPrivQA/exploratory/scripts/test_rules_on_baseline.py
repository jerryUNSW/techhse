#!/usr/bin/env python3
"""
Test rule-based decision making on existing baseline data.
Uses Local and Local+CoT results from database, applies rules to decide which to use,
and reports the accuracy without running new experiments.
"""

import sqlite3
import json
import sys
import os
from typing import Dict, Tuple, Any
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dpprivqa.utils.config import load_config


def rule_very_short_question(tuple, attr) -> bool:
    """
    Rule: Very short questions (<10 words) should use local-only.
    Degradation cases average 10.7 words, CoT-helpful average 13.6 words.
    """
    question, options = tuple
    question_words = question.split()
    question_length = len(question_words)
    # return True if use local-only, False otherwise
    return question_length < 10


def rule_low_complexity(tuple, attr) -> bool:
    """
    Rule: Low complexity questions (score <17) should use local-only.
    Degradation cases average 16.6 complexity, CoT-helpful average 18.5.
    Complexity = question_length + avg_option_length.
    """
    question, options = tuple
    question_words = question.split()
    question_length = len(question_words)
    
    option_lengths = [len(v.split()) for v in options.values()]
    avg_option_length = sum(option_lengths) / len(option_lengths) if option_lengths else 0
    complexity_score = question_length + avg_option_length
    
    # return True if use local-only, False otherwise
    return complexity_score < 17


def rule_short_factual(tuple, attr) -> bool:
    """
    Rule: Short factual questions (<12 words) should use local-only.
    Degradation cases are often short factual recall questions.
    """
    question, options = tuple
    question_words = question.split()
    question_length = len(question_words)
    
    question_lower = question.lower()
    factual_keywords = ['what is', 'which of the following', 'what are', 'define', 'identify', 'name']
    is_factual = any(kw in question_lower for kw in factual_keywords)
    
    # return True if use local-only, False otherwise
    return is_factual and question_length < 12


def rule_negative_question(tuple, attr) -> bool:
    """
    Rule: Negative questions should use CoT (return False).
    CoT-helpful cases have 18.4% negative questions vs 4.3% in degradation.
    This is a strong signal to use CoT, so return False (don't use local-only).
    """
    question, options = tuple
    question_lower = question.lower()
    negative_keywords = ['not', 'false', 'except', 'never', 'none']
    is_negative = any(kw in question_lower for kw in negative_keywords)
    
    # return True if use local-only, False otherwise
    # For negative questions, we want CoT, so return False
    return False if is_negative else None  # None means "not applicable"


def rule_short_comparison(tuple, attr) -> bool:
    """
    Rule: Short comparison questions (<12 words) should use local-only.
    Longer comparison questions benefit from CoT.
    """
    question, options = tuple
    question_words = question.split()
    question_length = len(question_words)
    
    question_lower = question.lower()
    comparison_keywords = ['most appropriate', 'most likely', 'best', 'least', 'greatest', 'most common']
    is_comparison = any(kw in question_lower for kw in comparison_keywords)
    
    # return True if use local-only, False otherwise
    return is_comparison and question_length < 12


def rule_very_simple_structure(tuple, attr) -> bool:
    """
    Rule: Questions with very simple structure (ends with colon, <10 words) should use local-only.
    Example: "With an increasing number of sprints the:" (6 words, degradation case)
    """
    question, options = tuple
    question_words = question.split()
    question_length = len(question_words)
    
    # Check if question ends with colon (incomplete sentence structure)
    ends_with_colon = question.strip().endswith(':')
    
    # return True if use local-only, False otherwise
    return ends_with_colon and question_length < 10


def rule_short_which_question(tuple, attr) -> bool:
    """
    Rule: Short "Which of the following" questions (<12 words) should use local-only.
    Many degradation cases follow this pattern.
    """
    question, options = tuple
    question_words = question.split()
    question_length = len(question_words)
    
    question_lower = question.lower()
    starts_with_which = question_lower.startswith('which')
    
    # return True if use local-only, False otherwise
    return starts_with_which and question_length < 12


def should_use_local_only(question: str, options: Dict[str, str]) -> Tuple[bool, str]:
    """
    Apply all rules to decide whether to use local-only.
    
    Returns:
        (use_local_only: bool, reason: str)
    """
    tuple_data = (question, options)
    
    # Rule 1: Negative questions → use CoT (strong signal)
    negative_result = rule_negative_question(tuple_data, None)
    if negative_result is False:
        return False, "negative_question_use_cot"
    
    # Rule 2: Very short questions → use local-only
    if rule_very_short_question(tuple_data, None):
        return True, "very_short_question"
    
    # Rule 3: Low complexity → use local-only
    if rule_low_complexity(tuple_data, None):
        return True, "low_complexity"
    
    # Rule 4: Short factual → use local-only
    if rule_short_factual(tuple_data, None):
        return True, "short_factual"
    
    # Rule 5: Short comparison → use local-only
    if rule_short_comparison(tuple_data, None):
        return True, "short_comparison"
    
    # Rule 6: Very simple structure → use local-only
    if rule_very_simple_structure(tuple_data, None):
        return True, "very_simple_structure"
    
    # Rule 7: Short which question → use local-only
    if rule_short_which_question(tuple_data, None):
        return True, "short_which_question"
    
    # Default: use CoT (since CoT helps more often than it hurts)
    return False, "default_use_cot"


def load_baseline_data(db_path: str, experiment_id_local: int = 9, experiment_id_cot: int = 23):
    """Load baseline Local and Local+CoT results from database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Load local results
    local_results = {}
    cursor = conn.execute("""
        SELECT question_idx, original_question, options, is_correct, generated_answer
        FROM mmlu_clinical_knowledge_epsilon_independent_results
        WHERE experiment_id = ? AND scenario = 'local'
        ORDER BY question_idx
    """, (experiment_id_local,))
    
    for row in cursor:
        local_results[row['question_idx']] = {
            'question': row['original_question'],
            'options': json.loads(row['options']),
            'is_correct': bool(row['is_correct']),
            'answer': row['generated_answer']
        }
    
    # Load local+CoT results
    cot_results = {}
    cursor = conn.execute("""
        SELECT question_idx, original_question, options, is_correct, generated_answer
        FROM mmlu_clinical_knowledge_epsilon_independent_results
        WHERE experiment_id = ? AND scenario = 'local_cot'
        ORDER BY question_idx
    """, (experiment_id_cot,))
    
    for row in cursor:
        cot_results[row['question_idx']] = {
            'question': row['original_question'],
            'options': json.loads(row['options']),
            'is_correct': bool(row['is_correct']),
            'answer': row['generated_answer']
        }
    
    conn.close()
    
    return local_results, cot_results


def main():
    config = load_config()
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    
    print("="*80)
    print("Testing Rule-Based Decision Making on Baseline Data")
    print("Dataset: Clinical Knowledge")
    print("="*80)
    
    # Load baseline data
    print("\nLoading baseline data from database...")
    local_results, cot_results = load_baseline_data(db_path)
    
    print(f"Loaded {len(local_results)} Local results")
    print(f"Loaded {len(cot_results)} Local+CoT results")
    
    # Calculate baseline accuracies
    local_correct = sum(1 for r in local_results.values() if r['is_correct'])
    local_total = len(local_results)
    local_accuracy = local_correct / local_total if local_total > 0 else 0.0
    
    cot_correct = sum(1 for r in cot_results.values() if r['is_correct'])
    cot_total = len(cot_results)
    cot_accuracy = cot_correct / cot_total if cot_total > 0 else 0.0
    
    print(f"\nBaseline Accuracies:")
    print(f"  Local only: {local_accuracy:.3f} ({local_correct}/{local_total})")
    print(f"  Local+CoT: {cot_accuracy:.3f} ({cot_correct}/{cot_total})")
    
    # Apply rules to decide which result to use
    print("\nApplying rules to decide Local-only vs CoT for each question...")
    
    rule_based_correct = 0
    rule_based_total = 0
    local_only_count = 0
    cot_count = 0
    decision_reasons = Counter()
    
    # Track cases
    degradation_avoided = 0  # Local ✓ but CoT ✗, we used Local
    cot_helpful_captured = 0  # Local ✗ but CoT ✓, we used CoT
    degradation_missed = 0  # Local ✓ but CoT ✗, we used CoT (wrong decision)
    cot_helpful_missed = 0  # Local ✗ but CoT ✓, we used Local (wrong decision)
    
    for q_idx in sorted(local_results.keys()):
        if q_idx not in cot_results:
            continue
        
        local_result = local_results[q_idx]
        cot_result = cot_results[q_idx]
        
        question = local_result['question']
        options = local_result['options']
        
        # Apply rules
        use_local_only, reason = should_use_local_only(question, options)
        
        # Select result based on rule
        if use_local_only:
            selected_result = local_result
            local_only_count += 1
        else:
            selected_result = cot_result
            cot_count += 1
        
        decision_reasons[reason] += 1
        
        # Track correctness
        if selected_result['is_correct']:
            rule_based_correct += 1
        
        rule_based_total += 1
        
        # Track specific cases
        local_correct = local_result['is_correct']
        cot_correct = cot_result['is_correct']
        
        if local_correct and not cot_correct:
            # Degradation case: Local ✓ but CoT ✗
            if use_local_only:
                degradation_avoided += 1
            else:
                degradation_missed += 1
        elif not local_correct and cot_correct:
            # CoT helpful case: Local ✗ but CoT ✓
            if not use_local_only:
                cot_helpful_captured += 1
            else:
                cot_helpful_missed += 1
    
    rule_based_accuracy = rule_based_correct / rule_based_total if rule_based_total > 0 else 0.0
    
    print(f"\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nRule-Based Selective Accuracy: {rule_based_accuracy:.3f} ({rule_based_correct}/{rule_based_total})")
    print(f"\nDecision Breakdown:")
    print(f"  Local-only: {local_only_count} ({local_only_count/rule_based_total*100:.1f}%)")
    print(f"  CoT: {cot_count} ({cot_count/rule_based_total*100:.1f}%)")
    
    print(f"\nDecision Reasons:")
    for reason, count in decision_reasons.most_common():
        print(f"  {reason}: {count} ({count/rule_based_total*100:.1f}%)")
    
    print(f"\nComparison with Baselines:")
    print(f"  Baseline Local only: {local_accuracy:.3f} ({local_correct}/{local_total})")
    print(f"  Baseline Local+CoT: {cot_accuracy:.3f} ({cot_correct}/{cot_total})")
    print(f"  Rule-Based Selective: {rule_based_accuracy:.3f} ({rule_based_correct}/{rule_based_total})")
    
    improvement_over_local = (rule_based_accuracy - local_accuracy) * 100
    improvement_over_cot = (rule_based_accuracy - cot_accuracy) * 100
    
    print(f"\nImprovement:")
    print(f"  Over Local: {improvement_over_local:+.1f} percentage points")
    print(f"  Over Local+CoT: {improvement_over_cot:+.1f} percentage points")
    
    print(f"\nCase Analysis:")
    print(f"  Degradation cases avoided (Local ✓, CoT ✗, used Local): {degradation_avoided}")
    print(f"  Degradation cases missed (Local ✓, CoT ✗, used CoT): {degradation_missed}")
    print(f"  CoT helpful captured (Local ✗, CoT ✓, used CoT): {cot_helpful_captured}")
    print(f"  CoT helpful missed (Local ✗, CoT ✓, used Local): {cot_helpful_missed}")
    
    if rule_based_accuracy >= cot_accuracy:
        print(f"\n✓ SUCCESS: Rule-based approach matches or exceeds baseline Local+CoT!")
    else:
        print(f"\n⚠ Rule-based accuracy is {cot_accuracy - rule_based_accuracy:.3f} lower than baseline Local+CoT")
    
    print("="*80)


if __name__ == "__main__":
    main()


