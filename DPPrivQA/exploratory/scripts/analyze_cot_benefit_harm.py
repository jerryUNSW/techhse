#!/usr/bin/env python3
r"""
Comprehensive Analysis: When CoT Helps vs Hurts
===============================================

Analyzes:
1. Degradation cases (S1 \ S2): Local ✓ but Local+CoT ✗
2. CoT-helpful cases (S2 \ S1): Local ✗ but Local+CoT ✓
3. Question characteristics that predict CoT benefit/harm
"""

import sqlite3
import json
import os
import sys
import re
from typing import List, Dict, Any
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DB_PATH = os.path.join(PROJECT_ROOT, "exp-results", "results.db")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "exploratory")

DATASET_CONFIGS = [
    {
        "name": "professional_law",
        "display_name": "Professional Law",
        "table": "mmlu_professional_law_epsilon_independent_results",
        "local_exp_id": 7,
        "cot_exp_id": 21,
    },
    {
        "name": "professional_medicine",
        "display_name": "Professional Medicine",
        "table": "mmlu_professional_medicine_epsilon_independent_results",
        "local_exp_id": 8,
        "cot_exp_id": 22,
    },
    {
        "name": "clinical_knowledge",
        "display_name": "Clinical Knowledge",
        "table": "mmlu_clinical_knowledge_epsilon_independent_results",
        "local_exp_id": 9,
        "cot_exp_id": 23,
    },
    {
        "name": "college_medicine",
        "display_name": "College Medicine",
        "table": "mmlu_college_medicine_epsilon_independent_results",
        "local_exp_id": 10,
        "cot_exp_id": 24,
    }
]


def analyze_question_characteristics(question: str, options: Dict[str, str]) -> Dict[str, Any]:
    """Analyze question characteristics."""
    # Question length
    question_words = question.split()
    question_length = len(question_words)
    
    # Options analysis
    option_lengths = [len(v.split()) for v in options.values()]
    avg_option_length = sum(option_lengths) / len(option_lengths) if option_lengths else 0
    
    # Question type indicators
    question_lower = question.lower()
    
    # Factual recall indicators
    factual_keywords = ['what is', 'which of the following', 'what are', 'define', 'identify']
    is_factual = any(kw in question_lower for kw in factual_keywords)
    
    # Reasoning indicators
    reasoning_keywords = ['why', 'how', 'explain', 'analyze', 'compare', 'evaluate', 'determine', 'most likely', 'most appropriate']
    requires_reasoning = any(kw in question_lower for kw in reasoning_keywords)
    
    # Legal/medical terminology density
    legal_terms = ['statute', 'law', 'legal', 'court', 'plaintiff', 'defendant', 'evidence', 'privilege', 'waiver']
    medical_terms = ['patient', 'diagnosis', 'symptom', 'treatment', 'disease', 'syndrome', 'clinical', 'medical']
    
    legal_density = sum(1 for term in legal_terms if term in question_lower) / max(len(question_words), 1) * 100
    medical_density = sum(1 for term in medical_terms if term in question_lower) / max(len(question_words), 1) * 100
    
    # Complexity indicators
    has_scenario = 'patient' in question_lower or 'suspect' in question_lower or 'plaintiff' in question_lower
    has_multiple_factors = question_lower.count('and') + question_lower.count(',') > 3
    
    # Question structure
    is_negative = 'not' in question_lower or 'false' in question_lower or 'except' in question_lower
    is_comparison = 'most' in question_lower or 'best' in question_lower or 'least' in question_lower
    
    return {
        'question_length': question_length,
        'avg_option_length': avg_option_length,
        'is_factual': is_factual,
        'requires_reasoning': requires_reasoning,
        'legal_density': legal_density,
        'medical_density': medical_density,
        'has_scenario': has_scenario,
        'has_multiple_factors': has_multiple_factors,
        'is_negative': is_negative,
        'is_comparison': is_comparison,
        'complexity_score': question_length + avg_option_length + (10 if has_scenario else 0) + (5 if has_multiple_factors else 0)
    }


def analyze_cot_quality(cot_text: str) -> Dict[str, Any]:
    """Analyze CoT text quality."""
    if not cot_text or cot_text.startswith("Error"):
        return {
            'is_error': True,
            'length': 0,
            'word_count': 0,
            'has_direct_answer': False,
            'has_reasoning_steps': False,
            'answer_hint_strength': 0
        }
    
    cot_lower = cot_text.lower()
    words = cot_text.split()
    
    # Check for direct answer hints
    direct_answer_patterns = [
        r'answer:\s*([A-D])',
        r'correct.*?([A-D])',
        r'most likely.*?([A-D])',
        r'most appropriate.*?([A-D])',
        r'conclusion.*?([A-D])',
        r'([A-D])\s*is\s*(correct|right|answer)',
        r'([A-D])\s*\.',
        r'choice\s*([A-D])',
    ]
    
    has_direct_answer = False
    answer_hint_strength = 0
    
    for pattern in direct_answer_patterns:
        matches = re.findall(pattern, cot_lower)
        if matches:
            has_direct_answer = True
            answer_hint_strength = len(matches)
            break
    
    # Also check for explicit answer statements
    explicit_patterns = [
        r'greatest risk:\s*([A-D])',
        r'most likely:\s*([A-D])',
        r'answer:\s*([A-D])',
        r'conclusion:\s*([A-D])',
    ]
    
    for pattern in explicit_patterns:
        matches = re.findall(pattern, cot_lower)
        if matches:
            has_direct_answer = True
            answer_hint_strength += len(matches)
    
    # Check for reasoning steps
    reasoning_markers = ['step', 'first', 'second', 'then', 'therefore', 'because', 'reasoning', 'analyze', 'consider']
    has_reasoning_steps = any(marker in cot_lower for marker in reasoning_markers)
    
    # Check for key points format
    has_key_points = 'key points' in cot_lower or 'key considerations' in cot_lower
    
    return {
        'is_error': False,
        'length': len(cot_text),
        'word_count': len(words),
        'has_direct_answer': has_direct_answer,
        'has_reasoning_steps': has_reasoning_steps,
        'has_key_points': has_key_points,
        'answer_hint_strength': answer_hint_strength
    }


def extract_cases(conn: sqlite3.Connection, config: Dict[str, Any], case_type: str) -> List[Dict[str, Any]]:
    r"""Extract degradation cases (S1 \ S2) or CoT-helpful cases (S2 \ S1)."""
    table = config['table']
    
    if case_type == 'degradation':  # S1 \ S2: Local ✓ but Local+CoT ✗
        query = f"""
            SELECT 
                local_res.question_idx,
                local_res.original_question as question,
                local_res.options,
                local_res.generated_answer as local_answer,
                cot_res.generated_answer as cot_answer,
                cot_res.cot_text,
                local_res.ground_truth_answer as ground_truth
            FROM {table} local_res
            JOIN {table} cot_res
                ON local_res.question_idx = cot_res.question_idx
            WHERE local_res.experiment_id = ?
                AND local_res.scenario = 'local'
                AND local_res.is_correct = 1
                AND cot_res.experiment_id = ?
                AND cot_res.scenario = 'local_cot'
                AND cot_res.is_correct = 0
            ORDER BY local_res.question_idx
        """
    else:  # S2 \ S1: Local ✗ but Local+CoT ✓
        query = f"""
            SELECT 
                local_res.question_idx,
                local_res.original_question as question,
                local_res.options,
                local_res.generated_answer as local_answer,
                cot_res.generated_answer as cot_answer,
                cot_res.cot_text,
                local_res.ground_truth_answer as ground_truth
            FROM {table} local_res
            JOIN {table} cot_res
                ON local_res.question_idx = cot_res.question_idx
            WHERE local_res.experiment_id = ?
                AND local_res.scenario = 'local'
                AND local_res.is_correct = 0
                AND cot_res.experiment_id = ?
                AND cot_res.scenario = 'local_cot'
                AND cot_res.is_correct = 1
            ORDER BY local_res.question_idx
        """
    
    cursor = conn.execute(query, (config['local_exp_id'], config['cot_exp_id']))
    rows = cursor.fetchall()
    
    columns = [desc[0] for desc in cursor.description]
    cases = [dict(zip(columns, row)) for row in rows]
    
    # Add analysis
    for case in cases:
        # Parse options
        if isinstance(case['options'], str):
            try:
                case['options_dict'] = json.loads(case['options'])
            except:
                case['options_dict'] = {}
        else:
            case['options_dict'] = case['options']
        
        # Analyze question
        case['question_analysis'] = analyze_question_characteristics(
            case['question'], case['options_dict']
        )
        
        # Analyze CoT
        case['cot_analysis'] = analyze_cot_quality(case.get('cot_text', ''))
    
    return cases


def aggregate_statistics(all_cases: Dict[str, List]) -> Dict[str, Any]:
    """Aggregate statistics across all datasets."""
    all_question_analyses = []
    all_cot_analyses = []
    
    for cases in all_cases.values():
        for case in cases:
            all_question_analyses.append(case['question_analysis'])
            all_cot_analyses.append(case['cot_analysis'])
    
    if not all_question_analyses:
        return {}
    
    stats = {
        'total_cases': len(all_question_analyses),
        'question_length_avg': sum(qa['question_length'] for qa in all_question_analyses) / len(all_question_analyses),
        'question_length_median': sorted([qa['question_length'] for qa in all_question_analyses])[len(all_question_analyses)//2],
        'factual_pct': sum(1 for qa in all_question_analyses if qa['is_factual']) / len(all_question_analyses) * 100,
        'reasoning_pct': sum(1 for qa in all_question_analyses if qa['requires_reasoning']) / len(all_question_analyses) * 100,
        'scenario_pct': sum(1 for qa in all_question_analyses if qa['has_scenario']) / len(all_question_analyses) * 100,
        'multiple_factors_pct': sum(1 for qa in all_question_analyses if qa['has_multiple_factors']) / len(all_question_analyses) * 100,
        'negative_pct': sum(1 for qa in all_question_analyses if qa['is_negative']) / len(all_question_analyses) * 100,
        'comparison_pct': sum(1 for qa in all_question_analyses if qa['is_comparison']) / len(all_question_analyses) * 100,
        'complexity_avg': sum(qa['complexity_score'] for qa in all_question_analyses) / len(all_question_analyses),
        'cot_direct_answer_pct': sum(1 for ca in all_cot_analyses if ca.get('has_direct_answer', False)) / len(all_cot_analyses) * 100 if all_cot_analyses else 0,
        'cot_hint_strength_avg': sum(ca.get('answer_hint_strength', 0) for ca in all_cot_analyses) / len(all_cot_analyses) if all_cot_analyses else 0,
        'cot_length_avg': sum(ca.get('length', 0) for ca in all_cot_analyses) / len(all_cot_analyses) if all_cot_analyses else 0,
    }
    
    return stats


def format_statistics(stats: Dict[str, Any]) -> str:
    """Format statistics as markdown."""
    if not stats:
        return "No data available."
    
    lines = []
    lines.append(f"- **Total cases**: {stats['total_cases']}")
    lines.append(f"- **Average question length**: {stats['question_length_avg']:.1f} words (median: {stats['question_length_median']})")
    lines.append(f"- **Factual questions**: {stats['factual_pct']:.1f}%")
    lines.append(f"- **Requires reasoning**: {stats['reasoning_pct']:.1f}%")
    lines.append(f"- **Has scenario**: {stats['scenario_pct']:.1f}%")
    lines.append(f"- **Multiple factors**: {stats['multiple_factors_pct']:.1f}%")
    lines.append(f"- **Negative questions**: {stats['negative_pct']:.1f}%")
    lines.append(f"- **Comparison questions**: {stats['comparison_pct']:.1f}%")
    lines.append(f"- **Average complexity score**: {stats['complexity_avg']:.1f}")
    lines.append(f"- **CoT has direct answer hint**: {stats['cot_direct_answer_pct']:.1f}%")
    lines.append(f"- **CoT hint strength (avg)**: {stats['cot_hint_strength_avg']:.2f}")
    lines.append(f"- **CoT length (avg)**: {stats['cot_length_avg']:.0f} characters")
    
    return "\n".join(lines)


def compare_statistics(degradation_stats: Dict[str, Any], helpful_stats: Dict[str, Any]) -> str:
    """Compare degradation vs helpful statistics."""
    lines = []
    
    if not degradation_stats or not helpful_stats:
        return "Insufficient data for comparison."
    
    comparisons = [
        ('question_length_avg', 'Question Length', 'words'),
        ('factual_pct', 'Factual Questions', '%'),
        ('reasoning_pct', 'Requires Reasoning', '%'),
        ('scenario_pct', 'Has Scenario', '%'),
        ('complexity_avg', 'Complexity Score', ''),
        ('cot_direct_answer_pct', 'CoT Direct Answer Hint', '%'),
        ('cot_hint_strength_avg', 'CoT Hint Strength', ''),
    ]
    
    lines.append("| Characteristic | Degradation | CoT-Helpful | Difference |")
    lines.append("|----------------|-------------|-------------|------------|")
    
    for key, label, unit in comparisons:
        deg_val = degradation_stats.get(key, 0)
        help_val = helpful_stats.get(key, 0)
        diff = help_val - deg_val
        diff_str = f"{diff:+.1f}" if diff != 0 else "0.0"
        lines.append(f"| {label} | {deg_val:.1f}{unit} | {help_val:.1f}{unit} | {diff_str}{unit} |")
    
    return "\n".join(lines)


def identify_patterns(degradation_cases: Dict[str, List], helpful_cases: Dict[str, List]) -> str:
    """Identify patterns in question types."""
    lines = []
    
    # Analyze question starters
    degradation_starters = Counter()
    helpful_starters = Counter()
    
    for cases in degradation_cases.values():
        for case in cases:
            question = case['question']
            first_words = ' '.join(question.split()[:3]).lower()
            degradation_starters[first_words] += 1
    
    for cases in helpful_cases.values():
        for case in cases:
            question = case['question']
            first_words = ' '.join(question.split()[:3]).lower()
            helpful_starters[first_words] += 1
    
    lines.append("### Common Question Patterns")
    lines.append("")
    lines.append("**Degradation Cases (S1 \\ S2) - Top Question Starters:**")
    lines.append("")
    for starter, count in degradation_starters.most_common(5):
        lines.append(f"- \"{starter}...\" ({count} cases)")
    lines.append("")
    
    lines.append("**CoT-Helpful Cases (S2 \\ S1) - Top Question Starters:**")
    lines.append("")
    for starter, count in helpful_starters.most_common(5):
        lines.append(f"- \"{starter}...\" ({count} cases)")
    lines.append("")
    
    return "\n".join(lines)


def generate_analysis_report(all_degradation_cases: Dict[str, List], all_cot_helpful_cases: Dict[str, List]) -> str:
    """Generate comprehensive analysis report."""
    md_lines = []
    
    md_lines.append("# Comprehensive Analysis: When CoT Helps vs Hurts")
    md_lines.append("")
    md_lines.append("## Overview")
    md_lines.append("")
    md_lines.append("This analysis compares:")
    md_lines.append("- **Degradation Cases (S1 \\ S2)**: Local ✓ but Local+CoT ✗")
    md_lines.append("- **CoT-Helpful Cases (S2 \\ S1)**: Local ✗ but Local+CoT ✓")
    md_lines.append("")
    
    total_degradation = sum(len(cases) for cases in all_degradation_cases.values())
    total_helpful = sum(len(cases) for cases in all_cot_helpful_cases.values())
    
    md_lines.append(f"**Total degradation cases**: {total_degradation}")
    md_lines.append(f"**Total CoT-helpful cases**: {total_helpful}")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # Aggregate statistics
    md_lines.append("## Aggregate Statistics")
    md_lines.append("")
    
    # Degradation cases stats
    md_lines.append("### Degradation Cases (S1 \\ S2) Characteristics")
    md_lines.append("")
    
    degradation_stats = aggregate_statistics(all_degradation_cases)
    md_lines.append(format_statistics(degradation_stats))
    md_lines.append("")
    
    # CoT-helpful cases stats
    md_lines.append("### CoT-Helpful Cases (S2 \\ S1) Characteristics")
    md_lines.append("")
    
    helpful_stats = aggregate_statistics(all_cot_helpful_cases)
    md_lines.append(format_statistics(helpful_stats))
    md_lines.append("")
    
    md_lines.append("---")
    md_lines.append("")
    
    # Comparison
    md_lines.append("## Key Differences: Degradation vs CoT-Helpful")
    md_lines.append("")
    
    comparisons = compare_statistics(degradation_stats, helpful_stats)
    md_lines.append(comparisons)
    md_lines.append("")
    
    md_lines.append("---")
    md_lines.append("")
    
    # Patterns
    md_lines.append("## Question Type Patterns")
    md_lines.append("")
    
    patterns = identify_patterns(all_degradation_cases, all_cot_helpful_cases)
    md_lines.append(patterns)
    md_lines.append("")
    
    md_lines.append("---")
    md_lines.append("")
    
    # Per-dataset breakdown
    md_lines.append("## Per-Dataset Breakdown")
    md_lines.append("")
    
    for config in DATASET_CONFIGS:
        name = config['display_name']
        deg_cases = all_degradation_cases.get(config['name'], [])
        help_cases = all_cot_helpful_cases.get(config['name'], [])
        
        md_lines.append(f"### {name}")
        md_lines.append("")
        md_lines.append(f"- Degradation cases: {len(deg_cases)}")
        md_lines.append(f"- CoT-helpful cases: {len(help_cases)}")
        
        if deg_cases:
            deg_stats = aggregate_statistics({config['name']: deg_cases})
            md_lines.append("")
            md_lines.append("**Degradation characteristics:**")
            md_lines.append(format_statistics(deg_stats))
        
        if help_cases:
            help_stats = aggregate_statistics({config['name']: help_cases})
            md_lines.append("")
            md_lines.append("**CoT-helpful characteristics:**")
            md_lines.append(format_statistics(help_stats))
        
        md_lines.append("")
        md_lines.append("---")
        md_lines.append("")
    
    # Recommendations
    md_lines.append("## Recommendations")
    md_lines.append("")
    md_lines.append("### Questions That Benefit from CoT:")
    md_lines.append("")
    
    # Analyze what makes CoT helpful
    if helpful_stats:
        md_lines.append("Based on analysis of CoT-helpful cases:")
        md_lines.append("")
        if helpful_stats.get('reasoning_pct', 0) > degradation_stats.get('reasoning_pct', 0):
            md_lines.append("1. **Complex reasoning questions** requiring multi-step analysis")
        if helpful_stats.get('scenario_pct', 0) > degradation_stats.get('scenario_pct', 0):
            md_lines.append("2. **Scenario-based questions** with multiple factors to consider")
        if helpful_stats.get('comparison_pct', 0) > degradation_stats.get('comparison_pct', 0):
            md_lines.append("3. **Comparison questions** asking for 'most appropriate' or 'best'")
        if helpful_stats.get('complexity_avg', 0) > degradation_stats.get('complexity_avg', 0):
            md_lines.append("4. **High complexity questions** with longer text and multiple factors")
        md_lines.append("")
    
    md_lines.append("### Questions Better Answered Locally:")
    md_lines.append("")
    
    # Analyze what makes CoT harmful
    if degradation_stats:
        md_lines.append("Based on analysis of degradation cases:")
        md_lines.append("")
        if degradation_stats.get('factual_pct', 0) > helpful_stats.get('factual_pct', 0):
            md_lines.append("1. **Simple factual recall** questions")
        if degradation_stats.get('question_length_avg', 0) < helpful_stats.get('question_length_avg', 0):
            md_lines.append("2. **Shorter, straightforward** questions")
        if degradation_stats.get('scenario_pct', 0) < helpful_stats.get('scenario_pct', 0):
            md_lines.append("3. **Questions without complex scenarios**")
        md_lines.append("")
    
    md_lines.append("---")
    md_lines.append("")
    
    # Detailed examples
    md_lines.append("## Detailed Examples")
    md_lines.append("")
    
    # Degradation examples
    md_lines.append("### Degradation Case Examples")
    md_lines.append("")
    for config in DATASET_CONFIGS:
        name = config['display_name']
        cases = all_degradation_cases.get(config['name'], [])
        if cases:
            md_lines.append(f"#### {name} ({len(cases)} cases)")
            md_lines.append("")
            for i, case in enumerate(cases[:2], 1):  # Show first 2
                md_lines.append(f"**Example {i}: Question {case['question_idx']}**")
                md_lines.append("")
                md_lines.append(f"Question: {case['question'][:200]}...")
                md_lines.append("")
                md_lines.append(f"Characteristics:")
                qa = case['question_analysis']
                md_lines.append(f"- Length: {qa['question_length']} words")
                md_lines.append(f"- Factual: {qa['is_factual']}")
                md_lines.append(f"- Requires reasoning: {qa['requires_reasoning']}")
                md_lines.append(f"- Has scenario: {qa['has_scenario']}")
                md_lines.append(f"- Complexity score: {qa['complexity_score']:.1f}")
                md_lines.append("")
                cot_a = case['cot_analysis']
                md_lines.append(f"CoT Quality:")
                md_lines.append(f"- Has direct answer hint: {cot_a.get('has_direct_answer', False)}")
                md_lines.append(f"- Answer hint strength: {cot_a.get('answer_hint_strength', 0)}")
                md_lines.append(f"- CoT preview: {case.get('cot_text', '')[:200]}...")
                md_lines.append("")
                md_lines.append("---")
                md_lines.append("")
    
    # CoT-helpful examples
    md_lines.append("### CoT-Helpful Case Examples")
    md_lines.append("")
    for config in DATASET_CONFIGS:
        name = config['display_name']
        cases = all_cot_helpful_cases.get(config['name'], [])
        if cases:
            md_lines.append(f"#### {name} ({len(cases)} cases)")
            md_lines.append("")
            for i, case in enumerate(cases[:2], 1):  # Show first 2
                md_lines.append(f"**Example {i}: Question {case['question_idx']}**")
                md_lines.append("")
                md_lines.append(f"Question: {case['question'][:200]}...")
                md_lines.append("")
                md_lines.append(f"Characteristics:")
                qa = case['question_analysis']
                md_lines.append(f"- Length: {qa['question_length']} words")
                md_lines.append(f"- Factual: {qa['is_factual']}")
                md_lines.append(f"- Requires reasoning: {qa['requires_reasoning']}")
                md_lines.append(f"- Has scenario: {qa['has_scenario']}")
                md_lines.append(f"- Complexity score: {qa['complexity_score']:.1f}")
                md_lines.append("")
                cot_a = case['cot_analysis']
                md_lines.append(f"CoT Quality:")
                md_lines.append(f"- Has direct answer hint: {cot_a.get('has_direct_answer', False)}")
                md_lines.append(f"- Answer hint strength: {cot_a.get('answer_hint_strength', 0)}")
                md_lines.append(f"- CoT preview: {case.get('cot_text', '')[:200]}...")
                md_lines.append("")
                md_lines.append("---")
                md_lines.append("")
    
    return "\n".join(md_lines)


def main():
    """Main function."""
    print("="*80)
    print("Comprehensive Analysis: When CoT Helps vs Hurts")
    print("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    all_degradation_cases = {}
    all_cot_helpful_cases = {}
    
    for config in DATASET_CONFIGS:
        print(f"\nProcessing {config['display_name']}...")
        
        # Degradation cases
        degradation = extract_cases(conn, config, 'degradation')
        all_degradation_cases[config['name']] = degradation
        print(f"  Degradation cases: {len(degradation)}")
        
        # CoT-helpful cases
        helpful = extract_cases(conn, config, 'helpful')
        all_cot_helpful_cases[config['name']] = helpful
        print(f"  CoT-helpful cases: {len(helpful)}")
    
    conn.close()
    
    # Generate report
    print("\nGenerating analysis report...")
    report = generate_analysis_report(all_degradation_cases, all_cot_helpful_cases)
    
    # Save report
    output_file = os.path.join(OUTPUT_DIR, "cot_benefit_harm_analysis.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nSaved analysis report to: {output_file}")
    
    # Print summary
    total_degradation = sum(len(cases) for cases in all_degradation_cases.values())
    total_helpful = sum(len(cases) for cases in all_cot_helpful_cases.values())
    
    print(f"\nSummary:")
    print(f"  Degradation cases: {total_degradation}")
    print(f"  CoT-helpful cases: {total_helpful}")
    
    # Print key insights
    if all_degradation_cases and all_cot_helpful_cases:
        deg_stats = aggregate_statistics(all_degradation_cases)
        help_stats = aggregate_statistics(all_cot_helpful_cases)
        
        print(f"\nKey Insights:")
        print(f"  Degradation cases avg complexity: {deg_stats.get('complexity_avg', 0):.1f}")
        print(f"  CoT-helpful cases avg complexity: {help_stats.get('complexity_avg', 0):.1f}")
        print(f"  Degradation cases reasoning %: {deg_stats.get('reasoning_pct', 0):.1f}%")
        print(f"  CoT-helpful cases reasoning %: {help_stats.get('reasoning_pct', 0):.1f}%")


if __name__ == "__main__":
    main()

