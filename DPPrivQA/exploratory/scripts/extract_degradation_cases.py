#!/usr/bin/env python3
r"""
Extract Degradation Cases: S1 \ S2 (Local ✓ but Local+CoT ✗)
================================================================

Finds and formats questions where Local model got correct but Local+CoT failed.
These are cases where CoT guidance backfired.
"""

import sqlite3
import json
import os
import sys
from typing import List, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Paths
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


def extract_degradation_cases(conn: sqlite3.Connection, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract cases where Local worked but Local+CoT failed."""
    table = config['table']
    
    query = f"""
        SELECT 
            local_res.question_idx,
            local_res.original_question as question,
            local_res.options,
            local_res.generated_answer as local_answer,
            local_res.generated_answer as local_answer_text,
            cot_res.generated_answer as cot_answer,
            cot_res.generated_answer as cot_answer_text,
            cot_res.cot_text,
            local_res.ground_truth_answer as ground_truth,
            local_res.processing_time as local_time,
            cot_res.processing_time as cot_time,
            cot_res.processing_time as cot_generation_time
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
    
    cursor = conn.execute(query, (config['local_exp_id'], config['cot_exp_id']))
    rows = cursor.fetchall()
    
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in rows]


def format_options(options_str: str) -> str:
    """Format options dictionary as markdown."""
    if isinstance(options_str, str):
        try:
            options = json.loads(options_str)
        except:
            return options_str
    else:
        options = options_str
    
    if isinstance(options, dict):
        return "\n".join([f"- **{k}**: {v}" for k, v in sorted(options.items())])
    return str(options)


def analyze_cot_quality(cot_text: str) -> Dict[str, Any]:
    """Analyze CoT quality characteristics."""
    if not cot_text or cot_text.startswith("Error"):
        return {
            "is_error": True,
            "length": 0,
            "word_count": 0,
            "has_reasoning_steps": False,
            "preview": "Error or empty CoT"
        }
    
    words = cot_text.split()
    has_steps = any(marker in cot_text.lower() for marker in 
                   ['step', 'first', 'second', 'then', 'therefore', 'because', 'reasoning', 'analyze'])
    
    return {
        "is_error": False,
        "length": len(cot_text),
        "word_count": len(words),
        "has_reasoning_steps": has_steps,
        "preview": cot_text[:300] + "..." if len(cot_text) > 300 else cot_text
    }


def generate_markdown(all_cases: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate markdown report."""
    md_lines = []
    
    md_lines.append("# Degradation Cases Analysis: S1 \\ S2")
    md_lines.append("")
    md_lines.append("## Overview")
    md_lines.append("")
    md_lines.append("This document contains questions where the **Local model (S1) got correct** but **Local+CoT (S2) failed**.")
    md_lines.append("These are cases where Chain-of-Thought guidance backfired.")
    md_lines.append("")
    
    total_cases = sum(len(cases) for cases in all_cases.values())
    md_lines.append(f"**Total degradation cases across all datasets: {total_cases}**")
    md_lines.append("")
    
    # Summary table
    md_lines.append("### Summary by Dataset")
    md_lines.append("")
    md_lines.append("| Dataset | Degradation Cases | Percentage |")
    md_lines.append("|---------|------------------|------------|")
    
    for config in DATASET_CONFIGS:
        name = config['display_name']
        cases = all_cases.get(config['name'], [])
        count = len(cases)
        total_q = config.get('total_questions', 0)
        pct = (count / total_q * 100) if total_q > 0 else 0
        md_lines.append(f"| {name} | {count} | {pct:.1f}% |")
    
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    
    # Detailed cases by dataset
    for config in DATASET_CONFIGS:
        name = config['display_name']
        cases = all_cases.get(config['name'], [])
        
        if not cases:
            continue
        
        md_lines.append(f"## {name}")
        md_lines.append("")
        md_lines.append(f"**Total degradation cases: {len(cases)}**")
        md_lines.append("")
        
        for idx, case in enumerate(cases, 1):
            md_lines.append(f"### Case {idx}: Question {case['question_idx']}")
            md_lines.append("")
            
            # Question
            md_lines.append("#### Question")
            md_lines.append("")
            md_lines.append(case['question'])
            md_lines.append("")
            
            # Options
            md_lines.append("#### Options")
            md_lines.append("")
            md_lines.append(format_options(case['options']))
            md_lines.append("")
            
            # Ground truth
            md_lines.append(f"#### Ground Truth: **{case['ground_truth']}**")
            md_lines.append("")
            
            # Local result (correct)
            md_lines.append("#### Local Model Result (✓ Correct)")
            md_lines.append("")
            md_lines.append(f"- **Answer**: {case['local_answer']}")
            md_lines.append(f"- **Answer Text**: {case['local_answer_text']}")
            md_lines.append(f"- **Processing Time**: {case['local_time']:.2f}s")
            md_lines.append("")
            
            # Local+CoT result (incorrect)
            md_lines.append("#### Local+CoT Result (✗ Incorrect)")
            md_lines.append("")
            md_lines.append(f"- **Answer**: {case['cot_answer']}")
            md_lines.append(f"- **Answer Text**: {case['cot_answer_text']}")
            md_lines.append(f"- **Processing Time**: {case['cot_time']:.2f}s")
            md_lines.append(f"- **CoT Generation Time**: {case.get('cot_generation_time', 0):.2f}s")
            md_lines.append("")
            
            # CoT analysis
            cot_analysis = analyze_cot_quality(case.get('cot_text', ''))
            md_lines.append("#### Chain-of-Thought Analysis")
            md_lines.append("")
            if cot_analysis['is_error']:
                md_lines.append("- **Status**: ❌ Error or Empty")
            else:
                md_lines.append("- **Status**: ✓ Generated")
                md_lines.append(f"- **Length**: {cot_analysis['length']} characters")
                md_lines.append(f"- **Word Count**: {cot_analysis['word_count']} words")
                md_lines.append(f"- **Has Reasoning Steps**: {'✓ Yes' if cot_analysis['has_reasoning_steps'] else '✗ No'}")
            md_lines.append("")
            
            md_lines.append("#### CoT Text")
            md_lines.append("")
            md_lines.append("```")
            md_lines.append(cot_analysis['preview'])
            md_lines.append("```")
            md_lines.append("")
            
            md_lines.append("---")
            md_lines.append("")
    
    # Analysis section
    md_lines.append("## Analysis Notes")
    md_lines.append("")
    md_lines.append("### Patterns to Investigate")
    md_lines.append("")
    md_lines.append("1. **Question Types**: What types of questions don't benefit from CoT?")
    md_lines.append("2. **CoT Quality**: Are there patterns in the CoT text that correlate with failure?")
    md_lines.append("3. **Question Complexity**: Are simpler questions more likely to be degraded by CoT?")
    md_lines.append("4. **Domain-Specific**: Are certain domains more prone to degradation?")
    md_lines.append("")
    
    return "\n".join(md_lines)


def main():
    """Main function."""
    print("="*80)
    print("Extracting Degradation Cases: S1 \\ S2")
    print("="*80)
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    all_cases = {}
    
    for config in DATASET_CONFIGS:
        print(f"\nProcessing {config['display_name']}...")
        cases = extract_degradation_cases(conn, config)
        all_cases[config['name']] = cases
        print(f"  Found {len(cases)} degradation cases")
    
    conn.close()
    
    # Generate markdown
    print("\nGenerating markdown report...")
    md_content = generate_markdown(all_cases)
    
    # Save to file
    output_file = os.path.join(OUTPUT_DIR, "degradation_cases_s1_minus_s2.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"\nSaved degradation cases report to: {output_file}")
    
    # Print summary
    total = sum(len(cases) for cases in all_cases.values())
    print(f"\nTotal degradation cases: {total}")
    for config in DATASET_CONFIGS:
        count = len(all_cases.get(config['name'], []))
        print(f"  {config['display_name']}: {count}")


if __name__ == "__main__":
    main()

