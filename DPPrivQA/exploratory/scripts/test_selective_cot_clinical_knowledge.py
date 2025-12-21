#!/usr/bin/env python3
"""
Test script for Local + Selective CoT scenario
Tests on Clinical Knowledge dataset (lowest CoT boost: +5.7%)
Exploratory experiment - logs only, no DB writes
"""

import argparse
import sys
import os
import sqlite3
from typing import Dict, Any
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dpprivqa.datasets import MMLUDataset
from dpprivqa.qa.scenarios import run_local_with_selective_cot
from dpprivqa.qa.models import get_nebius_client, get_remote_llm_client
from dpprivqa.utils.logging import setup_logging
from dpprivqa.utils.config import load_config
from dpprivqa.qa.prompts import check_mcq_correctness


def get_baseline_results(db_path: str) -> Dict[str, Any]:
    """
    Query baseline results from database.
    
    Args:
        db_path: Path to database file
    
    Returns:
        Dictionary with baseline accuracies and statistics
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    results = {}
    
    # Query local baseline (experiment_id=9, scenario='local')
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(is_correct) as correct
        FROM mmlu_clinical_knowledge_epsilon_independent_results
        WHERE experiment_id = 9 AND scenario = 'local'
    """)
    row = cursor.fetchone()
    if row and row['total']:
        local_accuracy = row['correct'] / row['total'] if row['total'] > 0 else 0.0
        results['local'] = {
            'accuracy': local_accuracy,
            'total': row['total'],
            'correct': row['correct']
        }
    else:
        results['local'] = {'accuracy': 0.732, 'total': 265, 'correct': 194}  # Fallback to known values
    
    # Query local+CoT baseline (experiment_id=23, scenario='local_cot')
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(is_correct) as correct
        FROM mmlu_clinical_knowledge_epsilon_independent_results
        WHERE experiment_id = 23 AND scenario = 'local_cot'
    """)
    row = cursor.fetchone()
    if row and row['total']:
        cot_accuracy = row['correct'] / row['total'] if row['total'] > 0 else 0.0
        results['local_cot'] = {
            'accuracy': cot_accuracy,
            'total': row['total'],
            'correct': row['correct']
        }
    else:
        results['local_cot'] = {'accuracy': 0.789, 'total': 265, 'correct': 209}  # Fallback to known values
    
    conn.close()
    return results


def main():
    parser = argparse.ArgumentParser(description="Test Local + Selective CoT on Clinical Knowledge")
    parser.add_argument("--num-questions", type=int, default=None, help="Number of questions to test (None = all)")
    parser.add_argument("--start-index", type=int, default=0, help="Starting question index")
    
    args = parser.parse_args()
    
    config = load_config()
    dataset_name = "mmlu_clinical_knowledge"
    subset = "clinical_knowledge"
    
    # Setup logging
    logger = setup_logging(f"{dataset_name}_selective_cot_exploratory")
    logger.info("="*80)
    logger.info("Local + Selective CoT (Exploratory)")
    logger.info("Dataset: Clinical Knowledge (lowest CoT boost: +5.7%)")
    logger.info("="*80)
    
    # Query baseline results from database
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    logger.info(f"Querying baseline results from database: {db_path}")
    baseline_results = get_baseline_results(db_path)
    
    logger.info("\nBaseline Results (from database):")
    logger.info(f"  Local only: {baseline_results['local']['accuracy']:.3f} ({baseline_results['local']['accuracy']*100:.1f}%)")
    logger.info(f"    - Total: {baseline_results['local']['total']}, Correct: {baseline_results['local']['correct']}")
    logger.info(f"  Local + CoT: {baseline_results['local_cot']['accuracy']:.3f} ({baseline_results['local_cot']['accuracy']*100:.1f}%)")
    logger.info(f"    - Total: {baseline_results['local_cot']['total']}, Correct: {baseline_results['local_cot']['correct']}")
    
    if args.num_questions:
        logger.info(f"\nTesting {args.num_questions} questions starting from index {args.start_index}")
    else:
        logger.info(f"\nTesting all questions starting from index {args.start_index}")
    
    # Load dataset
    logger.info(f"Loading MMLU {subset} dataset...")
    dataset = MMLUDataset(subset=subset)
    questions = dataset.load(split="test")
    
    if args.num_questions:
        end_index = min(args.start_index + args.num_questions, len(questions))
        questions = questions[args.start_index:end_index]
    else:
        questions = questions[args.start_index:]
    
    logger.info(f"Loaded {len(questions)} questions (indices {args.start_index} to {args.start_index + len(questions) - 1})")
    
    # Initialize clients
    logger.info("Initializing Nebius client...")
    try:
        local_client = get_nebius_client()
        logger.info("Nebius client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Nebius client: {e}")
        return
    
    logger.info("Initializing OpenAI client...")
    try:
        remote_client = get_remote_llm_client()
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return
    
    local_model = config.get("models", {}).get("local", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    remote_cot_model = config.get("models", {}).get("remote_cot", "gpt-5")
    logger.info(f"Using local model: {local_model}")
    logger.info(f"Using remote CoT model: {remote_cot_model}")
    logger.info("NOTE: This is an exploratory experiment - results logged only, not saved to DB")
    
    logger.info("\n" + "="*80)
    logger.info("Running Local + Selective CoT scenario")
    logger.info("="*80)
    
    correct_count = 0
    cot_used_count = 0
    local_only_count = 0
    decision_reasons = Counter()
    error_count = 0
    
    for idx, q in enumerate(questions):
        question = q['question']
        options = q['options']
        ground_truth = q['answer_idx']
        
        logger.info(f"\nQuestion {idx+1}/{len(questions)}:")
        logger.info(f"  Question: {question[:150]}...")
        logger.info(f"  Ground truth: {ground_truth}")
        
        try:
            result = run_local_with_selective_cot(
                local_client, remote_client, local_model, remote_cot_model,
                question, options,
                dataset_name='clinical_knowledge'  # Use dataset-specific rules
            )
            
            predicted = result['answer']
            cot_used = result.get('cot_used', False)
            cot_reason = result.get('cot_reason', 'unknown')
            
            # Track decision
            if cot_used:
                cot_used_count += 1
            else:
                local_only_count += 1
            decision_reasons[cot_reason] += 1
            
            is_correct = check_mcq_correctness(predicted, ground_truth)
            if is_correct:
                correct_count += 1
            
            logger.info(f"  Decision: {'USE CoT' if cot_used else 'Local Only'} (reason: {cot_reason})")
            logger.info(f"  Predicted: {predicted}")
            logger.info(f"  Correct: {is_correct}")
            logger.info(f"  Processing time: {result['processing_time']:.2f}s")
            if cot_used:
                logger.info(f"  CoT generation time: {result.get('cot_generation_time', 0):.2f}s")
                logger.info(f"  CoT preview: {result.get('cot_text', '')[:100]}...")
            
        except Exception as e:
            error_count += 1
            logger.error(f"  Error processing question: {e}")
            continue
        
        # Progress update every 10 questions
        if (idx + 1) % 10 == 0:
            current_accuracy = correct_count / (idx + 1)
            logger.info(f"\n  Progress: {idx+1}/{len(questions)} questions")
            logger.info(f"  Current accuracy: {current_accuracy:.3f} ({current_accuracy*100:.1f}%)")
            logger.info(f"  CoT usage: {cot_used_count}/{idx+1} ({cot_used_count/(idx+1)*100:.1f}%)")
    
    # Final summary
    accuracy = correct_count / len(questions) if questions else 0
    cot_usage_rate = cot_used_count / len(questions) if questions else 0
    
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total questions: {len(questions)}")
    logger.info(f"Correct answers: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    logger.info(f"Errors: {error_count}")
    
    logger.info("\nDecision Breakdown:")
    logger.info(f"  CoT used: {cot_used_count} ({cot_usage_rate*100:.1f}%)")
    logger.info(f"  Local only: {local_only_count} ({(1-cot_usage_rate)*100:.1f}%)")
    logger.info("\nDecision Reasons:")
    for reason, count in decision_reasons.most_common():
        pct = count / len(questions) * 100 if questions else 0
        logger.info(f"  {reason}: {count} ({pct:.1f}%)")
    
    logger.info("\nComparison with Baseline:")
    logger.info(f"  Baseline Local (S1): {baseline_results['local']['accuracy']*100:.1f}% accuracy")
    logger.info(f"  Baseline Local+CoT (S2): {baseline_results['local_cot']['accuracy']*100:.1f}% accuracy")
    logger.info(f"  Selective CoT (this test): {accuracy*100:.1f}% accuracy")
    
    improvement_over_local = (accuracy - baseline_results['local']['accuracy']) * 100
    improvement_over_cot = (accuracy - baseline_results['local_cot']['accuracy']) * 100
    
    logger.info(f"\nImprovement:")
    logger.info(f"  Over Local: {improvement_over_local:+.1f} percentage points")
    logger.info(f"  Over Local+CoT: {improvement_over_cot:+.1f} percentage points")
    
    if accuracy >= baseline_results['local_cot']['accuracy']:
        logger.info("\n✓ SUCCESS: Selective CoT matches or exceeds baseline Local+CoT accuracy!")
    else:
        logger.info(f"\n⚠ Selective CoT accuracy is {baseline_results['local_cot']['accuracy'] - accuracy:.3f} lower than baseline Local+CoT")
    
    logger.info(f"\nCost Savings:")
    logger.info(f"  Baseline CoT usage: 100% (always uses CoT)")
    logger.info(f"  Selective CoT usage: {cot_usage_rate*100:.1f}%")
    logger.info(f"  CoT calls saved: {(1-cot_usage_rate)*100:.1f}%")


if __name__ == "__main__":
    main()

