#!/usr/bin/env python3
"""
Test script for Local + CoT with Option 1 Prompt Optimization
Tests on Clinical Knowledge dataset (lowest CoT boost: +5.7%)
Exploratory experiment - logs only, no DB writes
"""

import argparse
import sys
import os
import time
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dpprivqa.datasets import MMLUDataset
from dpprivqa.qa.models import get_nebius_client, get_remote_llm_client, create_completion_with_model_support, find_working_nebius_model
from dpprivqa.utils.logging import setup_logging
from dpprivqa.utils.config import load_config
from dpprivqa.qa.prompts import check_mcq_correctness, format_question_with_options


def run_local_with_cot_option1(
    local_client,
    remote_client,
    local_model_name: str,
    remote_model_name: str,
    question: str,
    options: Dict[str, str],
    max_tokens: int = 256,
    cot_max_tokens: int = 512
) -> Dict[str, Any]:
    """
    Run local + CoT scenario with Option 1 prompt optimization.
    Uses "concise" format that works with GPT-5's behavior.
    
    This is a workaround that doesn't modify existing code.
    """
    start_time = time.time()
    
    # Step 1: Generate CoT using Option 1 prompt (embrace "concise")
    cot_start = time.time()
    
    # Option 1: Embrace the "Concise" Format
    prompt_lines = [
        "Here is the question:",
        question,
        "",
        "Analyze this question and provide key insights that would help solve it:",
        "1. What concepts, principles, or knowledge are relevant?",
        "2. What factors should be considered?",
        "3. What is the analytical approach or framework?",
        "",
        "Provide analytical guidance only - do NOT give the final answer."
    ]
    cot_prompt = "\n".join(prompt_lines)
    
    try:
        cot_response = create_completion_with_model_support(
            remote_client, remote_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert consultant. Provide clear, structured analytical guidance to help solve domain-specific questions. Focus on relevant principles, key considerations, and logical approaches."
                },
                {"role": "user", "content": cot_prompt}
            ],
            max_tokens=cot_max_tokens,
            temperature=0.0
        )
        cot_text = cot_response.choices[0].message.content.strip()
    except Exception as e:
        cot_text = f"Error: {e}"
    
    cot_time = time.time() - cot_start
    
    # Step 2: Get answer from local model with CoT (same as original)
    formatted_question = format_question_with_options(question, options)
    full_prompt = f"{formatted_question}\n\nAnalytical Guidance:\n{cot_text}\n\nBased on the analytical guidance above, what is the correct answer? Provide only the letter (A, B, C, or D):"
    
    try:
        local_model = find_working_nebius_model(local_client, local_model_name)
        response = create_completion_with_model_support(
            local_client, local_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert. Use the provided analytical guidance to answer the multiple choice question. Provide only the letter (A, B, C, or D) of the correct option."
                },
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        answer_text = response.choices[0].message.content.strip()
        predicted = answer_text[0] if answer_text else "Error"
    except Exception as e:
        predicted = "Error"
        answer_text = f"Error: {e}"
    
    processing_time = time.time() - start_time
    
    return {
        "answer": predicted,
        "answer_text": answer_text,
        "cot_text": cot_text,
        "question": question,
        "options": options,
        "processing_time": processing_time,
        "cot_generation_time": cot_time,
        "local_model": local_model_name,
        "remote_model": remote_model_name,
        "scenario": "local_cot_option1"
    }


def main():
    parser = argparse.ArgumentParser(description="Test Local + CoT with Option 1 prompt optimization on Clinical Knowledge")
    parser.add_argument("--num-questions", type=int, default=None, help="Number of questions to test (None = all)")
    parser.add_argument("--start-index", type=int, default=0, help="Starting question index")
    
    args = parser.parse_args()
    
    config = load_config()
    dataset_name = "mmlu_clinical_knowledge"
    subset = "clinical_knowledge"
    
    # Setup logging with special name for this experiment
    logger = setup_logging(f"{dataset_name}_option1_exploratory")
    logger.info("="*80)
    logger.info("Local + CoT with Option 1 Prompt Optimization (Exploratory)")
    logger.info("Dataset: Clinical Knowledge (lowest CoT boost: +5.7%)")
    logger.info("="*80)
    
    if args.num_questions:
        logger.info(f"Testing {args.num_questions} questions starting from index {args.start_index}")
    else:
        logger.info(f"Testing all questions starting from index {args.start_index}")
    
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
    logger.info("Using Option 1 prompt: Embrace 'concise' format")
    logger.info("NOTE: This is an exploratory experiment - results logged only, not saved to DB")
    
    logger.info("\n" + "="*80)
    logger.info("Running Local + CoT (Option 1) scenario")
    logger.info("="*80)
    
    correct_count = 0
    refusal_count = 0
    error_count = 0
    
    for idx, q in enumerate(questions):
        question = q['question']
        options = q['options']
        ground_truth = q['answer_idx']
        
        logger.info(f"\nQuestion {idx+1}/{len(questions)}:")
        logger.info(f"  Question: {question[:150]}...")
        logger.info(f"  Ground truth: {ground_truth}")
        
        try:
            result = run_local_with_cot_option1(
                local_client, remote_client, local_model, remote_cot_model, 
                question, options
            )
            
            # Check for refusal pattern
            cot_text = result.get('cot_text', '')
            if cot_text and (cot_text.startswith("I can't") or cot_text.startswith("Sorry") or 
                            "can't share" in cot_text.lower() or "cannot share" in cot_text.lower()):
                refusal_count += 1
            
            is_correct = check_mcq_correctness(result['answer'], ground_truth)
            if is_correct:
                correct_count += 1
            
            if result['answer'] == "Error":
                error_count += 1
            
            logger.info(f"  Predicted answer: {result['answer']}")
            logger.info(f"  Correct: {is_correct}")
            logger.info(f"  CoT generation time: {result.get('cot_generation_time', 0):.2f}s")
            logger.info(f"  Total processing time: {result.get('processing_time', 0):.2f}s")
            logger.info(f"  CoT preview: {cot_text[:150]}...")
            if cot_text.startswith("I can't") or cot_text.startswith("Sorry") or "can't share" in cot_text.lower():
                logger.info(f"  ⚠️  Refusal pattern detected")
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing question {idx+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    accuracy = correct_count / len(questions) if questions else 0
    refusal_rate = refusal_count / len(questions) if questions else 0
    error_rate = error_count / len(questions) if questions else 0
    
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total questions: {len(questions)}")
    logger.info(f"Correct answers: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    logger.info(f"Refusal pattern count: {refusal_count} ({refusal_rate*100:.1f}%)")
    logger.info(f"Error count: {error_count} ({error_rate*100:.1f}%)")
    logger.info(f"\nComparison with baseline:")
    logger.info(f"  Baseline Local (S1): 73.2% accuracy")
    logger.info(f"  Baseline Local+CoT (S2, original prompt): 78.9% accuracy")
    logger.info(f"  Option 1 Local+CoT (this test): {accuracy*100:.1f}% accuracy")
    logger.info(f"  Improvement over Local: {accuracy*100 - 73.2:.1f} percentage points")
    logger.info(f"  Improvement over baseline CoT: {accuracy*100 - 78.9:.1f} percentage points")
    logger.info(f"\nRefusal pattern comparison:")
    logger.info(f"  Baseline CoT refusal rate: ~94%")
    logger.info(f"  Option 1 refusal rate: {refusal_rate*100:.1f}%")
    logger.info(f"  Reduction: {94.0 - refusal_rate*100:.1f} percentage points")
    logger.info("\n" + "="*80)
    logger.info("Experiment complete. Results logged to file.")


if __name__ == "__main__":
    main()


