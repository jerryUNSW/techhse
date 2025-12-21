#!/usr/bin/env python3
"""
Test script for Remote model (GPT-5 via OpenAI API) only on MMLU datasets.
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpprivqa.datasets import MMLUDataset
from dpprivqa.qa.scenarios import run_remote_only
from dpprivqa.qa.models import get_remote_llm_client
from dpprivqa.database.writer import ExperimentDBWriter
from dpprivqa.utils.logging import setup_logging
from dpprivqa.utils.config import load_config
from dpprivqa.qa.prompts import check_mcq_correctness


def main():
    parser = argparse.ArgumentParser(description="Test Remote model (GPT-5 via Lightning) on MMLU dataset")
    parser.add_argument("--subset", type=str, default="professional_law", 
                       choices=["professional_law", "professional_medicine", "clinical_knowledge", "college_medicine"],
                       help="MMLU subset to test")
    parser.add_argument("--num-questions", type=int, default=10, help="Number of questions to test")
    parser.add_argument("--start-index", type=int, default=0, help="Starting question index")
    
    args = parser.parse_args()
    
    config = load_config()
    
    # Map subset to dataset name
    dataset_name_map = {
        "professional_law": "mmlu_professional_law",
        "professional_medicine": "mmlu_professional_medicine",
        "clinical_knowledge": "mmlu_clinical_knowledge",
        "college_medicine": "mmlu_college_medicine"
    }
    dataset_name = dataset_name_map[args.subset]
    
    logger = setup_logging(dataset_name)
    logger.info(f"Starting Remote model (GPT-5 via OpenAI API) test on MMLU {args.subset}")
    logger.info(f"Testing {args.num_questions} questions")
    
    # Initialize database
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    db_writer = ExperimentDBWriter(db_path)
    logger.info(f"Database: {db_path}")
    
    # Load dataset
    logger.info(f"Loading MMLU {args.subset} dataset...")
    dataset = MMLUDataset(subset=args.subset)
    questions = dataset.load(split="test")
    
    # Limit number of questions
    end_index = min(args.start_index + args.num_questions, len(questions))
    questions = questions[args.start_index:end_index]
    
    logger.info(f"Loaded {len(questions)} questions (indices {args.start_index} to {end_index-1})")
    
    # Initialize OpenAI client
    logger.info("Initializing OpenAI client...")
    try:
        remote_client = get_remote_llm_client()
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        return
    
    remote_model = config.get("models", {}).get("remote_qa", "gpt-5")
    logger.info(f"Using remote model: {remote_model}")
    
    # Create experiment record
    ei_experiment_id = db_writer.create_experiment(
        dataset_name=dataset_name,
        experiment_type="epsilon_independent",
        total_questions=len(questions),
        remote_qa_model=remote_model,
        description=f"Remote model (GPT-5 via OpenAI API) test on MMLU {args.subset} ({len(questions)} questions)"
    )
    logger.info(f"Created experiment record: ID {ei_experiment_id}")
    
    # Run Remote scenario
    logger.info("\n" + "="*80)
    logger.info("Running Remote scenario (GPT-5 via OpenAI API)")
    logger.info("="*80)
    
    correct_count = 0
    
    for idx, q in enumerate(questions):
        question = q['question']
        options = q['options']
        ground_truth = q['answer_idx']
        
        logger.info(f"\nQuestion {idx+1}/{len(questions)}:")
        logger.info(f"  Question: {question[:150]}...")
        logger.info(f"  Options: {options}")
        logger.info(f"  Ground truth: {ground_truth}")
        
        try:
            result = run_remote_only(remote_client, remote_model, question, options)
            
            is_correct = check_mcq_correctness(result['answer'], ground_truth)
            if is_correct:
                correct_count += 1
            
            logger.info(f"  Predicted answer: {result['answer']}")
            logger.info(f"  Correct: {is_correct}")
            logger.info(f"  Processing time: {result.get('processing_time', 0):.2f}s")
            
            # Write to database
            db_result = {
                'question': question,
                'options': options,
                'answer': result['answer'],
                'ground_truth': ground_truth,
                'is_correct': is_correct,
                'processing_time': result.get('processing_time'),
                'remote_model': result.get('remote_model', remote_model)
            }
            
            db_writer.write_epsilon_independent_result(
                dataset_name, ei_experiment_id, idx, 'remote', db_result
            )
            
        except Exception as e:
            logger.error(f"Error processing question {idx+1}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # Summary
    accuracy = correct_count / len(questions) if questions else 0
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total questions: {len(questions)}")
    logger.info(f"Correct answers: {correct_count}")
    logger.info(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    db_writer.close()
    logger.info("\nResults saved to database successfully!")
    logger.info(f"Experiment ID: {ei_experiment_id}")


if __name__ == "__main__":
    main()

