#!/usr/bin/env python3
"""
MedQA-USMLE Experiment Script

Runs QA tests on MedQA-USMLE dataset with two phases:
1. Epsilon-independent tests (Local, Local+CoT, Remote)
2. Epsilon-dependent tests (DPPrivQA with epsilon 1.0, 2.0, 3.0)
"""

import argparse
import time
from sentence_transformers import SentenceTransformer

from dpprivqa.datasets import MedQADataset
from dpprivqa.qa.scenarios import run_local_only, run_local_with_cot, run_remote_only
from dpprivqa.qa.dpprivqa import run_dpprivqa
from dpprivqa.qa.models import get_nebius_client, get_remote_llm_client
from dpprivqa.database.writer import ExperimentDBWriter
from dpprivqa.utils.logging import setup_logging
from dpprivqa.utils.config import load_config
from dpprivqa.qa.prompts import check_mcq_correctness


def main():
    parser = argparse.ArgumentParser(description="Run MedQA-USMLE experiments")
    parser.add_argument("--num-questions", type=int, default=None, help="Number of questions to test (default: all)")
    parser.add_argument("--start-index", type=int, default=0, help="Starting question index")
    parser.add_argument("--epsilon-values", type=float, nargs="+", default=[1.0, 2.0, 3.0], help="Epsilon values to test")
    parser.add_argument("--use-phrasedp-plus", action="store_true", help="Use PhraseDP+ instead of PhraseDP")
    parser.add_argument("--skip-epsilon-independent", action="store_true", help="Skip epsilon-independent tests")
    parser.add_argument("--skip-epsilon-dependent", action="store_true", help="Skip epsilon-dependent tests")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    dataset_name = "medqa"
    
    # Setup logging
    logger = setup_logging(dataset_name)
    logger.info("Starting MedQA-USMLE experiment")
    logger.info(f"Configuration: {config}")
    
    # Initialize database writer
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    db_writer = ExperimentDBWriter(db_path)
    logger.info(f"Database: {db_path}")
    
    # Load dataset
    logger.info("Loading MedQA-USMLE dataset...")
    dataset = MedQADataset()
    questions = dataset.load(split="test")
    
    # Limit number of questions if specified
    if args.num_questions:
        end_index = min(args.start_index + args.num_questions, len(questions))
        questions = questions[args.start_index:end_index]
    else:
        questions = questions[args.start_index:]
    
    logger.info(f"Testing {len(questions)} questions (starting from index {args.start_index})")
    
    # Initialize clients
    logger.info("Initializing LLM clients...")
    local_client = get_nebius_client()
    remote_client = get_remote_llm_client()
    
    local_model = config.get("models", {}).get("local", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    remote_cot_model = config.get("models", {}).get("remote_cot", "gpt-5")
    remote_qa_model = config.get("models", {}).get("remote_qa", "gpt-5")
    
    logger.info(f"Local model: {local_model}")
    logger.info(f"Remote CoT model: {remote_cot_model}")
    logger.info(f"Remote QA model: {remote_qa_model}")
    
    # Load Sentence-BERT for PhraseDP
    logger.info("Loading Sentence-BERT model...")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # ===== PHASE 1: Epsilon-Independent Tests =====
    if not args.skip_epsilon_independent:
        logger.info("=" * 80)
        logger.info("PHASE 1: Epsilon-Independent Tests")
        logger.info("=" * 80)
        
        ei_experiment_id = db_writer.create_experiment(
            dataset_name=dataset_name,
            experiment_type="epsilon_independent",
            total_questions=len(questions),
            local_model=local_model,
            remote_cot_model=remote_cot_model,
            remote_qa_model=remote_qa_model,
            description=f"MedQA-USMLE epsilon-independent tests ({len(questions)} questions)"
        )
        logger.info(f"Created experiment record: ID {ei_experiment_id}")
        
        scenarios = ['local', 'local_cot', 'remote']
        
        for scenario in scenarios:
            logger.info(f"\nRunning scenario: {scenario}")
            correct_count = 0
            
            for idx, question_data in enumerate(questions):
                question = question_data['question']
                options = question_data['options']
                ground_truth = question_data['answer_idx']
                
                logger.info(f"Question {idx+1}/{len(questions)}: {question[:100]}...")
                
                try:
                    if scenario == 'local':
                        result = run_local_only(
                            local_client, local_model, question, options
                        )
                    elif scenario == 'local_cot':
                        result = run_local_with_cot(
                            local_client, remote_client,
                            local_model, remote_cot_model,
                            question, options
                        )
                    elif scenario == 'remote':
                        result = run_remote_only(
                            remote_client, remote_qa_model, question, options
                        )
                    
                    # Check correctness
                    is_correct = check_mcq_correctness(result['answer'], ground_truth)
                    if is_correct:
                        correct_count += 1
                    
                    # Write to database
                    db_result = {
                        'question': question,
                        'options': options,
                        'answer': result['answer'],
                        'ground_truth': ground_truth,
                        'is_correct': is_correct,
                        'processing_time': result.get('processing_time'),
                        'cot_text': result.get('cot_text'),
                        'local_model': result.get('local_model'),
                        'remote_model': result.get('remote_model')
                    }
                    
                    db_writer.write_epsilon_independent_result(
                        dataset_name, ei_experiment_id, idx, scenario, db_result
                    )
                    
                    logger.info(f"  Answer: {result['answer']}, Ground truth: {ground_truth}, Correct: {is_correct}")
                
                except Exception as e:
                    logger.error(f"Error processing question {idx+1}: {e}")
                    continue
            
            accuracy = correct_count / len(questions) if questions else 0
            logger.info(f"\nScenario {scenario} accuracy: {correct_count}/{len(questions)} = {accuracy:.3f}")
    
    # ===== PHASE 2: Epsilon-Dependent Tests =====
    if not args.skip_epsilon_dependent:
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: Epsilon-Dependent Tests (DPPrivQA)")
        logger.info("=" * 80)
        
        mechanism = 'phrasedp_plus' if args.use_phrasedp_plus else 'phrasedp'
        logger.info(f"Using mechanism: {mechanism}")
        
        ed_experiment_id = db_writer.create_experiment(
            dataset_name=dataset_name,
            experiment_type="epsilon_dependent",
            total_questions=len(questions),
            mechanisms=[mechanism],
            epsilon_values=args.epsilon_values,
            local_model=local_model,
            remote_cot_model=remote_cot_model,
            remote_qa_model=remote_qa_model,
            description=f"MedQA-USMLE DPPrivQA tests ({len(questions)} questions, {mechanism})"
        )
        logger.info(f"Created experiment record: ID {ed_experiment_id}")
        
        for epsilon in args.epsilon_values:
            logger.info(f"\nRunning DPPrivQA with epsilon={epsilon}")
            correct_count = 0
            
            for idx, question_data in enumerate(questions):
                question = question_data['question']
                options = question_data['options']
                ground_truth = question_data['answer_idx']
                
                logger.info(f"Question {idx+1}/{len(questions)} (epsilon={epsilon}): {question[:100]}...")
                
                try:
                    result = run_dpprivqa(
                        local_client, remote_client,
                        local_model, remote_cot_model,
                        question, options,
                        mechanism=mechanism,
                        epsilon=epsilon,
                        sbert_model=sbert_model,
                        medical_mode=True  # MedQA is medical dataset
                    )
                    
                    # Check correctness
                    is_correct = check_mcq_correctness(result['answer'], ground_truth)
                    if is_correct:
                        correct_count += 1
                    
                    # Write to database
                    db_result = {
                        'question': question,
                        'sanitized_question': result.get('sanitized_question', ''),
                        'options': options,
                        'cot_text': result.get('cot_text', ''),
                        'answer': result['answer'],
                        'ground_truth': ground_truth,
                        'is_correct': is_correct,
                        'processing_time': result.get('processing_time'),
                        'sanitization_time': result.get('sanitization_time'),
                        'cot_generation_time': result.get('cot_generation_time'),
                        'local_model': result.get('local_model'),
                        'remote_model': result.get('remote_model')
                    }
                    
                    db_writer.write_epsilon_dependent_result(
                        dataset_name, ed_experiment_id, idx, mechanism, epsilon, db_result
                    )
                    
                    logger.info(f"  Answer: {result['answer']}, Ground truth: {ground_truth}, Correct: {is_correct}")
                
                except Exception as e:
                    logger.error(f"Error processing question {idx+1} (epsilon={epsilon}): {e}")
                    continue
            
            accuracy = correct_count / len(questions) if questions else 0
            logger.info(f"\nDPPrivQA (epsilon={epsilon}) accuracy: {correct_count}/{len(questions)} = {accuracy:.3f}")
    
    # Close database
    db_writer.close()
    logger.info("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()


