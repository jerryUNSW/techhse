#!/usr/bin/env python3
"""
MedMCQA Experiment Script

Similar to MedQA but uses MedMCQA dataset.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from test_medqa and adapt
from experiments.test_medqa import main as medqa_main
from dpprivqa.datasets import MedMCQADataset

# The structure is identical to MedQA, just change the dataset
if __name__ == "__main__":
    import argparse
    from sentence_transformers import SentenceTransformer
    from dpprivqa.qa.scenarios import run_local_only, run_local_with_cot, run_remote_only
    from dpprivqa.qa.dpprivqa import run_dpprivqa
    from dpprivqa.qa.models import get_nebius_client, get_remote_llm_client
    from dpprivqa.database.writer import ExperimentDBWriter
    from dpprivqa.utils.logging import setup_logging
    from dpprivqa.utils.config import load_config
    from dpprivqa.qa.prompts import check_mcq_correctness
    
    parser = argparse.ArgumentParser(description="Run MedMCQA experiments")
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--epsilon-values", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--use-phrasedp-plus", action="store_true")
    parser.add_argument("--skip-epsilon-independent", action="store_true")
    parser.add_argument("--skip-epsilon-dependent", action="store_true")
    args = parser.parse_args()
    
    config = load_config()
    dataset_name = "medmcqa"
    logger = setup_logging(dataset_name)
    logger.info("Starting MedMCQA experiment")
    
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    db_writer = ExperimentDBWriter(db_path)
    
    dataset = MedMCQADataset()
    questions = dataset.load(split="test")
    
    if args.num_questions:
        end_index = min(args.start_index + args.num_questions, len(questions))
        questions = questions[args.start_index:end_index]
    else:
        questions = questions[args.start_index:]
    
    logger.info(f"Testing {len(questions)} questions")
    
    local_client = get_nebius_client()
    remote_client = get_remote_llm_client()
    local_model = config.get("models", {}).get("local")
    remote_cot_model = config.get("models", {}).get("remote_cot")
    remote_qa_model = config.get("models", {}).get("remote_qa")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Run experiments (same structure as MedQA)
    # Epsilon-independent tests
    if not args.skip_epsilon_independent:
        ei_experiment_id = db_writer.create_experiment(
            dataset_name=dataset_name,
            experiment_type="epsilon_independent",
            total_questions=len(questions),
            local_model=local_model,
            remote_cot_model=remote_cot_model,
            remote_qa_model=remote_qa_model
        )
        
        for scenario in ['local', 'local_cot', 'remote']:
            logger.info(f"Running scenario: {scenario}")
            for idx, q in enumerate(questions):
                question = q['question']
                options = q['options']
                ground_truth = q['answer_idx']
                
                if scenario == 'local':
                    result = run_local_only(local_client, local_model, question, options)
                elif scenario == 'local_cot':
                    result = run_local_with_cot(local_client, remote_client, local_model, remote_cot_model, question, options)
                else:
                    result = run_remote_only(remote_client, remote_qa_model, question, options)
                
                is_correct = check_mcq_correctness(result['answer'], ground_truth)
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
                db_writer.write_epsilon_independent_result(dataset_name, ei_experiment_id, idx, scenario, db_result)
    
    # Epsilon-dependent tests
    if not args.skip_epsilon_dependent:
        mechanism = 'phrasedp_plus' if args.use_phrasedp_plus else 'phrasedp'
        ed_experiment_id = db_writer.create_experiment(
            dataset_name=dataset_name,
            experiment_type="epsilon_dependent",
            total_questions=len(questions),
            mechanisms=[mechanism],
            epsilon_values=args.epsilon_values,
            local_model=local_model,
            remote_cot_model=remote_cot_model,
            remote_qa_model=remote_qa_model
        )
        
        for epsilon in args.epsilon_values:
            logger.info(f"Running DPPrivQA with epsilon={epsilon}")
            for idx, q in enumerate(questions):
                question = q['question']
                options = q['options']
                ground_truth = q['answer_idx']
                
                result = run_dpprivqa(
                    local_client, remote_client,
                    local_model, remote_cot_model,
                    question, options,
                    mechanism=mechanism,
                    epsilon=epsilon,
                    sbert_model=sbert_model,
                    medical_mode=True
                )
                
                is_correct = check_mcq_correctness(result['answer'], ground_truth)
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
                db_writer.write_epsilon_dependent_result(dataset_name, ed_experiment_id, idx, mechanism, epsilon, db_result)
    
    db_writer.close()
    logger.info("Experiment completed!")


