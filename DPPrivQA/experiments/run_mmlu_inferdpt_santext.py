#!/usr/bin/env python3
"""
Run InferDPT and SANTEXT+ QA experiments on MMLU datasets.

This script runs InferDPT and SANTEXT+ mechanisms on MMLU datasets:
- Professional Law (first 200 questions)
- Professional Medicine (all 272 questions)
- Clinical Knowledge (all 265 questions)
- College Medicine (all 173 questions)

Usage:
    python experiments/run_mmlu_inferdpt_santext.py \
        --dataset professional_law \
        --num-questions 200 \
        --epsilon 2.0
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dpprivqa.datasets import MMLUDataset
from dpprivqa.qa.dpprivqa import run_dpprivqa
from dpprivqa.qa.models import get_nebius_client, get_remote_llm_client
from dpprivqa.database.writer import ExperimentDBWriter
from dpprivqa.utils.logging import setup_logging
from dpprivqa.utils.config import load_config
from dpprivqa.qa.prompts import check_mcq_correctness
from dpprivqa.qa.async_scenarios import async_run_dpprivqa


# Dataset configuration
DATASET_CONFIG = {
    'professional_law': {
        'display_name': 'Professional Law',
        'dataset_name': 'mmlu_professional_law',
        'default_num_questions': 200
    },
    'professional_medicine': {
        'display_name': 'Professional Medicine',
        'dataset_name': 'mmlu_professional_medicine',
        'default_num_questions': 272
    },
    'clinical_knowledge': {
        'display_name': 'Clinical Knowledge',
        'dataset_name': 'mmlu_clinical_knowledge',
        'default_num_questions': 265
    },
    'college_medicine': {
        'display_name': 'College Medicine',
        'dataset_name': 'mmlu_college_medicine',
        'default_num_questions': 173
    }
}


async def process_question(
    question_data: Dict[str, Any],
    question_idx: int,
    mechanism: str,
    epsilon: float,
    local_client: OpenAI,
    remote_client: OpenAI,
    local_model: str,
    remote_cot_model: str,
    sbert_model: SentenceTransformer,
    semaphore: asyncio.Semaphore,
    db_writer: ExperimentDBWriter,
    experiment_id: int,
    dataset_name: str
) -> Optional[Dict[str, Any]]:
    """Process one question with InferDPT or SANTEXT+."""
    async with semaphore:
        try:
            question = question_data['question']
            options = question_data['options']
            ground_truth = question_data['answer_idx']
            
            # Convert ground_truth to letter if it's a number
            if isinstance(ground_truth, int):
                ground_truth_letter = chr(ord('A') + ground_truth)
            else:
                ground_truth_letter = str(ground_truth)
            
            result = await async_run_dpprivqa(
                local_client, remote_client,
                local_model, remote_cot_model,
                question, options,
                mechanism=mechanism,
                epsilon=epsilon,
                sbert_model=sbert_model
            )
            
            result['is_correct'] = check_mcq_correctness(result['answer'], ground_truth_letter)
            result['ground_truth'] = ground_truth_letter
            
            # Write to database
            db_result = {
                'question': question,
                'sanitized_question': result.get('sanitized_question', ''),
                'options': options,
                'induced_cot': result.get('cot_text', ''),
                'answer': result['answer'],
                'ground_truth': ground_truth_letter,
                'is_correct': result['is_correct'],
                'processing_time': result.get('processing_time'),
                'sanitization_time': result.get('sanitization_time'),
                'cot_generation_time': result.get('cot_generation_time'),
                'local_model': local_model,
                'remote_model': remote_cot_model
            }
            
            db_writer.write_epsilon_dependent_result(
                dataset_name, experiment_id, question_idx, mechanism, epsilon, db_result
            )
            
            return result
            
        except Exception as e:
            print(f"    [Q{question_idx}] ERROR: {e}", flush=True)
            return None


async def run_experiments_parallel(
    questions: List[Dict[str, Any]],
    mechanism: str,
    epsilon: float,
    local_client: OpenAI,
    remote_client: OpenAI,
    local_model: str,
    remote_cot_model: str,
    sbert_model: SentenceTransformer,
    db_writer: ExperimentDBWriter,
    experiment_id: int,
    dataset_name: str,
    max_concurrent: int = 10
):
    """Run experiments in parallel."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    tasks = []
    for idx, q in enumerate(questions):
        task = process_question(
            q, idx, mechanism, epsilon,
            local_client, remote_client,
            local_model, remote_cot_model,
            sbert_model, semaphore,
            db_writer, experiment_id, dataset_name
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # Count successes
    successful = [r for r in results if r is not None]
    correct = [r for r in successful if r.get('is_correct', False)]
    
    return len(successful), len(correct)


def main():
    parser = argparse.ArgumentParser(
        description="Run InferDPT and SANTEXT+ QA experiments on MMLU datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_CONFIG.keys()),
        help="MMLU dataset name"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        help="Number of questions to test (default: dataset default)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting question index (default: 0)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help="Epsilon value (default: 2.0)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        choices=['inferdpt', 'santext', 'both'],
        default='both',
        help="Mechanism to run: inferdpt, santext, or both (default: both)"
    )
    
    args = parser.parse_args()
    
    config = load_config()
    dataset_info = DATASET_CONFIG[args.dataset]
    dataset_name = dataset_info['dataset_name']
    display_name = dataset_info['display_name']
    
    # Setup logging
    logger = setup_logging(dataset_name)
    logger.info(f"Starting InferDPT/SANTEXT+ experiments on {display_name}")
    
    # Initialize database writer
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    db_writer = ExperimentDBWriter(db_path)
    
    # Load dataset
    print(f"\n{'='*80}")
    print(f"Loading {display_name} dataset...")
    print(f"{'='*80}")
    
    dataset = MMLUDataset(subset=args.dataset)
    all_questions = dataset.load(split="test")
    
    # Determine number of questions
    num_questions = args.num_questions or dataset_info['default_num_questions']
    end_index = min(args.start_index + num_questions, len(all_questions))
    questions = all_questions[args.start_index:end_index]
    
    print(f"Loaded {len(all_questions)} total questions")
    print(f"Processing questions {args.start_index} to {end_index-1} ({len(questions)} questions)")
    
    # Initialize clients and models
    local_client = get_nebius_client()
    remote_client = get_remote_llm_client()
    local_model = config.get("models", {}).get("local", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    remote_cot_model = config.get("models", {}).get("remote_cot", "gpt-5")
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Determine which mechanisms to run
    mechanisms_to_run = []
    if args.mechanism == 'both':
        mechanisms_to_run = ['inferdpt', 'santext']
    else:
        mechanisms_to_run = [args.mechanism]
    
    # Run experiments for each mechanism
    for mechanism in mechanisms_to_run:
        print(f"\n{'='*80}")
        print(f"Running {mechanism.upper()} experiments (ε={args.epsilon})")
        print(f"{'='*80}")
        
        # Create experiment record
        experiment_id = db_writer.create_experiment(
            dataset_name=dataset_name,
            experiment_type="epsilon_dependent",
            total_questions=len(questions),
            mechanisms=[mechanism],
            epsilon_values=[args.epsilon],
            local_model=local_model,
            remote_cot_model=remote_cot_model,
            remote_qa_model=remote_cot_model,
            description=f"InferDPT/SANTEXT+ experiments on {display_name}"
        )
        print(f"Created experiment record: ID {experiment_id}")
        
        # Run experiments
        successful, correct = asyncio.run(
            run_experiments_parallel(
                questions, mechanism, args.epsilon,
                local_client, remote_client,
                local_model, remote_cot_model,
                sbert_model, db_writer,
                experiment_id, dataset_name,
                max_concurrent=args.max_concurrent
            )
        )
        
        accuracy = (correct / successful * 100) if successful > 0 else 0
        print(f"\n{mechanism.upper()} Results:")
        print(f"  Successful: {successful}/{len(questions)}")
        print(f"  Correct: {correct}/{successful}")
        print(f"  Accuracy: {accuracy:.1f}%")
    
    db_writer.close()
    print(f"\n{'='*80}")
    print("✅ All experiments completed!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

