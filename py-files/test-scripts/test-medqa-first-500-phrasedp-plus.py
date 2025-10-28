#!/usr/bin/env python3
"""
MedQA First 500 Questions - PhraseDP+ (Medical Mode) Testing
============================================================

This script tests PhraseDP+ (Medical Mode) on the first 500 questions (0-499)
with configurable epsilon values via command line arguments.

Usage:
    python test-medqa-first-500-phrasedp-plus.py --epsilon 1.0
    python test-medqa-first-500-phrasedp-plus.py --epsilon 2.0
    python test-medqa-first-500-phrasedp-plus.py --epsilon 3.0

Author: Tech4HSE Team
Date: 2025-10-01
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Import local modules
import utils

# Color codes for console output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

class MedQAExperimentResults:
    """Results container for MedQA experiments."""
    
    def __init__(self):
        self.phrase_dp_plus_correct = 0
        self.total_questions = 0
    
    def print_accuracy(self):
        """Print current accuracy results."""
        print(f"\n{GREEN}=== CURRENT ACCURACY RESULTS ==={RESET}")
        print(f"PhraseDP+ (Medical Mode): {self.phrase_dp_plus_correct}/{self.total_questions} = {self.phrase_dp_plus_correct/max(1,self.total_questions)*100:.2f}%")
        print(f"Total Questions: {self.total_questions}")

def get_remote_llm_client():
    """Get remote LLM client (OpenAI)."""
    try:
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return client
    except Exception as e:
        print(f"Failed to initialize remote LLM client: {e}")
        return None

def get_local_llm_client():
    """Get local LLM client (Nebius)."""
    try:
        return utils.get_nebius_client()
    except Exception as e:
        print(f"Failed to initialize local LLM client: {e}")
        return None

def load_sentence_bert():
    """Load Sentence-BERT model for similarity computation."""
    print(f"{CYAN}Loading Sentence-BERT model...{RESET}")
    return SentenceTransformer('all-MiniLM-L6-v2')

def run_scenario_phrase_dp_plus(question: str, metamap_phrases: List[str], epsilon: float, 
                               local_client, remote_client, sbert_model) -> bool:
    """Run PhraseDP+ (Medical Mode) scenario."""
    print(f"--- Scenario: PhraseDP+ (Medical Mode) ---")
    
    try:
        # Apply PhraseDP+ perturbation with medical mode
        print(f"1a. Applying PhraseDP+ sanitization...")
        print(f"   - Setting up PhraseDP+ with medical mode...")
        print(f"   - Epsilon: {epsilon}")
        print(f"   - Metamap phrases: {len(metamap_phrases)} medical terms")
        print(f"   - Mode: medqa-ume (medical mode)")
        
        # Get model name from config.yaml directly
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        nebius_model_name = config.get('local_model')
        
        sanitized_question = utils.phrase_DP_perturbation_old(
            nebius_client=local_client,
            nebius_model_name=nebius_model_name,
            input_sentence=question,
            epsilon=epsilon,
            sbert_model=sbert_model,
            mode="medqa-ume",
            metamap_phrases=metamap_phrases
        )
        print(f"   - Sanitized question length: {len(sanitized_question)} chars")
        
        # Generate CoT from sanitized question
        print(f"1b. Generating CoT from sanitized question...")
        try:
            cot_response = remote_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical expert. Provide detailed step-by-step reasoning for medical questions."},
                    {"role": "user", "content": sanitized_question}
                ],
                max_tokens=512,
                temperature=0.0
            )
            cot_text = cot_response.choices[0].message.content
            print(f"   - CoT length: {len(cot_text)} chars")
        except Exception as e:
            print(f"   - Error generating CoT: {e}")
            cot_text = "Let me think through this step by step."
        
        # Get local model response with CoT
        print(f"1c. Running Local Model with PhraseDP+ CoT...")
        formatted_question = question
        full_prompt = f"{formatted_question}\n\nChain of Thought:\n{cot_text}\n\nBased on the chain of thought above, what is the correct answer? Provide only the letter (A, B, C, or D):"
        
        try:
            response = local_client.chat.completions.create(
                model=nebius_model_name,
                messages=[
                    {"role": "system", "content": "You are a medical expert. Use the provided chain of thought to answer the multiple choice question. Provide only the letter (A, B, C, or D) of the correct option."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=256,
                temperature=0.0
            )
            response_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"   - Error with local model: {e}")
            response_text = "A"
        
        print(f"   - Raw response length: {len(response_text)} chars")
        print(f"Local Answer (PhraseDP+ CoT-Aided): {response_text}")
        
        # Extract answer
        answer_letter = response_text[-1] if response_text and response_text[-1] in 'ABCD' else 'A'
        print(f"Extracted Letter: {answer_letter}")
        
        return answer_letter
        
    except Exception as e:
        print(f"Error in PhraseDP+ scenario: {e}")
        return 'A'  # Default fallback

def run_experiment_for_question(question: str, options: Dict[str, str], correct_answer: str, 
                               metamap_phrases: List[str], epsilon: float, 
                               local_client, remote_client, sbert_model) -> bool:
    """Run experiment for a single question."""
    print(f"{CYAN}Epsilon: {epsilon}{RESET}")
    
    # Run PhraseDP+ scenario
    phrase_dp_plus_answer = run_scenario_phrase_dp_plus(question, metamap_phrases, epsilon, 
                                                       local_client, remote_client, sbert_model)
    
    if phrase_dp_plus_answer == correct_answer:
        print(f"Result: Correct")
        return True
    else:
        print(f"Result: Incorrect")
        return False

def main():
    """Main function to run MedQA experiment."""
    parser = argparse.ArgumentParser(
        description="MedQA First 500 Questions - PhraseDP+ (Medical Mode) Testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        required=True,
        choices=[1.0, 2.0, 3.0],
        help="Epsilon value for differential privacy (1.0, 2.0, or 3.0)"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting question index (default: 0)"
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=500,
        help="Number of questions to test (default: 500)"
    )
    parser.add_argument(
        "--test-single",
        type=int,
        help="Test a single question by index (0-based)"
    )
    
    args = parser.parse_args()
    
    print(f"{GREEN}=== MedQA First 500 Questions - PhraseDP+ (Medical Mode) ==={RESET}")
    print(f"Epsilon: {args.epsilon}")
    print(f"Questions: {args.start_index} to {args.start_index + args.num_questions - 1}")
    
    # Initialize clients
    remote_client = get_remote_llm_client()
    if not remote_client:
        print(f"{RED}Cannot proceed without remote client. Exiting.{RESET}")
        return
    
    local_client = get_local_llm_client()
    if not local_client:
        print(f"{RED}Cannot proceed without local client. Exiting.{RESET}")
        return
    
    # Load Sentence-BERT
    sbert_model = load_sentence_bert()
    
    # Load dataset
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    
    # Initialize results
    results = MedQAExperimentResults()
    
    # Initialize incremental saving
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.test_single is not None:
        output_file = f"medqa_first_500_phrasedp_plus_epsilon_{args.epsilon}_single_{args.test_single}_{timestamp}.json"
        question_indices = [args.test_single]
    else:
        output_file = f"medqa_first_500_phrasedp_plus_epsilon_{args.epsilon}_{timestamp}.json"
        question_indices = list(range(args.start_index, args.start_index + args.num_questions))
    
    all_results = []
    print(f"{GREEN}Results will be saved to: {output_file}{RESET}")
    
    for i, dataset_idx in enumerate(question_indices):
        print(f"\n{GREEN}--- Question {i+1}/{len(question_indices)} (Dataset idx: {dataset_idx}) ---{RESET}")
        
        # Get question data
        item = dataset[dataset_idx]
        question = item["question"]
        options = item["options"]
        correct_answer = item["answer_idx"]
        meta_info = item.get("meta_info", {})
        
        # Generate MetaMap phrases (simplified)
        metamap_phrases = question.split()[:20]
        
        print(f"Question: {question[:200]}...")
        print(f"Options: {options}")
        print(f"Correct Answer: {correct_answer}")
        if isinstance(meta_info, dict):
            print(f"Complexity Level: {meta_info.get('complexity_level', 'unknown')}")
        else:
            print(f"Complexity Level: {meta_info}")
        print(f"MetaMap Phrases ({len(metamap_phrases)}): {metamap_phrases[:10]}...")
        
        # Run experiment
        is_correct = run_experiment_for_question(question, options, correct_answer, 
                                                metamap_phrases, args.epsilon,
                                                local_client, remote_client, sbert_model)
        
        # Update results
        results.total_questions += 1
        if is_correct:
            results.phrase_dp_plus_correct += 1
        
        # Save progress incrementally after each question
        question_result = {
            'question_id': dataset_idx,
            'question': question,
            'options': options,
            'correct_answer': correct_answer,
            'meta_info': meta_info,
            'metamap_phrases': metamap_phrases,
            'epsilon': args.epsilon,
            'is_correct': is_correct,
            'timestamp': datetime.now().isoformat()
        }
        all_results.append(question_result)
        
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"{GREEN}Progress saved: {len(all_results)} questions completed{RESET}")
        
        # Print cumulative accuracy
        results.print_accuracy()
    
    print(f"\n{GREEN}=== EXPERIMENT COMPLETED ==={RESET}")
    print(f"Final Results File: {output_file}")
    results.print_accuracy()
    
    return results

if __name__ == "__main__":
    main()
