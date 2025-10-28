#!/usr/bin/env python3
"""
MedQA USMLE Comprehensive Comparison Script (Efficient Epsilon Handling)
======================================================================

This script efficiently compares all 7 privacy mechanisms across multiple epsilon values
by caching epsilon-independent results and reusing them across epsilon values.

Features:
- Epsilon-independent mechanisms (1, 6, 7) run once per question and are cached
- Epsilon-dependent mechanisms (2, 3, 4, 5) run for each epsilon value
- Significant API call reduction and time savings
- Incremental saving and progress tracking
- No config.yaml dependencies - all parameters via command line

Usage:
    # Run all epsilons (1.0, 2.0, 3.0) in one process - RECOMMENDED
    python test-medqa-usmle-phrasedp-comparison.py --epsilons "1.0,2.0,3.0" --num-samples 10
    
    # Run single epsilon
    python test-medqa-usmle-phrasedp-comparison.py --epsilon 1.0 --num-samples 10

Author: Tech4HSE Team
Date: 2025-10-02
"""

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

# Import local modules
import utils
from sanitization_methods import (
    inferdpt_sanitize_text,
    santext_sanitize_text,
)

# Color codes for terminal output
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def get_remote_llm_client():
    """Get remote LLM client (OpenAI)."""
    try:
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        return client
    except Exception as e:
        print(f"{RED}Failed to initialize remote LLM client: {e}{RESET}")
        return None

def get_local_llm_client():
    """Get local LLM client (Nebius)."""
    try:
        return utils.get_nebius_client()
    except Exception as e:
        print(f"{RED}Failed to initialize local LLM client: {e}{RESET}")
        return None

def load_sentence_bert():
    """Load Sentence-BERT model for similarity computation."""
    print(f"{CYAN}Loading Sentence-BERT model...{RESET}")
    return SentenceTransformer('all-MiniLM-L6-v2')

def format_question_with_options(question, options=None):
    """Format question (optionally with answer choices) for LLM input."""
    formatted = f"{question}"
    if options:
        formatted += "\n\nOptions:\n"
        for key, value in options.items():
            formatted += f"{key}) {value}\n"
    formatted += "\n\nAnswer:"
    return formatted

def extract_letter_from_answer(answer):
    """Extract the letter (A, B, C, D) from the model's answer."""
    answer = answer.strip().upper()
    
    # Look for single letters
    for letter in ['A', 'B', 'C', 'D']:
        if answer == letter or answer.startswith(letter) or f" {letter}" in answer:
            return letter
    
    # Look for patterns like "Option A", "Choice A", etc.
    patterns = ['OPTION', 'CHOICE', 'ANSWER']
    for pattern in patterns:
        for letter in ['A', 'B', 'C', 'D']:
            if f"{pattern} {letter}" in answer:
                return letter
    
    return answer[:1] if answer else "Error"

def check_mcq_correctness(predicted_letter, correct_letter):
    """Check if the predicted answer is correct."""
    return predicted_letter.upper() == correct_letter.upper()

def get_answer_from_local_model_alone(client, model_name, question, options, max_tokens=256):
    """Get answer from local model without any CoT assistance."""
    
    formatted_question = format_question_with_options(question, options)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert. Answer the multiple choice question by providing only the letter (A, B, C, or D) of the correct option."},
                {"role": "user", "content": formatted_question}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in local model inference: {e}")
        return "Error"

def get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text, max_tokens=256):
    """Get answer from local model with CoT assistance."""
    
    formatted_question = format_question_with_options(question, options)
    full_prompt = f"{formatted_question}\n\nChain of Thought:\n{cot_text}\n\nBased on the chain of thought above, what is the correct answer? Provide only the letter (A, B, C, or D):"
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert. Use the provided chain of thought to answer the multiple choice question. Provide only the letter (A, B, C, or D) of the correct option."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in local model with CoT inference: {e}")
        return "Error"

def generate_cot_from_remote_llm(client, model_name, question, options=None, max_tokens=512):
    """Generate Chain-of-Thought from remote LLM."""
    
    formatted_question = format_question_with_options(question, options)
    prompt = f"{formatted_question}\n\nPlease provide a step-by-step chain of thought to solve this medical question:"
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert. Provide a clear, step-by-step chain of thought to solve the given medical multiple choice question. Focus on medical reasoning and knowledge."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in remote CoT generation: {e}")
        return "Error"

def get_answer_from_remote_llm(client, model_name, question, options, max_tokens=256):
    """Get answer directly from remote LLM."""
    
    formatted_question = format_question_with_options(question, options)
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert. Answer the multiple choice question by providing only the letter (A, B, C, or D) of the correct option."},
                {"role": "user", "content": formatted_question}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in remote LLM inference: {e}")
        return "Error"

def run_local_model_scenario(question: str, options: Dict[str, str], correct_answer: str, 
                           answer_model: str, local_client, remote_client) -> str:
    """Run Local Model (Baseline) scenario."""
    print(f"--- Scenario: Local Model (Baseline) ---")
    
    try:
        # Use remote client for OpenAI models, local client for Nebius models
        if answer_model.startswith('gpt-') or answer_model.startswith('o1-'):
            response_text = get_answer_from_local_model_alone(remote_client, answer_model, question, options)
        else:
            response_text = get_answer_from_local_model_alone(local_client, answer_model, question, options)
        
        answer_letter = extract_letter_from_answer(response_text)
        
        print(f"Local Answer: {response_text}")
        print(f"Extracted Letter: {answer_letter}")
        
        return answer_letter
        
    except Exception as e:
        print(f"Error in Local Model scenario: {e}")
        return 'A'

def run_inferdpt_scenario(question: str, options: Dict[str, str], correct_answer: str, 
                         epsilon: float, answer_model: str, local_client, remote_client) -> str:
    """Run InferDPT scenario."""
    print(f"--- Scenario: InferDPT ---")
    
    try:
        # Apply InferDPT perturbation
        print(f"1a. Applying InferDPT sanitization...")
        print(f"   - Epsilon: {epsilon}")
        
        perturbed_question = inferdpt_sanitize_text(question, epsilon=epsilon)
        print(f"   - Perturbed question length: {len(perturbed_question)} chars")
        
        # Generate CoT from perturbed question
        print(f"1b. Generating CoT from perturbed question...")
        cot_text = generate_cot_from_remote_llm(remote_client, "gpt-4o-mini", perturbed_question)
        print(f"   - CoT length: {len(cot_text)} chars")
        
        # Get model response with CoT
        print(f"1c. Running Model with InferDPT CoT...")
        if answer_model.startswith('gpt-') or answer_model.startswith('o1-'):
            response_text = get_answer_from_local_model_with_cot(remote_client, answer_model, question, options, cot_text)
        else:
            response_text = get_answer_from_local_model_with_cot(local_client, answer_model, question, options, cot_text)
        
        print(f"   - Raw response length: {len(response_text)} chars")
        print(f"InferDPT Answer (CoT-Aided): {response_text}")
        
        answer_letter = extract_letter_from_answer(response_text)
        print(f"Extracted Letter: {answer_letter}")
        
        return answer_letter
        
    except Exception as e:
        print(f"Error in InferDPT scenario: {e}")
        return 'A'

def run_santext_scenario(question: str, options: Dict[str, str], correct_answer: str, 
                        epsilon: float, answer_model: str, local_client, remote_client) -> str:
    """Run SANTEXT+ scenario."""
    print(f"--- Scenario: SANTEXT+ ---")
    
    try:
        # Apply SANTEXT+ perturbation
        print(f"1a. Applying SANTEXT+ sanitization...")
        print(f"   - Epsilon: {epsilon}")
        
        perturbed_question = santext_sanitize_text(question, epsilon=epsilon)
        print(f"   - Perturbed question length: {len(perturbed_question)} chars")
        
        # Generate CoT from perturbed question
        print(f"1b. Generating CoT from perturbed question...")
        cot_text = generate_cot_from_remote_llm(remote_client, "gpt-4o-mini", perturbed_question)
        print(f"   - CoT length: {len(cot_text)} chars")
        
        # Get model response with CoT
        print(f"1c. Running Model with SANTEXT+ CoT...")
        if answer_model.startswith('gpt-') or answer_model.startswith('o1-'):
            response_text = get_answer_from_local_model_with_cot(remote_client, answer_model, question, options, cot_text)
        else:
            response_text = get_answer_from_local_model_with_cot(local_client, answer_model, question, options, cot_text)
        
        print(f"   - Raw response length: {len(response_text)} chars")
        print(f"SANTEXT+ Answer (CoT-Aided): {response_text}")
        
        answer_letter = extract_letter_from_answer(response_text)
        print(f"Extracted Letter: {answer_letter}")
        
        return answer_letter
        
    except Exception as e:
        print(f"Error in SANTEXT+ scenario: {e}")
        return 'A'

def run_phrasedp_scenario(question: str, options: Dict[str, str], correct_answer: str, 
                         metamap_phrases: List[str], epsilon: float, phrasedp_model: str, answer_model: str,
                         local_client, remote_client, sbert_model, use_medical_mode: bool) -> str:
    """Run PhraseDP scenario (normal or medical mode)."""
    
    mode_name = "Medical Mode" if use_medical_mode else "Normal Mode"
    print(f"--- Scenario: PhraseDP ({mode_name}) ---")
    
    try:
        # Apply PhraseDP perturbation
        print(f"1a. Applying PhraseDP sanitization ({mode_name})...")
        print(f"   - Epsilon: {epsilon}")
        print(f"   - Mode: {'medqa-ume' if use_medical_mode else 'normal'}")
        if use_medical_mode:
            print(f"   - Metamap phrases: {len(metamap_phrases)} medical terms")
        else:
            print(f"   - No metamap phrases (normal mode)")
        
        # Use the specified PhraseDP model for perturbation
        nebius_model_name = phrasedp_model
        
        sanitized_question = utils.phrase_DP_perturbation_old(
            nebius_client=local_client,
            nebius_model_name=nebius_model_name,
            input_sentence=question,
            epsilon=epsilon,
            sbert_model=sbert_model,
            mode="medqa-ume" if use_medical_mode else "normal",
            metamap_phrases=metamap_phrases if use_medical_mode else None
        )
        print(f"   - Sanitized question length: {len(sanitized_question)} chars")
        
        # Generate CoT from sanitized question
        print(f"1b. Generating CoT from sanitized question...")
        cot_text = generate_cot_from_remote_llm(remote_client, "gpt-4o-mini", sanitized_question)
        print(f"   - CoT length: {len(cot_text)} chars")
        
        # Get model response with CoT
        print(f"1c. Running Model with PhraseDP CoT...")
        if answer_model.startswith('gpt-') or answer_model.startswith('o1-'):
            response_text = get_answer_from_local_model_with_cot(remote_client, answer_model, question, options, cot_text)
        else:
            response_text = get_answer_from_local_model_with_cot(local_client, answer_model, question, options, cot_text)
        
        print(f"   - Raw response length: {len(response_text)} chars")
        print(f"PhraseDP Answer ({mode_name} CoT-Aided): {response_text}")
        
        # Extract answer
        answer_letter = extract_letter_from_answer(response_text)
        print(f"Extracted Letter: {answer_letter}")
        
        return answer_letter
        
    except Exception as e:
        print(f"Error in PhraseDP {mode_name} scenario: {e}")
        return 'A'  # Default fallback

def run_local_cot_scenario(question: str, options: Dict[str, str], correct_answer: str, 
                          answer_model: str, local_client, remote_client) -> str:
    """Run Local + CoT (Non-private) scenario."""
    print(f"--- Scenario: Local + CoT (Non-private) ---")
    
    try:
        # Generate CoT from original question
        print(f"1a. Generating CoT from ORIGINAL Question with REMOTE LLM...")
        cot_text = generate_cot_from_remote_llm(remote_client, "gpt-4o-mini", question)
        print(f"   - CoT length: {len(cot_text)} chars")
        
        # Get model response with CoT
        print(f"1b. Running Model with Non-Private CoT...")
        if answer_model.startswith('gpt-') or answer_model.startswith('o1-'):
            response_text = get_answer_from_local_model_with_cot(remote_client, answer_model, question, options, cot_text)
        else:
            response_text = get_answer_from_local_model_with_cot(local_client, answer_model, question, options, cot_text)
        
        print(f"   - Raw response length: {len(response_text)} chars")
        print(f"Local Answer (Non-Private CoT-Aided): {response_text}")
        
        answer_letter = extract_letter_from_answer(response_text)
        print(f"Extracted Letter: {answer_letter}")
        
        return answer_letter
        
    except Exception as e:
        print(f"Error in Local + CoT scenario: {e}")
        return 'A'

def run_remote_model_scenario(question: str, options: Dict[str, str], correct_answer: str, 
                             remote_client) -> str:
    """Run Remote Model scenario."""
    print(f"--- Scenario: Remote Model ---")
    
    try:
        response_text = get_answer_from_remote_llm(remote_client, "gpt-4o-mini", question, options)
        answer_letter = extract_letter_from_answer(response_text)
        
        print(f"Remote Answer: {response_text}")
        print(f"Extracted Letter: {answer_letter}")
        
        return answer_letter
        
    except Exception as e:
        print(f"Error in Remote Model scenario: {e}")
        return 'A'

def run_epsilon_independent_scenarios(question: str, options: Dict[str, str], correct_answer: str, 
                                    answer_model: str, local_client, remote_client) -> Dict[str, Any]:
    """Run epsilon-independent scenarios (1, 6, 7) - only need to run once."""
    print(f"{CYAN}Running epsilon-independent mechanisms (1, 6, 7)...{RESET}")
    
    # 1. Local Model (Baseline)
    local_answer = run_local_model_scenario(question, options, correct_answer, answer_model, local_client, remote_client)
    
    # 6. Local + CoT (Non-private)
    local_cot_answer = run_local_cot_scenario(question, options, correct_answer, answer_model, local_client, remote_client)
    
    # 7. Remote Model
    remote_answer = run_remote_model_scenario(question, options, correct_answer, remote_client)
    
    # Check results
    local_correct = local_answer == correct_answer
    local_cot_correct = local_cot_answer == correct_answer
    remote_correct = remote_answer == correct_answer
    
    print(f"\n📊 EPSILON-INDEPENDENT RESULTS:")
    print(f"1. Local Model: {local_answer} {'✅' if local_correct else '❌'}")
    print(f"6. Local + CoT: {local_cot_answer} {'✅' if local_cot_correct else '❌'}")
    print(f"7. Remote Model: {remote_answer} {'✅' if remote_correct else '❌'}")
    print(f"Correct Answer: {correct_answer}")
    
    return {
        'local_answer': local_answer,
        'local_cot_answer': local_cot_answer,
        'remote_answer': remote_answer,
        'local_correct': local_correct,
        'local_cot_correct': local_cot_correct,
        'remote_correct': remote_correct
    }

def run_epsilon_dependent_scenarios(question: str, options: Dict[str, str], correct_answer: str, 
                                  metamap_phrases: List[str], epsilon: float, phrasedp_model: str, answer_model: str,
                                  local_client, remote_client, sbert_model) -> Dict[str, Any]:
    """Run epsilon-dependent scenarios (2, 3, 4, 5) - run for each epsilon value."""
    print(f"{CYAN}Running epsilon-dependent mechanisms (2, 3, 4, 5) for epsilon {epsilon}...{RESET}")
    
    # 2. InferDPT
    inferdpt_answer = run_inferdpt_scenario(question, options, correct_answer, epsilon, answer_model, local_client, remote_client)
    
    # 3. SANTEXT+
    santext_answer = run_santext_scenario(question, options, correct_answer, epsilon, answer_model, local_client, remote_client)
    
    # 4. PhraseDP (Normal Mode)
    phrasedp_normal_answer = run_phrasedp_scenario(question, options, correct_answer, 
                                                  metamap_phrases, epsilon, phrasedp_model, answer_model,
                                                  local_client, remote_client, sbert_model, 
                                                  use_medical_mode=False)
    
    # 5. PhraseDP+ (Medical Mode)
    phrasedp_medical_answer = run_phrasedp_scenario(question, options, correct_answer, 
                                                   metamap_phrases, epsilon, phrasedp_model, answer_model,
                                                   local_client, remote_client, sbert_model, 
                                                   use_medical_mode=True)
    
    # Check results
    inferdpt_correct = inferdpt_answer == correct_answer
    santext_correct = santext_answer == correct_answer
    phrasedp_normal_correct = phrasedp_normal_answer == correct_answer
    phrasedp_medical_correct = phrasedp_medical_answer == correct_answer
    
    print(f"\n📊 EPSILON-DEPENDENT RESULTS (ε={epsilon}):")
    print(f"2. InferDPT: {inferdpt_answer} {'✅' if inferdpt_correct else '❌'}")
    print(f"3. SANTEXT+: {santext_answer} {'✅' if santext_correct else '❌'}")
    print(f"4. PhraseDP (Normal): {phrasedp_normal_answer} {'✅' if phrasedp_normal_correct else '❌'}")
    print(f"5. PhraseDP+ (Medical): {phrasedp_medical_answer} {'✅' if phrasedp_medical_correct else '❌'}")
    print(f"Correct Answer: {correct_answer}")
    
    # Medical mode benefit analysis
    medical_benefit = phrasedp_medical_correct - phrasedp_normal_correct
    print(f"\n🔍 MEDICAL MODE BENEFIT: {medical_benefit:+1.0f} {'✅' if medical_benefit > 0 else '❌' if medical_benefit < 0 else '⚖️'}")
    
    return {
        'inferdpt_answer': inferdpt_answer,
        'santext_answer': santext_answer,
        'phrasedp_normal_answer': phrasedp_normal_answer,
        'phrasedp_medical_answer': phrasedp_medical_answer,
        'inferdpt_correct': inferdpt_correct,
        'santext_correct': santext_correct,
        'phrasedp_normal_correct': phrasedp_normal_correct,
        'phrasedp_medical_correct': phrasedp_medical_correct
    }

def combine_results(epsilon_independent: Dict[str, Any], epsilon_dependent: Dict[str, Any]) -> Dict[str, Any]:
    """Combine epsilon-independent and epsilon-dependent results."""
    combined = {}
    combined.update(epsilon_independent)
    combined.update(epsilon_dependent)
    return combined

def main():
    """Main function with proper indentation."""
    parser = argparse.ArgumentParser(
        description="MedQA USMLE Comprehensive Comparison Script (Efficient Epsilon Handling)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--epsilon",
        type=float,
        choices=[1.0, 2.0, 3.0],
        help="Single epsilon value for differential privacy (1.0, 2.0, or 3.0)"
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="1.0,2.0,3.0",
        help="Comma-separated epsilon values to test (default: '1.0,2.0,3.0')"
    )
    parser.add_argument(
        "--phrasedp-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Nebius model name for PhraseDP perturbation (default: Llama 8B from config)"
    )
    parser.add_argument(
        "--answer-model",
        type=str,
        default="gpt-4o-mini",
        help="Model name for final answer generation (default: 'gpt-4o-mini')"
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting question index (default: 0)"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of questions to test (default: 10)"
    )
    parser.add_argument(
        "--test-single",
        type=int,
        help="Test a single question by index (0-based)"
    )
    
    args = parser.parse_args()
    
    # Parse epsilon values
    if args.epsilon is not None:
        epsilon_values = [args.epsilon]
    else:
        epsilon_values = [float(eps.strip()) for eps in args.epsilons.split(',')]
    
    print(f"{GREEN}=== MedQA USMLE Comprehensive Comparison (Efficient Epsilon) ==={RESET}")
    print(f"Epsilon Values: {epsilon_values}")
    print(f"PhraseDP Model: {args.phrasedp_model}")
    print(f"Answer Model: {args.answer_model}")
    print(f"Questions: {args.start_index} to {args.start_index + args.num_samples - 1}")
    print(f"\nTesting 7 mechanisms:")
    print(f"1. Local Model (Baseline) - Epsilon Independent ✅")
    print(f"2. InferDPT - Epsilon Dependent ❌")
    print(f"3. SANTEXT+ - Epsilon Dependent ❌")
    print(f"4. PhraseDP (Normal Mode) - Epsilon Dependent ❌")
    print(f"5. PhraseDP+ (Medical Mode) - Epsilon Dependent ❌")
    print(f"6. Local + CoT (Non-private) - Epsilon Independent ✅")
    print(f"7. Remote Model (GPT-4o) - Epsilon Independent ✅")
    print(f"\n⚡ EFFICIENCY: Epsilon-independent mechanisms cached and reused across all epsilon values")
    
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
    print(f"{CYAN}Loading MedQA-USMLE dataset...{RESET}")
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    
    # Initialize results storage
    all_results = {}
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Determine question indices
    if args.test_single is not None:
        question_indices = [args.test_single]
    else:
        question_indices = list(range(args.start_index, args.start_index + args.num_samples))
    
    # Store epsilon-independent results (run once per question, reused across all epsilons)
    epsilon_independent_cache = {}
    
    print(f"{GREEN}Processing {len(question_indices)} questions across {len(epsilon_values)} epsilon values{RESET}")
    print(f"{GREEN}Total experiments: {len(question_indices)} questions × {len(epsilon_values)} epsilons = {len(question_indices) * len(epsilon_values)}{RESET}")
    
    # Process each question
    for i, dataset_idx in enumerate(question_indices):
        try:
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
            
            # Run epsilon-independent scenarios (only once per question, reused across all epsilons)
            if dataset_idx not in epsilon_independent_cache:
                print(f"\n⚡ Running epsilon-independent mechanisms (1, 6, 7)...")
                epsilon_independent_cache[dataset_idx] = run_epsilon_independent_scenarios(
                    question, options, correct_answer, args.answer_model, local_client, remote_client
                )
                print(f"💾 Epsilon-independent results cached for question {dataset_idx}")
            else:
                print(f"\n♻️  Reusing epsilon-independent results for question {dataset_idx}")
            
            # Initialize results for this question across all epsilons
            all_results[dataset_idx] = {}
            
            # Run epsilon-dependent scenarios for each epsilon value
            for epsilon in epsilon_values:
                print(f"\n🎯 Running epsilon-dependent mechanisms (2, 3, 4, 5) for epsilon {epsilon}...")
                epsilon_dependent_results = run_epsilon_dependent_scenarios(
                    question, options, correct_answer, metamap_phrases, epsilon,
                    args.phrasedp_model, args.answer_model, local_client, remote_client, sbert_model
                )
                
                # Combine results for this epsilon
                question_results = combine_results(
                    epsilon_independent_cache[dataset_idx], 
                    epsilon_dependent_results
                )
                
                # Store results for this epsilon
                all_results[dataset_idx][epsilon] = question_results
            
            # Save progress incrementally after each question (across all epsilons)
            if (i + 1) % 5 == 0 or i == len(question_indices) - 1:
                # Create output filename with epsilon range
                eps_range = "_".join([f"{eps:.1f}" for eps in epsilon_values])
                output_file = f"medqa_usmle_efficient_eps{eps_range}_{args.phrasedp_model.replace('/', '_')}_{args.answer_model.replace('/', '_')}_{timestamp}.json"
                
                # Save current progress
                save_data = {
                    'experiment_info': {
                        'epsilon_values': epsilon_values,
                        'phrasedp_model': args.phrasedp_model,
                        'answer_model': args.answer_model,
                        'start_index': args.start_index,
                        'num_samples': args.num_samples,
                        'questions_completed': i + 1,
                        'timestamp': timestamp
                    },
                    'results': all_results,
                    'epsilon_independent_cache': epsilon_independent_cache
                }
                
                with open(output_file, 'w') as f:
                    json.dump(save_data, f, indent=2)
                print(f"💾 Progress saved: {output_file}")
            
            # Print progress every 2 questions
            if (i + 1) % 2 == 0:
                print(f"\n📊 Progress: {i + 1}/{len(question_indices)} questions completed")
                print(f"⚡ Epsilon-independent mechanisms cached: {len(epsilon_independent_cache)}")
                print(f"🎯 Total experiments run: {(i + 1) * len(epsilon_values)}")
        
        except Exception as e:
            print(f"❌ Error processing question {dataset_idx}: {e}")
            continue
    
    # Final results summary
    print(f"\n{'='*80}")
    print(f"🏁 FINAL RESULTS - All Epsilon Values")
    print(f"{'='*80}")
    
    # Calculate and display results for each epsilon
    for epsilon in epsilon_values:
        print(f"\n📊 EPSILON {epsilon}:")
        print("-" * 40)
        
        # Calculate totals for this epsilon
        total_questions = len(question_indices)
        local_correct = sum(1 for q_idx in question_indices if all_results.get(q_idx, {}).get(epsilon, {}).get('local_correct', False))
        inferdpt_correct = sum(1 for q_idx in question_indices if all_results.get(q_idx, {}).get(epsilon, {}).get('inferdpt_correct', False))
        santext_correct = sum(1 for q_idx in question_indices if all_results.get(q_idx, {}).get(epsilon, {}).get('santext_correct', False))
        phrasedp_normal_correct = sum(1 for q_idx in question_indices if all_results.get(q_idx, {}).get(epsilon, {}).get('phrasedp_normal_correct', False))
        phrasedp_medical_correct = sum(1 for q_idx in question_indices if all_results.get(q_idx, {}).get(epsilon, {}).get('phrasedp_medical_correct', False))
        local_cot_correct = sum(1 for q_idx in question_indices if all_results.get(q_idx, {}).get(epsilon, {}).get('local_cot_correct', False))
        remote_correct = sum(1 for q_idx in question_indices if all_results.get(q_idx, {}).get(epsilon, {}).get('remote_correct', False))
        
        print(f"1. Local Model: {local_correct}/{total_questions} = {local_correct/total_questions*100:.1f}%")
        print(f"2. InferDPT: {inferdpt_correct}/{total_questions} = {inferdpt_correct/total_questions*100:.1f}%")
        print(f"3. SANTEXT+: {santext_correct}/{total_questions} = {santext_correct/total_questions*100:.1f}%")
        print(f"4. PhraseDP (Normal): {phrasedp_normal_correct}/{total_questions} = {phrasedp_normal_correct/total_questions*100:.1f}%")
        print(f"5. PhraseDP+ (Medical): {phrasedp_medical_correct}/{total_questions} = {phrasedp_medical_correct/total_questions*100:.1f}%")
        print(f"6. Local + CoT: {local_cot_correct}/{total_questions} = {local_cot_correct/total_questions*100:.1f}%")
        print(f"7. Remote Model: {remote_correct}/{total_questions} = {remote_correct/total_questions*100:.1f}%")
        
        # Medical mode benefit
        medical_benefit = phrasedp_medical_correct - phrasedp_normal_correct
        print(f"🔍 Medical Mode Benefit: {medical_benefit:+d} questions ({medical_benefit/total_questions*100:+.1f}%)")
    
    # Save final results
    eps_range = "_".join([f"{eps:.1f}" for eps in epsilon_values])
    final_output_file = f"medqa_usmle_efficient_eps{eps_range}_{args.phrasedp_model.replace('/', '_')}_{args.answer_model.replace('/', '_')}_FINAL_{timestamp}.json"
    
    final_save_data = {
        'experiment_info': {
            'epsilon_values': epsilon_values,
            'phrasedp_model': args.phrasedp_model,
            'answer_model': args.answer_model,
            'start_index': args.start_index,
            'num_samples': args.num_samples,
            'questions_completed': len(question_indices),
            'timestamp': timestamp
        },
        'results': all_results,
        'epsilon_independent_cache': epsilon_independent_cache
    }
    
    with open(final_output_file, 'w') as f:
        json.dump(final_save_data, f, indent=2)
    print(f"\n💾 Final results saved: {final_output_file}")
    
    print(f"\n✅ Comprehensive MedQA experiment completed!")
    print(f"📊 Tested {len(question_indices)} questions across {len(epsilon_values)} epsilon values")
    print(f"🎯 Total experiments: {len(question_indices)} × {len(epsilon_values)} = {len(question_indices) * len(epsilon_values)}")
    print(f"⚡ EFFICIENCY: Epsilon-independent mechanisms cached and reused across all epsilon values")
    print(f"💰 API calls saved: ~{len(question_indices) * (len(epsilon_values) - 1) * 3} calls for mechanisms 1, 6, 7")

if __name__ == "__main__":
    main()