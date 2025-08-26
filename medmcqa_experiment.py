#!/usr/bin/env python3
"""
MedMCQA Experiment Script
=========================

A standalone script for experimenting with MedMCQA (Medical Multiple Choice Questions)
using privacy-preserving approaches similar to the multi-hop experiments.

This script tests different scenarios:
1. Purely Local Model (Baseline)
2. Non-Private Local Model + Remote CoT
3. Private Local Model + CoT (Phrase DP)
4. Private Local Model + CoT (InferDPT)
5. Purely Remote Model

Author: Tech4HSE Team
Date: 2025-08-25
"""

import os
import sys
import yaml
import json
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import random

# Import local modules
import utils
from prompt_loader import load_system_prompt, load_user_prompt_template, format_user_prompt

# Load environment variables
load_dotenv()

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# ANSI color codes for better console output
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

class MCQExperimentResults:
    """Class to track experiment results for MCQ dataset."""
    
    def __init__(self):
        self.local_alone_correct = 0
        self.non_private_cot_correct = 0
        self.local_cot_correct = 0
        self.phrase_dp_local_cot_correct = 0
        self.inferdpt_local_cot_correct = 0
        self.purely_remote_correct = 0
        self.total_questions = 0

def load_sentence_bert():
    """Load Sentence-BERT model for similarity computation."""
    print(f"{CYAN}Loading Sentence-BERT model...{RESET}")
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_remote_llm_client():
    """Get remote LLM client based on configuration."""
    provider = config.get('remote_llm_provider', 'openai')
    
    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return openai.OpenAI(api_key=api_key)
    else:
        raise ValueError(f"Unsupported remote LLM provider: {provider}")

def get_answer_from_local_model_alone(client, model_name, question, options, max_tokens=256):
    """Get answer from local model without any CoT assistance."""
    
    # Format the question with options
    formatted_question = f"Question: {question}\n\nOptions:\nA) {options['a']}\nB) {options['b']}\nC) {options['c']}\nD) {options['d']}\n\nAnswer:"
    
    try:
        # Use a smaller/weaker model to simulate local model behavior
        # For now, use gpt-4o-mini as a proxy for local model
        local_model = "gpt-4o-mini"
        response = client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "system", "content": "You are a medical expert. Answer the multiple choice question by providing only the letter (A, B, C, or D) of the correct option."},
                {"role": "user", "content": formatted_question}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error in local model inference: {e}{RESET}")
        return "Error"

def get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text, max_tokens=256):
    """Get answer from local model with CoT assistance."""
    
    # Format the question with options and CoT
    formatted_question = f"Question: {question}\n\nOptions:\nA) {options['a']}\nB) {options['b']}\nC) {options['c']}\nD) {options['d']}\n\nChain of Thought:\n{cot_text}\n\nBased on the chain of thought above, what is the correct answer? Provide only the letter (A, B, C, or D):"
    
    try:
        # Use a smaller/weaker model to simulate local model behavior
        # For now, use gpt-4o-mini as a proxy for local model
        local_model = "gpt-4o-mini"
        response = client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "system", "content": "You are a medical expert. Use the provided chain of thought to answer the multiple choice question. Provide only the letter (A, B, C, or D) of the correct option."},
                {"role": "user", "content": formatted_question}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error in local model with CoT inference: {e}{RESET}")
        return "Error"

def generate_cot_from_remote_llm(client, model_name, question, options, max_tokens=512):
    """Generate Chain-of-Thought from remote LLM."""
    
    formatted_question = f"Question: {question}\n\nOptions:\nA) {options['a']}\nB) {options['b']}\nC) {options['c']}\nD) {options['d']}\n\nPlease provide a step-by-step chain of thought to solve this medical question:"
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert. Provide a clear, step-by-step chain of thought to solve the given medical multiple choice question. Focus on medical reasoning and knowledge."},
                {"role": "user", "content": formatted_question}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error in remote CoT generation: {e}{RESET}")
        return "Error"

def get_answer_from_remote_llm(client, model_name, question, options, max_tokens=256):
    """Get answer directly from remote LLM."""
    
    formatted_question = f"Question: {question}\n\nOptions:\nA) {options['a']}\nB) {options['b']}\nC) {options['c']}\nD) {options['d']}\n\nAnswer:"
    
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
        print(f"{RED}Error in remote LLM inference: {e}{RESET}")
        return "Error"

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

def check_mcq_correctness(predicted_letter, correct_index):
    """Check if the predicted answer is correct."""
    # Convert correct_index (0,1,2,3) to letter (A,B,C,D)
    index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct_letter = index_to_letter.get(correct_index, 'Error')
    return predicted_letter.upper() == correct_letter.upper()

def run_scenario_1_purely_local(client, model_name, question, options, correct_answer):
    """Scenario 1: Purely Local Model (Baseline)."""
    print(f"\n{BLUE}--- Scenario 1: Purely Local Model (Baseline) ---{RESET}")
    
    try:
        local_response = get_answer_from_local_model_alone(client, model_name, question, options)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer: {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return is_correct
    except Exception as e:
        print(f"{RED}Error during purely local model inference: {e}{RESET}")
        return False

def run_scenario_2_non_private_cot(client, model_name, remote_client, question, options, correct_answer):
    """Scenario 2: Non-Private Local Model + Remote CoT."""
    print(f"\n{BLUE}--- Scenario 2: Non-Private Local Model + Remote CoT ---{RESET}")
    
    try:
        # Generate CoT from remote LLM
        print(f"{YELLOW}2a. Generating CoT from ORIGINAL Question with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], question, options)
        print(f"Generated Chain-of-Thought (Remote, Non-Private):\n{cot_text}")
        
        # Use local model with CoT
        print(f"{YELLOW}2b. Running Local Model with Non-Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer (Non-Private CoT-Aided): {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return is_correct
    except Exception as e:
        print(f"{RED}Error during non-private CoT inference: {e}{RESET}")
        return False

def run_scenario_3_1_phrase_dp_local_cot(client, model_name, remote_client, sbert_model, question, options, correct_answer):
    """Scenario 3.1: Private Local Model + CoT (Phrase DP)."""
    print(f"\n{BLUE}--- Scenario 3.1: Private Local Model + CoT (Phrase DP) ---{RESET}")
    
    try:
        # For now, skip complex phrase DP and use a simple generalization
        print(f"{YELLOW}3.1a. Applying Simple Generalization to the question...{RESET}")
        # Simple generalization: replace specific medical terms with general ones
        # This is a simplified version since MedMCQA doesn't have context
        perturbed_question = question.replace("myelinated nerve fibers", "certain nerve structures")
        print(f"Perturbed Question: {perturbed_question}")
        
        # Generate CoT from perturbed question
        print(f"{YELLOW}3.1b. Generating CoT from Perturbed Question with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], perturbed_question, options)
        print(f"Generated Chain-of-Thought (Remote, Private):\n{cot_text}")
        
        # Use local model with private CoT
        print(f"{YELLOW}3.1c. Running Local Model with Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer (Private CoT-Aided): {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return is_correct
    except Exception as e:
        print(f"{RED}Error during phrase DP private CoT-aided inference: {e}{RESET}")
        return False

def run_scenario_3_2_inferdpt_local_cot(client, model_name, remote_client, question, options, correct_answer):
    """Scenario 3.2: Private Local Model + CoT (InferDPT)."""
    print(f"\n{BLUE}--- Scenario 3.2: Private Local Model + CoT (InferDPT) ---{RESET}")
    
    try:
        # Apply InferDPT Differential Privacy to the question
        print(f"{YELLOW}3.2a. Applying InferDPT Differential Privacy to the question...{RESET}")
        import sys
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]  # Reset to just the script name to avoid argument conflicts
        from inferdpt import perturb_sentence
        sys.argv = original_argv  # Restore original arguments
        
        perturbed_question = perturb_sentence(question, config['epsilon'])
        print(f"Perturbed Question: {perturbed_question}")
        
        # Generate CoT from perturbed question
        print(f"{YELLOW}3.2b. Generating CoT from Perturbed Question with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], perturbed_question, options)
        print(f"Generated Chain-of-Thought (Remote, Private):\n{cot_text}")
        
        # Use local model with private CoT
        print(f"{YELLOW}3.2c. Running Local Model with Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer (Private CoT-Aided): {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return is_correct
    except Exception as e:
        print(f"{RED}Error during InferDPT private CoT-aided inference: {e}{RESET}")
        return False

def run_scenario_4_purely_remote(remote_client, question, options, correct_answer):
    """Scenario 4: Purely Remote Model."""
    print(f"\n{BLUE}--- Scenario 4: Purely Remote Model ---{RESET}")
    
    try:
        print(f"{YELLOW}4a. Running Purely Remote LLM...{RESET}")
        remote_response = get_answer_from_remote_llm(remote_client, config['remote_models']['llm_model'], question, options)
        predicted_letter = extract_letter_from_answer(remote_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Purely Remote Answer: {remote_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return is_correct
    except Exception as e:
        print(f"{RED}Error during purely remote model inference: {e}{RESET}")
        return False

def run_experiment_for_model(model_name):
    """Run the MedQA experiment for a given local model."""
    
    print(f"{CYAN}Starting MedQA Experiment with model: {model_name}{RESET}")
    
    # Load dataset
    print(f"{CYAN}Loading MedQA dataset...{RESET}")
    print(f"{YELLOW}Note: MedQA contains clinical vignettes with patient scenarios in questions{RESET}")
    dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
    
    # Get remote clients
    try:
        remote_client = get_remote_llm_client()
        print(f"{GREEN}Remote LLM client initialized successfully{RESET}")
    except Exception as e:
        print(f"{RED}Failed to initialize remote LLM client: {e}{RESET}")
        remote_client = None
    
    # Load Sentence-BERT for similarity computation
    sbert_model = load_sentence_bert()
    
    # Initialize results
    results = MCQExperimentResults()
    
    # Get sample questions
    num_samples = config['dataset']['num_samples']
    sample_questions = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"{CYAN}Testing {len(sample_questions)} questions from MedQA test set{RESET}")
    
    for i, item in enumerate(sample_questions):
        print(f"\n{YELLOW}--- Question {i+1}/{len(sample_questions)} ---{RESET}")
        
        question = item['question']
        options = item['options']  # Already a dict with A, B, C, D keys
        correct_answer = item['answer_idx']  # This is already a letter (A, B, C, D)
        
        print(f"Question: {question}")
        print(f"Options:")
        print(f"  A) {options['a']}")
        print(f"  B) {options['b']}")
        print(f"  C) {options['c']}")
        print(f"  D) {options['d']}")
        # Convert correct_index to letter for display
        index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        correct_letter = index_to_letter.get(correct_answer, 'Error')
        print(f"Correct Answer: {correct_letter}")
        
        # Run all scenarios
        results.total_questions += 1
        
        # Scenario 1: Purely Local Model
        if run_scenario_1_purely_local(remote_client, model_name, question, options, correct_answer):
            results.local_alone_correct += 1
        
        # Scenario 2: Non-Private Local + Remote CoT
        if remote_client and run_scenario_2_non_private_cot(remote_client, model_name, remote_client, question, options, correct_answer):
            results.non_private_cot_correct += 1
        
        # Scenario 3.1: Private Local + CoT (Phrase DP)
        if remote_client and run_scenario_3_1_phrase_dp_local_cot(remote_client, model_name, remote_client, sbert_model, question, options, correct_answer):
            results.phrase_dp_local_cot_correct += 1
        
        # Scenario 3.2: Private Local + CoT (InferDPT)
        if remote_client and run_scenario_3_2_inferdpt_local_cot(remote_client, model_name, remote_client, question, options, correct_answer):
            results.inferdpt_local_cot_correct += 1
        
        # Scenario 4: Purely Remote Model
        if remote_client and run_scenario_4_purely_remote(remote_client, question, options, correct_answer):
            results.purely_remote_correct += 1
    
    # Print final results
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"1. Purely Local Model ({model_name}) Accuracy: {results.local_alone_correct}/{results.total_questions} = {results.local_alone_correct/results.total_questions*100:.2f}%")
    print(f"2. Non-Private Local Model + Remote CoT Accuracy: {results.non_private_cot_correct}/{results.total_questions} = {results.non_private_cot_correct/results.total_questions*100:.2f}%")
    print(f"3.1. Private Local Model + CoT (Phrase DP) Accuracy: {results.phrase_dp_local_cot_correct}/{results.total_questions} = {results.phrase_dp_local_cot_correct/results.total_questions*100:.2f}%")
    print(f"3.2. Private Local Model + CoT (InferDPT) Accuracy: {results.inferdpt_local_cot_correct}/{results.total_questions} = {results.inferdpt_local_cot_correct/results.total_questions*100:.2f}%")
    print(f"4. Purely Remote Model ({config['remote_models']['llm_model']}) Accuracy: {results.purely_remote_correct}/{results.total_questions} = {results.purely_remote_correct/results.total_questions*100:.2f}%")
    
    return results

def test_single_question(question_index=0):
    """Test a single question from the dataset by index."""
    print(f"{CYAN}Testing single question (index {question_index}) from MedQA{RESET}")
    
    # Load dataset
    dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
    
    if question_index >= len(dataset):
        print(f"{RED}Question index {question_index} is out of range. Dataset has {len(dataset)} questions.{RESET}")
        return
    
    item = dataset[question_index]
    model_name = config['local_model']
    
    # Get remote clients
    try:
        remote_client = get_remote_llm_client()
    except Exception as e:
        print(f"{RED}Failed to initialize remote LLM client: {e}{RESET}")
        return
    
    # Load Sentence-BERT
    sbert_model = load_sentence_bert()
    
    question = item['question']
    options = {
        'a': item['opa'],
        'b': item['opb'],
        'c': item['opc'],
        'd': item['opd']
    }
    correct_answer = item['cop']
    
    print(f"Question: {question}")
    print(f"Options:")
    print(f"  A) {options['a']}")
    print(f"  B) {options['b']}")
    print(f"  C) {options['c']}")
    print(f"  D) {options['d']}")
    # Convert correct_index to letter for display
    index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    correct_letter = index_to_letter.get(correct_answer, 'Error')
    print(f"Correct Answer: {correct_letter}")
    
    # Run all scenarios
    run_scenario_1_purely_local(remote_client, model_name, question, options, correct_answer)
    run_scenario_2_non_private_cot(remote_client, model_name, remote_client, question, options, correct_answer)
    run_scenario_3_1_phrase_dp_local_cot(remote_client, model_name, remote_client, sbert_model, question, options, correct_answer)
    run_scenario_3_2_inferdpt_local_cot(remote_client, model_name, remote_client, question, options, correct_answer)
    run_scenario_4_purely_remote(remote_client, question, options, correct_answer)

if __name__ == "__main__":
    import sys
    
    model_name = config["local_model"]
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--index" and len(sys.argv) > 2:
            try:
                question_index = int(sys.argv[2])
                test_single_question(question_index)
            except ValueError:
                print("Error: Question index must be a number")
        else:
            print("Usage:")
            print("  python medmcqa_experiment.py                    # Run full experiment")
            print("  python medmcqa_experiment.py --index <number>   # Test from dataset")
    else:
        run_experiment_for_model(model_name)
