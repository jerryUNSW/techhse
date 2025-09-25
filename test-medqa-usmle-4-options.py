#!/usr/bin/env python3
"""
MedQA Experiment Script
=======================

A script for experimenting with MedQA (Medical Question Answering) dataset
using privacy-preserving approaches similar to the multi-hop experiments.

This script tests different scenarios:
1. Purely Local Model (Baseline)
2. Non-Private Local Model + Remote CoT
3.1. Private Local Model + CoT (Phrase DP)
3.2. Private Local Model + CoT (InferDPT)
3.3. Private Local Model + CoT (SANTEXT+)
3.4. Private Local Model + CoT (CUSTEXT+)
3.5. Private Local Model + CoT (CluSanT)
4. Purely Remote Model

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
from santext_integration import create_santext_mechanism
from cus_text_ppi_protection_experiment import run_custext_ppi_protection
from sanitization_methods import (
    phrasedp_sanitize_text,
    inferdpt_sanitize_text,
    santext_sanitize_text,
    custext_sanitize_text,
    clusant_sanitize_text,
)

def _resolve_local_model_name_for_nebius(client, fallback_model_name: str) -> str:
    """Return local model name from YAML config for Nebius invocations."""
    return config.get('local_model', fallback_model_name)

def _find_working_nebius_model(client) -> str:
    """Probe Nebius with candidate models to find a working local model ID.

    Order: config['local_model'] first, then config['local_models'] list, then 'microsoft/phi-4'.
    Returns first model that successfully completes a 1-token ping.
    """
    candidates = []
    preferred = config.get('local_model')
    if preferred:
        candidates.append(preferred)
    candidates.extend(config.get('local_models', []))
    # Ensure a common fallback exists at the end
    if 'microsoft/phi-4' not in candidates:
        candidates.append('microsoft/phi-4')

    seen = set()
    ordered = [m for m in candidates if not (m in seen or seen.add(m))]
    for model in ordered:
        try:
            resp = create_completion_with_model_support(
                client,
                model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
            )
            # If no exception, model works
            return model
        except Exception:
            continue
    # If none worked, return preferred to surface original error upstream
    return preferred or 'microsoft/phi-4'

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

class MedQAExperimentResults:
    """Class to track experiment results for MedQA dataset."""
    
    def __init__(self):
        self.local_alone_correct = 0
        self.non_private_cot_correct = 0
        self.local_cot_correct = 0
        self.phrase_dp_local_cot_correct = 0
        self.phrase_dp_with_options_local_cot_correct = 0
        self.inferdpt_local_cot_correct = 0
        self.santext_local_cot_correct = 0
        self.custext_local_cot_correct = 0
        self.clusant_local_cot_correct = 0
        self.purely_remote_correct = 0
        self.total_questions = 0

def load_sentence_bert():
    """Load Sentence-BERT model for similarity computation."""
    print(f"{CYAN}Loading Sentence-BERT model...{RESET}")
    return SentenceTransformer('all-MiniLM-L6-v2')

def initialize_santext_mechanism():
    """Initialize SANTEXT+ mechanism once for all questions."""
    print(f"{CYAN}Initializing SANTEXT+ mechanism...{RESET}")
    santext_mechanism = create_santext_mechanism(epsilon=config['epsilon'], p=0.3)
    # Note: build_vocabulary() will be called with actual texts when needed
    # For now, we'll build it with a sample text or skip it
    print(f"{GREEN}SANTEXT+ mechanism initialized successfully{RESET}")
    return santext_mechanism

def initialize_custext_components():
    """Initialize CUSTEXT+ components once for all questions."""
    print(f"{CYAN}Initializing CUSTEXT+ components...{RESET}")
    # TODO: Load actual CUSTEXT+ components (emb_matrix, idx2word, word2idx, stop_set)
    # For now, return None values - these would need to be loaded from the PPI experiment
    custext_components = {
        'emb_matrix': None,
        'idx2word': None, 
        'word2idx': None,
        'stop_set': set()
    }
    print(f"{YELLOW}CUSTEXT+ components placeholder initialized (need actual loading){RESET}")
    return custext_components

def initialize_clusant_mechanism():
    """Initialize CluSanT mechanism once for all questions."""
    print(f"{CYAN}Initializing CluSanT mechanism...{RESET}")
    # TODO: Load actual CluSanT model and components
    # For now, return None - this would need to be loaded from the CluSanT source
    clusant_mechanism = None
    print(f"{YELLOW}CluSanT mechanism placeholder initialized (need actual loading){RESET}")
    return clusant_mechanism

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

def create_completion_with_model_support(client, model_name, messages, max_tokens=256, temperature=0.0):
    """
    Create a chat completion with proper parameter support for different models.
    GPT-5 uses max_completion_tokens and doesn't support temperature=0.0, others use max_tokens.
    """
    if "gpt-5" in model_name or "gpt-5-chat-latest" in model_name:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens
            # GPT-5 doesn't support temperature parameter
        )
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

def format_question_with_options(question, options):
    """Format question with options for LLM input."""
    formatted = f"{question}\n\nOptions:\n"
    for key, value in options.items():
        formatted += f"{key}) {value}\n"
    formatted += "\nAnswer:"
    return formatted

def get_answer_from_local_model_alone(client, model_name, question, options, max_tokens=256):
    """Get answer from local model without any CoT assistance."""
    
    formatted_question = format_question_with_options(question, options)
    
    try:
        # Resolve Nebius-valid model; probe fallback if needed
        local_model = _find_working_nebius_model(client)
        response = create_completion_with_model_support(
            client, local_model,
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
    
    formatted_question = format_question_with_options(question, options)
    full_prompt = f"{formatted_question}\n\nChain of Thought:\n{cot_text}\n\nBased on the chain of thought above, what is the correct answer? Provide only the letter (A, B, C, or D):"
    
    try:
        # Resolve Nebius-valid model; probe fallback if needed
        local_model = _find_working_nebius_model(client)
        response = create_completion_with_model_support(
            client, local_model,
            messages=[
                {"role": "system", "content": "You are a medical expert. Use the provided chain of thought to answer the multiple choice question. Provide only the letter (A, B, C, or D) of the correct option."},
                {"role": "user", "content": full_prompt}
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
    
    formatted_question = format_question_with_options(question, options)
    prompt = f"{formatted_question}\n\nPlease provide a step-by-step chain of thought to solve this medical question:"
    
    try:
        response = create_completion_with_model_support(
            client, model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert. Provide a clear, step-by-step chain of thought to solve the given medical multiple choice question. Focus on medical reasoning and knowledge."},
                {"role": "user", "content": prompt}
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
    
    formatted_question = format_question_with_options(question, options)
    
    try:
        response = create_completion_with_model_support(
            client, model_name,
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

def check_mcq_correctness(predicted_letter, correct_letter):
    """Check if the predicted answer is correct."""
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
        # Initialize Nebius client for DP perturbation via unified helper
        from sanitization_methods import config as sm_config
        nebius_client = utils.get_nebius_client()
        nebius_model_name = sm_config.get('local_model', 'microsoft/phi-4')

        # Apply Phrase-Level Differential Privacy to the question (wrapped)
        print(f"{YELLOW}3.1a. Applying PhraseDP sanitization...{RESET}")
        perturbed_question = phrasedp_sanitize_text(
            question,
            epsilon=config['epsilon'],
            nebius_client=nebius_client,
            nebius_model_name=nebius_model_name,
        )
        print(f"Perturbed Question: {perturbed_question}")
        
        # Keep options unchanged - only perturb the question for privacy
        print(f"{YELLOW}3.1b. Keeping options unchanged for local model...{RESET}")
        print(f"Original Options: {options}")
        
        # Generate CoT from perturbed question with original options
        print(f"{YELLOW}3.1c. Generating CoT from Perturbed Question with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], perturbed_question, options)
        print(f"Generated Chain-of-Thought (Remote, Private):\n{cot_text}")
        
        # Use local model with private CoT
        print(f"{YELLOW}3.1d. Running Local Model with Private CoT...{RESET}")
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

def run_scenario_3_1_2_phrase_dp_with_options_local_cot(client, model_name, remote_client, sbert_model, question, options, correct_answer):
    """Scenario 3.1.2: Private Local Model + CoT (Phrase DP with Perturbed Options)."""
    print(f"\n{BLUE}--- Scenario 3.1.2: Private Local Model + CoT (Phrase DP with Perturbed Options) ---{RESET}")

    try:
        # Initialize Nebius client for DP perturbation via unified helper
        from sanitization_methods import config as sm_config
        nebius_client = utils.get_nebius_client()
        nebius_model_name = sm_config.get('local_model', 'microsoft/phi-4')

        # Apply Phrase-Level Differential Privacy to the question
        print(f"{YELLOW}3.1.2a. Applying PhraseDP sanitization to question...{RESET}")
        perturbed_question = phrasedp_sanitize_text(
            question,
            epsilon=config['epsilon'],
            nebius_client=nebius_client,
            nebius_model_name=nebius_model_name,
        )
        print(f"Perturbed Question: {perturbed_question}")

        # Apply Phrase-Level Differential Privacy to each option
        print(f"{YELLOW}3.1.2b. Applying PhraseDP sanitization to each option...{RESET}")
        perturbed_options = {}
        for key, value in options.items():
            perturbed_option = phrasedp_sanitize_text(
                value,
                epsilon=config['epsilon'],
                nebius_client=nebius_client,
                nebius_model_name=nebius_model_name,
            )
            perturbed_options[key] = perturbed_option
            print(f"Option {key} - Original: {value}")
            print(f"Option {key} - Perturbed: {perturbed_option}")

        # Generate CoT from perturbed question with perturbed options
        print(f"{YELLOW}3.1.2c. Generating CoT from Perturbed Question AND Options with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], perturbed_question, perturbed_options)
        print(f"Generated Chain-of-Thought (Remote, Fully Private):\n{cot_text}")

        # Use local model with original question/options but private CoT
        print(f"{YELLOW}3.1.2d. Running Local Model with original question/options and Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)

        print(f"Local Answer (Fully Private CoT-Aided): {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")

        return is_correct
    except Exception as e:
        print(f"{RED}Error during phrase DP with options private CoT-aided inference: {e}{RESET}")
        return False

def run_scenario_3_2_inferdpt_local_cot(client, model_name, remote_client, question, options, correct_answer):
    """Scenario 3.2: Private Local Model + CoT (InferDPT)."""
    print(f"\n{BLUE}--- Scenario 3.2: Private Local Model + CoT (InferDPT) ---{RESET}")
    
    try:
        # Apply InferDPT sanitization (wrapped)
        print(f"{YELLOW}3.2a. Applying InferDPT sanitization...{RESET}")
        perturbed_question = inferdpt_sanitize_text(question, epsilon=config['epsilon'])
        print(f"Perturbed Question: {perturbed_question}")
        
        # Keep options unchanged - only perturb the question for privacy
        print(f"{YELLOW}3.2b. Keeping options unchanged for local model...{RESET}")
        print(f"Original Options: {options}")
        
        # Generate CoT from perturbed question with original options
        print(f"{YELLOW}3.2c. Generating CoT from Perturbed Question with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], perturbed_question, options)
        print(f"Generated Chain-of-Thought (Remote, Private):\n{cot_text}")
        
        # Use local model with private CoT
        print(f"{YELLOW}3.2d. Running Local Model with Private CoT...{RESET}")
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

def run_scenario_3_3_santext_local_cot(client, model_name, remote_client, question, options, correct_answer, santext_mechanism):
    """Scenario 3.3: Private Local Model + CoT (SANTEXT+)."""
    print(f"\n{BLUE}--- Scenario 3.3: Private Local Model + CoT (SANTEXT+) ---{RESET}")
    
    try:
        print(f"{YELLOW}3.3a. Applying SANTEXT+ sanitization...{RESET}")
        # Use unified wrapper (handles vocabulary caching internally)
        sanitized_question = santext_sanitize_text(question, epsilon=config['epsilon'])
        print(f"Sanitized Question: {sanitized_question}")
        
        print(f"{YELLOW}3.3b. Getting CoT from Remote LLM...{RESET}")
        cot_response = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], sanitized_question, options)
        print(f"Remote CoT: {cot_response}")

        print(f"{YELLOW}3.3c. Running Local Model with CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(client, model_name, sanitized_question, options, cot_response)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer with SANTEXT+ CoT: {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return is_correct
    except Exception as e:
        print(f"{RED}Error in SANTEXT+ scenario: {e}{RESET}")
        return False

def run_scenario_3_4_custext_local_cot(client, model_name, remote_client, question, options, correct_answer, custext_components):
    """Scenario 3.4: Private Local Model + CoT (CUSTEXT+)."""
    print(f"\n{BLUE}--- Scenario 3.4: Private Local Model + CoT (CUSTEXT+) ---{RESET}")
    
    try:
        print(f"{YELLOW}3.4a. Applying CUSTEXT+ sanitization...{RESET}")
        sanitized_question = custext_sanitize_text(
            question,
            epsilon=config['epsilon'],
            top_k=5,
        )
        print(f"Sanitized Question: {sanitized_question}")
        
        print(f"{YELLOW}3.4b. Getting CoT from Remote LLM...{RESET}")
        cot_response = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], sanitized_question, options)
        print(f"Remote CoT: {cot_response}")

        print(f"{YELLOW}3.4c. Running Local Model with CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(client, model_name, sanitized_question, options, cot_response)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer with CUSTEXT+ CoT: {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return is_correct
    except Exception as e:
        print(f"{RED}Error in CUSTEXT+ scenario: {e}{RESET}")
        return False

def run_scenario_3_5_clusant_local_cot(client, model_name, remote_client, question, options, correct_answer, clusant_mechanism):
    """Scenario 3.5: Private Local Model + CoT (CluSanT)."""
    print(f"\n{BLUE}--- Scenario 3.5: Private Local Model + CoT (CluSanT) ---{RESET}")
    
    try:
        print(f"{YELLOW}3.5a. Applying CluSanT sanitization...{RESET}")
        sanitized_question = clusant_sanitize_text(question, epsilon=config['epsilon'])
        print(f"Sanitized Question: {sanitized_question}")
        
        print(f"{YELLOW}3.5b. Getting CoT from Remote LLM...{RESET}")
        cot_response = get_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], sanitized_question)
        print(f"Remote CoT: {cot_response}")
        
        print(f"{YELLOW}3.5c. Running Local Model with CoT...{RESET}")
        local_response = get_answer_from_local_llm_with_cot(client, model_name, sanitized_question, options, cot_response)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer with CluSanT CoT: {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return is_correct
    except Exception as e:
        print(f"{RED}Error in CluSanT scenario: {e}{RESET}")
        return False

def run_experiment_for_model(model_name):
    """Run the MedQA experiment for a given local model."""
    
    print(f"{CYAN}Starting MedQA Experiment with model: {model_name}{RESET}")
    
    # Load dataset
    print(f"{CYAN}Loading MedQA dataset...{RESET}")
    print(f"{YELLOW}Note: MedQA contains clinical vignettes with patient scenarios in questions{RESET}")
    dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
    
    # Get clients
    try:
        remote_client = get_remote_llm_client()
        print(f"{GREEN}Remote LLM client initialized successfully{RESET}")
    except Exception as e:
        print(f"{RED}Failed to initialize remote LLM client: {e}{RESET}")
        remote_client = None
    # Initialize Nebius local client for local model invocations
    try:
        local_client = utils.get_nebius_client()
        print(f"{GREEN}Local (Nebius) client initialized successfully{RESET}")
    except Exception as e:
        print(f"{RED}Failed to initialize Nebius local client: {e}{RESET}")
        local_client = None
    
    # Load Sentence-BERT for similarity computation
    sbert_model = load_sentence_bert()
    
    # Initialize privacy mechanisms once for all questions
    print(f"{CYAN}Initializing privacy mechanisms...{RESET}")
    santext_mechanism = initialize_santext_mechanism()
    custext_components = initialize_custext_components()
    clusant_mechanism = initialize_clusant_mechanism()
    
    # Initialize results
    results = MedQAExperimentResults()
    
    # Get sample questions - test 500 questions starting from 51st (instances 51-550)
    start_index = 50  # Start from question 51 (0-indexed)
    num_samples = 10  # Test 10 questions
    sample_questions = dataset.select(range(start_index, start_index + num_samples))
    
    print(f"{CYAN}Testing {len(sample_questions)} questions from MedQA test set{RESET}")
    
    for i, item in enumerate(sample_questions):
        print(f"\n{YELLOW}--- Question {i+1}/{len(sample_questions)} ---{RESET}")
        
        question = item['question']
        options = item['options']  # Already a dict with A, B, C, D keys
        correct_answer = item['answer_idx']  # This is already a letter (A, B, C, D)
        
        print(f"Question: {question}")
        print(f"Options:")
        for key, value in options.items():
            print(f"  {key}) {value}")
        print(f"Correct Answer: {correct_answer}")
        
        # Run all scenarios
        results.total_questions += 1
        
        # Scenario 1: Purely Local Model
        if local_client and run_scenario_1_purely_local(local_client, model_name, question, options, correct_answer):
            results.local_alone_correct += 1
        
        # Scenario 2: Non-Private Local + Remote CoT
        if remote_client and run_scenario_2_non_private_cot(local_client or remote_client, model_name, remote_client, question, options, correct_answer):
            results.non_private_cot_correct += 1
        
        # Scenario 3.1: Private Local + CoT (Phrase DP)
        if remote_client and local_client and run_scenario_3_1_phrase_dp_local_cot(local_client, model_name, remote_client, sbert_model, question, options, correct_answer):
            results.phrase_dp_local_cot_correct += 1

        # Scenario 3.1.2: Private Local + CoT (Phrase DP with Perturbed Options)
        if remote_client and local_client and run_scenario_3_1_2_phrase_dp_with_options_local_cot(local_client, model_name, remote_client, sbert_model, question, options, correct_answer):
            results.phrase_dp_with_options_local_cot_correct += 1

        # Scenario 3.2: Private Local + CoT (InferDPT)
        if remote_client and local_client and run_scenario_3_2_inferdpt_local_cot(local_client, model_name, remote_client, question, options, correct_answer):
            results.inferdpt_local_cot_correct += 1
        
        # Scenario 3.3: Private Local + CoT (SANTEXT+)
        if remote_client and local_client and run_scenario_3_3_santext_local_cot(local_client, model_name, remote_client, question, options, correct_answer, santext_mechanism):
            results.santext_local_cot_correct += 1
        
        # Scenario 3.4: Private Local + CoT (CUSTEXT+)
        if remote_client and local_client and run_scenario_3_4_custext_local_cot(local_client, model_name, remote_client, question, options, correct_answer, custext_components):
            results.custext_local_cot_correct += 1
        
        # Scenario 3.5: Private Local + CoT (CluSanT)
        # Temporarily disabled per request
        # if remote_client and run_scenario_3_5_clusant_local_cot(remote_client, model_name, remote_client, question, options, correct_answer, clusant_mechanism):
        #     results.clusant_local_cot_correct += 1
        
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
    print(f"3.1.2. Private Local Model + CoT (Phrase DP with Perturbed Options) Accuracy: {results.phrase_dp_with_options_local_cot_correct}/{results.total_questions} = {results.phrase_dp_with_options_local_cot_correct/results.total_questions*100:.2f}%")
    print(f"3.2. Private Local Model + CoT (InferDPT) Accuracy: {results.inferdpt_local_cot_correct}/{results.total_questions} = {results.inferdpt_local_cot_correct/results.total_questions*100:.2f}%")
    print(f"3.3. Private Local Model + CoT (SANTEXT+) Accuracy: {results.santext_local_cot_correct}/{results.total_questions} = {results.santext_local_cot_correct/results.total_questions*100:.2f}%")
    print(f"3.4. Private Local Model + CoT (CUSTEXT+) Accuracy: {results.custext_local_cot_correct}/{results.total_questions} = {results.custext_local_cot_correct/results.total_questions*100:.2f}%")
    # 3.5 disabled for now
    # print(f"3.5. Private Local Model + CoT (CluSanT) Accuracy: {results.clusant_local_cot_correct}/{results.total_questions} = {results.clusant_local_cot_correct/results.total_questions*100:.2f}%")
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
    
    # Get remote and local clients
    try:
        remote_client = get_remote_llm_client()
        print(f"{GREEN}Remote LLM client initialized successfully{RESET}")
    except Exception as e:
        print(f"{RED}Failed to initialize remote LLM client: {e}{RESET}")
        remote_client = None

    # Initialize Nebius local client for local model invocations
    try:
        local_client = utils.get_nebius_client()
        print(f"{GREEN}Local (Nebius) client initialized successfully{RESET}")
    except Exception as e:
        print(f"{RED}Failed to initialize Nebius local client: {e}{RESET}")
        local_client = None

    if not remote_client:
        print(f"{RED}Cannot run tests without remote client{RESET}")
        return
    
    # Load Sentence-BERT
    sbert_model = load_sentence_bert()
    
    # Initialize privacy mechanisms once for this test
    print(f"{CYAN}Initializing privacy mechanisms for single question test...{RESET}")
    # Skip initializing unused mechanisms to speed up testing
    santext_mechanism = None  # initialize_santext_mechanism()
    custext_components = None  # initialize_custext_components()
    clusant_mechanism = None   # initialize_clusant_mechanism()
    
    question = item['question']
    options = item['options']
    correct_answer = item['answer_idx']
    
    print(f"Question: {question}")
    print(f"Options:")
    for key, value in options.items():
        print(f"  {key}) {value}")
    print(f"Correct Answer: {correct_answer}")
    
    # Run all scenarios
    # if local_client:
    #     run_scenario_1_purely_local(local_client, model_name, question, options, correct_answer)

    if remote_client and local_client:
        # run_scenario_2_non_private_cot(local_client, model_name, remote_client, question, options, correct_answer)
        # run_scenario_3_1_phrase_dp_local_cot(local_client, model_name, remote_client, sbert_model, question, options, correct_answer)
        run_scenario_3_1_2_phrase_dp_with_options_local_cot(local_client, model_name, remote_client, sbert_model, question, options, correct_answer)
        # run_scenario_3_2_inferdpt_local_cot(local_client, model_name, remote_client, question, options, correct_answer)
        # run_scenario_3_3_santext_local_cot(local_client, model_name, remote_client, question, options, correct_answer, santext_mechanism)
        # run_scenario_3_4_custext_local_cot(local_client, model_name, remote_client, question, options, correct_answer, custext_components)
        # run_scenario_3_5_clusant_local_cot(local_client, model_name, remote_client, question, options, correct_answer, clusant_mechanism)

    # if remote_client:
    #     run_scenario_4_purely_remote(remote_client, question, options, correct_answer)

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
            print("  python medqa_experiment.py                    # Run full experiment")
            print("  python medqa_experiment.py --index <number>   # Test from dataset")
    else:
        run_experiment_for_model(model_name)
