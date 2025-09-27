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
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import random
from datasets import load_dataset

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

    Order: config['local_model'] first, then config['local_models'] list.
    Returns first model that successfully completes a 1-token ping.
    """
    candidates = []
    preferred = config.get('local_model')
    if preferred:
        candidates.append(preferred)
    candidates.extend(config.get('local_models', []))

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
    # If none worked, raise an error
    raise ValueError(f"None of the configured models worked: {candidates}")

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
        self.old_phrase_dp_local_cot_correct = 0
        self.phrase_dp_local_cot_correct = 0
        self.phrase_dp_with_options_local_cot_correct = 0
        self.phrase_dp_batch_options_local_cot_correct = 0
        self.inferdpt_local_cot_correct = 0
        self.inferdpt_batch_options_local_cot_correct = 0
        self.santext_local_cot_correct = 0
        self.santext_batch_options_local_cot_correct = 0
        self.custext_local_cot_correct = 0
        self.custext_batch_options_local_cot_correct = 0
        self.clusant_local_cot_correct = 0
        self.clusant_batch_options_local_cot_correct = 0
        self.purely_remote_correct = 0
        self.total_questions = 0

def load_sentence_bert():
    """Load Sentence-BERT model for similarity computation."""
    print(f"{CYAN}Loading Sentence-BERT model...{RESET}")
    return SentenceTransformer('all-MiniLM-L6-v2')

def initialize_santext_mechanism():
    """Initialize SANTEXT+ mechanism once for all questions."""
    print(f"{CYAN}Initializing SANTEXT+ mechanism...{RESET}")
    try:
        santext_sanitize_text("Warm-up text for SANTEXT+", epsilon=config['epsilon'])
        print(f"{GREEN}SANTEXT+ mechanism initialized successfully{RESET}")
    except Exception as exc:
        print(f"{YELLOW}SANTEXT+ warm-up encountered an issue but will proceed lazily: {exc}{RESET}")

def initialize_custext_components():
    """Initialize CUSTEXT+ components once for all questions."""
    print(f"{CYAN}Initializing CUSTEXT+ components...{RESET}")
    try:
        custext_sanitize_text("Warm-up text for CUSTEXT+", epsilon=config['epsilon'])
        print(f"{GREEN}CUSTEXT+ components initialized successfully{RESET}")
    except Exception as exc:
        print(f"{YELLOW}CUSTEXT+ warm-up encountered an issue but will proceed lazily: {exc}{RESET}")

def initialize_clusant_mechanism():
    """Initialize CluSanT mechanism once for all questions."""
    print(f"{CYAN}Initializing CluSanT mechanism...{RESET}")
    try:
        clusant_sanitize_text("Warm-up text for CluSanT", epsilon=config['epsilon'])
        print(f"{GREEN}CluSanT mechanism initialized successfully{RESET}")
    except Exception as exc:
        print(f"{YELLOW}CluSanT warm-up encountered an issue but will proceed lazily: {exc}{RESET}")

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

def check_quota_error(exception):
    """Check if the exception is due to quota/rate limit issues."""
    error_message = str(exception).lower()
    quota_indicators = [
        'quota',
        'rate limit',
        '429',
        'exceeded',
        'insufficient',
        'limit reached',
        'too many requests'
    ]
    return any(indicator in error_message for indicator in quota_indicators)

def abort_on_quota_error(exception, api_type="API"):
    """Abort the program if quota error is detected."""
    if check_quota_error(exception):
        print(f"\n{RED}{'='*60}{RESET}")
        print(f"{RED}QUOTA ERROR DETECTED{RESET}")
        print(f"{RED}{'='*60}{RESET}")
        print(f"{RED}Error: {exception}{RESET}")
        print(f"{RED}The {api_type} quota has been exceeded.{RESET}")
        print(f"{RED}Aborting the entire program to prevent further API calls.{RESET}")
        print(f"{RED}Please check your API quotas and try again later.{RESET}")
        print(f"{RED}{'='*60}{RESET}")
        import sys
        sys.exit(1)

def create_completion_with_model_support(client, model_name, messages, max_tokens=256, temperature=0.0):
    """
    Create a chat completion with proper parameter support for different models.
    GPT-5 uses max_completion_tokens and doesn't support temperature=0.0, others use max_tokens.
    """
    try:
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
    except Exception as e:
        abort_on_quota_error(e, "API")
        raise e

def format_question_with_options(question, options=None):
    """Format question (optionally with answer choices) for LLM input."""
    formatted = f"{question}"
    if options:
        formatted += "\n\nOptions:\n"
        for key, value in options.items():
            formatted += f"{key}) {value}\n"
    formatted += "\n\nAnswer:"
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

def generate_cot_from_remote_llm(client, model_name, question, options=None, max_tokens=512):
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

def generate_cot_from_remote_llm_with_perturbed_options(client, model_name, perturbed_question, perturbed_options_text, max_tokens=512):
    """Generate Chain-of-Thought from remote LLM using both perturbed question and perturbed options."""
    
    # Format the input with clear separation between perturbed question and options
    prompt = f"""Perturbed Question: {perturbed_question}

Perturbed Options: {perturbed_options_text}

Please provide step-by-step medical reasoning guidance to help analyze this question. This guidance will be fed back to a local model to guide its decision-making process. Focus on:
1. Key medical concepts and diagnostic frameworks
2. Clinical reasoning principles
3. Differential diagnosis approaches
4. Important considerations for each option

Do NOT provide a final answer (A, B, C, or D). Instead, give reasoning guidance that will help the local model make the decision."""
    
    try:
        response = create_completion_with_model_support(
            client, model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert providing reasoning guidance. Your output will be fed back to a local model to guide its decision-making process. Provide step-by-step medical reasoning principles, diagnostic frameworks, and clinical considerations. Do NOT solve the problem or choose an answer - only provide reasoning guidance that the local model can use to make the final decision."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error in remote CoT generation with perturbed options: {e}{RESET}")
        return "Error"

def generate_cot_from_remote_llm_with_perturbed_options_dict(client, model_name, perturbed_question, perturbed_options_dict, max_tokens=512):
    """Generate Chain-of-Thought from remote LLM using both perturbed question and perturbed options (dictionary format)."""
    
    # Format the perturbed options from dictionary
    options_text = ""
    for key, value in perturbed_options_dict.items():
        options_text += f"{key}) {value}\n"
    
    # Format the input with clear separation between perturbed question and options
    prompt = f"""Perturbed Question: {perturbed_question}

Perturbed Options:
{options_text}
Please provide step-by-step medical reasoning guidance to help analyze this question. This guidance will be fed back to a local model to guide its decision-making process. Focus on:
1. Key medical concepts and diagnostic frameworks
2. Clinical reasoning principles
3. Differential diagnosis approaches
4. Important considerations for each option

Do NOT provide a final answer (A, B, C, or D). Instead, give reasoning guidance that will help the local model make the decision."""
    
    try:
        response = create_completion_with_model_support(
            client, model_name,
            messages=[
                {"role": "system", "content": "You are a medical expert providing reasoning guidance. Your output will be fed back to a local model to guide its decision-making process. Provide step-by-step medical reasoning principles, diagnostic frameworks, and clinical considerations. Do NOT solve the problem or choose an answer - only provide reasoning guidance that the local model can use to make the final decision."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error in remote CoT generation with perturbed options dict: {e}{RESET}")
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
        abort_on_quota_error(e, "Nebius")
        print(f"{RED}Error during purely local model inference: {e}{RESET}")
        return False

def run_scenario_2_non_private_cot(client, model_name, remote_client, question, options, correct_answer):
    """Scenario 2: Non-Private Local Model + Remote CoT."""
    print(f"\n{BLUE}--- Scenario 2: Non-Private Local Model + Remote CoT ---{RESET}")
    
    try:
        # Generate CoT from remote LLM
        print(f"{YELLOW}2a. Generating CoT from ORIGINAL Question with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], question)
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
        abort_on_quota_error(e, "API")
        print(f"{RED}Error during non-private CoT inference: {e}{RESET}")
        return False

# Removed outdated function - now using run_scenario_3_private_local_cot('phrasedp', use_old_phrasedp=True)

def run_scenario_3_private_local_cot(client, model_name, remote_client, sbert_model, question, options, correct_answer, privacy_mechanism, use_old_phrasedp=False):
    """Scenario 3: Private Local Model + CoT (Generic function for all privacy mechanisms without batch options)."""
    mechanism_names = {
        'phrasedp': 'Phrase DP',
        'inferdpt': 'InferDPT', 
        'santext': 'SANTEXT+',
        'custext': 'CUSTEXT+',
        'clusant': 'CluSanT'
    }
    
    mechanism_name = mechanism_names.get(privacy_mechanism, privacy_mechanism.upper())
    if privacy_mechanism == 'phrasedp' and use_old_phrasedp:
        mechanism_name = 'Phrase DP (Old)'
    elif privacy_mechanism == 'phrasedp' and not use_old_phrasedp:
        mechanism_name = 'Phrase DP (New)'
    
    print(f"\n{BLUE}--- Scenario 3: Private Local Model + CoT ({mechanism_name}) ---{RESET}")
    
    try:
        # Apply privacy mechanism to the question
        print(f"{YELLOW}3a. Applying {mechanism_name} sanitization...{RESET}")
        
        if privacy_mechanism == 'phrasedp':
            if use_old_phrasedp:
                # Use old PhraseDP (single API call, no band diversity)
                from sanitization_methods import config as sm_config
                nebius_client = utils.get_nebius_client()
                nebius_model_name = sm_config.get('local_model')
                perturbed_question = utils.phrase_DP_perturbation_old(
                    nebius_client=nebius_client,
                    nebius_model_name=nebius_model_name,
                    input_sentence=question,
                    epsilon=config['epsilon'],
                    sbert_model=sbert_model
                )
            else:
                # Use new PhraseDP (10 API calls, 10-band diversity)
                from sanitization_methods import config as sm_config
                nebius_client = utils.get_nebius_client()
                nebius_model_name = sm_config.get('local_model')
                perturbed_question = phrasedp_sanitize_text(
                    question,
                    epsilon=config['epsilon'],
                    nebius_client=nebius_client,
                    nebius_model_name=nebius_model_name,
                )
        elif privacy_mechanism == 'inferdpt':
            perturbed_question = inferdpt_sanitize_text(question, epsilon=config['epsilon'])
        elif privacy_mechanism == 'santext':
            perturbed_question = santext_sanitize_text(question, epsilon=config['epsilon'])
        elif privacy_mechanism == 'custext':
            perturbed_question = custext_sanitize_text(question, epsilon=config['epsilon'])
        elif privacy_mechanism == 'clusant':
            perturbed_question = clusant_sanitize_text(question, epsilon=config['epsilon'])
        else:
            raise ValueError(f"Unknown privacy mechanism: {privacy_mechanism}")
            
        print(f"Perturbed Question: {perturbed_question}")
        
        # Keep options unchanged - only perturb the question for privacy
        print(f"{YELLOW}3b. Keeping options unchanged for local model...{RESET}")
        print(f"Original Options: {options}")
        
        # Generate CoT from perturbed question with original options
        print(f"{YELLOW}3c. Generating CoT from Perturbed Question with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, config['remote_models']['cot_model'], perturbed_question)
        print(f"Generated Chain-of-Thought (Remote, Private):\n{cot_text}")
        
        # Use local model with private CoT
        print(f"{YELLOW}3d. Running Local Model with Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer (Private CoT-Aided): {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return is_correct
    except Exception as e:
        print(f"{RED}Error during {mechanism_name} private CoT-aided inference: {e}{RESET}")
        return False

# def run_scenario_3_1_2_phrase_dp_with_options_local_cot(client, model_name, remote_client, sbert_model, question, options, correct_answer):
#     """Scenario 3.1.2: Private Local Model + CoT (Phrase DP with Perturbed Options)."""
#     print(f"\n{BLUE}--- Scenario 3.1.2: Private Local Model + CoT (Phrase DP with Perturbed Options) ---{RESET}")

def batch_perturb_options_with_phrasedp(options, epsilon, nebius_client, nebius_model_name):
    print(f"{CYAN}Batch Perturbation Starting:{RESET}")
    print(f"Options Count: {len(options)}")
    print(f"Epsilon: {epsilon}")
    """Batch perturb all options together - return single perturbed text for CoT generation."""
    print(f"{CYAN}Original Options:{RESET}")
    for key, value in options.items():
        print(f"  {key}) {value}")

    # Simple concatenation without semicolon enforcement
    combined_text = " ".join(options.values())

    print(f"{CYAN}Combined Text:{RESET}\n{combined_text}")

    # Apply PhraseDP to the combined text
    print(f"{YELLOW}Applying PhraseDP to combined text...{RESET}")
    perturbed_combined = phrasedp_sanitize_text(
        combined_text,
        epsilon=epsilon,
        nebius_client=nebius_client,
        nebius_model_name=nebius_model_name,
    )

    print(f"{CYAN}Perturbed Combined Text:{RESET}\n{perturbed_combined}")
    print(f"{GREEN}Batch perturbation complete - returning single perturbed text for CoT{RESET}")

    # Return the single perturbed text - no need to parse back to individual options
    return perturbed_combined

def batch_perturb_options_with_old_phrasedp(options, epsilon, nebius_client, nebius_model_name, sbert_model):
    """Batch perturb all options together using OLD PhraseDP (single API call, no band diversity)."""
    print(f"{CYAN}OLD Batch Perturbation Starting:{RESET}")
    print(f"Options Count: {len(options)}")
    print(f"Epsilon: {epsilon}")
    print(f"Model: {nebius_model_name}")
    print(f"{CYAN}Original Options:{RESET}")
    for key, value in options.items():
        print(f"  {key}) {value}")

    # Simple concatenation without semicolon enforcement
    combined_text = " ".join(options.values())

    print(f"{CYAN}Combined Text:{RESET}\n{combined_text}")

    # Apply OLD PhraseDP to the combined text
    print(f"{YELLOW}Applying OLD PhraseDP to combined text...{RESET}")
    perturbed_combined = utils.phrase_DP_perturbation_old(
        nebius_client=nebius_client,
        nebius_model_name=nebius_model_name,
        input_sentence=combined_text,
        epsilon=epsilon,
        sbert_model=sbert_model
    )

    print(f"{CYAN}Perturbed Combined Text (OLD PhraseDP):{RESET}\n{perturbed_combined}")
    print(f"{GREEN}OLD Batch perturbation complete - returning single perturbed text for CoT{RESET}")

    # Return the single perturbed text - no need to parse back to individual options
    return perturbed_combined

def run_scenario_3_private_local_cot_with_batch_options(client, model_name, remote_client, sbert_model, question, options, correct_answer, privacy_mechanism, use_old_phrasedp=False):
    """Scenario 3: Private Local Model + CoT with Batch Options (Generic function for all privacy mechanisms with batch options)."""
    mechanism_names = {
        'phrasedp': 'Phrase DP',
        'inferdpt': 'InferDPT', 
        'santext': 'SANTEXT+',
        'custext': 'CUSTEXT+',
        'clusant': 'CluSanT'
    }
    
    mechanism_name = mechanism_names.get(privacy_mechanism, privacy_mechanism.upper())
    if privacy_mechanism == 'phrasedp' and use_old_phrasedp:
        mechanism_name = 'Phrase DP (Old)'
    elif privacy_mechanism == 'phrasedp' and not use_old_phrasedp:
        mechanism_name = 'Phrase DP (New)'
    
    print(f"\n{BLUE}--- Scenario 3: Private Local Model + CoT ({mechanism_name} with Batch Options) ---{RESET}")

    try:
        # Apply privacy mechanism to the question
        print(f"{YELLOW}3a. Applying {mechanism_name} sanitization to question...{RESET}")
        
        if privacy_mechanism == 'phrasedp':
            if use_old_phrasedp:
                # Use old PhraseDP (single API call, no band diversity)
                from sanitization_methods import config as sm_config
                nebius_client = utils.get_nebius_client()
                nebius_model_name = sm_config.get('local_model')
                perturbed_question = utils.phrase_DP_perturbation_old(
                    nebius_client=nebius_client,
                    nebius_model_name=nebius_model_name,
                    input_sentence=question,
                    epsilon=config['epsilon'],
                    sbert_model=sbert_model
                )
            else:
                # Use new PhraseDP (10 API calls, 10-band diversity)
                from sanitization_methods import config as sm_config
                nebius_client = utils.get_nebius_client()
                nebius_model_name = sm_config.get('local_model')
                perturbed_question = phrasedp_sanitize_text(
                    question,
                    epsilon=config['epsilon'],
                    nebius_client=nebius_client,
                    nebius_model_name=nebius_model_name,
                )
        elif privacy_mechanism == 'inferdpt':
            perturbed_question = inferdpt_sanitize_text(question, epsilon=config['epsilon'])
        elif privacy_mechanism == 'santext':
            perturbed_question = santext_sanitize_text(question, epsilon=config['epsilon'])
        elif privacy_mechanism == 'custext':
            perturbed_question = custext_sanitize_text(question, epsilon=config['epsilon'])
        elif privacy_mechanism == 'clusant':
            perturbed_question = clusant_sanitize_text(question, epsilon=config['epsilon'])
        else:
            raise ValueError(f"Unknown privacy mechanism: {privacy_mechanism}")
            
        print(f"Perturbed Question: {perturbed_question}")

        # Apply privacy mechanism to all options in batch
        print(f"{YELLOW}3b. Applying {mechanism_name} batch sanitization to all options...{RESET}")
        
        if privacy_mechanism == 'phrasedp':
            if use_old_phrasedp:
                # Use OLD PhraseDP batch perturbation (single API call, no band diversity)
                perturbed_options_text = batch_perturb_options_with_old_phrasedp(
                    options, config['epsilon'], nebius_client, nebius_model_name, sbert_model
                )
            else:
                # Use new PhraseDP batch perturbation (10 API calls, 10-band diversity)
                perturbed_options_text = batch_perturb_options_with_phrasedp(
                    options, config['epsilon'], nebius_client, nebius_model_name
                )
        elif privacy_mechanism == 'inferdpt':
            perturbed_options_text = batch_perturb_options_with_inferdpt(options, config['epsilon'])
        elif privacy_mechanism == 'santext':
            perturbed_options_text = batch_perturb_options_with_santext(options, config['epsilon'])
        elif privacy_mechanism == 'custext':
            perturbed_options_text = batch_perturb_options_with_custext(options, config['epsilon'])
        elif privacy_mechanism == 'clusant':
            perturbed_options = batch_perturb_options_with_clusant(options, config['epsilon'])
            print("Batch Perturbed Options:")
            for key, value in perturbed_options.items():
                print(f"  {key}) {value}")
        else:
            raise ValueError(f"Unknown privacy mechanism: {privacy_mechanism}")

        if privacy_mechanism != 'clusant':
            print(f"Batch Perturbed Options Text: {perturbed_options_text}")

        # Generate CoT from perturbed question AND perturbed options
        print(f"{YELLOW}3c. Generating CoT from Perturbed Question AND Options with REMOTE LLM...{RESET}")
        
        if privacy_mechanism == 'clusant':
            cot_text = generate_cot_from_remote_llm_with_perturbed_options_dict(
                remote_client, 
                config['remote_models']['cot_model'], 
                perturbed_question, 
                perturbed_options
            )
        else:
            cot_text = generate_cot_from_remote_llm_with_perturbed_options(
                remote_client, 
                config['remote_models']['cot_model'], 
                perturbed_question, 
                perturbed_options_text
            )
        print(f"Generated Chain-of-Thought (Remote, Fully Private with Perturbed Options):\n{cot_text}")

        # Use local model with original question/options but private CoT
        print(f"{YELLOW}3d. Running Local Model with original question/options and Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)

        print(f"Local Answer (Fully Private CoT-Aided): {local_response}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")

        return is_correct
    except Exception as e:
        print(f"{RED}Error during {mechanism_name} with batch options private CoT-aided inference: {e}{RESET}")
        return False

# Removed individual function - now using run_scenario_3_private_local_cot('inferdpt', ...)

def batch_perturb_options_with_inferdpt(options, epsilon):
    """Batch perturb all options together using InferDPT for efficiency."""
    print(f"{CYAN}Batch Perturbation Starting:{RESET}")
    print(f"Options Count: {len(options)}")
    print(f"Epsilon: {epsilon}")
    print(f"{CYAN}Original Options:{RESET}")
    for key, value in options.items():
        print(f"  {key}) {value}")

    # Simple concatenation without semicolon enforcement
    combined_text = " ".join(options.values())

    print(f"{CYAN}Combined Text:{RESET}\n{combined_text}")

    # Apply InferDPT to the combined text
    print(f"{YELLOW}Applying InferDPT to combined text...{RESET}")
    perturbed_combined = inferdpt_sanitize_text(combined_text, epsilon=epsilon)

    print(f"{CYAN}Perturbed Combined Text:{RESET}\n{perturbed_combined}")
    print(f"{GREEN}Batch perturbation complete - returning single perturbed text for CoT{RESET}")

    # Return the single perturbed text - no need to parse back to individual options
    return perturbed_combined

def batch_perturb_options_with_santext(options, epsilon):
    """Batch perturb all options together using SANTEXT+ for efficiency."""
    print(f"{CYAN}Batch Perturbation Starting:{RESET}")
    print(f"Options Count: {len(options)}")
    print(f"Epsilon: {epsilon}")
    print(f"{CYAN}Original Options:{RESET}")
    for key, value in options.items():
        print(f"  {key}) {value}")

    # Simple concatenation without semicolon enforcement
    combined_text = " ".join(options.values())

    print(f"{CYAN}Combined Text:{RESET}\n{combined_text}")

    # Apply SANTEXT+ to the combined text
    print(f"{YELLOW}Applying SANTEXT+ to combined text...{RESET}")
    perturbed_combined = santext_sanitize_text(combined_text, epsilon=epsilon)

    print(f"{CYAN}Perturbed Combined Text:{RESET}\n{perturbed_combined}")
    print(f"{GREEN}Batch perturbation complete - returning single perturbed text for CoT{RESET}")

    # Return the single perturbed text - no need to parse back to individual options
    return perturbed_combined

def batch_perturb_options_with_custext(options, epsilon):
    """Batch perturb all options together using CUSTEXT+ for efficiency."""
    print(f"{CYAN}Batch Perturbation Starting:{RESET}")
    print(f"Options Count: {len(options)}")
    print(f"Epsilon: {epsilon}")
    print(f"{CYAN}Original Options:{RESET}")
    for key, value in options.items():
        print(f"  {key}) {value}")

    # Simple concatenation without semicolon enforcement
    combined_text = " ".join(options.values())

    print(f"{CYAN}Combined Text:{RESET}\n{combined_text}")

    # Apply CUSTEXT+ to the combined text
    print(f"{YELLOW}Applying CUSTEXT+ to combined text...{RESET}")
    perturbed_combined = custext_sanitize_text(combined_text, epsilon=epsilon)

    print(f"{CYAN}Perturbed Combined Text:{RESET}\n{perturbed_combined}")
    print(f"{GREEN}Batch perturbation complete - returning single perturbed text for CoT{RESET}")

    # Return the single perturbed text - no need to parse back to individual options
    return perturbed_combined

def batch_perturb_options_with_clusant(options, epsilon):
    """Batch perturb all options together using CluSanT for efficiency."""
    # Combine all options into a single text for batch processing
    combined_text = ""
    for key, value in options.items():
        combined_text += f"Option {key}: {value}\n"

    # Apply CluSanT to the combined text
    perturbed_combined = clusant_sanitize_text(combined_text.strip(), epsilon=epsilon)

    # Parse back to individual options
    perturbed_options = {}
    lines = perturbed_combined.split('\n')

    for line in lines:
        line = line.strip()
        if line.startswith('Option ') and ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                option_key = parts[0].replace('Option ', '').strip()
                option_value = parts[1].strip()
                if option_key in ['A', 'B', 'C', 'D']:
                    perturbed_options[option_key] = option_value

    # Fallback: if parsing fails, use original keys with split content
    if len(perturbed_options) != len(options):
        print(f"{YELLOW}Warning: CluSanT batch parsing partially failed, using fallback approach{RESET}")
        option_keys = list(options.keys())
        lines = [l.strip() for l in perturbed_combined.split('\n') if l.strip()]
        for i, key in enumerate(option_keys):
            if i < len(lines):
                # Remove "Option X:" prefix if it exists
                line = lines[i]
                if line.startswith(f'Option {key}:'):
                    line = line[len(f'Option {key}:'):].strip()
                perturbed_options[key] = line
            else:
                perturbed_options[key] = f"Perturbed option {key}"

    return perturbed_options

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

def run_experiment_for_model(
    model_name,
    question_indices: list[int] | None = None,
    start_index: int = 0,
    num_samples: int = 500,
):
    """Run the MedQA experiment for a given local model."""
    
    print(f"{CYAN}Starting MedQA Experiment with model: {model_name}{RESET}")
    
    # Load dataset
    print(f"{CYAN}Loading MedQA dataset...{RESET}")
    print(f"{YELLOW}Note: MedQA contains clinical vignettes with patient scenarios in questions{RESET}")
    dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
    
    if question_indices is not None and len(question_indices) == 0:
        print(f"{YELLOW}No question indices provided. Nothing to run.{RESET}")
        return MedQAExperimentResults()

    if question_indices is not None:
        selected_indices = [idx for idx in question_indices if 0 <= idx < len(dataset)]
        if not selected_indices:
            print(f"{RED}Provided question indices are out of range. Dataset has {len(dataset)} questions.{RESET}")
            return MedQAExperimentResults()
    else:
        selected_indices = list(range(start_index, min(start_index + num_samples, len(dataset))))

    print(
        f"{CYAN}Testing {len(selected_indices)} question(s) from MedQA test set "
        f"(indices: {selected_indices}){RESET}"
    )

    # Get clients - exit if any are not available
    try:
        remote_client = get_remote_llm_client()
        print(f"{GREEN}Remote LLM client initialized successfully{RESET}")
    except Exception as e:
        abort_on_quota_error(e, "OpenAI")
        print(f"{RED}Failed to initialize remote LLM client: {e}{RESET}")
        print(f"{RED}Cannot proceed without remote client. Exiting.{RESET}")
        return MedQAExperimentResults()
    
    try:
        local_client = utils.get_nebius_client()
        print(f"{GREEN}Local (Nebius) client initialized successfully{RESET}")
    except Exception as e:
        abort_on_quota_error(e, "Nebius")
        print(f"{RED}Failed to initialize Nebius local client: {e}{RESET}")
        print(f"{RED}Cannot proceed without local client. Exiting.{RESET}")
        return MedQAExperimentResults()
    
    # Load Sentence-BERT for similarity computation
    sbert_model = load_sentence_bert()
    
    # Initialize privacy mechanisms once for all questions
    print(f"{CYAN}Initializing privacy mechanisms...{RESET}")
    santext_mechanism = initialize_santext_mechanism()
    custext_components = initialize_custext_components()
    clusant_mechanism = initialize_clusant_mechanism()
    
    # Initialize results
    results = MedQAExperimentResults()
    
    sample_questions = dataset.select(selected_indices)
    
    for i, item in enumerate(sample_questions):
        dataset_idx = selected_indices[i]
        print(f"\n{YELLOW}--- Question {i+1}/{len(sample_questions)} (Dataset idx: {dataset_idx}) ---{RESET}")
        
        question = item['question']
        options = item['options']  # Already a dict with A, B, C, D keys
        correct_answer = item['answer_idx']  # This is already a letter (A, B, C, D)
        
        # Extract additional metadata
        meta_info = item.get('meta_info', 'N/A')
        metamap_phrases = item.get('metamap_phrases', [])
        
        print(f"Question: {question}")
        print(f"Options:")
        for key, value in options.items():
            print(f"  {key}) {value}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Complexity Level: {meta_info}")
        print(f"MetaMap Phrases ({len(metamap_phrases)}): {', '.join(metamap_phrases[:10])}{'...' if len(metamap_phrases) > 10 else ''}")
        
        # Run all scenarios
        results.total_questions += 1
        
        # DISABLED: Scenario 1: Purely Local Model
        # if run_scenario_1_purely_local(local_client, model_name, question, options, correct_answer):
        #     results.local_alone_correct += 1
        
        # DISABLED: Scenario 2: Non-Private Local + Remote CoT
        # if run_scenario_2_non_private_cot(local_client, model_name, remote_client, question, options, correct_answer):
        #     results.non_private_cot_correct += 1

        # DISABLED: Scenario 3.0: Private Local + CoT (Old Phrase DP - Single API Call)
        # if run_scenario_3_private_local_cot(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'phrasedp', use_old_phrasedp=True):
        #     results.old_phrase_dp_local_cot_correct += 1
        

        # ACTIVE: Scenario 3.1.2: Private Local + CoT (Old Phrase DP with Batch Perturbed Options) - FIXED BUG
        if run_scenario_3_private_local_cot_with_batch_options(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'phrasedp', use_old_phrasedp=True):
            results.phrase_dp_batch_options_local_cot_correct += 1
        
        # ACTIVE: Scenario 3.2: Private Local + CoT (InferDPT WITHOUT Batch Options)
        if run_scenario_3_private_local_cot(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'inferdpt'):
            results.inferdpt_local_cot_correct += 1

        # DISABLED: Scenario 3.2.new: Private Local + CoT (InferDPT with Batch Perturbed Options)
        # if run_scenario_3_private_local_cot_with_batch_options(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'inferdpt'):
        #     results.inferdpt_batch_options_local_cot_correct += 1
        
        # ACTIVE: Scenario 3.3: Private Local + CoT (SANTEXT+ WITHOUT Batch Options)
        if run_scenario_3_private_local_cot(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'santext'):
            results.santext_local_cot_correct += 1

        # DISABLED: Scenario 3.3.new: Private Local + CoT (SANTEXT+ with Batch Perturbed Options)
        # if run_scenario_3_private_local_cot_with_batch_options(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'santext'):
        #     results.santext_batch_options_local_cot_correct += 1
        
        # Scenario 3.4: Private Local + CoT (CUSTEXT+)
        # if run_scenario_3_private_local_cot(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'custext'):
        #     results.custext_local_cot_correct += 1

        # Scenario 3.4.new: Private Local + CoT (CUSTEXT+ with Batch Perturbed Options)
        # if run_scenario_3_private_local_cot_with_batch_options(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'custext'):
        #     results.custext_batch_options_local_cot_correct += 1

        # Scenario 3.5: Private Local + CoT (CluSanT)
        # Temporarily disabled per request
        # if run_scenario_3_private_local_cot(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'clusant'):
        #     results.clusant_local_cot_correct += 1

        # Scenario 3.5.new: Private Local + CoT (CluSanT with Batch Perturbed Options)
        # if run_scenario_3_private_local_cot_with_batch_options(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'clusant'):
        #     results.clusant_batch_options_local_cot_correct += 1
        
        # Scenario 4: Purely Remote Model
        if run_scenario_4_purely_remote(remote_client, question, options, correct_answer):
            results.purely_remote_correct += 1
    
    # Print final results
    def print_accuracy(name, correct, total):
        """Helper function to print accuracy results."""
        percentage = correct/total*100 if total > 0 else 0
        print(f"{name} Accuracy: {correct}/{total} = {percentage:.2f}%")
    
    print(f"\n{'='*50}")
    print(f"FINAL RESULTS - THREE SCENARIO TEST")
    print(f"{'='*50}")
    print_accuracy("3.1.2. Private Local Model + CoT (Old Phrase DP with Batch Perturbed Options) - FIXED", results.phrase_dp_batch_options_local_cot_correct, results.total_questions)
    print_accuracy("3.2. Private Local Model + CoT (InferDPT WITHOUT Batch Options)", results.inferdpt_local_cot_correct, results.total_questions)
    print_accuracy("3.3. Private Local Model + CoT (SANTEXT+ WITHOUT Batch Options)", results.santext_local_cot_correct, results.total_questions)
    
    return results


def test_single_question(question_index=0):
    """Test a single question from the dataset by index."""
    print(f"{CYAN}Testing single question (index {question_index}) from MedQA{RESET}")
    return run_experiment_for_model(
        config['local_model'],
        question_indices=[question_index],
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MedQA USMLE four-option experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Test a single MedQA question by index (0-based). If omitted, run the full batch configured in the script.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config["local_model"],
        help="Override the local model defined in config.yaml.",
    )

    args = parser.parse_args(argv)

    model_name = args.model

    if args.index is not None:
        test_single_question(args.index)
    else:
        run_experiment_for_model(model_name)


if __name__ == "__main__":
    main()

