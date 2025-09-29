#!/usr/bin/env python3
"""
MedMCQA Experiment Script
=========================

A script for experimenting with MedMCQA (Medical Multiple Choice Questions) dataset
using privacy-preserving approaches similar to the MedQA experiments.

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
Date: 2025-01-27
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
import datetime

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

class MedMCQAExperimentResults:
    """Class to track experiment results for MedMCQA dataset."""
    
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
        self.output_file = None
        self.start_time = datetime.datetime.now()
    
    def initialize_output_file(self, model_name, num_samples, epsilon=None):
        """Initialize output file for incremental writing."""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        if epsilon is not None:
            self.output_file = f"QA-results/medmcqa/medmcqa_results_{model_name.replace('/', '_')}_{num_samples}q_eps{epsilon}_{timestamp}.json"
        else:
            self.output_file = f"QA-results/medmcqa/medmcqa_results_{model_name.replace('/', '_')}_{num_samples}q_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Initialize file with header
        initial_data = {
            "experiment_info": {
                "dataset": "MedMCQA",
                "model": model_name,
                "num_samples": num_samples,
                "start_time": self.start_time.isoformat(),
                "epsilon": epsilon if epsilon is not None else config['epsilon'],
                "scenarios": [
                    "1. Purely Local Model (Baseline)",
                    "2. Non-Private Local Model + Remote CoT", 
                    "3.0. Private Local Model + CoT (Old Phrase DP)",
                    "3.2. Private Local Model + CoT (InferDPT)",
                    "3.3. Private Local Model + CoT (SANTEXT+)",
                    "4. Purely Remote Model"
                ]
            },
            "results": {
                "local_alone_correct": 0,
                "non_private_cot_correct": 0,
                "old_phrase_dp_local_cot_correct": 0,
                "inferdpt_local_cot_correct": 0,
                "santext_local_cot_correct": 0,
                "purely_remote_correct": 0,
                "total_questions": 0
            },
            "question_results": []
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
        
        print(f"{GREEN}Output file initialized: {self.output_file}{RESET}")
    
    def write_incremental_results(self, question_idx, question_data, scenario_results):
        """Write incremental results to output file."""
        if self.output_file is None:
            return
        
        try:
            # Read current data
            with open(self.output_file, 'r') as f:
                data = json.load(f)
            
            # Update results
            data["results"]["total_questions"] = self.total_questions
            data["results"]["local_alone_correct"] = self.local_alone_correct
            data["results"]["non_private_cot_correct"] = self.non_private_cot_correct
            data["results"]["old_phrase_dp_local_cot_correct"] = self.old_phrase_dp_local_cot_correct
            data["results"]["inferdpt_local_cot_correct"] = self.inferdpt_local_cot_correct
            data["results"]["santext_local_cot_correct"] = self.santext_local_cot_correct
            data["results"]["purely_remote_correct"] = self.purely_remote_correct
            
            # Add question result
            question_result = {
                "question_index": question_idx,
                "question": question_data.get("question", ""),
                "subject": question_data.get("subject_name", ""),
                "topic": question_data.get("topic_name", ""),
                "correct_answer": question_data.get("correct_answer", ""),
                "scenario_results": scenario_results,
                "timestamp": datetime.datetime.now().isoformat()
            }
            data["question_results"].append(question_result)
            
            # Write back to file
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"{YELLOW}Warning: Failed to write incremental results: {e}{RESET}")
    
    def finalize_results(self):
        """Finalize results and write summary."""
        if self.output_file is None:
            return
        
        try:
            # Read current data
            with open(self.output_file, 'r') as f:
                data = json.load(f)
            
            # Update final results
            data["results"]["total_questions"] = self.total_questions
            data["results"]["local_alone_correct"] = self.local_alone_correct
            data["results"]["non_private_cot_correct"] = self.non_private_cot_correct
            data["results"]["old_phrase_dp_local_cot_correct"] = self.old_phrase_dp_local_cot_correct
            data["results"]["inferdpt_local_cot_correct"] = self.inferdpt_local_cot_correct
            data["results"]["santext_local_cot_correct"] = self.santext_local_cot_correct
            data["results"]["purely_remote_correct"] = self.purely_remote_correct
            
            # Add final summary
            data["final_summary"] = {
                "end_time": datetime.datetime.now().isoformat(),
                "duration_minutes": (datetime.datetime.now() - self.start_time).total_seconds() / 60,
                "accuracies": {
                    "local_alone": self.local_alone_correct / self.total_questions * 100 if self.total_questions > 0 else 0,
                    "non_private_cot": self.non_private_cot_correct / self.total_questions * 100 if self.total_questions > 0 else 0,
                    "old_phrase_dp": self.old_phrase_dp_local_cot_correct / self.total_questions * 100 if self.total_questions > 0 else 0,
                    "inferdpt": self.inferdpt_local_cot_correct / self.total_questions * 100 if self.total_questions > 0 else 0,
                    "santext": self.santext_local_cot_correct / self.total_questions * 100 if self.total_questions > 0 else 0,
                    "purely_remote": self.purely_remote_correct / self.total_questions * 100 if self.total_questions > 0 else 0
                }
            }
            
            # Write final data
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            print(f"{GREEN}Final results written to: {self.output_file}{RESET}")
            
        except Exception as e:
            print(f"{YELLOW}Warning: Failed to finalize results: {e}{RESET}")

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

def run_scenario_3_private_local_cot_with_epsilon(client, model_name, remote_client, sbert_model, question, options, correct_answer, privacy_mechanism, use_old_phrasedp=False, epsilon=3.0):
    """Scenario 3: Private Local Model + CoT with custom epsilon value."""
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
    
    print(f"\n{BLUE}--- Scenario 3: Private Local Model + CoT ({mechanism_name}, ε={epsilon}) ---{RESET}")
    
    try:
        # Apply privacy mechanism to the question
        print(f"{YELLOW}3a. Applying {mechanism_name} sanitization with ε={epsilon}...{RESET}")
        
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
                    epsilon=epsilon,
                    sbert_model=sbert_model
                )
            else:
                # Use new PhraseDP (10 API calls, 10-band diversity)
                from sanitization_methods import config as sm_config
                nebius_client = utils.get_nebius_client()
                nebius_model_name = sm_config.get('local_model')
                perturbed_question = phrasedp_sanitize_text(
                    question,
                    epsilon=epsilon,
                    nebius_client=nebius_client,
                    nebius_model_name=nebius_model_name,
                )
        elif privacy_mechanism == 'inferdpt':
            perturbed_question = inferdpt_sanitize_text(question, epsilon=epsilon)
        elif privacy_mechanism == 'santext':
            perturbed_question = santext_sanitize_text(question, epsilon=epsilon)
        elif privacy_mechanism == 'custext':
            perturbed_question = custext_sanitize_text(question, epsilon=epsilon)
        elif privacy_mechanism == 'clusant':
            perturbed_question = clusant_sanitize_text(question, epsilon=epsilon)
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
    num_samples: int = 100,
    epsilon_values: list[float] = [1.0, 2.0, 3.0],
):
    """Run the MedMCQA experiment for a given local model with multiple epsilon values."""
    
    print(f"{CYAN}Starting MedMCQA Experiment with model: {model_name}{RESET}")
    print(f"{CYAN}Testing epsilon values: {epsilon_values}{RESET}")
    
    # Load dataset
    print(f"{CYAN}Loading MedMCQA dataset...{RESET}")
    print(f"{YELLOW}Note: MedMCQA contains medical multiple choice questions from Indian medical entrance exams{RESET}")
    dataset = load_dataset('medmcqa', split='validation')
    
    if question_indices is not None and len(question_indices) == 0:
        print(f"{YELLOW}No question indices provided. Nothing to run.{RESET}")
        return MedMCQAExperimentResults()

    if question_indices is not None:
        selected_indices = [idx for idx in question_indices if 0 <= idx < len(dataset)]
        if not selected_indices:
            print(f"{RED}Provided question indices are out of range. Dataset has {len(dataset)} questions.{RESET}")
            return MedMCQAExperimentResults()
    else:
        selected_indices = list(range(start_index, min(start_index + num_samples, len(dataset))))

    print(
        f"{CYAN}Testing {len(selected_indices)} question(s) from MedMCQA validation set "
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
        return MedMCQAExperimentResults()
    
    try:
        local_client = utils.get_nebius_client()
        print(f"{GREEN}Local (Nebius) client initialized successfully{RESET}")
    except Exception as e:
        abort_on_quota_error(e, "Nebius")
        print(f"{RED}Failed to initialize Nebius local client: {e}{RESET}")
        print(f"{RED}Cannot proceed without local client. Exiting.{RESET}")
        return MedMCQAExperimentResults()
    
    # Load Sentence-BERT for similarity computation
    sbert_model = load_sentence_bert()
    
    # Initialize privacy mechanisms once for all questions
    print(f"{CYAN}Initializing privacy mechanisms...{RESET}")
    santext_mechanism = initialize_santext_mechanism()
    custext_components = initialize_custext_components()
    clusant_mechanism = initialize_clusant_mechanism()
    
    # Initialize results for each epsilon
    all_results = {}
    
    for epsilon in epsilon_values:
        print(f"\n{'='*80}")
        print(f"{CYAN}RUNNING EXPERIMENT WITH EPSILON = {epsilon}{RESET}")
        print(f"{'='*80}")
        
        # Initialize results for this epsilon
        results = MedMCQAExperimentResults()
        
        # Initialize output file for incremental writing
        results.initialize_output_file(model_name, len(selected_indices), epsilon)
        
        all_results[epsilon] = results
    
    sample_questions = dataset.select(selected_indices)
    
    for i, item in enumerate(sample_questions):
        dataset_idx = selected_indices[i]
        print(f"\n{YELLOW}--- Question {i+1}/{len(sample_questions)} (Dataset idx: {dataset_idx}) ---{RESET}")
        
        question = item['question']
        # MedMCQA has opa, opb, opc, opd fields and cop (correct option as integer 0-3)
        options = {
            'A': item['opa'],
            'B': item['opb'], 
            'C': item['opc'],
            'D': item['opd']
        }
        # Convert cop (0-3) to letter (A-D)
        correct_answer = ['A', 'B', 'C', 'D'][item['cop']]
        
        # Extract additional metadata
        subject_name = item.get('subject_name', 'N/A')
        topic_name = item.get('topic_name', 'N/A')
        explanation = item.get('exp', 'N/A')
        
        print(f"Question: {question}")
        print(f"Options:")
        for key, value in options.items():
            print(f"  {key}) {value}")
        print(f"Correct Answer: {correct_answer}")
        print(f"Subject: {subject_name}")
        print(f"Topic: {topic_name}")
        print(f"Has Explanation: {'Yes' if explanation and explanation != 'N/A' else 'No'}")
        
        # Run scenarios that don't depend on epsilon (1, 2, 4) - only once per question
        print(f"\n{CYAN}--- Running Epsilon-Independent Scenarios ---{RESET}")
        
        # Track shared scenario results
        shared_scenario_results = {}
        
        # ACTIVE: Scenario 1: Purely Local Model (run once, share across all epsilon)
        scenario_1_result = run_scenario_1_purely_local(local_client, model_name, question, options, correct_answer)
        if scenario_1_result:
            for eps in epsilon_values:
                all_results[eps].local_alone_correct += 1
        shared_scenario_results["scenario_1_purely_local"] = scenario_1_result
        
        # ACTIVE: Scenario 2: Non-Private Local + Remote CoT (run once, share across all epsilon)
        scenario_2_result = run_scenario_2_non_private_cot(local_client, model_name, remote_client, question, options, correct_answer)
        if scenario_2_result:
            for eps in epsilon_values:
                all_results[eps].non_private_cot_correct += 1
        shared_scenario_results["scenario_2_non_private_cot"] = scenario_2_result
        
        # ACTIVE: Scenario 4: Purely Remote Model (run once, share across all epsilon)
        scenario_4_result = run_scenario_4_purely_remote(remote_client, question, options, correct_answer)
        if scenario_4_result:
            for eps in epsilon_values:
                all_results[eps].purely_remote_correct += 1
        shared_scenario_results["scenario_4_purely_remote"] = scenario_4_result
        
        # Run epsilon-dependent scenarios (3.0, 3.2, 3.3) for each epsilon value
        for epsilon in epsilon_values:
            print(f"\n{CYAN}--- Testing Epsilon-Dependent Scenarios with ε = {epsilon} ---{RESET}")
            results = all_results[epsilon]
            results.total_questions += 1
            
            # Track scenario results for this question and epsilon
            scenario_results = shared_scenario_results.copy()  # Start with shared results
            
            # ACTIVE: Scenario 3.0: Private Local + CoT (Old Phrase DP - Single API Call)
            scenario_3_0_result = run_scenario_3_private_local_cot_with_epsilon(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'phrasedp', use_old_phrasedp=True, epsilon=epsilon)
            if scenario_3_0_result:
                results.old_phrase_dp_local_cot_correct += 1
            scenario_results["scenario_3_0_old_phrase_dp"] = scenario_3_0_result
            
            # ACTIVE: Scenario 3.2: Private Local + CoT (InferDPT WITHOUT Batch Options)
            scenario_3_2_result = run_scenario_3_private_local_cot_with_epsilon(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'inferdpt', epsilon=epsilon)
            if scenario_3_2_result:
                results.inferdpt_local_cot_correct += 1
            scenario_results["scenario_3_2_inferdpt"] = scenario_3_2_result
            
            # ACTIVE: Scenario 3.3: Private Local + CoT (SANTEXT+ WITHOUT Batch Options)
            scenario_3_3_result = run_scenario_3_private_local_cot_with_epsilon(local_client, model_name, remote_client, sbert_model, question, options, correct_answer, 'santext', epsilon=epsilon)
            if scenario_3_3_result:
                results.santext_local_cot_correct += 1
            scenario_results["scenario_3_3_santext"] = scenario_3_3_result
            
            # DISABLED: Scenario 3.4: Private Local + CoT (CUSTEXT+) - DISABLED
            scenario_results["scenario_3_4_custext"] = "DISABLED"

            # DISABLED: Scenario 3.5: Private Local + CoT (CluSanT) - DISABLED
            scenario_results["scenario_3_5_clusant"] = "DISABLED"
            
            # Write incremental results for this epsilon
            question_data = {
                "question": question,
                "subject_name": subject_name,
                "topic_name": topic_name,
                "correct_answer": correct_answer
            }
            results.write_incremental_results(dataset_idx, question_data, scenario_results)
    
    # Finalize results and write to file for each epsilon
    for epsilon, results in all_results.items():
        results.finalize_results()
    
    # Print final results for each epsilon
    def print_accuracy(name, correct, total):
        """Helper function to print accuracy results."""
        percentage = correct/total*100 if total > 0 else 0
        print(f"{name} Accuracy: {correct}/{total} = {percentage:.2f}%")
    
    for epsilon in epsilon_values:
        results = all_results[epsilon]
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS FOR EPSILON = {epsilon} - ACTIVE MECHANISMS (3.4 & 3.5 DISABLED)")
        print(f"{'='*80}")
        print_accuracy("1. Purely Local Model (Baseline)", results.local_alone_correct, results.total_questions)
        print_accuracy("2. Non-Private Local Model + Remote CoT", results.non_private_cot_correct, results.total_questions)
        print_accuracy("3.0. Private Local Model + CoT (Old Phrase DP)", results.old_phrase_dp_local_cot_correct, results.total_questions)
        print_accuracy("3.2. Private Local Model + CoT (InferDPT)", results.inferdpt_local_cot_correct, results.total_questions)
        print_accuracy("3.3. Private Local Model + CoT (SANTEXT+)", results.santext_local_cot_correct, results.total_questions)
        print(f"3.4. Private Local Model + CoT (CUSTEXT+) - DISABLED")
        print(f"3.5. Private Local Model + CoT (CluSanT) - DISABLED")
        print_accuracy("4. Purely Remote Model", results.purely_remote_correct, results.total_questions)
    
    return all_results


def test_single_question(question_index=0):
    """Test a single question from the dataset by index."""
    print(f"{CYAN}Testing single question (index {question_index}) from MedMCQA{RESET}")
    return run_experiment_for_model(
        config['local_model'],
        question_indices=[question_index],
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MedMCQA experiment runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Test a single MedMCQA question by index (0-based). If omitted, run the full batch configured in the script.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config["local_model"],
        help="Override the local model defined in config.yaml.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of questions to test (default: 100).",
    )

    args = parser.parse_args(argv)

    model_name = args.model

    if args.index is not None:
        test_single_question(args.index)
    else:
        run_experiment_for_model(model_name, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
