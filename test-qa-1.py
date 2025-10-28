#!/usr/bin/env python3
"""
MedQA USMLE Experiment Script
============================

A script for experimenting with MedQA USMLE (Medical Question Answering) dataset
using privacy-preserving approaches. This script tests different scenarios
without feeding the multiple choice options to the LLMs - only the question text.

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
from sanitization_methods import (
    phrasedp_sanitize_text,
    inferdpt_sanitize_text,
    santext_sanitize_text,
    custext_sanitize_text,
    clusant_sanitize_text,
)

def _resolve_local_model_name_for_nebius(client, fallback_model_name: str) -> str:
    """Return local model name for Nebius invocations."""
    return fallback_model_name

def _find_working_nebius_model(client, local_model: str) -> str:
    """Probe Nebius with candidate models to find a working local model ID.

    Returns the provided local model if it works, otherwise raises an error.
    """
    try:
        resp = create_completion_with_model_support(
            client,
            local_model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        # If no exception, model works
        return local_model
    except Exception:
        raise ValueError(f"Local model {local_model} is not working")

# Load environment variables
load_dotenv()

# Configuration is now handled via command-line arguments

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
        self.phrase_dp_plus_correct = 0
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

def initialize_santext_mechanism(epsilon: float):
    """Initialize SANTEXT+ mechanism once for all questions."""
    print(f"{CYAN}Initializing SANTEXT+ mechanism...{RESET}")
    try:
        santext_sanitize_text("Warm-up text for SANTEXT+", epsilon=epsilon)
        print(f"{GREEN}SANTEXT+ mechanism initialized successfully{RESET}")
    except Exception as exc:
        print(f"{YELLOW}SANTEXT+ warm-up encountered an issue but will proceed lazily: {exc}{RESET}")

def initialize_custext_components(epsilon: float):
    """Initialize CUSTEXT+ components once for all questions."""
    print(f"{CYAN}Initializing CUSTEXT+ components...{RESET}")
    try:
        custext_sanitize_text("Warm-up text for CUSTEXT+", epsilon=epsilon)
        print(f"{GREEN}CUSTEXT+ components initialized successfully{RESET}")
    except Exception as exc:
        print(f"{YELLOW}CUSTEXT+ warm-up encountered an issue but will proceed lazily: {exc}{RESET}")

def initialize_clusant_mechanism(epsilon: float):
    """Initialize CluSanT mechanism once for all questions."""
    print(f"{CYAN}Initializing CluSanT mechanism...{RESET}")
    try:
        clusant_sanitize_text("Warm-up text for CluSanT", epsilon=epsilon)
        print(f"{GREEN}CluSanT mechanism initialized successfully{RESET}")
    except Exception as exc:
        print(f"{YELLOW}CluSanT warm-up encountered an issue but will proceed lazily: {exc}{RESET}")

def get_remote_llm_client():
    """Get remote LLM client (OpenAI)."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return openai.OpenAI(api_key=api_key)

def get_anthropic_client():
    """Get Anthropic client."""
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        raise ValueError("Anthropic library not installed. Run: pip install anthropic")

def is_anthropic_model(model_name):
    """Check if the model is an Anthropic model."""
    return model_name.startswith('claude') or 'claude' in model_name.lower()

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
    Create a chat completion with proper parameter support for different models and providers.
    Automatically detects if model is Anthropic and uses appropriate client.
    """
    try:
        if is_anthropic_model(model_name):
            # Use Anthropic client for Claude models
            anthropic_client = get_anthropic_client()
            
            # Convert OpenAI format to Anthropic format
            system_content = ""
            user_content = ""
            
            for msg in messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    if user_content:
                        user_content += "\n\n" + msg["content"]
                    else:
                        user_content = msg["content"]
            
            # Anthropic models don't support temperature=0.0, use 0.1 as minimum
            anthropic_temp = max(0.1, temperature) if temperature == 0.0 else temperature
            
            response = anthropic_client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=anthropic_temp,
                system=system_content if system_content else None,
                messages=[{"role": "user", "content": user_content}]
            )
            
            # Convert Anthropic response to OpenAI-like format
            class AnthropicResponse:
                def __init__(self, anthropic_response):
                    self.choices = [type('Choice', (), {
                        'message': type('Message', (), {
                            'content': anthropic_response.content[0].text
                        })()
                    })()]
            
            return AnthropicResponse(response)
            
        else:
            # OpenAI API format for non-Anthropic models
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
        local_model = _find_working_nebius_model(client, model_name)
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
        local_model = _find_working_nebius_model(client, model_name)
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
    # Build a clear, generic CoT prompt that works for medical and non-medical questions.
    # Avoid including the trailing 'Answer:' token that `format_question_with_options` appends.
    prompt_lines = []
    prompt_lines.append("Here is the (possibly perturbed) question:")
    prompt_lines.append(question)
    if options:
        prompt_lines.append("")
        prompt_lines.append("Options:")
        for k, v in options.items():
            prompt_lines.append(f"{k}) {v}")
    prompt_lines.append("")
    prompt_lines.append("Please provide a clear, step-by-step chain-of-thought reasoning to solve this question. Do NOT provide the final answer; provide only the reasoning steps.")
    prompt = "\n".join(prompt_lines)

    try:
        # Print the prompt for visibility (debug)
        print(f"\n{CYAN}=== Remote CoT Prompt ==={RESET}\n{prompt}\n{CYAN}=== End Prompt ==={RESET}\n")

        response = create_completion_with_model_support(
            client, model_name,
            messages=[
                {"role": "system", "content": "You are an expert reasoner. Provide a clear, step-by-step chain of thought to analyze the given question. Focus on domain-appropriate reasoning and knowledge."},
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

def run_scenario_2_non_private_cot(client, model_name, remote_client, remote_model, question, options, correct_answer):
    """Scenario 2: Non-Private Local Model + Remote CoT."""
    print(f"\n{BLUE}--- Scenario 2: Non-Private Local Model + Remote CoT ---{RESET}")
    
    try:
        # Generate CoT from remote LLM
        print(f"{YELLOW}2a. Generating CoT from ORIGINAL Question with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, remote_model, question)
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

def run_scenario_3_private_local_cot(client, model_name, remote_client, remote_model, sbert_model, question, options, correct_answer, privacy_mechanism, epsilon, metamap_phrases=None):
    """Scenario 3: Private Local Model + CoT (Generic function for all privacy mechanisms without batch options)."""
    mechanism_names = {
        'phrasedp': 'Phrase DP',
        'inferdpt': 'InferDPT', 
        'santext': 'SANTEXT+',
        'custext': 'CUSTEXT+',
        'clusant': 'CluSanT'
    }
    
    mechanism_name = mechanism_names.get(privacy_mechanism, privacy_mechanism.upper())
    if privacy_mechanism == 'phrasedp':
        mechanism_name = 'Phrase DP (single API call)'
    
    print(f"\n{BLUE}--- Scenario 3: Private Local Model + CoT ({mechanism_name}) ---{RESET}")
    
    try:
        # Apply privacy mechanism to the question
        print(f"{YELLOW}3a. Applying {mechanism_name} sanitization...{RESET}")
        
        if privacy_mechanism == 'phrasedp':
            # Always use the old PhraseDP single-call pipeline.
            # If metamap_phrases is provided, enable medical mode; otherwise use normal mode.
                from sanitization_methods import config as sm_config
                nebius_client = utils.get_nebius_client()
                nebius_model_name = sm_config.get('local_model')
            mode = "medqa-ume" if metamap_phrases else "normal"
            # Debug: show mode and metamap phrases summary for visibility between 3.0 and 3.1
            print(f"{YELLOW}PhraseDP mode: {mode}{RESET}")
            if metamap_phrases:
                sample_phrases = ", ".join(metamap_phrases[:20])
                sample_phrases += "..." if len(metamap_phrases) > 20 else ""
                print(f"Metamap phrases ({len(metamap_phrases)}): {sample_phrases}")
            else:
                print("Metamap phrases: None")

                perturbed_question = utils.phrase_DP_perturbation_old(
                    nebius_client=nebius_client,
                    nebius_model_name=nebius_model_name,
                    input_sentence=question,
                    epsilon=epsilon,
                    sbert_model=sbert_model,
                mode=mode,
                    metamap_phrases=metamap_phrases
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
            
        # Print only a short preview to avoid duplicating the full question when the CoT prompt is printed
        preview = perturbed_question[:300] + ("..." if len(perturbed_question) > 300 else "")
        print(f"Perturbed Question (preview): {preview}")

        # (CoT prompt is generated and printed inside the remote CoT helper function.
        #  Avoid duplicate/rigid prompt printing here.)
        
        # Keep options unchanged - only perturb the question for privacy
        print(f"{YELLOW}3b. Keeping options unchanged for local model...{RESET}")
        print(f"Original Options: {options}")
        
        # Generate CoT from perturbed question with original options
        print(f"{YELLOW}3c. Generating CoT from Perturbed Question with REMOTE LLM...{RESET}")
        cot_text = generate_cot_from_remote_llm(remote_client, remote_model, perturbed_question)
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

# def run_scenario_3_private_local_cot_with_batch_options(client, model_name, remote_client, remote_model, sbert_model, question, options, correct_answer, privacy_mechanism, epsilon, use_old_phrasedp=False, metamap_phrases=None):
#     """Scenario 3: Private Local Model + CoT with Batch Options (Generic function for all privacy mechanisms with batch options)."""
#     mechanism_names = {
#         'phrasedp': 'Phrase DP',
#         'inferdpt': 'InferDPT', 
#         'santext': 'SANTEXT+',
#         'custext': 'CUSTEXT+',
#         'clusant': 'CluSanT'
#     }
    
#     mechanism_name = mechanism_names.get(privacy_mechanism, privacy_mechanism.upper())
#     if privacy_mechanism == 'phrasedp' and use_old_phrasedp:
#         mechanism_name = 'Phrase DP (Old)'
#     elif privacy_mechanism == 'phrasedp' and not use_old_phrasedp:
#         mechanism_name = 'Phrase DP (New)'
    
#     print(f"\n{BLUE}--- Scenario 3: Private Local Model + CoT ({mechanism_name} with Batch Options) ---{RESET}")

#     try:
#         # Apply privacy mechanism to the question
#         print(f"{YELLOW}3a. Applying {mechanism_name} sanitization to question...{RESET}")
        
#         if privacy_mechanism == 'phrasedp':
#             if use_old_phrasedp:
#                 # Use old PhraseDP (single API call, no band diversity)
#                 from sanitization_methods import config as sm_config
#                 nebius_client = utils.get_nebius_client()
#                 nebius_model_name = sm_config.get('local_model')
#                 perturbed_question = utils.phrase_DP_perturbation_old(
#                     nebius_client=nebius_client,
#                     nebius_model_name=nebius_model_name,
#                     input_sentence=question,
#                     epsilon=epsilon,
#                     sbert_model=sbert_model,
#                     mode="medqa-ume",
#                     metamap_phrases=metamap_phrases
#                 )
#             else:
#                 # Use new PhraseDP (10 API calls, 10-band diversity)
#                 from sanitization_methods import config as sm_config
#                 nebius_client = utils.get_nebius_client()
#                 nebius_model_name = sm_config.get('local_model')
#                 perturbed_question = phrasedp_sanitize_text(
#                     question,
#                     epsilon=epsilon,
#                     nebius_client=nebius_client,
#                     nebius_model_name=nebius_model_name,
#                 )
#         elif privacy_mechanism == 'inferdpt':
#             perturbed_question = inferdpt_sanitize_text(question, epsilon=epsilon)
#         elif privacy_mechanism == 'santext':
#             perturbed_question = santext_sanitize_text(question, epsilon=epsilon)
#         elif privacy_mechanism == 'custext':
#             perturbed_question = custext_sanitize_text(question, epsilon=epsilon)
#         elif privacy_mechanism == 'clusant':
#             perturbed_question = clusant_sanitize_text(question, epsilon=epsilon)
#         else:
#             raise ValueError(f"Unknown privacy mechanism: {privacy_mechanism}")
            
#         print(f"Perturbed Question: {perturbed_question}")

#         # Apply privacy mechanism to all options in batch
#         print(f"{YELLOW}3b. Applying {mechanism_name} batch sanitization to all options...{RESET}")
        
#         if privacy_mechanism == 'phrasedp':
#             if use_old_phrasedp:
#                 # Use OLD PhraseDP batch perturbation (single API call, no band diversity)
#                 perturbed_options_text = batch_perturb_options_with_old_phrasedp(
#                     options, epsilon, nebius_client, nebius_model_name, sbert_model
#                 )
#             else:
#                 # Use new PhraseDP batch perturbation (10 API calls, 10-band diversity)
#                 perturbed_options_text = batch_perturb_options_with_phrasedp(
#                     options, epsilon, nebius_client, nebius_model_name
#                 )
#         elif privacy_mechanism == 'inferdpt':
#             perturbed_options_text = batch_perturb_options_with_inferdpt(options, epsilon)
#         elif privacy_mechanism == 'santext':
#             perturbed_options_text = batch_perturb_options_with_santext(options, epsilon)
#         elif privacy_mechanism == 'custext':
#             perturbed_options_text = batch_perturb_options_with_custext(options, epsilon)
#         elif privacy_mechanism == 'clusant':
#             perturbed_options = batch_perturb_options_with_clusant(options, epsilon)
#             print("Batch Perturbed Options:")
#             for key, value in perturbed_options.items():
#                 print(f"  {key}) {value}")
#         else:
#             raise ValueError(f"Unknown privacy mechanism: {privacy_mechanism}")

#         if privacy_mechanism != 'clusant':
#             print(f"Batch Perturbed Options Text: {perturbed_options_text}")

#         # Generate CoT from perturbed question AND perturbed options
#         print(f"{YELLOW}3c. Generating CoT from Perturbed Question AND Options with REMOTE LLM...{RESET}")
        
#         # NOTE: Batch-perturbed-option CoT generation is disabled in main flow.
#         # The functions to generate CoT from perturbed options have been moved to the end of the file
#         # and are commented/disabled. For now, use perturbed_question only.
#         cot_text = generate_cot_from_remote_llm(remote_client, remote_model, perturbed_question)
#         print(f"Generated Chain-of-Thought (Remote, Fully Private with Perturbed Options):\n{cot_text}")

#         # Use local model with original question/options but private CoT
#         print(f"{YELLOW}3d. Running Local Model with original question/options and Private CoT...{RESET}")
#         local_response = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
#         predicted_letter = extract_letter_from_answer(local_response)
#         is_correct = check_mcq_correctness(predicted_letter, correct_answer)

#         print(f"Local Answer (Fully Private CoT-Aided): {local_response}")
#         print(f"Extracted Letter: {predicted_letter}")
#         print(f"Result: {'Correct' if is_correct else 'Incorrect'}")

#         return is_correct
#     except Exception as e:
#         print(f"{RED}Error during {mechanism_name} with batch options private CoT-aided inference: {e}{RESET}")
#         return False

# Removed individual function - now using run_scenario_3_private_local_cot('inferdpt', ...)

# def batch_perturb_options_with_inferdpt(options, epsilon):
#     raise NotImplementedError("Batch option perturbation (InferDPT) is disabled. Original implementation moved to end of file.")

# def batch_perturb_options_with_santext(options, epsilon):
#     raise NotImplementedError("Batch option perturbation (SANTEXT+) is disabled. Original implementation moved to end of file.")

# def batch_perturb_options_with_custext(options, epsilon):
#     raise NotImplementedError("Batch option perturbation (CusText+) is disabled. Original implementation moved to end of file.")

# def batch_perturb_options_with_clusant(options, epsilon):
#     raise NotImplementedError("Batch option perturbation (CluSanT) is disabled. Original implementation moved to end of file.")

def run_scenario_4_purely_remote(remote_client, remote_model, question, options, correct_answer):
    """Scenario 4: Purely Remote Model."""
    print(f"\n{BLUE}--- Scenario 4: Purely Remote Model ---{RESET}")
    
    try:
        print(f"{YELLOW}4a. Running Purely Remote LLM...{RESET}")
        remote_response = get_answer_from_remote_llm(remote_client, remote_model, question, options)
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
    epsilon,
    remote_cot_model,
    remote_llm_model,
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
        # Try to get OpenAI client first, but don't fail if only Anthropic is available
        try:
            remote_client = get_remote_llm_client()
            print(f"{GREEN}OpenAI client initialized successfully{RESET}")
        except:
            # If OpenAI fails, try Anthropic
            try:
                remote_client = get_anthropic_client()
                print(f"{GREEN}Anthropic client initialized successfully{RESET}")
            except Exception as e:
                print(f"{RED}Failed to initialize any remote LLM client: {e}{RESET}")
                print(f"{RED}Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env{RESET}")
                return MedQAExperimentResults()
    except Exception as e:
        abort_on_quota_error(e, "API")
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
    santext_mechanism = initialize_santext_mechanism(epsilon)
    custext_components = initialize_custext_components(epsilon)
    clusant_mechanism = initialize_clusant_mechanism(epsilon)
    
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
        
        # # Scenario 1: Purely Local Model
        # if run_scenario_1_purely_local(local_client, model_name, question, options, correct_answer):
        #     results.local_alone_correct += 1
        
        # # Scenario 2: Non-Private Local + Remote CoT
        # if run_scenario_2_non_private_cot(local_client, model_name, remote_client, remote_cot_model, question, options, correct_answer):
        #     results.non_private_cot_correct += 1

        # Scenario 3.0: Private Local + CoT (PhraseDP+ with metamap phrases)
        if run_scenario_3_private_local_cot(local_client, model_name, remote_client, remote_cot_model, sbert_model, question, options, correct_answer, 'phrasedp', epsilon):
            results.old_phrase_dp_local_cot_correct += 1
        
        # Scenario 3.1: Private Local + CoT (PhraseDP+ with metamap phrases)
        if run_scenario_3_private_local_cot(local_client, model_name, remote_client, remote_cot_model, sbert_model, question, options, correct_answer, 'phrasedp', epsilon, metamap_phrases=metamap_phrases):
            results.phrase_dp_plus_correct += 1
    

        exit(0)
        # # Scenario 3.2: Private Local + CoT (InferDPT WITHOUT Batch Options)
        # if run_scenario_3_private_local_cot(local_client, model_name, remote_client, remote_cot_model, sbert_model, question, options, correct_answer, 'inferdpt', epsilon):
        #     results.inferdpt_local_cot_correct += 1
        
        # # Scenario 3.3: Private Local + CoT (SANTEXT+ WITHOUT Batch Options)
        # if run_scenario_3_private_local_cot(local_client, model_name, remote_client, remote_cot_model, sbert_model, question, options, correct_answer, 'santext', epsilon):
        #     results.santext_local_cot_correct += 1
        
        # ACTIVE: Scenario 3.4: Private Local + CoT (CUSTEXT+)
        # if run_scenario_3_private_local_cot(local_client, model_name, remote_client, remote_cot_model, sbert_model, question, options, correct_answer, 'custext', epsilon):
        #     results.custext_local_cot_correct += 1

        # ACTIVE: Scenario 3.5: Private Local + CoT (CluSanT)
        # if run_scenario_3_private_local_cot(local_client, model_name, remote_client, remote_cot_model, sbert_model, question, options, correct_answer, 'clusant', epsilon):
        #     results.clusant_local_cot_correct += 1
        
        # Scenario 4: Purely Remote Model
        if run_scenario_4_purely_remote(remote_client, remote_llm_model, question, options, correct_answer):
            results.purely_remote_correct += 1
    
    # Print final results
    def print_accuracy(name, correct, total):
        """Helper function to print accuracy results."""
        percentage = correct/total*100 if total > 0 else 0
        print(f"{name} Accuracy: {correct}/{total} = {percentage:.2f}%")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - ALL MECHANISMS (NO BATCH OPTIONS)")
    print(f"{'='*60}")
    print_accuracy("1. Purely Local Model (Baseline)", results.local_alone_correct, results.total_questions)
    print_accuracy("2. Non-Private Local Model + Remote CoT", results.non_private_cot_correct, results.total_questions)
    print_accuracy("3.0. Private Local Model + CoT (Phrase DP)", results.old_phrase_dp_local_cot_correct, results.total_questions)
    print_accuracy("3.1. Private Local Model + CoT (Phrase DP+)", results.phrase_dp_plus_correct, results.total_questions)
    print_accuracy("3.2. Private Local Model + CoT (InferDPT)", results.inferdpt_local_cot_correct, results.total_questions)
    print_accuracy("3.3. Private Local Model + CoT (SANTEXT+)", results.santext_local_cot_correct, results.total_questions)
    print_accuracy("4. Purely Remote Model", results.purely_remote_correct, results.total_questions)
    
    return results


def test_single_question(question_index=0, local_model="meta-llama/Meta-Llama-3.1-8B-Instruct", epsilon=1.0, remote_cot_model="gpt-4o-mini", remote_llm_model="gpt-4o"):
    """Test a single question from the dataset by index."""
    print(f"{CYAN}Testing single question (index {question_index}) from MedQA{RESET}")
    return run_experiment_for_model(
        local_model,
        epsilon,
        remote_cot_model,
        remote_llm_model,
        question_indices=[question_index],
    )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="MedQA USMLE experiment runner (question-only, no options fed to LLMs)",
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
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Local model name for Nebius.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        help="Epsilon value for differential privacy mechanisms.",
    )
    parser.add_argument(
        "--remote-model",
        type=str,
        default="claude-3-haiku-20240307",
        help="Remote model to use for both CoT generation and final LLM inference.",
    )

    args = parser.parse_args(argv)

    final_remote_model = args.remote_model

    if args.index is not None:
        test_single_question(
            args.index, 
            args.model, 
            args.epsilon, 
            final_remote_model,
        )
    else:
        run_experiment_for_model(
            args.model, 
            args.epsilon, 
            final_remote_model,
        )


if __name__ == "__main__":
    main()

