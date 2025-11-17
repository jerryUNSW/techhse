#!/usr/bin/env python3
"""
MedQA USMLE Experiment Script (New Version with Few-Shot Support)
=================================================================

A script for experimenting with MedQA USMLE (Medical Question Answering) dataset
using privacy-preserving approaches. This script tests different scenarios
without feeding the multiple choice options to the LLMs - only the question text.

This script tests different scenarios:
1. Purely Local Model (Baseline)
2. Non-Private Local Model + Remote CoT
3.0. Private Local Model + CoT (Phrase DP)
3.1. Private Local Model + CoT (Phrase DP+)
3.2. Private Local Model + CoT (Phrase DP+ with Few-Shot)
4. Purely Remote Model

Author: Tech4HSE Team
Date: 2025-01-XX
"""

import os
import sys
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import random
from datasets import load_dataset
from datetime import datetime

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
from experiment_db_writer import ExperimentDBWriter

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

# ANSI color codes for better console output
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def load_sentence_bert():
    """Load Sentence-BERT model for similarity computation."""
    print(f"{CYAN}Loading Sentence-BERT model...{RESET}", flush=True)
    return SentenceTransformer('all-MiniLM-L6-v2')

def get_lightning_client():
    """Get Lightning AI client."""
    api_key = os.getenv('LIGHTNING_API')
    if not api_key:
        raise ValueError("LIGHTNING_API not found in environment variables")
    return openai.OpenAI(
        base_url="https://lightning.ai/api/v1/",
        api_key=api_key,
    )

def get_remote_llm_client():
    """
    Get remote LLM client - tries Lightning API first, falls back to OpenAI.
    
    Returns:
        OpenAI client (either Lightning or standard OpenAI)
    """
    # Try Lightning API first
    try:
        client = get_lightning_client()
        print(f"{GREEN}Lightning API client initialized successfully{RESET}", flush=True)
        return client
    except Exception as e:
        # Fall back to OpenAI if Lightning fails
        print(f"{YELLOW}Lightning API not available ({e}), falling back to OpenAI API{RESET}", flush=True)
        
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Neither LIGHTNING_API nor OPENAI_API_KEY found in environment variables")
        print(f"{GREEN}OpenAI client initialized successfully{RESET}", flush=True)
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
        print(f"{RED}Error in local model inference: {e}{RESET}", flush=True)
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
        print(f"{RED}Error in local model with CoT inference: {e}{RESET}", flush=True)
        return "Error"

def get_default_few_shot_examples():
    """Return default 3-shot examples for CoT generation with perturbed questions."""
    return [
        (
            "Here is the (possibly perturbed) question:\nA patient who is experiencing symptoms of a serious infection is brought to the emergency department. She exhibits signs of high fever, chills, and severe abdominal pain in the lower left quadrant. Her vital signs indicate elevated body temperature, rapid pulse, and increased respiratory rate. Blood pressure is critically low. A pelvic examination reveals signs of infection in the reproductive organs, accompanied by blood discharge and tenderness. Laboratory results indicate low platelet count, low fibrinogen levels, and elevated D-dimer levels, suggesting a coagulation disorder. Upon further analysis of the patient's blood, a specific chemical compound is identified when treated with phenol at a high temperature. Considering these findings, which microorganism is most likely to be isolated from a blood culture?\n\nPlease provide a clear, step-by-step chain-of-thought reasoning to solve this question. Do NOT provide the final answer; provide only the reasoning steps.",
            "Step 1: Identify key clinical findings.\n- High fever, chills, severe LLQ abdominal pain → suggests pelvic infection/sepsis\n- Hypotension, tachycardia, tachypnea → septic shock\n- Low platelets, low fibrinogen, elevated D-dimer → disseminated intravascular coagulation (DIC)\n- Pelvic exam findings → reproductive tract infection source\n- Chemical compound identified with phenol at high temperature → this is endotoxin (LPS)\n\nStep 2: Link DIC to endotoxin.\n- DIC is commonly triggered by Gram-negative bacterial endotoxin (LPS)\n- Endotoxin activates coagulation cascade → consumptive coagulopathy\n- The phenol-water extraction method isolates LPS from Gram-negative bacteria\n\nStep 3: Determine likely organism.\n- Pelvic sepsis with DIC and endotoxin detection → Gram-negative bacteria\n- Common organisms in pelvic sepsis: E. coli, Klebsiella, other Enterobacteriaceae\n- The presence of endotoxin (LPS) confirms Gram-negative etiology"
        ),
        (
            "Here is the (possibly perturbed) question:\nA patient presents to a physician with right flank pain, has a background of diabetes and peripheral artery disease, exhibits elevated blood pressure and abdominal tenderness, and undergoes imaging showing dilation of the ureter and renal pelvis, necessitating identification of the underlying cause.\n\nPlease provide a clear, step-by-step chain-of-thought reasoning to solve this question. Do NOT provide the final answer; provide only the reasoning steps.",
            "Step 1: Interpret the imaging finding.\n- Dilation of ureter and renal pelvis → hydronephrosis\n- Hydronephrosis indicates obstruction to urine outflow\n- Both ureter and renal pelvis dilated → obstruction is distal (at or below ureterovesical junction)\n\nStep 2: Correlate with clinical presentation.\n- Right flank pain → renal/ureteral origin\n- Acute onset with pain → suggests obstructive cause\n- Diabetes and PAD are comorbidities but not directly causing hydronephrosis\n\nStep 3: Identify most likely cause.\n- Unilateral hydronephrosis with acute flank pain → ureteric stone most common\n- Stone causes obstruction → dilation upstream\n- Other causes (stricture, tumor) less common and usually more gradual onset"
        ),
        (
            "Here is the (possibly perturbed) question:\nA patient with a history of hypertension and asthma, who presented with acute chest pain, is being treated for a cardiovascular condition and is undergoing a specific medical procedure. He has certain medications, appears to be experiencing symptoms, and has undergone certain diagnostic tests. Which medication should be added to his regimen, in addition to aspirin, to ensure optimal treatment for his condition?\n\nPlease provide a clear, step-by-step chain-of-thought reasoning to solve this question. Do NOT provide the final answer; provide only the reasoning steps.",
            "Step 1: Identify the clinical scenario.\n- Acute chest pain + cardiovascular condition + medical procedure → acute coronary syndrome (ACS) with percutaneous coronary intervention (PCI)\n- Already on aspirin → standard antiplatelet therapy initiated\n\nStep 2: Recall standard ACS/PCI treatment.\n- Dual antiplatelet therapy (DAPT) is cornerstone: aspirin + P2Y12 inhibitor\n- P2Y12 inhibitors: clopidogrel, prasugrel, or ticagrelor\n- DAPT reduces stent thrombosis and recurrent ischemic events\n\nStep 3: Consider comorbidities.\n- Asthma → beta-blockers must be used cautiously (can cause bronchospasm)\n- Question asks what to add 'in addition to aspirin' → classic clue for DAPT\n- P2Y12 inhibitor is the standard addition, not beta-blockers or anticoagulants"
        )
    ]

def generate_cot_from_remote_llm(client, model_name, question, options=None, max_tokens=512, 
                                  use_few_shot_cot=False, few_shot_examples=None, few_shot_style="dialog"):
    """Generate Chain-of-Thought from remote LLM.
    
    Args:
        client: Remote LLM client
        model_name: Model name
        question: Question text (possibly perturbed)
        options: Optional answer choices dict
        max_tokens: Max tokens for response
        use_few_shot_cot: If True, include few-shot examples (default: False for backward compatibility)
        few_shot_examples: List of (prompt, reasoning) tuples. If None and use_few_shot_cot=True, uses defaults.
        few_shot_style: "dialog" (conversation format) or "system_block" (text block)
    
    Returns:
        CoT reasoning text
    """
    # GPT-5 requires higher max_completion_tokens to avoid empty responses
    # Known issue: GPT-5 can return empty content with finish_reason: length if limit is too low
    # Longer questions need more tokens - use 2048 to handle all cases
    if "gpt-5" in model_name.lower():
        if max_tokens < 2048:
            print(f"{CYAN}INFO: Increasing max_tokens from {max_tokens} to 2048 for GPT-5 to avoid empty responses (handles longer questions){RESET}", flush=True)
            max_tokens = 2048
    
    # Build a clear, generic CoT prompt that works for medical and non-medical questions.
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
        # Prepare system message
        system_content = "You are an expert reasoner. Provide a clear, step-by-step chain of thought to analyze the given question. Focus on domain-appropriate reasoning and knowledge."
        
        # Add few-shot guidance if enabled
        if use_few_shot_cot:
            system_content += " When reasoning about perturbed or sanitized questions, treat any placeholders or noise as opaque - reason from available evidence and context without attempting to reconstruct masked content."
        
        # Build messages list
        messages = []
        
        if use_few_shot_cot:
            # Get examples (use provided or defaults)
            examples = few_shot_examples if few_shot_examples is not None else get_default_few_shot_examples()
            
            if few_shot_style == "dialog":
                # Add system message
                messages.append({"role": "system", "content": system_content})
                
                # Add few-shot examples as conversation pairs
                for example_prompt, example_reasoning in examples:
                    messages.append({"role": "user", "content": example_prompt})
                    messages.append({"role": "assistant", "content": example_reasoning})
                
                # Add current question as final user message
                messages.append({"role": "user", "content": prompt})
            else:  # system_block
                # Append examples as text block to system message
                example_block = "\n\nFew-shot examples:\n\n"
                for i, (example_prompt, example_reasoning) in enumerate(examples, 1):
                    example_block += f"Example {i}:\nUser: {example_prompt}\nAssistant: {example_reasoning}\n\n"
                system_content += example_block
                messages.append({"role": "system", "content": system_content})
                messages.append({"role": "user", "content": prompt})
        else:
            # Standard flow (backward compatible)
            messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": prompt})

        # Print the prompt for visibility (debug)
        print(f"\n{CYAN}=== Remote CoT Prompt ==={RESET}\n{prompt}\n{CYAN}=== End Prompt ==={RESET}\n", flush=True)
        if use_few_shot_cot:
            examples = few_shot_examples if few_shot_examples is not None else get_default_few_shot_examples()
            print(f"{CYAN}Few-shot CoT enabled ({few_shot_style} style, {len(examples)} examples){RESET}\n", flush=True)

        response = create_completion_with_model_support(
            client, model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0
        )
        
        # Debug: Print full response structure for investigation
        print(f"{CYAN}DEBUG: Response type: {type(response)}{RESET}", flush=True)
        print(f"{CYAN}DEBUG: Response choices length: {len(response.choices)}{RESET}", flush=True)
        
        choice = response.choices[0]
        print(f"{CYAN}DEBUG: Choice type: {type(choice)}{RESET}", flush=True)
        print(f"{CYAN}DEBUG: Choice attributes: {dir(choice)}{RESET}", flush=True)
        
        message = choice.message
        print(f"{CYAN}DEBUG: Message type: {type(message)}{RESET}", flush=True)
        print(f"{CYAN}DEBUG: Message attributes: {dir(message)}{RESET}", flush=True)
        
        content = message.content
        print(f"{CYAN}DEBUG: Content type: {type(content)}, Content value: {repr(content)}{RESET}", flush=True)
        
        # Check for finish_reason
        finish_reason = getattr(choice, 'finish_reason', None) or getattr(message, 'finish_reason', None)
        if finish_reason:
            print(f"{CYAN}DEBUG: Finish reason: {finish_reason}{RESET}", flush=True)
        
        # Throw error if content is None or empty
        if content is None:
            error_msg = f"ERROR: Remote CoT model returned None content. Model: {model_name}, Finish reason: {finish_reason}"
            print(f"{RED}{error_msg}{RESET}", flush=True)
            raise ValueError(error_msg)
        
        content_str = content.strip() if content else ""
        if not content_str:
            error_msg = f"ERROR: Remote CoT model returned empty content. Model: {model_name}, Finish reason: {finish_reason}, Content: {repr(content)}"
            print(f"{RED}{error_msg}{RESET}", flush=True)
            raise ValueError(error_msg)
        
        return content_str
    except Exception as e:
        print(f"{RED}Error in remote CoT generation: {e}{RESET}", flush=True)
        return "Error"

def get_answer_from_remote_llm(client, model_name, question, options, max_tokens=256):
    """Get answer directly from remote LLM."""
    
    # GPT-5 requires higher max_completion_tokens to avoid empty responses
    # Known issue: GPT-5 can return empty content with finish_reason: length if limit is too low
    if "gpt-5" in model_name.lower():
        if max_tokens < 2048:
            print(f"{CYAN}INFO: Increasing max_tokens from {max_tokens} to 2048 for GPT-5 to avoid empty responses{RESET}", flush=True)
            max_tokens = 2048
    
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
        content = response.choices[0].message.content
        # Debug: Check for empty/None content (especially for GPT-5 Mini)
        if content is None:
            print(f"{YELLOW}Warning: Remote model returned None content{RESET}", flush=True)
            return "Error"
        content_str = content.strip() if content else ""
        if not content_str and "gpt-5" in model_name.lower():
            # GPT-5 Mini specific: check finish_reason
            choice = response.choices[0]
            finish_reason = getattr(choice, 'finish_reason', None) or getattr(choice.message, 'finish_reason', None)
            if finish_reason:
                print(f"{YELLOW}Warning: GPT-5 Mini empty response, finish_reason={finish_reason}{RESET}", flush=True)
        return content_str
    except Exception as e:
        print(f"{RED}Error in remote LLM inference: {e}{RESET}", flush=True)
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
    print(f"\n{BLUE}--- Scenario 1: Purely Local Model (Baseline) ---{RESET}", flush=True)
    
    try:
        local_response = get_answer_from_local_model_alone(client, model_name, question, options)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer: {local_response}", flush=True)
        print(f"Extracted Letter: {predicted_letter}", flush=True)
        print(f"Correct Answer: {correct_answer}", flush=True)
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}", flush=True)
        
        return {
            "answer": predicted_letter,
            "is_correct": is_correct,
            "response_text": local_response
        }
    except Exception as e:
        abort_on_quota_error(e, "Nebius")
        print(f"{RED}Error during purely local model inference: {e}{RESET}", flush=True)
        return {
            "answer": "Error",
            "is_correct": False,
            "response_text": f"Error: {str(e)}"
        }

def run_scenario_2_non_private_cot(client, model_name, remote_client, remote_model, question, options, correct_answer):
    """Scenario 2: Non-Private Local Model + Remote CoT."""
    print(f"\n{BLUE}--- Scenario 2: Non-Private Local Model + Remote CoT ---{RESET}", flush=True)
    
    try:
        # Generate CoT from remote LLM
        print(f"{YELLOW}2a. Generating CoT from ORIGINAL Question with REMOTE LLM...{RESET}", flush=True)
        cot_text = generate_cot_from_remote_llm(remote_client, remote_model, question)
        print(f"Generated Chain-of-Thought (Remote, Non-Private):\n{cot_text}", flush=True)
        
        # Use local model with CoT
        print(f"{YELLOW}2b. Running Local Model with Non-Private CoT...{RESET}", flush=True)
        local_response = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer (Non-Private CoT-Aided): {local_response}", flush=True)
        print(f"Extracted Letter: {predicted_letter}", flush=True)
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}", flush=True)
        
        return {
            "answer": predicted_letter,
            "is_correct": is_correct,
            "response_text": local_response,
            "cot_text": cot_text
        }
    except Exception as e:
        abort_on_quota_error(e, "API")
        print(f"{RED}Error during non-private CoT inference: {e}{RESET}", flush=True)
        return {
            "answer": "Error",
            "is_correct": False,
            "response_text": f"Error: {str(e)}",
            "cot_text": ""
        }

def run_scenario_3_private_local_cot(client, model_name, remote_client, remote_model, sbert_model, question, options, correct_answer, privacy_mechanism, epsilon, metamap_phrases=None, use_few_shot_cot=False, few_shot_examples=None, few_shot_style="dialog"):
    """Scenario 3: Private Local Model + CoT (Generic function for all privacy mechanisms without batch options).
    
    Args:
        use_few_shot_cot: If True, enable few-shot prompting for CoT generation (default: False for backward compatibility)
        few_shot_examples: List of (prompt, reasoning) tuples for few-shot. If None and use_few_shot_cot=True, uses defaults.
        few_shot_style: "dialog" or "system_block" style for few-shot examples
    """
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
    
    print(f"\n{BLUE}--- Scenario 3: Private Local Model + CoT ({mechanism_name}) ---{RESET}", flush=True)
    if use_few_shot_cot:
        print(f"{CYAN}Few-shot CoT enabled for this scenario{RESET}", flush=True)
    
    try:
        # Apply privacy mechanism to the question
        print(f"{YELLOW}3a. Applying {mechanism_name} sanitization...{RESET}", flush=True)
        
        if privacy_mechanism == 'phrasedp':
            # Always use the old PhraseDP single-call pipeline.
            # If metamap_phrases is provided, enable medical mode; otherwise use normal mode.
            from sanitization_methods import config as sm_config
            nebius_client = utils.get_nebius_client()
            nebius_model_name = sm_config.get('local_model')
            mode = "medqa-ume" if metamap_phrases else "normal"
            # Debug: show mode and metamap phrases summary for visibility between 3.0 and 3.1
            print(f"{YELLOW}PhraseDP mode: {mode}{RESET}", flush=True)
            if metamap_phrases:
                sample_phrases = ", ".join(metamap_phrases[:20])
                sample_phrases += "..." if len(metamap_phrases) > 20 else ""
                print(f"Metamap phrases ({len(metamap_phrases)}): {sample_phrases}", flush=True)
            else:
                print("Metamap phrases: None", flush=True)
            
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
        print(f"Perturbed Question (preview): {preview}", flush=True)

        # Keep options unchanged - only perturb the question for privacy
        print(f"{YELLOW}3b. Keeping options unchanged for local model...{RESET}", flush=True)
        print(f"Original Options: {options}", flush=True)
        
        # Generate CoT from perturbed question with original options
        print(f"{YELLOW}3c. Generating CoT from Perturbed Question with REMOTE LLM...{RESET}", flush=True)
        cot_text = generate_cot_from_remote_llm(
            remote_client, remote_model, perturbed_question,
            use_few_shot_cot=use_few_shot_cot,
            few_shot_examples=few_shot_examples,
            few_shot_style=few_shot_style
        )
        print(f"Generated Chain-of-Thought (Remote, Private):\n{cot_text}", flush=True)
        
        # Use local model with private CoT
        print(f"{YELLOW}3d. Running Local Model with Private CoT...{RESET}", flush=True)
        local_response = get_answer_from_local_model_with_cot(client, model_name, question, options, cot_text)
        predicted_letter = extract_letter_from_answer(local_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Local Answer (Private CoT-Aided): {local_response}", flush=True)
        print(f"Extracted Letter: {predicted_letter}", flush=True)
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}", flush=True)
        
        return {
            "answer": predicted_letter,
            "is_correct": is_correct,
            "response_text": local_response,
            "perturbed_question": perturbed_question,
            "cot_text": cot_text,
            "metamap_phrases_used": metamap_phrases if metamap_phrases else None,
            "few_shot_cot": use_few_shot_cot
        }
    except Exception as e:
        print(f"{RED}Error during {mechanism_name} private CoT-aided inference: {e}{RESET}", flush=True)
        return {
            "answer": "Error",
            "is_correct": False,
            "response_text": f"Error: {str(e)}",
            "perturbed_question": "",
            "cot_text": "",
            "metamap_phrases_used": None,
            "few_shot_cot": use_few_shot_cot
        }

def run_scenario_4_purely_remote(remote_client, remote_model, question, options, correct_answer):
    """Scenario 4: Purely Remote Model."""
    print(f"\n{BLUE}--- Scenario 4: Purely Remote Model ---{RESET}", flush=True)
    
    try:
        print(f"{YELLOW}4a. Running Purely Remote LLM...{RESET}", flush=True)
        remote_response = get_answer_from_remote_llm(remote_client, remote_model, question, options)
        predicted_letter = extract_letter_from_answer(remote_response)
        is_correct = check_mcq_correctness(predicted_letter, correct_answer)
        
        print(f"Purely Remote Answer: {remote_response}", flush=True)
        print(f"Extracted Letter: {predicted_letter}", flush=True)
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}", flush=True)
        
        return {
            "answer": predicted_letter,
            "is_correct": is_correct,
            "response_text": remote_response
        }
    except Exception as e:
        print(f"{RED}Error during purely remote model inference: {e}{RESET}", flush=True)
        return {
            "answer": "Error",
            "is_correct": False,
            "response_text": f"Error: {str(e)}"
        }

def run_experiment_for_model(
    model_name,
    epsilon_values,
    remote_cot_model,
    remote_llm_model,
    question_indices: list[int] | None = None,
    start_index: int = 0,
    num_samples: int = 100,
    run_epsilon_independent: bool = True,
    run_epsilon_dependent: bool = True,
    skip_phrasedp_normal: bool = False,
    skip_phrasedp_plus_normal: bool = False,
    skip_phrasedp_plus_fewshot: bool = False,
):
    """Run the MedQA experiment for a given local model with multiple epsilon values.
    
    Args:
        model_name: Local model name
        epsilon_values: List of epsilon values to test
        remote_cot_model: Remote model for CoT generation
        remote_llm_model: Remote model for direct answering
        question_indices: Optional list of specific question indices to test
        start_index: Starting index for question selection
        num_samples: Number of questions to test
        run_epsilon_independent: If True, run scenarios that don't depend on epsilon (Local, Local+CoT, Remote)
        run_epsilon_dependent: If True, run scenarios that depend on epsilon (PhraseDP, PhraseDP+, PhraseDP+ few-shot)
    
    Returns:
        Tuple of (MedQAExperimentResults object, per_question_results dict)
    """
    
    print(f"{CYAN}Starting MedQA Experiment with model: {model_name}{RESET}", flush=True)
    print(f"{CYAN}Epsilon values: {epsilon_values}{RESET}", flush=True)
    
    # Load dataset
    print(f"{CYAN}Loading MedQA dataset...{RESET}", flush=True)
    print(f"{YELLOW}Note: MedQA contains clinical vignettes with patient scenarios in questions{RESET}", flush=True)
    dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
    
    if question_indices is not None and len(question_indices) == 0:
        print(f"{YELLOW}No question indices provided. Nothing to run.{RESET}", flush=True)
        return None, {}
    
    if question_indices is not None:
        selected_indices = [idx for idx in question_indices if 0 <= idx < len(dataset)]
        if not selected_indices:
            print(f"{RED}Provided question indices are out of range. Dataset has {len(dataset)} questions.{RESET}", flush=True)
            return None, {}
    else:
        selected_indices = list(range(start_index, min(start_index + num_samples, len(dataset))))
    
    print(
        f"{CYAN}Testing {len(selected_indices)} question(s) from MedQA test set "
        f"(indices: {selected_indices[:10]}{'...' if len(selected_indices) > 10 else ''}){RESET}", flush=True
    )
    
    # Get clients - exit if any are not available
    try:
        # Try to get OpenAI/Lightning client first, but don't fail if only Anthropic is available
        try:
            remote_client = get_remote_llm_client()
            # Note: get_remote_llm_client() already prints success message
        except:
            # If OpenAI fails, try Anthropic
            try:
                remote_client = get_anthropic_client()
                print(f"{GREEN}Anthropic client initialized successfully{RESET}", flush=True)
            except Exception as e:
                print(f"{RED}Failed to initialize any remote LLM client: {e}{RESET}", flush=True)
                print(f"{RED}Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env{RESET}", flush=True)
                return None, {}
    except Exception as e:
        abort_on_quota_error(e, "API")
        print(f"{RED}Failed to initialize remote LLM client: {e}{RESET}", flush=True)
        print(f"{RED}Cannot proceed without remote client. Exiting.{RESET}", flush=True)
        return None, {}
    
    try:
        local_client = utils.get_nebius_client()
        print(f"{GREEN}Local (Nebius) client initialized successfully{RESET}", flush=True)
    except Exception as e:
        abort_on_quota_error(e, "Nebius")
        print(f"{RED}Failed to initialize Nebius local client: {e}{RESET}", flush=True)
        print(f"{RED}Cannot proceed without local client. Exiting.{RESET}", flush=True)
        return None, {}
    
    # Load Sentence-BERT for similarity computation
    sbert_model = load_sentence_bert()
    
    # Initialize results tracking
    per_question_results = {}
    
    # Track counts for summary
    local_correct = 0
    local_cot_correct = 0
    phrasedp_correct = {eps: 0 for eps in epsilon_values}
    phrasedp_plus_correct = {eps: 0 for eps in epsilon_values}
    phrasedp_plus_fewshot_correct = {eps: 0 for eps in epsilon_values}
    remote_correct = 0
    total_questions = 0
    
    # Prepare output path for incremental saving
    output_dir = "exp/new-exp"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename early so we can save incrementally
    local_model_clean = model_name.replace("/", "_").replace("-", "_")
    remote_model_clean = remote_cot_model.replace("/", "_").replace("-", "_")
    eps_str = "_".join([str(int(eps) if eps.is_integer() else eps) for eps in epsilon_values])
    filename = f"medqa_usmle_local_{local_model_clean}_remote_{remote_model_clean}_{num_samples}q_eps{eps_str}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Initialize empty results file for incremental saving
    experiment_info = {
            "dataset": "MedQA-USMLE",
            "local_model": model_name,
            "remote_model": remote_cot_model,
            "num_samples": num_samples,
            "epsilon_values": epsilon_values,
            "start_index": start_index,
        "run_epsilon_independent": run_epsilon_independent,
        "run_epsilon_dependent": run_epsilon_dependent,
    }
    
    initial_data = {
        "experiment_info": {
            **experiment_info,
            "questions_completed": 0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "summary_results": {},
        "per_question_results": {}
    }
    
    # Initialize database writer
    try:
        db_writer = ExperimentDBWriter('tech4hse_results.db', experiment_info)
    except Exception as e:
        print(f"{YELLOW}Warning: Could not initialize database writer: {e}{RESET}", flush=True)
        db_writer = None
    with open(output_path, 'w') as f:
        json.dump(initial_data, f, indent=2)
    print(f"{CYAN}Initialized results file: {output_path}{RESET}", flush=True)
    
    sample_questions = dataset.select(selected_indices)
    
    for i, item in enumerate(sample_questions):
        dataset_idx = selected_indices[i]
        print(f"\n{YELLOW}{'='*60}{RESET}", flush=True)
        print(f"{YELLOW}--- Question {i+1}/{len(sample_questions)} (Dataset idx: {dataset_idx}) ---{RESET}", flush=True)
        print(f"{YELLOW}{'='*60}{RESET}", flush=True)
        
        question = item['question']
        options = item['options']  # Already a dict with A, B, C, D keys
        correct_answer = item['answer_idx']  # This is already a letter (A, B, C, D)
        
        # Extract additional metadata
        meta_info = item.get('meta_info', 'N/A')
        metamap_phrases = item.get('metamap_phrases', [])
        
        print(f"Question: {question[:200]}{'...' if len(question) > 200 else ''}", flush=True)
        print(f"Options:", flush=True)
        for key, value in options.items():
            print(f"  {key}) {value}", flush=True)
        print(f"Correct Answer: {correct_answer}", flush=True)
        print(f"MetaMap Phrases ({len(metamap_phrases)}): {', '.join(metamap_phrases[:10])}{'...' if len(metamap_phrases) > 10 else ''}", flush=True)
        
        total_questions += 1
        
        # Initialize per-question result structure
        q_result = {
            "question_idx": dataset_idx,
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "metamap_phrases": metamap_phrases,
            "epsilon_results": {}
        }
        
        # Run epsilon-independent scenarios once
        if run_epsilon_independent:
            # Scenario 1: Purely Local Model
            local_result = run_scenario_1_purely_local(local_client, model_name, question, options, correct_answer)
            if local_result["is_correct"]:
                local_correct += 1
            q_result["local_answer"] = local_result["answer"]
            q_result["local_response"] = local_result["response_text"]
            
            # Write to database
            if db_writer:
                try:
                    db_writer.write_epsilon_independent(dataset_idx, 'Local', {
                        'original_question': question,
                        'generated_answer': local_result['answer'],
                        'ground_truth_answer': correct_answer,
                        'options': options,
                        'metamap_phrases': metamap_phrases,
                        'is_correct': local_result['is_correct'],
                        'local_model': model_name,
                        'remote_model': None,
                    })
                except Exception as e:
                    print(f"{YELLOW}Warning: Could not write Local result to DB: {e}{RESET}", flush=True)
            
            # Scenario 2: Non-Private Local + Remote CoT
            local_cot_result = run_scenario_2_non_private_cot(local_client, model_name, remote_client, remote_cot_model, question, options, correct_answer)
            if local_cot_result["is_correct"]:
                local_cot_correct += 1
            q_result["local_cot_answer"] = local_cot_result["answer"]
            q_result["local_cot_response"] = local_cot_result["response_text"]
            q_result["local_cot_cot_text"] = local_cot_result["cot_text"]
            
            # Write to database
            if db_writer:
                try:
                    db_writer.write_epsilon_independent(dataset_idx, 'Local+CoT', {
                        'original_question': question,
                        'generated_answer': local_cot_result['answer'],
                        'ground_truth_answer': correct_answer,
                        'options': options,
                        'metamap_phrases': metamap_phrases,
                        'is_correct': local_cot_result['is_correct'],
                        'cot_text': local_cot_result.get('cot_text'),
                        'local_model': model_name,
                        'remote_model': remote_cot_model,
                    })
                except Exception as e:
                    print(f"{YELLOW}Warning: Could not write Local+CoT result to DB: {e}{RESET}", flush=True)
            
            # Scenario 4: Purely Remote Model
            remote_result = run_scenario_4_purely_remote(remote_client, remote_llm_model, question, options, correct_answer)
            if remote_result["is_correct"]:
                remote_correct += 1
            q_result["remote_answer"] = remote_result["answer"]
            q_result["remote_response"] = remote_result["response_text"]
            
            # Write to database
            if db_writer:
                try:
                    db_writer.write_epsilon_independent(dataset_idx, 'Remote', {
                        'original_question': question,
                        'generated_answer': remote_result['answer'],
                        'ground_truth_answer': correct_answer,
                        'options': options,
                        'metamap_phrases': metamap_phrases,
                        'is_correct': remote_result['is_correct'],
                        'local_model': model_name,
                        'remote_model': remote_llm_model,
                    })
                except Exception as e:
                    print(f"{YELLOW}Warning: Could not write Remote result to DB: {e}{RESET}", flush=True)
        
        # Run epsilon-dependent scenarios for each epsilon
        if run_epsilon_dependent:
            for epsilon in epsilon_values:
                eps_key = f"epsilon_{epsilon}"
                eps_result = {}
                
                # Scenario 3.0: Private Local + CoT (PhraseDP without metamap phrases)
                if not skip_phrasedp_normal:
                    phrasedp_result = run_scenario_3_private_local_cot(
                        local_client, model_name, remote_client, remote_cot_model, sbert_model, 
                        question, options, correct_answer, 'phrasedp', epsilon,
                        use_few_shot_cot=False
                    )
                    if phrasedp_result["is_correct"]:
                        phrasedp_correct[epsilon] += 1
                    eps_result["phrasedp_answer"] = phrasedp_result["answer"]
                    eps_result["phrasedp_response"] = phrasedp_result["response_text"]
                    eps_result["phrasedp_perturbed_question"] = phrasedp_result["perturbed_question"]
                    eps_result["phrasedp_cot_text"] = phrasedp_result["cot_text"]
                    
                    # Write to database
                    if db_writer:
                        try:
                            db_writer.write_epsilon_dependent(dataset_idx, 'PhraseDP', epsilon, {
                                'original_question': question,
                                'perturbed_question': phrasedp_result['perturbed_question'],
                                'options': options,
                                'metamap_phrases': metamap_phrases,
                                'induced_cot': phrasedp_result.get('cot_text'),
                                'generated_answer': phrasedp_result['answer'],
                                'ground_truth_answer': correct_answer,
                                'is_correct': phrasedp_result['is_correct'],
                                'local_model': model_name,
                                'remote_model': remote_cot_model,
                            })
                        except Exception as e:
                            print(f"{YELLOW}Warning: Could not write PhraseDP result to DB: {e}{RESET}", flush=True)
                
                # Scenario 3.1: Private Local + CoT (PhraseDP+ with metamap phrases)
                if not skip_phrasedp_plus_normal:
                    phrasedp_plus_result = run_scenario_3_private_local_cot(
                        local_client, model_name, remote_client, remote_cot_model, sbert_model, 
                        question, options, correct_answer, 'phrasedp', epsilon,
                        metamap_phrases=metamap_phrases,
                        use_few_shot_cot=False
                    )
                    if phrasedp_plus_result["is_correct"]:
                        phrasedp_plus_correct[epsilon] += 1
                    eps_result["phrasedp_plus_answer"] = phrasedp_plus_result["answer"]
                    eps_result["phrasedp_plus_response"] = phrasedp_plus_result["response_text"]
                    eps_result["phrasedp_plus_perturbed_question"] = phrasedp_plus_result["perturbed_question"]
                    eps_result["phrasedp_plus_cot_text"] = phrasedp_plus_result["cot_text"]
                    eps_result["phrasedp_plus_metamap_phrases"] = phrasedp_plus_result["metamap_phrases_used"]
                    
                    # Write to database
                    if db_writer:
                        try:
                            db_writer.write_epsilon_dependent(dataset_idx, 'PhraseDP+', epsilon, {
                                'original_question': question,
                                'perturbed_question': phrasedp_plus_result['perturbed_question'],
                                'options': options,
                                'metamap_phrases': phrasedp_plus_result.get('metamap_phrases_used', metamap_phrases),
                                'induced_cot': phrasedp_plus_result.get('cot_text'),
                                'generated_answer': phrasedp_plus_result['answer'],
                                'ground_truth_answer': correct_answer,
                                'is_correct': phrasedp_plus_result['is_correct'],
                                'local_model': model_name,
                                'remote_model': remote_cot_model,
                            })
                        except Exception as e:
                            print(f"{YELLOW}Warning: Could not write PhraseDP+ result to DB: {e}{RESET}", flush=True)
                
                # Scenario 3.2: Private Local + CoT (PhraseDP+ with few-shot)
                if not skip_phrasedp_plus_fewshot:
                    phrasedp_plus_fewshot_result = run_scenario_3_private_local_cot(
                        local_client, model_name, remote_client, remote_cot_model, sbert_model, 
                        question, options, correct_answer, 'phrasedp', epsilon,
                        metamap_phrases=metamap_phrases,
                        use_few_shot_cot=True
                    )
                    if phrasedp_plus_fewshot_result["is_correct"]:
                        phrasedp_plus_fewshot_correct[epsilon] += 1
                    eps_result["phrasedp_plus_fewshot_answer"] = phrasedp_plus_fewshot_result["answer"]
                    eps_result["phrasedp_plus_fewshot_response"] = phrasedp_plus_fewshot_result["response_text"]
                    eps_result["phrasedp_plus_fewshot_perturbed_question"] = phrasedp_plus_fewshot_result["perturbed_question"]
                    eps_result["phrasedp_plus_fewshot_cot_text"] = phrasedp_plus_fewshot_result["cot_text"]
                    eps_result["phrasedp_plus_fewshot_metamap_phrases"] = phrasedp_plus_fewshot_result["metamap_phrases_used"]
                    
                    # Write to database
                    if db_writer:
                        try:
                            db_writer.write_epsilon_dependent(dataset_idx, 'PhraseDP++', epsilon, {
                                'original_question': question,
                                'perturbed_question': phrasedp_plus_fewshot_result['perturbed_question'],
                                'options': options,
                                'metamap_phrases': phrasedp_plus_fewshot_result.get('metamap_phrases_used', metamap_phrases),
                                'induced_cot': phrasedp_plus_fewshot_result.get('cot_text'),
                                'generated_answer': phrasedp_plus_fewshot_result['answer'],
                                'ground_truth_answer': correct_answer,
                                'is_correct': phrasedp_plus_fewshot_result['is_correct'],
                                'local_model': model_name,
                                'remote_model': remote_cot_model,
                            })
                        except Exception as e:
                            print(f"{YELLOW}Warning: Could not write PhraseDP++ result to DB: {e}{RESET}", flush=True)
                
                q_result["epsilon_results"][eps_key] = eps_result
        
        per_question_results[str(dataset_idx)] = q_result
        
        # Incremental save: Save results after each question
        # This allows monitoring scripts to track progress and prevents data loss
        try:
            # Calculate current summary
            current_summary = {}
            if run_epsilon_independent:
                current_summary["local"] = {"correct": local_correct, "total": total_questions, "accuracy": (local_correct / total_questions * 100) if total_questions > 0 else 0}
                current_summary["local_cot"] = {"correct": local_cot_correct, "total": total_questions, "accuracy": (local_cot_correct / total_questions * 100) if total_questions > 0 else 0}
                current_summary["remote"] = {"correct": remote_correct, "total": total_questions, "accuracy": (remote_correct / total_questions * 100) if total_questions > 0 else 0}
            
            for epsilon in epsilon_values:
                eps_key = f"epsilon_{epsilon}"
                eps_summary = {
                    "phrasedp_plus_fewshot": {"correct": phrasedp_plus_fewshot_correct[epsilon], "total": total_questions, "accuracy": (phrasedp_plus_fewshot_correct[epsilon] / total_questions * 100) if total_questions > 0 else 0},
                }
                if not skip_phrasedp_plus_normal:
                    eps_summary["phrasedp_plus"] = {"correct": phrasedp_plus_correct[epsilon], "total": total_questions, "accuracy": (phrasedp_plus_correct[epsilon] / total_questions * 100) if total_questions > 0 else 0}
                if not skip_phrasedp_normal:
                    eps_summary["phrasedp"] = {"correct": phrasedp_correct[epsilon], "total": total_questions, "accuracy": (phrasedp_correct[epsilon] / total_questions * 100) if total_questions > 0 else 0}
                current_summary[eps_key] = eps_summary
            
            # Save incrementally
            incremental_data = {
                "experiment_info": {
                    "dataset": "MedQA-USMLE",
                    "local_model": model_name,
                    "remote_model": remote_cot_model,
                    "num_samples": num_samples,
                    "epsilon_values": epsilon_values,
                    "start_index": start_index,
                    "questions_completed": len(per_question_results),
                    "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                },
                "summary_results": current_summary,
                "per_question_results": per_question_results
            }
            with open(output_path, 'w') as f:
                json.dump(incremental_data, f, indent=2)
        except Exception as e:
            print(f"{YELLOW}Warning: Could not save incremental results: {e}{RESET}", flush=True)
    
    # Create results summary
    results_summary = {
        "local": {"correct": local_correct, "total": total_questions, "accuracy": (local_correct / total_questions * 100) if total_questions > 0 else 0},
        "local_cot": {"correct": local_cot_correct, "total": total_questions, "accuracy": (local_cot_correct / total_questions * 100) if total_questions > 0 else 0},
        "remote": {"correct": remote_correct, "total": total_questions, "accuracy": (remote_correct / total_questions * 100) if total_questions > 0 else 0},
    }
    
    for epsilon in epsilon_values:
        eps_key = f"epsilon_{epsilon}"
        eps_summary = {}
        if not skip_phrasedp_plus_fewshot:
            eps_summary["phrasedp_plus_fewshot"] = {"correct": phrasedp_plus_fewshot_correct[epsilon], "total": total_questions, "accuracy": (phrasedp_plus_fewshot_correct[epsilon] / total_questions * 100) if total_questions > 0 else 0}
        if not skip_phrasedp_plus_normal:
            eps_summary["phrasedp_plus"] = {"correct": phrasedp_plus_correct[epsilon], "total": total_questions, "accuracy": (phrasedp_plus_correct[epsilon] / total_questions * 100) if total_questions > 0 else 0}
        if not skip_phrasedp_normal:
            eps_summary["phrasedp"] = {"correct": phrasedp_correct[epsilon], "total": total_questions, "accuracy": (phrasedp_correct[epsilon] / total_questions * 100) if total_questions > 0 else 0}
        results_summary[eps_key] = eps_summary
    
    # Finalize database writer
    if db_writer:
        try:
            db_writer.finalize_experiment()
        except Exception as e:
            print(f"{YELLOW}Warning: Could not finalize database writer: {e}{RESET}", flush=True)
    
    return results_summary, per_question_results

def main(argv: list[str] | None = None) -> None:
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MedQA USMLE experiment runner (question-only, no options fed to LLMs) with few-shot support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--epsilons",
        type=str,
        default="1.0,2.0,3.0",
        help="Comma-separated epsilon values (e.g., '1.0,2.0,3.0').",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of questions to test (default: 100).",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Local model name for Nebius.",
    )
    parser.add_argument(
        "--remote-model",
        type=str,
        default="gpt-4o-mini",
        help="Remote model to use for both CoT generation and final LLM inference.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for question selection.",
    )
    parser.add_argument(
        "--skip-epsilon-independent",
        action="store_true",
        help="Skip epsilon-independent scenarios (Local, Local+CoT, Remote).",
    )
    parser.add_argument(
        "--skip-phrasedp-normal",
        action="store_true",
        help="Skip PhraseDP normal mode (without metamap phrases).",
    )
    parser.add_argument(
        "--only-phrasedp-plus",
        action="store_true",
        help="Only test PhraseDP+ (medical mode) and PhraseDP+ Few-Shot. Equivalent to --skip-epsilon-independent --skip-phrasedp-normal.",
    )
    parser.add_argument(
        "--only-phrasedp-plus-fewshot",
        action="store_true",
        help="Only test PhraseDP+ (medical mode + few-shot). Skips PhraseDP+ without few-shot.",
    )
    parser.add_argument(
        "--skip-phrasedp-plus-fewshot",
        action="store_true",
        help="Skip PhraseDP++ (PhraseDP+ with few-shot). Keeps PhraseDP and PhraseDP+.",
    )
    
    args = parser.parse_args(argv)
    
    # Handle --only-phrasedp-plus flag
    if args.only_phrasedp_plus:
        args.skip_epsilon_independent = True
        args.skip_phrasedp_normal = True
    
    # Handle --only-phrasedp-plus-fewshot flag
    if args.only_phrasedp_plus_fewshot:
        args.skip_epsilon_independent = True
        args.skip_phrasedp_normal = True
        args.skip_phrasedp_plus_normal = True
    
    # Parse epsilon values
    epsilon_values = [float(eps.strip()) for eps in args.epsilons.split(',')]
    
    print(f"{CYAN}{'='*60}{RESET}", flush=True)
    print(f"{CYAN}MedQA-USMLE Experiment Configuration{RESET}", flush=True)
    print(f"{CYAN}{'='*60}{RESET}", flush=True)
    print(f"Local Model: {args.local_model}", flush=True)
    print(f"Remote Model: {args.remote_model}", flush=True)
    print(f"Epsilon Values: {epsilon_values}", flush=True)
    print(f"Number of Samples: {args.num_samples}", flush=True)
    print(f"Start Index: {args.start_index}", flush=True)
    if args.skip_epsilon_independent:
        print(f"⚠️  Skipping epsilon-independent scenarios (Local, Local+CoT, Remote)", flush=True)
    if args.skip_phrasedp_normal:
        print(f"⚠️  Skipping PhraseDP normal mode (without metamap)", flush=True)
    if getattr(args, 'skip_phrasedp_plus_normal', False):
        print(f"⚠️  Skipping PhraseDP+ (medical mode) without few-shot", flush=True)
    if getattr(args, 'skip_phrasedp_plus_fewshot', False):
        print(f"⚠️  Skipping PhraseDP++ (PhraseDP+ with few-shot)", flush=True)
    if args.only_phrasedp_plus:
        print(f"✅ Testing ONLY: PhraseDP+ (medical mode) and PhraseDP+ Few-Shot", flush=True)
    if args.only_phrasedp_plus_fewshot:
        print(f"✅ Testing ONLY: PhraseDP+ (medical mode + few-shot)", flush=True)
    print(f"{CYAN}{'='*60}{RESET}", flush=True)
    
    # Run experiment
    summary_results, per_question_results = run_experiment_for_model(
        args.local_model,
        epsilon_values,
        args.remote_model,
        args.remote_model,
        num_samples=args.num_samples,
        start_index=args.start_index,
        run_epsilon_independent=not args.skip_epsilon_independent,
        run_epsilon_dependent=True,
        skip_phrasedp_normal=args.skip_phrasedp_normal,
        skip_phrasedp_plus_normal=getattr(args, 'skip_phrasedp_plus_normal', False),
        skip_phrasedp_plus_fewshot=getattr(args, 'skip_phrasedp_plus_fewshot', False),
    )
    
    if summary_results is None:
        print(f"{RED}Experiment failed to initialize. Exiting.{RESET}", flush=True)
        return
    
    # Prepare results for JSON output
    experiment_info = {
        "dataset": "MedQA-USMLE",
        "local_model": args.local_model,
        "remote_model": args.remote_model,
        "num_samples": args.num_samples,
        "epsilon_values": epsilon_values,
        "start_index": args.start_index,
        "questions_completed": len(per_question_results),
        "mechanisms_tested": (
            (["Local", "Local+CoT", "Remote"] if not args.skip_epsilon_independent else []) +
            (["PhraseDP"] if not args.skip_phrasedp_normal else []) +
            (["PhraseDP+"] if not getattr(args, 'skip_phrasedp_plus_normal', False) else []) +
            (["PhraseDP++"] if not getattr(args, 'skip_phrasedp_plus_fewshot', False) else [])
        ),
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Organize summary results by epsilon
    organized_summary = {}
    for epsilon in epsilon_values:
        eps_key = f"epsilon_{epsilon}"
        eps_data = {}
        if not getattr(args, 'skip_phrasedp_plus_fewshot', False):
            eps_data["phrasedp_plus_fewshot"] = summary_results[eps_key].get("phrasedp_plus_fewshot", {"correct": 0, "total": 0, "accuracy": 0})
        if not getattr(args, 'skip_phrasedp_plus_normal', False):
            eps_data["phrasedp_plus"] = summary_results[eps_key].get("phrasedp_plus", {"correct": 0, "total": 0, "accuracy": 0})
        if not args.skip_epsilon_independent:
            eps_data["local"] = summary_results["local"]
            eps_data["local_cot"] = summary_results["local_cot"]
            eps_data["remote"] = summary_results["remote"]
        if not args.skip_phrasedp_normal:
            eps_data["phrasedp"] = summary_results[eps_key].get("phrasedp", {"correct": 0, "total": 0, "accuracy": 0})
        organized_summary[eps_key] = eps_data
    
    # Create output directory
    output_dir = "exp/new-exp"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename
    local_model_clean = args.local_model.replace("/", "_").replace("-", "_")
    remote_model_clean = args.remote_model.replace("/", "_").replace("-", "_")
    eps_str = "_".join([str(int(eps) if eps.is_integer() else eps) for eps in epsilon_values])
    filename = f"medqa_usmle_local_{local_model_clean}_remote_{remote_model_clean}_{args.num_samples}q_eps{eps_str}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Save results
    output_data = {
        "experiment_info": experiment_info,
        "summary_results": organized_summary,
        "per_question_results": per_question_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*60}", flush=True)
    print(f"FINAL RESULTS - ALL MECHANISMS", flush=True)
    print(f"{'='*60}", flush=True)
    
    def print_accuracy(name, correct, total):
        """Helper function to print accuracy results."""
        percentage = correct/total*100 if total > 0 else 0
        print(f"{name} Accuracy: {correct}/{total} = {percentage:.2f}%", flush=True)
    
    if not args.skip_epsilon_independent:
        print(f"\nEpsilon-Independent Scenarios:", flush=True)
        print_accuracy("1. Purely Local Model (Baseline)", summary_results["local"]["correct"], summary_results["local"]["total"])
        print_accuracy("2. Non-Private Local Model + Remote CoT", summary_results["local_cot"]["correct"], summary_results["local_cot"]["total"])
        print_accuracy("4. Purely Remote Model", summary_results["remote"]["correct"], summary_results["remote"]["total"])
    
    for epsilon in epsilon_values:
        eps_key = f"epsilon_{epsilon}"
        print(f"\nEpsilon = {epsilon}:", flush=True)
        if not args.skip_phrasedp_normal:
            print_accuracy("3.0. Private Local Model + CoT (Phrase DP)", summary_results[eps_key].get("phrasedp", {}).get("correct", 0), summary_results[eps_key].get("phrasedp", {}).get("total", 0))
        if not getattr(args, 'skip_phrasedp_plus_normal', False):
            print_accuracy("3.1. Private Local Model + CoT (Phrase DP+)", summary_results[eps_key]["phrasedp_plus"]["correct"], summary_results[eps_key]["phrasedp_plus"]["total"])
        if not getattr(args, 'skip_phrasedp_plus_fewshot', False):
            print_accuracy("3.2. Private Local Model + CoT (Phrase DP+ Few-Shot)", summary_results[eps_key].get("phrasedp_plus_fewshot", {}).get("correct", 0), summary_results[eps_key].get("phrasedp_plus_fewshot", {}).get("total", 0))
    
    print(f"\n{'='*60}", flush=True)
    print(f"Results saved to: {output_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()

