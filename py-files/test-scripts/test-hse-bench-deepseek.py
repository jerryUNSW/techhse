#!/usr/bin/env python3
"""
HSE-Bench Privacy-Preserving QA Experiment
=========================================

This script tests privacy-preserving mechanisms on HSE-bench dataset:
- Regulation questions (448 questions)
- Court case questions (152 questions) 
- Safety exam questions (320 questions)

Mechanisms tested:
1. Purely Local (Baseline)
2. Non-Private Local + Remote CoT
3.0 Private Local + CoT (Old PhraseDP)
3.2 Private Local + CoT (InferDPT)
3.3 Private Local + CoT (SANTEXT+)
4. Purely Remote Model

Author: Tech4HSE Team
Date: 2025-01-27
"""

import os
import json
import yaml
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import openai
import time
import random
from sanitization_methods import (
    phrasedp_sanitize_text,
    inferdpt_sanitize_text,
    santext_sanitize_text,
)

# Color codes for console output
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"

def load_env_vars():
    """Load environment variables from .env file."""
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")

def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_local_llm_client():
    """Get local LLM client (Nebius)."""
    api_key = os.getenv('NEBIUS')
    if not api_key:
        raise ValueError("NEBIUS API key not found in environment variables")
    # Use OpenAI client with Nebius endpoint
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.studio.nebius.ai/v1/"
    )

def get_remote_llm_client():
    """Get remote LLM client - using DeepSeek for HSE-bench experiments."""
    # Force DeepSeek for HSE-bench experiments
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com"
    )

def generate_cot_prompt(question: str, options: List[str]) -> str:
    """Generate CoT prompt for HSE-bench questions."""
    options_text = "\n".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
    
    prompt = f"""You are an expert in health, safety, and environment (HSE) law and regulations. Please analyze the following question step by step and provide your reasoning.

Question: {question}

Options:
{options_text}

Please provide your step-by-step analysis and reasoning to determine the correct answer. Consider:
1. The legal framework and regulations that apply
2. The specific facts and circumstances in the scenario
3. How the law applies to these facts
4. Which option best represents the correct legal principle or outcome

Your analysis should be thorough and demonstrate your understanding of HSE law and legal reasoning."""
    
    return prompt

def get_remote_cot_response(question: str, options: List[str]) -> str:
    """Get chain-of-thought response from remote LLM."""
    try:
        remote_client = get_remote_llm_client()
        cot_prompt = generate_cot_prompt(question, options)
        
        response = remote_client.chat.completions.create(
            model='deepseek-chat',  # Use DeepSeek for HSE-bench
            messages=[{"role": "user", "content": cot_prompt}],
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens', 256)
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"{RED}Error getting remote CoT: {e}{RESET}")
        return "Error generating CoT"

def get_remote_cot_response_private(question: str) -> str:
    """Get chain-of-thought response from remote LLM for privacy scenarios (no options sent)."""
    try:
        remote_client = get_remote_llm_client()
        cot_prompt = f"""You are an expert in health, safety, and environment (HSE) law and regulations. Please analyze the following question step by step and provide your reasoning.

Question: {question}

Please provide your step-by-step analysis and reasoning to determine the correct answer. Consider:
1. The legal framework and regulations that apply
2. The specific facts and circumstances in the scenario
3. How the law applies to these facts
4. Which option best represents the correct legal principle or outcome

Your analysis should be thorough and demonstrate your understanding of HSE law and legal reasoning."""
        
        response = remote_client.chat.completions.create(
            model='deepseek-chat',  # Use DeepSeek for HSE-bench
            messages=[{"role": "user", "content": cot_prompt}],
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens', 256)
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"{RED}Error getting remote CoT: {e}{RESET}")
        return "Error generating CoT"

def _find_working_nebius_model(client) -> str:
    """Probe Nebius with candidate models to find a working local model ID."""
    # Use the same model as the working testing_medical_qa.py script
    model_name = config.get('local_model', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
    
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0
        )
        print(f"{GREEN}Found working Nebius model: {model_name}{RESET}")
        return model_name
    except Exception as e:
        print(f"{RED}Error with model {model_name}: {e}{RESET}")
        # Try alternative models
        alternative_models = [
            'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'meta-llama/Llama-3.1-8B-Instruct',
            'meta-llama/Llama-3-8B-Instruct'
        ]
        
        for alt_model in alternative_models:
            try:
                resp = client.chat.completions.create(
                    model=alt_model,
                    messages=[{"role": "user", "content": "ping"}],
                    max_tokens=1,
                    temperature=0.0
                )
                print(f"{GREEN}Found working Nebius model: {alt_model}{RESET}")
                return alt_model
            except Exception:
                continue
        
        raise Exception("No working Nebius model found")

def get_local_response(question: str, options: List[str], cot_guidance: str = None) -> str:
    """Get response from local LLM."""
    try:
        local_client = get_local_llm_client()
        
        # Find working model
        working_model = _find_working_nebius_model(local_client)
        
        if cot_guidance:
            prompt = f"""Based on the following legal analysis, answer the question:

Legal Analysis: {cot_guidance}

Question: {question}

Options:
{chr(10).join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])}

Answer with just the letter (A, B, C, or D):"""
        else:
            prompt = f"""Question: {question}

Options:
{chr(10).join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])}

Answer with just the letter (A, B, C, or D):"""
        
        response = local_client.chat.completions.create(
            model=working_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens', 256)
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error getting local response: {e}{RESET}")
        return "Error"

def get_remote_response(question: str, options: List[str]) -> str:
    """Get response from remote LLM."""
    try:
        remote_client = get_remote_llm_client()
        options_text = chr(10).join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
        prompt = f"""Question: {question}
Options:
{options_text}

Answer with just the letter (A, B, C, or D):"""
        
        response = remote_client.chat.completions.create(
            model='deepseek-chat',  # Use DeepSeek for HSE-bench
            messages=[{"role": "user", "content": prompt}],
            temperature=config.get('temperature', 0.0),
            max_tokens=config.get('max_tokens', 256)
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error getting remote response: {e}{RESET}")
        return "Error"

def initialize_santext_mechanism():
    """Initialize SANTEXT+ mechanism once for all questions."""
    print(f"{CYAN}Initializing SANTEXT+ mechanism...{RESET}")
    try:
        # Just test that the function is available, don't use any epsilon
        santext_sanitize_text("Warm-up text for SANTEXT+", epsilon=1.0)
        print(f"{GREEN}SANTEXT+ mechanism initialized successfully{RESET}")
        return True
    except Exception as exc:
        print(f"{YELLOW}SANTEXT+ warm-up encountered an issue but will proceed lazily: {exc}{RESET}")
        return True  # Still return True so the mechanism is available

def initialize_phrasedp_mechanism():
    """Initialize OLD PhraseDP mechanism once for all questions."""
    print(f"{CYAN}Initializing OLD PhraseDP mechanism...{RESET}")
    # PhraseDP does not require initialization - it's called directly when needed
    print(f"{GREEN}OLD PhraseDP mechanism ready{RESET}")
    return True  # Return True so the mechanism is available

def initialize_inferdpt_mechanism():
    """Initialize InferDPT mechanism once for all questions."""
    print(f"{CYAN}Initializing InferDPT mechanism...{RESET}")
    try:
        # Use a fixed epsilon for embedding loading (embeddings are epsilon-independent)
        inferdpt_sanitize_text("Warm-up text for InferDPT", epsilon=1.0)
        print(f"{GREEN}InferDPT mechanism initialized successfully{RESET}")
        return True
    except Exception as exc:
        print(f"{YELLOW}InferDPT warm-up encountered an issue but will proceed lazily: {exc}{RESET}")
        return True  # Still return True so the mechanism is available

def simple_perturb_text(text: str, epsilon: float, mechanism: str) -> str:
    """Simple text perturbation for testing (placeholder implementation)."""
    # This is a placeholder - replace with actual mechanism implementations
    if mechanism == "phrasedp":
        return text.replace("the", "[PERTURBED]").replace("a", "[PERTURBED]")
    elif mechanism == "inferdpt":
        return text.replace("was", "[PERTURBED]").replace("were", "[PERTURBED]")
    elif mechanism == "santext":
        return text.replace("he", "[PERTURBED]").replace("she", "[PERTURBED]")
    else:
        return text

def load_hse_bench_data(category: str, num_samples: int):
    """Load HSE-bench data from CSV files."""
    data_dir = Path("hse-bench/results")
    
    # Load data from different task types
    task_types = ['rule_recall', 'rule_application', 'issue_spotting', 'rule_conclusion']
    all_questions = []
    
    for task_type in task_types:
        csv_file = data_dir / category / task_type / "m1.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file, header=None)
            for _, row in df.iterrows():
                if len(row) >= 7:  # Ensure we have all required columns
                    question_data = {
                        'question': row[0],
                        'options': [row[1], row[2], row[3], row[4]],
                        'answer': row[5],
                        'reference': row[6] if len(row) > 6 else '',
                        'task_type': task_type
                    }
                    all_questions.append(question_data)
    
    # Shuffle and limit to requested number of samples
    random.shuffle(all_questions)
    
    # If num_samples is -1, return all questions
    if num_samples == -1:
        print(f"Loading ALL questions for category '{category}': {len(all_questions)} questions")
        return all_questions
    else:
        return all_questions[:num_samples]

class HSEBenchExperimentResults:
    """Class to handle HSE-bench experiment results."""
    
    def __init__(self, model_name: str, num_samples: int, epsilon_values: List[float]):
        self.model_name = model_name
        self.num_samples = num_samples
        self.epsilon_values = epsilon_values
        self.start_time = datetime.datetime.now()
        self.results = {}
        self.output_file = None
        
        # Initialize results structure
        for epsilon in epsilon_values:
            self.results[epsilon] = {
                'local_alone_correct': 0,
                'non_private_cot_correct': 0,
                'old_phrase_dp_local_cot_correct': 0,
                'inferdpt_local_cot_correct': 0,
                'santext_local_cot_correct': 0,
                'purely_remote_correct': 0,
                'total_questions': 0
            }
        
        # Initialize shared results (epsilon-independent scenarios 1, 2, 4)
        self.shared_results = {
            'local_alone_correct': 0,
            'non_private_cot_correct': 0,
            'purely_remote_correct': 0,
            'total_questions': 0
        }
    
    def initialize_output_file(self, model_name, num_samples, epsilon=None):
        """Initialize output file for incremental writing."""
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        # Get remote model name for better file naming (DeepSeek for HSE-bench)
        remote_model = 'deepseek-chat'
        remote_model_clean = remote_model.replace('-', '_').replace('.', '_')
        
        if epsilon is not None:
            self.output_file = f"QA-results/hse-bench/hse_bench_results_local_{model_name.replace('/', '_')}_remote_{remote_model_clean}_{num_samples}q_eps{epsilon}_{timestamp}.json"
        else:
            self.output_file = f"QA-results/hse-bench/hse_bench_results_local_{model_name.replace('/', '_')}_remote_{remote_model_clean}_{num_samples}q_{timestamp}.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Initialize file with metadata
        initial_data = {
            'experiment_type': 'HSE-bench',
            'model_name': model_name,
            'remote_model': remote_model,
            'num_samples': num_samples,
            'epsilon_values': self.epsilon_values,
            'start_time': self.start_time.strftime("%Y-%m-%d %H:%M:%S.%f"),
            'results': self.results
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
    
    def update_results(self, epsilon: float, mechanism: str, question_idx: int, response: str, correct_answer: str, is_correct: bool):
        """Update results for a specific mechanism."""
        if epsilon in self.results:
            if is_correct:
                self.results[epsilon][f'{mechanism}_correct'] += 1
    
    def update_shared_results(self, mechanism: str, question_idx: int, response: str, correct_answer: str, is_correct: bool):
        """Update shared results for epsilon-independent scenarios (1, 2, 4)."""
        if is_correct:
            self.shared_results[f'{mechanism}_correct'] += 1
    
    def increment_question_count(self, epsilon: float):
        """Increment the total question count for an epsilon."""
        if epsilon in self.results:
            self.results[epsilon]['total_questions'] += 1
    
    def increment_shared_question_count(self):
        """Increment the total question count for shared results."""
        self.shared_results['total_questions'] += 1
    
    def save_results(self, epsilon: float):
        """Save current results to file."""
        if self.output_file:
            with open(self.output_file, 'r') as f:
                data = json.load(f)
            
            data['results'] = self.results
            data['shared_results'] = self.shared_results
            data['last_updated'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
            
            with open(self.output_file, 'w') as f:
                json.dump(data, f, indent=2)

def run_experiment_for_model(model_name: str, category: str, num_samples: int, epsilon_values: List[float]):
    """Run HSE-bench experiment for a given model."""
    print(f"{CYAN}Starting HSE-bench experiment for {category} with {model_name}{RESET}")
    print(f"Testing {num_samples} questions with epsilon values: {epsilon_values}")
    
    # Load HSE-bench data
    print(f"{CYAN}Loading HSE-bench {category} data...{RESET}")
    questions = load_hse_bench_data(category, num_samples)
    print(f"Loaded {len(questions)} questions from {category}")
    
    # Initialize privacy mechanisms
    print(f"{CYAN}Initializing privacy mechanisms...{RESET}")
    santext_mechanism = initialize_santext_mechanism()
    phrasedp_mechanism = initialize_phrasedp_mechanism()
    inferdpt_mechanism = initialize_inferdpt_mechanism()
    
    # Initialize results handler
    results_handler = HSEBenchExperimentResults(model_name, num_samples, epsilon_values)
    
    # Main experiment loop: For each question, run all scenarios
    print(f"\n{'='*80}")
    print(f"RUNNING HSE-BENCH EXPERIMENT")
    print(f"{'='*80}")
    
    for i, question_data in enumerate(questions):
        print(f"\n{GREEN}Processing question {i+1}/{len(questions)}{RESET}")
        question = question_data['question']
        options = question_data['options']
        correct_answer = question_data['answer']
        
        print(f"Question: {question[:100]}...")
        print(f"Correct answer: {correct_answer}")
        
        # Run epsilon-independent scenarios (1, 2, 4) ONCE per question
        print(f"\n{CYAN}Epsilon-Independent Scenarios (1, 2, 4){RESET}")
        
        # Increment shared question count
        results_handler.increment_shared_question_count()
        
        # Scenario 1: Purely Local Model
        print(f"{YELLOW}Scenario 1: Purely Local Model{RESET}")
        local_response = get_local_response(question, options)
        is_correct = local_response.strip().upper() == correct_answer.upper()
        print(f"Local response: {local_response} (Correct: {is_correct})")
        results_handler.update_shared_results('local_alone', i, local_response, correct_answer, is_correct)
        
        # Scenario 2: Non-Private Local + Remote CoT (Privacy-Preserving)
        print(f"{YELLOW}Scenario 2: Non-Private Local + Remote CoT (Privacy-Preserving){RESET}")
        cot_guidance = get_remote_cot_response_private(question)  # Don't send options to remote
        non_private_response = get_local_response(question, options, cot_guidance)
        is_correct = non_private_response.strip().upper() == correct_answer.upper()
        print(f"Non-private response: {non_private_response} (Correct: {is_correct})")
        results_handler.update_shared_results('non_private_cot', i, non_private_response, correct_answer, is_correct)
        
        # Scenario 4: Purely Remote Model
        print(f"{YELLOW}Scenario 4: Purely Remote Model{RESET}")
        remote_response = get_remote_response(question, options)
        is_correct = remote_response.strip().upper() == correct_answer.upper()
        print(f"Remote response: {remote_response} (Correct: {is_correct})")
        results_handler.update_shared_results('purely_remote', i, remote_response, correct_answer, is_correct)
        
        # Run epsilon-dependent scenarios (3.0, 3.2, 3.3) for each epsilon
        for epsilon in epsilon_values:
            print(f"\n{CYAN}Epsilon {epsilon} - Privacy Mechanisms (3.0, 3.2, 3.3){RESET}")
            
            # Initialize output file for this epsilon (only once per epsilon)
            if i == 0:  # Only initialize on first question
                results_handler.initialize_output_file(model_name, num_samples, epsilon)
            
            # Increment question count for this epsilon
            results_handler.increment_question_count(epsilon)
            
            # Scenario 3.0: Private Local + CoT (Old PhraseDP)
            print(f"{YELLOW}Scenario 3.0: Private Local + CoT (Old PhraseDP, ε={epsilon}){RESET}")
            print(f"Original Question: {question}")
            print(f"Options: {options}")
            print(f"Correct Answer: {correct_answer}")
            
            if phrasedp_mechanism:
                # Apply OLD PhraseDP perturbation to question (single API call, no band diversity)
                print(f"{CYAN}Applying OLD PhraseDP perturbation...{RESET}")
                from sanitization_methods import config as sm_config
                import utils
                nebius_client = utils.get_nebius_client()
                nebius_model_name = sm_config.get('local_model')
                # Load Sentence-BERT for similarity computation
                from sentence_transformers import SentenceTransformer
                sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
                perturbed_question = utils.phrase_DP_perturbation_old(
                    nebius_client=nebius_client,
                    nebius_model_name=nebius_model_name,
                    input_sentence=question,
                    epsilon=epsilon,
                    sbert_model=sbert_model
                )
                print(f"Perturbed Question: {perturbed_question}")
                
                print(f"{CYAN}Generating CoT from perturbed question (options stay local)...{RESET}")
                cot_guidance = get_remote_cot_response_private(perturbed_question)  # Don't send options to remote
                print(f"Generated CoT: {cot_guidance}")
                
                phrasedp_response = get_local_response(question, options, cot_guidance)  # Use original question locally
                is_correct = phrasedp_response.strip().upper() == correct_answer.upper()
                print(f"PhraseDP response: {phrasedp_response} (Correct: {is_correct})")
                results_handler.update_results(epsilon, 'old_phrase_dp_local_cot', i, phrasedp_response, correct_answer, is_correct)
            else:
                # Use simple perturbation for testing
                print(f"{CYAN}Using simple PhraseDP perturbation...{RESET}")
                perturbed_question = simple_perturb_text(question, epsilon, "phrasedp")
                print(f"Perturbed Question: {perturbed_question}")
                
                print(f"{CYAN}Generating CoT from perturbed question (options stay local)...{RESET}")
                cot_guidance = get_remote_cot_response_private(perturbed_question)  # Don't send options to remote
                print(f"Generated CoT: {cot_guidance}")
                
                phrasedp_response = get_local_response(question, options, cot_guidance)  # Use original question locally
                is_correct = phrasedp_response.strip().upper() == correct_answer.upper()
                print(f"PhraseDP response (simple): {phrasedp_response} (Correct: {is_correct})")
                results_handler.update_results(epsilon, 'old_phrase_dp_local_cot', i, phrasedp_response, correct_answer, is_correct)
            
            # Scenario 3.2: Private Local + CoT (InferDPT)
            print(f"{YELLOW}Scenario 3.2: Private Local + CoT (InferDPT, ε={epsilon}){RESET}")
            print(f"Original Question: {question}")
            print(f"Options: {options}")
            print(f"Correct Answer: {correct_answer}")
            
            if inferdpt_mechanism:
                # Apply InferDPT perturbation to question
                print(f"{CYAN}Applying InferDPT perturbation...{RESET}")
                perturbed_question = inferdpt_sanitize_text(question, epsilon=epsilon)
                print(f"Perturbed Question: {perturbed_question}")
                
                print(f"{CYAN}Generating CoT from perturbed question (options stay local)...{RESET}")
                cot_guidance = get_remote_cot_response_private(perturbed_question)  # Don't send options to remote
                print(f"Generated CoT: {cot_guidance}")
                
                inferdpt_response = get_local_response(question, options, cot_guidance)  # Use original question locally
                is_correct = inferdpt_response.strip().upper() == correct_answer.upper()
                print(f"InferDPT response: {inferdpt_response} (Correct: {is_correct})")
                results_handler.update_results(epsilon, 'inferdpt_local_cot', i, inferdpt_response, correct_answer, is_correct)
            else:
                # Use simple perturbation for testing
                print(f"{CYAN}Using simple InferDPT perturbation...{RESET}")
                perturbed_question = simple_perturb_text(question, epsilon, "inferdpt")
                print(f"Perturbed Question: {perturbed_question}")
                
                print(f"{CYAN}Generating CoT from perturbed question (options stay local)...{RESET}")
                cot_guidance = get_remote_cot_response_private(perturbed_question)  # Don't send options to remote
                print(f"Generated CoT: {cot_guidance}")
                
                inferdpt_response = get_local_response(question, options, cot_guidance)  # Use original question locally
                is_correct = inferdpt_response.strip().upper() == correct_answer.upper()
                print(f"InferDPT response (simple): {inferdpt_response} (Correct: {is_correct})")
                results_handler.update_results(epsilon, 'inferdpt_local_cot', i, inferdpt_response, correct_answer, is_correct)
            
            # Scenario 3.3: Private Local + CoT (SANTEXT+)
            print(f"{YELLOW}Scenario 3.3: Private Local + CoT (SANTEXT+, ε={epsilon}){RESET}")
            print(f"Original Question: {question}")
            print(f"Options: {options}")
            print(f"Correct Answer: {correct_answer}")
            
            if santext_mechanism:
                # Apply SANTEXT+ perturbation to question
                print(f"{CYAN}Applying SANTEXT+ perturbation...{RESET}")
                perturbed_question = santext_sanitize_text(question, epsilon=epsilon)
                print(f"Perturbed Question: {perturbed_question}")
                
                print(f"{CYAN}Generating CoT from perturbed question (options stay local)...{RESET}")
                cot_guidance = get_remote_cot_response_private(perturbed_question)  # Don't send options to remote
                print(f"Generated CoT: {cot_guidance}")
                
                santext_response = get_local_response(question, options, cot_guidance)  # Use original question locally
                is_correct = santext_response.strip().upper() == correct_answer.upper()
                print(f"SANTEXT+ response: {santext_response} (Correct: {is_correct})")
                results_handler.update_results(epsilon, 'santext_local_cot', i, santext_response, correct_answer, is_correct)
            else:
                # Use simple perturbation for testing
                print(f"{CYAN}Using simple SANTEXT+ perturbation...{RESET}")
                perturbed_question = simple_perturb_text(question, epsilon, "santext")
                print(f"Perturbed Question: {perturbed_question}")
                
                print(f"{CYAN}Generating CoT from perturbed question (options stay local)...{RESET}")
                cot_guidance = get_remote_cot_response_private(perturbed_question)  # Don't send options to remote
                print(f"Generated CoT: {cot_guidance}")
                
                santext_response = get_local_response(question, options, cot_guidance)  # Use original question locally
                is_correct = santext_response.strip().upper() == correct_answer.upper()
                print(f"SANTEXT+ response (simple): {santext_response} (Correct: {is_correct})")
                results_handler.update_results(epsilon, 'santext_local_cot', i, santext_response, correct_answer, is_correct)
            
            # Save incremental results
            results_handler.save_results(epsilon)
        
        # Progress update after each question
        if (i + 1) % 10 == 0:
            print(f"{CYAN}Progress: {i+1}/{len(questions)} questions completed{RESET}")
    
    print(f"\n{GREEN}HSE-bench experiment completed for {category}!{RESET}")
    
    # Print final results summary
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Show shared results (epsilon-independent scenarios)
    if results_handler.shared_results:
        total = results_handler.shared_results['total_questions']
        print(f"\n{CYAN}Shared Results (Epsilon-Independent Scenarios){RESET}")
        print(f"{'='*50}")
        print(f"Total Questions: {total}")
        print(f"Local Alone: {results_handler.shared_results['local_alone_correct']}/{total} = {results_handler.shared_results['local_alone_correct']/total*100:.1f}%")
        print(f"Non-Private CoT: {results_handler.shared_results['non_private_cot_correct']}/{total} = {results_handler.shared_results['non_private_cot_correct']/total*100:.1f}%")
        print(f"Purely Remote: {results_handler.shared_results['purely_remote_correct']}/{total} = {results_handler.shared_results['purely_remote_correct']/total*100:.1f}%")
    
    for epsilon in epsilon_values:
        print(f"\n{CYAN}Epsilon = {epsilon} (Privacy Mechanisms){RESET}")
        print(f"{'='*50}")
        
        if epsilon in results_handler.results:
            results = results_handler.results[epsilon]
            total = results['total_questions']
            
            if total > 0:
                print(f"Total Questions: {total}")
                print(f"PhraseDP (Old): {results['old_phrase_dp_local_cot_correct']}/{total} = {results['old_phrase_dp_local_cot_correct']/total*100:.1f}%")
                print(f"InferDPT: {results['inferdpt_local_cot_correct']}/{total} = {results['inferdpt_local_cot_correct']/total*100:.1f}%")
                print(f"SANTEXT+: {results['santext_local_cot_correct']}/{total} = {results['santext_local_cot_correct']/total*100:.1f}%")
            else:
                print("No results available for this epsilon")
        else:
            print("No results found for this epsilon")
    
    return results_handler

def main():
    """Main function to run HSE-bench experiments."""
    # Load environment variables from .env file
    load_env_vars()
    
    parser = argparse.ArgumentParser(description='HSE-bench Privacy-Preserving QA Experiment')
    parser.add_argument('--category', type=str, choices=['regulation', 'court_case', 'safety_exam'], 
                       default='court_case', help='HSE-bench category to test')
    parser.add_argument('--num-samples', type=int, default=100, help='Number of questions to test (use -1 for all questions)')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', 
                       help='Local model to use')
    parser.add_argument('--epsilon-values', nargs='+', type=float, default=[1.0, 2.0, 3.0], 
                       help='Epsilon values to test')
    
    args = parser.parse_args()
    
    global config
    config = load_config()
    
    print(f"{CYAN}HSE-Bench Privacy-Preserving QA Experiment{RESET}")
    print(f"Category: {args.category}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Model: {args.model}")
    print(f"Epsilon values: {args.epsilon_values}")
    
    results_handler = run_experiment_for_model(
        model_name=args.model,
        category=args.category,
        num_samples=args.num_samples,
        epsilon_values=args.epsilon_values
    )
    
    print(f"\n{GREEN}Experiment completed successfully!{RESET}")
    if results_handler and results_handler.output_file:
        print(f"Results saved to: {results_handler.output_file}")

if __name__ == "__main__":
    main()
