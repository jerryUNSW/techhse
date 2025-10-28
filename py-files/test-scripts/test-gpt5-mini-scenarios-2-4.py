#!/usr/bin/env python3
"""
GPT-5 Mini Test for HSE-Bench Scenarios 2 & 4
============================================

This script tests GPT-5 Mini on scenarios 2 and 4 only:
- Scenario 2: Non-Private Local + Remote CoT
- Scenario 4: Purely Remote Model

Tests on 10 regulation questions to observe the gap between GPT-4o Mini and GPT-5 Mini.

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
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def load_env_vars():
    """Load environment variables from .env file."""
    env_file = Path('.env')
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

def get_local_llm_client():
    """Get local LLM client (Nebius)."""
    api_key = os.getenv('NEBIUS_API_KEY')
    if not api_key:
        raise ValueError("NEBIUS_API_KEY not found in environment variables")
    return openai.OpenAI(
        api_key=api_key,
        base_url="https://api.studio.nebius.ai/v1/"
    )

def get_remote_llm_client():
    """Get remote LLM client (OpenAI GPT-5 Mini)."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return openai.OpenAI(api_key=api_key)

def _find_working_nebius_model():
    """Find a working Nebius model."""
    client = get_local_llm_client()
    models_to_try = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "meta-llama/Meta-Llama-3.1-405B-Instruct"
    ]
    
    for model_name in models_to_try:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=10
            )
            print(f"Found working model: {model_name}")
            return model_name
        except Exception as e:
            print(f"Model {model_name} failed: {e}")
            continue
    
    raise RuntimeError("No working Nebius model found")

def get_remote_cot_response(question: str, options: List[str]) -> str:
    """Get chain-of-thought response from remote LLM (GPT-5 Mini) - Privacy-Preserving (no options sent)."""
    try:
        remote_client = get_remote_llm_client()
        cot_prompt = f"""You are an expert in health, safety, and environment (HSE) law and regulations. Please analyze the following question step by step and provide your reasoning.

Question: {question}

Please provide your step-by-step analysis and reasoning to determine the correct answer. Consider:
1. The relevant legal principles and regulations
2. The specific facts and circumstances
3. How the law applies to this situation
4. Which legal principle or outcome would be most appropriate

Your analysis should be thorough and demonstrate your understanding of HSE law and legal reasoning."""
        
        response = remote_client.chat.completions.create(
            model='gpt-5-mini',  # Use GPT-5 Mini
            messages=[{"role": "user", "content": cot_prompt}],
            max_completion_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error getting remote CoT response: {e}")
        return "Error generating CoT guidance"

def get_local_response(question: str, options: List[str], cot_guidance: str = "") -> str:
    """Get response from local LLM."""
    try:
        client = get_local_llm_client()
        model_name = _find_working_nebius_model()
        
        if cot_guidance:
            prompt = f"""Based on the following analysis, answer the question:

Analysis: {cot_guidance}

Question: {question}

Options:
{chr(10).join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])}

Answer with just the letter (A, B, C, or D):"""
        else:
            prompt = f"""Question: {question}

Options:
{chr(10).join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])}

Answer with just the letter (A, B, C, or D):"""
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip()
        if answer and answer[0] in 'ABCD':
            return answer[0]
        return "X"
    except Exception as e:
        print(f"Error getting local response: {e}")
        return "X"

def get_remote_response(question: str, options: List[str]) -> str:
    """Get response from remote LLM."""
    try:
        remote_client = get_remote_llm_client()
        prompt = f"""Question: {question}

Options:
{chr(10).join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])}

Answer with just the letter (A, B, C, or D):"""
        
        response = remote_client.chat.completions.create(
            model='gpt-5-mini',  # Use GPT-5 Mini
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=50
        )
        
        answer = response.choices[0].message.content.strip()
        if answer and answer[0] in 'ABCD':
            return answer[0]
        return "X"
    except Exception as e:
        print(f"Error getting remote response: {e}")
        return "X"

def load_hse_bench_data(category: str, num_samples: int = 10):
    """Load HSE-bench data from CSV files with task type information."""
    data_dir = Path("hse-bench/results")
    
    # Load data from different task types
    task_types = ['rule_recall', 'rule_application', 'issue_spotting', 'rule_conclusion']
    all_questions = []
    
    for task_type in task_types:
        csv_file = data_dir / category / task_type / "m1.csv"
        if csv_file.exists():
            df = pd.read_csv(csv_file, header=None)
            for _, row in df.iterrows():
                if len(row) >= 6:  # Ensure we have all required columns
                    question_data = {
                        'question': row[0],
                        'options': [row[1], row[2], row[3], row[4]],
                        'correct_answer': row[5],
                        'task_type': task_type
                    }
                    all_questions.append(question_data)
    
    if num_samples == -1:
        return all_questions
    else:
        return all_questions[:num_samples]

def run_gpt5_mini_test(category: str, num_samples: int = 10):
    """Run GPT-5 Mini test on scenarios 2 and 4."""
    print("Starting GPT-5 Mini Test for Scenarios 2 & 4")
    print(f"Category: {category}")
    print(f"Samples: {num_samples}")
    print(f"Remote model: GPT-5 Mini")
    print("=" * 80)
    
    # Load data
    questions = load_hse_bench_data(category, num_samples)
    print(f"Loaded {len(questions)} questions from {category}")
    
    # Initialize results
    results = {
        "experiment_type": "GPT-5-Mini-Scenarios-2-4",
        "remote_model": "gpt-5-mini",
        "num_samples": num_samples,
        "category": category,
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "scenario_2_results": [],  # Non-Private CoT
        "scenario_4_results": [],  # Purely Remote
        "summary": {}
    }
    
    scenario_2_correct = 0
    scenario_4_correct = 0
    
    for i, question_data in enumerate(questions, 1):
        print(f"\nProcessing question {i}/{len(questions)}")
        print(f"Question: {question_data['question'][:100]}...")
        print(f"Correct answer: {question_data['correct_answer']}")
        
        question = question_data['question']
        options = question_data['options']
        correct_answer = question_data['correct_answer']
        
        # Scenario 2: Non-Private Local + Remote CoT
        print("Scenario 2: Non-Private Local + Remote CoT (GPT-5 Mini)")
        cot_guidance = get_remote_cot_response(question, options)
        print(f"Generated CoT: {cot_guidance[:100]}...")
        
        local_response = get_local_response(question, options, cot_guidance)
        scenario_2_correct_bool = local_response == correct_answer
        scenario_2_correct += scenario_2_correct_bool
        print(f"Local response: {local_response} (Correct: {scenario_2_correct_bool})")
        
        results["scenario_2_results"].append({
            "question_id": i,
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "cot_guidance": cot_guidance,
            "local_response": local_response,
            "correct": scenario_2_correct_bool
        })
        
        # Scenario 4: Purely Remote Model
        print("Scenario 4: Purely Remote Model (GPT-5 Mini)")
        remote_response = get_remote_response(question, options)
        scenario_4_correct_bool = remote_response == correct_answer
        scenario_4_correct += scenario_4_correct_bool
        print(f"Remote response: {remote_response} (Correct: {scenario_4_correct_bool})")
        
        results["scenario_4_results"].append({
            "question_id": i,
            "question": question,
            "options": options,
            "correct_answer": correct_answer,
            "remote_response": remote_response,
            "correct": scenario_4_correct_bool
        })
        
        print(f"Progress: {i}/{len(questions)} questions completed")
    
    # Calculate final results
    total_questions = len(questions)
    scenario_2_accuracy = (scenario_2_correct / total_questions) * 100
    scenario_4_accuracy = (scenario_4_correct / total_questions) * 100
    
    results["summary"] = {
        "total_questions": total_questions,
        "scenario_2_accuracy": scenario_2_accuracy,
        "scenario_4_accuracy": scenario_4_accuracy,
        "scenario_2_correct": scenario_2_correct,
        "scenario_4_correct": scenario_4_correct,
        "gap": scenario_4_accuracy - scenario_2_accuracy
    }
    
    # Save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"QA-results/hse-bench/gpt5_mini_scenarios_2_4_{category}_{num_samples}q_{timestamp}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("GPT-5 MINI TEST RESULTS")
    print("=" * 80)
    print(f"Total Questions: {total_questions}")
    print(f"Scenario 2 (Non-Private CoT): {scenario_2_correct}/{total_questions} = {scenario_2_accuracy:.1f}%")
    print(f"Scenario 4 (Purely Remote): {scenario_4_correct}/{total_questions} = {scenario_4_accuracy:.1f}%")
    print(f"Gap (Remote - CoT): {scenario_4_accuracy - scenario_2_accuracy:.1f}%")
    print(f"Results saved to: {output_file}")
    
    return results, output_file

def main():
    """Main function."""
    load_env_vars()
    
    parser = argparse.ArgumentParser(description='GPT-5 Mini Test for HSE-Bench Scenarios 2 & 4')
    parser.add_argument('--category', type=str, default='regulation', 
                       choices=['regulation', 'court_case', 'safety_exam'],
                       help='HSE-bench category to test')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to test (-1 for all)')
    
    args = parser.parse_args()
    
    print("Starting GPT-5 Mini Test for HSE-Bench Scenarios 2 & 4")
    print(f"Category: {args.category}")
    print(f"Samples: {args.num_samples}")
    print(f"Remote model: GPT-5 Mini")
    
    try:
        results, output_file = run_gpt5_mini_test(args.category, args.num_samples)
        print("GPT-5 Mini test completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during GPT-5 Mini test: {e}")
        raise

if __name__ == "__main__":
    main()
