#!/usr/bin/env python3
"""
HSE-Bench Local Model Only Test Script
=====================================

This script tests ONLY the local model on HSE-bench dataset:
- All regulation questions (448 questions)
- Uses Meta-Llama-3.1-8B-Instruct for local inference
- No privacy mechanisms, no remote models
- Pure local baseline performance

Author: Tech4HSE Team
Date: 2025-01-30
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
import time
import random
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

# Import necessary functions for local model inference
import openai

# Load environment variables from .env file
load_dotenv()

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

def _find_working_nebius_model(client):
    """Find a working Nebius model."""
    models_to_try = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "microsoft/phi-4",
        "google/gemma-2-9b-it-fast",
        "google/gemma-2-2b-it",
        "Qwen/Qwen2.5-Coder-7B",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "Qwen/Qwen3-4B-fast",
        "Qwen/Qwen3-14B"
    ]
    
    for model in models_to_try:
        try:
            # Test the model with a simple request
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0.1
            )
            print(f"Using model: {model}")
            return model
        except Exception as e:
            print(f"Model {model} failed: {e}")
            continue
    
    raise RuntimeError("No working Nebius model found")

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
            max_tokens=10,
            temperature=0.1
        )
        
        answer = response.choices[0].message.content.strip()
        
        # Extract just the letter if possible
        for char in answer:
            if char.upper() in 'ABCD':
                return char.upper()
        return answer.upper()
    except Exception as e:
        print(f"Error getting local response: {e}")
        return "X"

def load_config():
    """Load configuration from config.yaml"""
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_hse_bench_data(category: str, num_samples: int):
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
                if len(row) >= 7:  # Ensure we have all required columns
                    question_data = {
                        'question': row[0],
                        'options': [row[1], row[2], row[3], row[4]],
                        'correct_answer': row[5],
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

class HSEBenchLocalOnlyResults:
    """Class to handle HSE-bench local-only experiment results."""
    
    def __init__(self, model_name: str, num_samples: int):
        self.model_name = model_name
        self.num_samples = num_samples
        self.start_time = datetime.datetime.now()
        self.results = {
            'local_alone_correct': 0,
            'total_questions': 0
        }
        # Task type specific results
        self.task_type_results = {
            'rule_recall': {'correct': 0, 'total': 0},
            'rule_application': {'correct': 0, 'total': 0},
            'issue_spotting': {'correct': 0, 'total': 0},
            'rule_conclusion': {'correct': 0, 'total': 0}
        }
        self.detailed_results = []  # Store detailed question data
        self.output_file = None
        
    def initialize_output_file(self, model_name: str, num_samples: int):
        """Initialize the output file with metadata."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f"experiment_results/QA-results/hse-bench/hse_bench_local_only_results_{model_name.replace('/', '_')}_{num_samples}q_{timestamp}.json"
        
        # Create directory if it doesn't exist
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the output file
        initial_data = {
            "experiment_type": "HSE-bench-local-only",
            "model_name": model_name,
            "num_samples": num_samples,
            "start_time": str(self.start_time),
            "results": self.results,
            "task_type_results": self.task_type_results,
            "detailed_results": []
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(initial_data, f, indent=2)
    
    def save_results(self):
        """Save current results to the output file."""
        if not self.output_file:
            return
            
        final_data = {
            "experiment_type": "HSE-bench-local-only",
            "model_name": self.model_name,
            "num_samples": self.num_samples,
            "start_time": str(self.start_time),
            "end_time": str(datetime.datetime.now()),
            "results": self.results,
            "task_type_results": self.task_type_results,
            "detailed_results": self.detailed_results
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(final_data, f, indent=2)
    
    def add_question_result(self, question_id: int, question_data: Dict, 
                          local_response: str, is_correct: bool):
        """Add a single question result."""
        self.results['total_questions'] += 1
        if is_correct:
            self.results['local_alone_correct'] += 1
        
        # Update task type specific results
        task_type = question_data.get('task_type', 'unknown')
        if task_type in self.task_type_results:
            self.task_type_results[task_type]['total'] += 1
            if is_correct:
                self.task_type_results[task_type]['correct'] += 1
        
        # Store detailed result
        detailed_result = {
            "question_id": question_id,
            "original_question": question_data.get('question', ''),
            "options": question_data.get('options', []),
            "correct_answer": question_data.get('correct_answer', ''),
            "reference": question_data.get('reference', ''),
            "task_type": question_data.get('task_type', ''),
            "local_alone": {
                "response": local_response,
                "is_correct": is_correct,
                "response_time": 0.0  # Placeholder
            }
        }
        
        self.detailed_results.append(detailed_result)
        
        # Save intermediate results every 10 questions
        if self.results['total_questions'] % 10 == 0:
            self.save_results()
            print(f"Progress: {self.results['total_questions']} questions completed")
            print(f"Current accuracy: {self.results['local_alone_correct']}/{self.results['total_questions']} = {self.results['local_alone_correct']/self.results['total_questions']*100:.1f}%")
            
            # Print task type breakdown
            print("Task Type Breakdown:")
            for task_type, stats in self.task_type_results.items():
                if stats['total'] > 0:
                    accuracy = stats['correct'] / stats['total'] * 100
                    print(f"  {task_type}: {stats['correct']}/{stats['total']} = {accuracy:.1f}%")

def send_completion_email(results_file: str, accuracy: float, total_questions: int):
    """Send email notification when experiment completes."""
    try:
        # Load email config
        with open('email_config.json', 'r') as f:
            email_config = json.load(f)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['from_email']
        msg['To'] = email_config['to_email']
        msg['Subject'] = f"HSE-bench Local-Only Test Complete - Host: {socket.gethostname()}"
        
        body = f"""
HSE-bench Local-Only Test Completed Successfully!

Results Summary:
- Total Questions: {total_questions}
- Local Model Accuracy: {accuracy:.1f}%
- Results File: {results_file}
- Host: {socket.gethostname()}
- Completion Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The experiment has completed successfully.
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['from_email'], email_config['password'])
        text = msg.as_string()
        server.sendmail(email_config['from_email'], email_config['to_email'], text)
        server.quit()
        
        print("Completion email sent successfully!")
        
    except Exception as e:
        print(f"Failed to send completion email: {e}")

def run_local_only_test(model_name: str, category: str, num_samples: int):
    """Run the local-only HSE-bench test."""
    print(f"Starting HSE-bench Local-Only Test")
    print(f"Model: {model_name}")
    print(f"Category: {category}")
    print(f"Number of samples: {num_samples}")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize results handler
    results_handler = HSEBenchLocalOnlyResults(model_name, num_samples)
    results_handler.initialize_output_file(model_name, num_samples)
    
    # Load HSE-bench data
    print(f"Loading HSE-bench {category} questions...")
    questions = load_hse_bench_data(category, num_samples)
    print(f"Loaded {len(questions)} questions")
    
    # Process each question
    for i, question_data in enumerate(questions):
        question_id = i + 1
        print(f"\nProcessing Question {question_id}/{len(questions)}")
        
        # Extract question components
        question_text = question_data.get('question', '')
        options = question_data.get('options', [])
        correct_answer = question_data.get('correct_answer', '')
        
        # Format question for local model
        formatted_question = f"Question: {question_text}\n\nOptions:\n"
        for j, option in enumerate(options):
            formatted_question += f"{chr(65+j)}. {option}\n"
        formatted_question += "\nPlease provide your answer as a single letter (A, B, C, or D)."
        
        try:
            # Get local model response
            print("Getting local model response...")
            local_response = get_local_response(question_text, options)
            
            # Extract answer from response
            local_answer = local_response.strip().upper()
            if len(local_answer) > 1:
                local_answer = local_answer[0]
            
            # Check if correct
            is_correct = local_answer == correct_answer.upper()
            
            print(f"Local response: {local_answer}")
            print(f"Correct answer: {correct_answer}")
            print(f"Correct: {is_correct}")
            
            # Add result
            results_handler.add_question_result(
                question_id, question_data, local_answer, is_correct
            )
            
        except Exception as e:
            print(f"Error processing question {question_id}: {e}")
            # Add failed result
            results_handler.add_question_result(
                question_id, question_data, "ERROR", False
            )
        
        # Small delay to avoid overwhelming the system
        time.sleep(0.1)
    
    # Save final results
    results_handler.save_results()
    
    # Calculate final accuracy
    total_questions = results_handler.results['total_questions']
    correct_answers = results_handler.results['local_alone_correct']
    final_accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
    
    print("\n" + "=" * 60)
    print("HSE-bench Local-Only Test Complete!")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Final Accuracy: {final_accuracy:.1f}%")
    print(f"Results saved to: {results_handler.output_file}")
    
    # Print final task type breakdown
    print("\nTask Type Accuracy Breakdown:")
    print("-" * 40)
    for task_type, stats in results_handler.task_type_results.items():
        if stats['total'] > 0:
            accuracy = stats['correct'] / stats['total'] * 100
            print(f"{task_type:20}: {stats['correct']:3d}/{stats['total']:3d} = {accuracy:5.1f}%")
        else:
            print(f"{task_type:20}: No questions")
    
    # Send completion email
    send_completion_email(results_handler.output_file, final_accuracy, total_questions)
    
    return results_handler.output_file, final_accuracy

def main():
    """Main function to run the HSE-bench local-only test."""
    parser = argparse.ArgumentParser(description='HSE-bench Local-Only Test')
    parser.add_argument('--model', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct',
                       help='Local model name')
    parser.add_argument('--category', type=str, default='regulation',
                       choices=['regulation', 'court_case', 'safety_exam', 'video'],
                       help='HSE-bench category to test')
    parser.add_argument('--num_samples', type=int, default=-1,
                       help='Number of samples to test (-1 for all)')
    
    args = parser.parse_args()
    
    # Run the test
    results_file, accuracy = run_local_only_test(
        args.model, args.category, args.num_samples
    )
    
    print(f"\nTest completed successfully!")
    print(f"Results file: {results_file}")
    print(f"Final accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    main()
