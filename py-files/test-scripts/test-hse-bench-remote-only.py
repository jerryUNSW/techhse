#!/usr/bin/env python3
"""
HSE-Bench Remote Model Only Test Script
======================================

This script tests ONLY the remote model (GPT-4o mini) on HSE-bench dataset:
- Tests all question categories: regulation, court_case, safety_exam
- Uses GPT-4o mini for remote inference
- No privacy mechanisms, no local models
- Pure remote baseline performance

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

# Import necessary functions for remote model inference
import openai

# Load environment variables from .env file
load_dotenv()

def get_remote_llm_client():
    """Get remote LLM client (OpenAI GPT-4o mini)."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return openai.OpenAI(api_key=api_key)

def get_remote_response(question: str, options: List[str]) -> str:
    """Get response from remote LLM (GPT-4o mini)."""
    try:
        client = get_remote_llm_client()
        
        prompt = f"""Question: {question}

Options:
{chr(10).join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])}

Answer with just the letter (A, B, C, or D):"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
        print(f"Error getting remote response: {e}")
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

class HSEBenchRemoteOnlyResults:
    """Class to handle HSE-bench remote-only experiment results."""
    
    def __init__(self, model_name: str, categories: List[str]):
        self.model_name = model_name
        self.categories = categories
        self.start_time = datetime.datetime.now()
        self.results = {}
        self.task_type_results = {
            'rule_recall': {'correct': 0, 'total': 0},
            'rule_application': {'correct': 0, 'total': 0},
            'issue_spotting': {'correct': 0, 'total': 0},
            'rule_conclusion': {'correct': 0, 'total': 0}
        }
        self.detailed_results = []  # Store detailed question data
        self.output_file = None
        
        # Initialize results for each category
        for category in categories:
            self.results[category] = {
                'remote_alone_correct': 0,
                'total_questions': 0
            }
    
    def initialize_output_file(self, model_name: str, categories: List[str]):
        """Initialize the output file with metadata."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        categories_str = "_".join(categories)
        self.output_file = f"experiment_results/QA-results/hse-bench/hse_bench_remote_only_results_{model_name.replace('/', '_')}_{categories_str}_{timestamp}.json"
        
        # Create directory if it doesn't exist
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize the output file
        initial_data = {
            "experiment_type": "HSE-bench-remote-only",
            "model_name": model_name,
            "categories": categories,
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
            "experiment_type": "HSE-bench-remote-only",
            "model_name": self.model_name,
            "categories": self.categories,
            "start_time": str(self.start_time),
            "end_time": str(datetime.datetime.now()),
            "results": self.results,
            "task_type_results": self.task_type_results,
            "detailed_results": self.detailed_results
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(final_data, f, indent=2)
    
    def add_question_result(self, category: str, question_id: int, question_data: Dict, 
                          remote_response: str, is_correct: bool):
        """Add a single question result."""
        self.results[category]['total_questions'] += 1
        if is_correct:
            self.results[category]['remote_alone_correct'] += 1
        
        # Update task type specific results
        task_type = question_data.get('task_type', 'unknown')
        if task_type in self.task_type_results:
            self.task_type_results[task_type]['total'] += 1
            if is_correct:
                self.task_type_results[task_type]['correct'] += 1
        
        # Store detailed result
        detailed_result = {
            "category": category,
            "question_id": question_id,
            "original_question": question_data.get('question', ''),
            "options": question_data.get('options', []),
            "correct_answer": question_data.get('correct_answer', ''),
            "reference": question_data.get('reference', ''),
            "task_type": question_data.get('task_type', ''),
            "remote_alone": {
                "response": remote_response,
                "is_correct": is_correct,
                "response_time": 0.0  # Placeholder
            }
        }
        
        self.detailed_results.append(detailed_result)
        
        # Save intermediate results every 20 questions
        total_questions = sum(cat_results['total_questions'] for cat_results in self.results.values())
        if total_questions % 20 == 0:
            self.save_results()
            print(f"Progress: {total_questions} questions completed across all categories")
            
            # Print current accuracy for each category
            for cat in self.categories:
                cat_total = self.results[cat]['total_questions']
                cat_correct = self.results[cat]['remote_alone_correct']
                if cat_total > 0:
                    cat_accuracy = cat_correct / cat_total * 100
                    print(f"  {cat}: {cat_correct}/{cat_total} = {cat_accuracy:.1f}%")

def send_completion_email(results_file: str, total_accuracy: float, total_questions: int):
    """Send email notification when experiment completes."""
    try:
        # Load email config
        with open('email_config.json', 'r') as f:
            email_config = json.load(f)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['from_email']
        msg['To'] = email_config['to_email']
        msg['Subject'] = f"HSE-bench Remote-Only Test Complete - Host: {socket.gethostname()}"
        
        body = f"""
HSE-bench Remote-Only Test Completed Successfully!

Results Summary:
- Total Questions: {total_questions}
- Remote Model Accuracy: {total_accuracy:.1f}%
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

def run_remote_only_test(model_name: str, categories: List[str]):
    """Run the remote-only HSE-bench test on multiple categories."""
    print(f"Starting HSE-bench Remote-Only Test")
    print(f"Model: {model_name}")
    print(f"Categories: {', '.join(categories)}")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Initialize results handler
    results_handler = HSEBenchRemoteOnlyResults(model_name, categories)
    results_handler.initialize_output_file(model_name, categories)
    
    total_questions_processed = 0
    
    # Process each category
    for category in categories:
        print(f"\n🔄 Processing {category.upper()} questions...")
        
        # Load HSE-bench data for this category
        questions = load_hse_bench_data(category, -1)  # Load all questions
        print(f"Loaded {len(questions)} {category} questions")
        
        # Process each question
        for i, question_data in enumerate(questions):
            question_id = i + 1
            print(f"Processing {category} Question {question_id}/{len(questions)}")
            
            # Extract question components
            question_text = question_data.get('question', '')
            options = question_data.get('options', [])
            correct_answer = question_data.get('correct_answer', '')
            
            try:
                # Get remote model response
                print("Getting remote model response...")
                remote_response = get_remote_response(question_text, options)
                
                # Extract answer from response
                remote_answer = remote_response.strip().upper()
                if len(remote_answer) > 1:
                    remote_answer = remote_answer[0]
                
                # Check if correct
                is_correct = remote_answer == correct_answer.upper()
                
                print(f"Remote response: {remote_answer}")
                print(f"Correct answer: {correct_answer}")
                print(f"Correct: {is_correct}")
                
                # Add result
                results_handler.add_question_result(
                    category, question_id, question_data, remote_answer, is_correct
                )
                
                total_questions_processed += 1
                
            except Exception as e:
                print(f"Error processing {category} question {question_id}: {e}")
                # Add failed result
                results_handler.add_question_result(
                    category, question_id, question_data, "ERROR", False
                )
                total_questions_processed += 1
            
            # Small delay to avoid overwhelming the API
            time.sleep(0.5)
    
    # Save final results
    results_handler.save_results()
    
    # Calculate final accuracies
    total_correct = 0
    total_questions = 0
    
    print("\n" + "=" * 60)
    print("HSE-bench Remote-Only Test Complete!")
    print()
    
    for category in categories:
        cat_total = results_handler.results[category]['total_questions']
        cat_correct = results_handler.results[category]['remote_alone_correct']
        cat_accuracy = (cat_correct / cat_total * 100) if cat_total > 0 else 0
        
        print(f"{category.upper()}: {cat_correct}/{cat_total} = {cat_accuracy:.1f}%")
        
        total_correct += cat_correct
        total_questions += cat_total
    
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0
    
    print(f"\nOVERALL: {total_correct}/{total_questions} = {overall_accuracy:.1f}%")
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
    send_completion_email(results_handler.output_file, overall_accuracy, total_questions)
    
    return results_handler.output_file, overall_accuracy

def main():
    """Main function to run the HSE-bench remote-only test."""
    parser = argparse.ArgumentParser(description='HSE-bench Remote-Only Test')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Remote model name')
    parser.add_argument('--categories', nargs='+', 
                       default=['regulation', 'court_case', 'safety_exam'],
                       choices=['regulation', 'court_case', 'safety_exam', 'video'],
                       help='HSE-bench categories to test (default: regulation court_case safety_exam)')
    
    args = parser.parse_args()
    
    # Run the test
    results_file, accuracy = run_remote_only_test(
        args.model, args.categories
    )
    
    print(f"\nTest completed successfully!")
    print(f"Results file: {results_file}")
    print(f"Overall accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    main()
