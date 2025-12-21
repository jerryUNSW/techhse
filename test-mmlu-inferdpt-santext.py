#!/usr/bin/env python3
"""
MMLU Dataset Experiments with InferDPT and SANTEXT+
===================================================

Run QA experiments on MMLU datasets using:
- InferDPT + QA (epsilon=2.0)
- SANTEXT+ + QA (epsilon=2.0)

Datasets:
- MMLU Professional Law: first 200 questions
- MMLU Professional Medicine: all 272 questions
- MMLU Clinical Knowledge: all 265 questions
- MMLU College Medicine: all 173 questions
"""

import os
import sys
import json
from datetime import datetime
from datasets import load_dataset
from dotenv import load_dotenv

# Import local modules
import utils
from sanitization_methods import inferdpt_sanitize_text, santext_sanitize_text
from experiment_db_writer import ExperimentDBWriter

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

load_dotenv()

# Configuration
EPSILON = 2.0
LOCAL_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
REMOTE_MODEL = "gpt-4o-mini"

# Dataset configurations
MMLU_DATASETS = {
    "professional_law": {
        "name": "MMLU Professional Law",
        "config": "professional_law",
        "num_questions": 200,  # First 200 out of 1534
        "start_index": 0
    },
    "professional_medicine": {
        "name": "MMLU Professional Medicine",
        "config": "professional_medicine",
        "num_questions": 272,  # All questions
        "start_index": 0
    },
    "clinical_knowledge": {
        "name": "MMLU Clinical Knowledge",
        "config": "clinical_knowledge",
        "num_questions": 265,  # All questions
        "start_index": 0
    },
    "college_medicine": {
        "name": "MMLU College Medicine",
        "config": "college_medicine",
        "num_questions": 173,  # All questions
        "start_index": 0
    }
}

def format_mmlu_options(choices):
    """Format MMLU choices into options dict."""
    if isinstance(choices, list):
        options = {}
        for i, choice in enumerate(choices):
            options[chr(65 + i)] = choice  # A, B, C, D
        return options
    return choices

def extract_answer_from_response(response_text, options):
    """Extract answer letter (A, B, C, D) from response text."""
    response_upper = response_text.upper().strip()
    
    # Look for single letter answer
    for letter in ['A', 'B', 'C', 'D']:
        if response_upper.startswith(letter) or f" {letter} " in response_upper or response_upper.endswith(f" {letter}"):
            return letter
    
    # Look for "Answer: A" pattern
    import re
    answer_match = re.search(r'answer[:\s]+([A-D])', response_upper)
    if answer_match:
        return answer_match.group(1)
    
    # Look for first occurrence of A, B, C, or D
    for char in response_upper:
        if char in ['A', 'B', 'C', 'D']:
            return char
    
    return None

def run_qa_with_sanitization(
    local_client,
    local_model,
    remote_client,
    remote_model,
    question,
    options,
    correct_answer,
    sanitization_method,
    epsilon
):
    """
    Run QA with sanitization: sanitize question, get CoT from remote, answer with local.
    
    Args:
        local_client: Local model client
        local_model: Local model name
        remote_client: Remote model client
        remote_model: Remote model name
        question: Original question text
        options: Dict of options {A: "...", B: "...", ...}
        correct_answer: Correct answer letter (A, B, C, D)
        sanitization_method: 'inferdpt' or 'santext'
        epsilon: Privacy parameter
    
    Returns:
        Dict with results
    """
    # Step 1: Sanitize the question
    if sanitization_method == 'inferdpt':
        perturbed_question = inferdpt_sanitize_text(question, epsilon=epsilon)
    elif sanitization_method == 'santext':
        perturbed_question = santext_sanitize_text(question, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown sanitization method: {sanitization_method}")
    
    # Step 2: Generate CoT from remote model using sanitized question
    cot_prompt = f"""Here is a question:

{perturbed_question}

Please provide a clear, step-by-step chain-of-thought reasoning to solve this question. Do NOT provide the final answer; provide only the reasoning steps."""
    
    try:
        cot_response = remote_client.chat.completions.create(
            model=remote_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides step-by-step reasoning."},
                {"role": "user", "content": cot_prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        cot_text = cot_response.choices[0].message.content
    except Exception as e:
        print(f"{YELLOW}Warning: CoT generation failed: {e}{RESET}", flush=True)
        cot_text = ""
    
    # Step 3: Format prompt with original question, options, and CoT
    options_text = "\n".join([f"{letter}. {text}" for letter, text in options.items()])
    
    answer_prompt = f"""Here is a medical question:

{question}

Options:
{options_text}

Here is some reasoning to help you answer:

{cot_text}

Based on the question and reasoning above, what is the correct answer? Respond with only the letter (A, B, C, or D)."""
    
    # Step 4: Get answer from local model
    try:
        answer_response = local_client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "user", "content": answer_prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        response_text = answer_response.choices[0].message.content
        predicted_answer = extract_answer_from_response(response_text, options)
        
        if predicted_answer is None:
            predicted_answer = "UNKNOWN"
            is_correct = False
        else:
            is_correct = (predicted_answer.upper() == correct_answer.upper())
    except Exception as e:
        print(f"{RED}Error in local model inference: {e}{RESET}", flush=True)
        response_text = f"Error: {e}"
        predicted_answer = "ERROR"
        is_correct = False
    
    return {
        "perturbed_question": perturbed_question,
        "induced_cot": cot_text,
        "local_answer": predicted_answer,
        "is_correct": is_correct,
        "response_text": response_text
    }

def run_experiment_for_dataset(dataset_key, dataset_config):
    """Run experiments for a single MMLU dataset."""
    print(f"\n{CYAN}{'='*60}{RESET}", flush=True)
    print(f"{CYAN}Running experiments on: {dataset_config['name']}{RESET}", flush=True)
    print(f"{CYAN}{'='*60}{RESET}", flush=True)
    
    # Load dataset
    print(f"{CYAN}Loading dataset: cais/mmlu/{dataset_config['config']}{RESET}", flush=True)
    try:
        dataset = load_dataset('cais/mmlu', dataset_config['config'], split='test')
        print(f"{GREEN}Loaded {len(dataset)} examples{RESET}", flush=True)
    except Exception as e:
        print(f"{RED}Failed to load dataset: {e}{RESET}", flush=True)
        return None
    
    # Select questions
    start_idx = dataset_config['start_index']
    num_questions = min(dataset_config['num_questions'], len(dataset) - start_idx)
    selected_indices = list(range(start_idx, start_idx + num_questions))
    sample_questions = dataset.select(selected_indices)
    
    print(f"{CYAN}Processing {num_questions} questions (indices {start_idx}-{start_idx + num_questions - 1}){RESET}", flush=True)
    
    # Initialize clients
    print(f"{CYAN}Initializing clients...{RESET}", flush=True)
    local_client = utils.get_nebius_client()
    remote_client = utils.get_openai_client()
    
    # Initialize results
    results = {
        "dataset": dataset_config['name'],
        "dataset_key": dataset_key,
        "epsilon": EPSILON,
        "local_model": LOCAL_MODEL,
        "remote_model": REMOTE_MODEL,
        "num_questions": num_questions,
        "start_index": start_idx,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "inferdpt_results": [],
        "santext_results": []
    }
    
    inferdpt_correct = 0
    santext_correct = 0
    
    # Process each question
    for i, item in enumerate(sample_questions):
        dataset_idx = selected_indices[i]
        question = item['question']
        choices = item['choices']  # List of choices
        correct_answer_idx = item['answer']  # Integer index (0, 1, 2, 3)
        
        # Convert to our format
        options = format_mmlu_options(choices)
        correct_answer_letter = chr(65 + correct_answer_idx)  # 0->A, 1->B, 2->C, 3->D
        
        print(f"\n{YELLOW}{'='*60}{RESET}", flush=True)
        print(f"{YELLOW}Question {i+1}/{num_questions} (Dataset idx: {dataset_idx}){RESET}", flush=True)
        print(f"{YELLOW}{'='*60}{RESET}", flush=True)
        print(f"Question: {question[:100]}...", flush=True)
        print(f"Correct Answer: {correct_answer_letter}", flush=True)
        
        # Run InferDPT + QA
        print(f"\n{CYAN}Running InferDPT + QA...{RESET}", flush=True)
        inferdpt_result = run_qa_with_sanitization(
            local_client, LOCAL_MODEL,
            remote_client, REMOTE_MODEL,
            question, options, correct_answer_letter,
            'inferdpt', EPSILON
        )
        
        if inferdpt_result['is_correct']:
            inferdpt_correct += 1
        
        print(f"InferDPT Answer: {inferdpt_result['local_answer']} ({'✓' if inferdpt_result['is_correct'] else '✗'})", flush=True)
        
        # Run SANTEXT+ + QA
        print(f"\n{CYAN}Running SANTEXT+ + QA...{RESET}", flush=True)
        santext_result = run_qa_with_sanitization(
            local_client, LOCAL_MODEL,
            remote_client, REMOTE_MODEL,
            question, options, correct_answer_letter,
            'santext', EPSILON
        )
        
        if santext_result['is_correct']:
            santext_correct += 1
        
        print(f"SANTEXT+ Answer: {santext_result['local_answer']} ({'✓' if santext_result['is_correct'] else '✗'})", flush=True)
        
        # Store results (only essential fields)
        results['inferdpt_results'].append({
            "question_index": dataset_idx,
            "perturbed_question": inferdpt_result['perturbed_question'],
            "induced_cot": inferdpt_result['induced_cot'],
            "local_answer": inferdpt_result['local_answer'],
            "is_correct": inferdpt_result['is_correct']
        })
        
        results['santext_results'].append({
            "question_index": dataset_idx,
            "perturbed_question": santext_result['perturbed_question'],
            "induced_cot": santext_result['induced_cot'],
            "local_answer": santext_result['local_answer'],
            "is_correct": santext_result['is_correct']
        })
        
        # Print running accuracy
        inferdpt_acc = (inferdpt_correct / (i + 1)) * 100
        santext_acc = (santext_correct / (i + 1)) * 100
        print(f"\n{GREEN}Running Accuracy - InferDPT: {inferdpt_correct}/{i+1} ({inferdpt_acc:.2f}%), SANTEXT+: {santext_correct}/{i+1} ({santext_acc:.2f}%){RESET}", flush=True)
        
        # Save incrementally
        output_dir = "exp/mmlu-results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"mmlu_{dataset_key}_eps{EPSILON}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Add summary
        results['summary'] = {
            "inferdpt": {
                "correct": inferdpt_correct,
                "total": i + 1,
                "accuracy": inferdpt_acc
            },
            "santext": {
                "correct": santext_correct,
                "total": i + 1,
                "accuracy": santext_acc
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Final summary
    final_inferdpt_acc = (inferdpt_correct / num_questions) * 100
    final_santext_acc = (santext_correct / num_questions) * 100
    
    print(f"\n{GREEN}{'='*60}{RESET}", flush=True)
    print(f"{GREEN}Final Results for {dataset_config['name']}{RESET}", flush=True)
    print(f"{GREEN}{'='*60}{RESET}", flush=True)
    print(f"InferDPT: {inferdpt_correct}/{num_questions} ({final_inferdpt_acc:.2f}%)", flush=True)
    print(f"SANTEXT+: {santext_correct}/{num_questions} ({final_santext_acc:.2f}%)", flush=True)
    print(f"{GREEN}{'='*60}{RESET}", flush=True)
    
    results['summary'] = {
        "inferdpt": {
            "correct": inferdpt_correct,
            "total": num_questions,
            "accuracy": final_inferdpt_acc
        },
        "santext": {
            "correct": santext_correct,
            "total": num_questions,
            "accuracy": final_santext_acc
        }
    }
    
    return results

def main():
    """Main function to run all experiments."""
    print(f"{CYAN}{'='*60}{RESET}", flush=True)
    print(f"{CYAN}MMLU Experiments: InferDPT + QA and SANTEXT+ + QA{RESET}", flush=True)
    print(f"{CYAN}Epsilon: {EPSILON}{RESET}", flush=True)
    print(f"{CYAN}{'='*60}{RESET}", flush=True)
    
    all_results = {}
    
    for dataset_key, dataset_config in MMLU_DATASETS.items():
        try:
            result = run_experiment_for_dataset(dataset_key, dataset_config)
            if result:
                all_results[dataset_key] = result
        except Exception as e:
            print(f"{RED}Error processing {dataset_config['name']}: {e}{RESET}", flush=True)
            import traceback
            traceback.print_exc()
    
    # Save combined results
    output_dir = "exp/mmlu-results"
    os.makedirs(output_dir, exist_ok=True)
    combined_file = os.path.join(output_dir, f"mmlu_all_datasets_eps{EPSILON}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{GREEN}All experiments completed!{RESET}", flush=True)
    print(f"Results saved to: {combined_file}", flush=True)

if __name__ == "__main__":
    main()

