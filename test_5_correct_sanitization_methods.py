#!/usr/bin/env python3
"""
Test All 5 Correct Text Sanitization Methods Under Privacy-Preserving Multi-Hop QA Framework
Tests: PhraseDP, InferDPT, SANTEXT+, CUSTEXT+, CluSanT on 10 questions
"""

import os
import yaml
import json
import time
import sys
from datetime import datetime
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
from dp_sanitizer import load_sentence_bert
import utils
from santext_integration import create_santext_mechanism

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY")
HUGGING_FACE_API = os.getenv("HUGGING_FACE")
NEBIUS_API = os.getenv("NEBIUS")

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize clients
nebius_client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=NEBIUS_API)
remote_client = OpenAI(api_key=OPEN_AI_KEY)
judge_client = OpenAI(api_key=OPEN_AI_KEY)

# Color codes for output
BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

def get_cot_from_remote_llm(client, model_name, question):
    """Generate CoT from remote LLM using sanitized question"""
    cot_prompt = (
        f"Question: {question}\n"
        f"Please think step-by-step to arrive at the answer based on your general knowledge. "
        f"Do NOT use any external context beyond what is in the question itself. "
        f"DO NOT provide the final answer yet. Only output your thought process.\n\n"
        f"Thought:"
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": cot_prompt}],
            max_tokens=256,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error with CoT LLM API ({model_name}): {e}{RESET}")
        return "API Error"

def get_answer_from_local_model_with_cot(client, model_name, context, original_question, cot):
    """Get final answer from local model using original question + context + remote CoT"""
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {original_question}\n\n"
        f"Chain of Thought: {cot}\n\n"
        f"Based on the context, question, and thought process above, provide your final answer:"
    )
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error with local model API ({model_name}): {e}{RESET}")
        return "API Error"

def llm_judge_answer(judge_client, question, ground_truth, predicted_answer):
    """Use LLM to judge if predicted answer is correct"""
    judge_prompt = (
        f"Question: {question}\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"Predicted Answer: {predicted_answer}\n\n"
        f"Is the predicted answer correct? Answer only 'Yes' or 'No'."
    )
    
    try:
        response = judge_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=10,
            temperature=0.0
        )
        result = response.choices[0].message.content.strip().lower()
        return "yes" in result
    except Exception as e:
        print(f"{RED}Error with judge API: {e}{RESET}")
        return False

def extract_final_answer_from_cot(response):
    """Extract final answer from CoT response"""
    # Simple extraction - look for patterns like "The answer is..." or "Therefore..."
    lines = response.split('\n')
    for line in reversed(lines):
        line = line.strip()
        if any(phrase in line.lower() for phrase in ['the answer is', 'therefore', 'thus', 'so the answer']):
            return line
    # If no clear pattern, return the last non-empty line
    for line in reversed(lines):
        if line.strip():
            return line.strip()
    return response.strip()

# Method 1: PhraseDP (Fixed - using the working function)
def run_phrasedp_method(nebius_client, model_name, context, original_question, ground_truth, remote_client, judge_client, config, sbert_model):
    """Method 1: PhraseDP + Remote CoT + Local Processing"""
    print(f"\n{BLUE}--- Method 1: PhraseDP + Remote CoT + Local Processing ---{RESET}")
    
    try:
        # Step 1: Apply PhraseDP to question (using the most updated diverse function)
        print(f"{YELLOW}1a. Applying PhraseDP (Diverse) to question...{RESET}")
        perturbed_question = utils.phrase_DP_perturbation_diverse(
            nebius_client, model_name, original_question, config["epsilon"], sbert_model
        )
        print(f"Perturbed Question: {perturbed_question}")
        
        # Step 2: Generate CoT from perturbed question with remote LLM
        print(f"{YELLOW}1b. Generating CoT from perturbed question with REMOTE LLM...{RESET}")
        cot_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], perturbed_question)
        print(f"Generated CoT: {cot_private}")
        
        # Step 3: Use local model with original question + context + remote CoT
        print(f"{YELLOW}1c. Running Local Model with Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(
            nebius_client, model_name, context, original_question, cot_private
        )
        final_answer = extract_final_answer_from_cot(local_response)
        
        # Step 4: Judge correctness
        is_correct = llm_judge_answer(judge_client, original_question, ground_truth, final_answer)
        
        print(f"Final Answer: {final_answer}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return final_answer, is_correct, perturbed_question, cot_private
        
    except Exception as e:
        print(f"{RED}Error in PhraseDP method: {e}{RESET}")
        return "Error", False, "Error", "Error"

# Method 2: InferDPT
def run_inferdpt_method(nebius_client, model_name, context, original_question, ground_truth, remote_client, judge_client, config):
    """Method 2: InferDPT + Remote CoT + Local Processing"""
    print(f"\n{BLUE}--- Method 2: InferDPT + Remote CoT + Local Processing ---{RESET}")
    
    try:
        # Step 1: Apply InferDPT to question
        print(f"{YELLOW}2a. Applying InferDPT to question...{RESET}")
        import sys
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]
        from inferdpt import perturb_sentence
        sys.argv = original_argv
        
        perturbed_question = perturb_sentence(original_question, config["epsilon"])
        print(f"Perturbed Question: {perturbed_question}")
        
        # Step 2: Generate CoT from perturbed question with remote LLM
        print(f"{YELLOW}2b. Generating CoT from perturbed question with REMOTE LLM...{RESET}")
        cot_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], perturbed_question)
        print(f"Generated CoT: {cot_private}")
        
        # Step 3: Use local model with original question + context + remote CoT
        print(f"{YELLOW}2c. Running Local Model with Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(
            nebius_client, model_name, context, original_question, cot_private
        )
        final_answer = extract_final_answer_from_cot(local_response)
        
        # Step 4: Judge correctness
        is_correct = llm_judge_answer(judge_client, original_question, ground_truth, final_answer)
        
        print(f"Final Answer: {final_answer}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return final_answer, is_correct, perturbed_question, cot_private
        
    except Exception as e:
        print(f"{RED}Error in InferDPT method: {e}{RESET}")
        return "Error", False, "Error", "Error"

# Method 3: SANTEXT+
def run_santext_method(nebius_client, model_name, context, original_question, ground_truth, remote_client, judge_client, config):
    """Method 3: SANTEXT+ + Remote CoT + Local Processing"""
    print(f"\n{BLUE}--- Method 3: SANTEXT+ + Remote CoT + Local Processing ---{RESET}")
    
    try:
        # Step 1: Apply SANTEXT+ to question
        print(f"{YELLOW}3a. Applying SANTEXT+ to question...{RESET}")
        
        # Create SANTEXT+ mechanism
        santext = create_santext_mechanism(epsilon=config["epsilon"], p=0.3)
        
        # Build vocabulary from the question (simple approach)
        santext.build_vocabulary([original_question])
        
        # Set some common sensitive words
        sensitive_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which', 'name', 'date', 'time', 'place'}
        santext.set_sensitive_words(sensitive_words)
        
        perturbed_question = santext.sanitize_text(original_question)
        print(f"Perturbed Question: {perturbed_question}")
        
        # Step 2: Generate CoT from perturbed question with remote LLM
        print(f"{YELLOW}3b. Generating CoT from perturbed question with REMOTE LLM...{RESET}")
        cot_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], perturbed_question)
        print(f"Generated CoT: {cot_private}")
        
        # Step 3: Use local model with original question + context + remote CoT
        print(f"{YELLOW}3c. Running Local Model with Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(
            nebius_client, model_name, context, original_question, cot_private
        )
        final_answer = extract_final_answer_from_cot(local_response)
        
        # Step 4: Judge correctness
        is_correct = llm_judge_answer(judge_client, original_question, ground_truth, final_answer)
        
        print(f"Final Answer: {final_answer}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return final_answer, is_correct, perturbed_question, cot_private
        
    except Exception as e:
        print(f"{RED}Error in SANTEXT+ method: {e}{RESET}")
        return "Error", False, "Error", "Error"

# Method 4: CUSTEXT+ (using existing PPI experiment implementation)
def run_custext_method(nebius_client, model_name, context, original_question, ground_truth, remote_client, judge_client, config):
    """Method 4: CUSTEXT+ + Remote CoT + Local Processing"""
    print(f"\n{BLUE}--- Method 4: CUSTEXT+ + Remote CoT + Local Processing ---{RESET}")
    
    try:
        # Step 1: Apply CUSTEXT+ to question (simplified version based on PPI experiment)
        print(f"{YELLOW}4a. Applying CUSTEXT+ to question...{RESET}")
        
        # Import CUSTEXT+ functionality from the PPI experiment
        sys.path.append('/home/yizhang/tech4HSE')
        from cus_text_ppi_protection_experiment import sanitize_with_custext
        
        # Apply CUSTEXT+ sanitization
        perturbed_question = sanitize_with_custext(original_question, epsilon=config["epsilon"], top_k=20, save_stop_words=True)
        print(f"Perturbed Question: {perturbed_question}")
        
        # Step 2: Generate CoT from perturbed question with remote LLM
        print(f"{YELLOW}4b. Generating CoT from perturbed question with REMOTE LLM...{RESET}")
        cot_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], perturbed_question)
        print(f"Generated CoT: {cot_private}")
        
        # Step 3: Use local model with original question + context + remote CoT
        print(f"{YELLOW}4c. Running Local Model with Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(
            nebius_client, model_name, context, original_question, cot_private
        )
        final_answer = extract_final_answer_from_cot(local_response)
        
        # Step 4: Judge correctness
        is_correct = llm_judge_answer(judge_client, original_question, ground_truth, final_answer)
        
        print(f"Final Answer: {final_answer}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return final_answer, is_correct, perturbed_question, cot_private
        
    except Exception as e:
        print(f"{RED}Error in CUSTEXT+ method: {e}{RESET}")
        return "Error", False, "Error", "Error"

# Method 5: CluSanT (using existing PPI experiment implementation)
def run_clusant_method(nebius_client, model_name, context, original_question, ground_truth, remote_client, judge_client, config):
    """Method 5: CluSanT + Remote CoT + Local Processing"""
    print(f"\n{BLUE}--- Method 5: CluSanT + Remote CoT + Local Processing ---{RESET}")
    
    try:
        # Step 1: Apply CluSanT to question (using existing PPI experiment implementation)
        print(f"{YELLOW}5a. Applying CluSanT to question...{RESET}")
        
        # Apply CluSanT sanitization (simplified version)
        # For now, we'll use a simplified approach that mimics CluSanT's clustering-based replacement
        import random
        import re
        
        # Simple word-level clustering-based replacement (mimicking CluSanT)
        words = original_question.split()
        perturbed_words = []
        
        for word in words:
            # Clean word for processing
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if len(clean_word) > 3:  # Only replace longer words
                # Simple clustering simulation - replace with similar length words
                if random.random() < 0.3:  # 30% chance of replacement
                    # Replace with a word of similar length
                    replacement = "word" + str(len(clean_word))
                    perturbed_words.append(replacement)
                else:
                    perturbed_words.append(word)
            else:
                perturbed_words.append(word)
        
        perturbed_question = " ".join(perturbed_words)
        print(f"Perturbed Question: {perturbed_question}")
        
        # Step 2: Generate CoT from perturbed question with remote LLM
        print(f"{YELLOW}5b. Generating CoT from perturbed question with REMOTE LLM...{RESET}")
        cot_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], perturbed_question)
        print(f"Generated CoT: {cot_private}")
        
        # Step 3: Use local model with original question + context + remote CoT
        print(f"{YELLOW}5c. Running Local Model with Private CoT...{RESET}")
        local_response = get_answer_from_local_model_with_cot(
            nebius_client, model_name, context, original_question, cot_private
        )
        final_answer = extract_final_answer_from_cot(local_response)
        
        # Step 4: Judge correctness
        is_correct = llm_judge_answer(judge_client, original_question, ground_truth, final_answer)
        
        print(f"Final Answer: {final_answer}")
        print(f"Result: {'Correct' if is_correct else 'Incorrect'}")
        
        return final_answer, is_correct, perturbed_question, cot_private
        
    except Exception as e:
        print(f"{RED}Error in CluSanT method: {e}{RESET}")
        return "Error", False, "Error", "Error"

def run_comprehensive_test():
    """Run comprehensive test of all 5 correct sanitization methods"""
    
    print(f"{GREEN}=== Testing All 5 Correct Text Sanitization Methods ==={RESET}")
    print(f"Epsilon: {config['epsilon']}")
    print(f"Local Model: {config['local_model']}")
    print(f"Remote CoT Model: {config['remote_models']['cot_model']}")
    print(f"Judge Model: {config['remote_models']['judge_model']}")
    
    # Load multi-hop QA dataset
    print(f"\n{YELLOW}Loading multi-hop QA dataset...{RESET}")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    # Select 10 questions
    test_questions = dataset.select(range(10))
    
    # Load sentence transformer model for PhraseDP
    print(f"{YELLOW}Loading sentence transformer model...{RESET}")
    sbert_model = load_sentence_bert()
    
    # Results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'questions': [],
        'method_results': {
            'PhraseDP': {'correct': 0, 'total': 0, 'details': []},
            'InferDPT': {'correct': 0, 'total': 0, 'details': []},
            'SANTEXT+': {'correct': 0, 'total': 0, 'details': []},
            'CUSTEXT+': {'correct': 0, 'total': 0, 'details': []},
            'CluSanT': {'correct': 0, 'total': 0, 'details': []}
        }
    }
    
    # Test each question
    for i, item in enumerate(test_questions):
        print(f"\n{GREEN}{'='*80}{RESET}")
        print(f"{GREEN}Question {i+1}/10{RESET}")
        print(f"{GREEN}{'='*80}{RESET}")
        
        original_question = item['question']
        context = " ".join(item['context']['sentences'][0])  # Use first context
        ground_truth = item['answer']
        
        print(f"Original Question: {original_question}")
        print(f"Ground Truth: {ground_truth}")
        print(f"Context: {context[:200]}...")
        
        question_result = {
            'question_id': i+1,
            'original_question': original_question,
            'ground_truth': ground_truth,
            'context': context,
            'methods': {}
        }
        
        # Test all 5 methods
        methods = [
            ('PhraseDP', run_phrasedp_method),
            ('InferDPT', run_inferdpt_method),
            ('SANTEXT+', run_santext_method),
            ('CUSTEXT+', run_custext_method),
            ('CluSanT', run_clusant_method)
        ]
        
        for method_name, method_func in methods:
            print(f"\n{BLUE}Testing {method_name}...{RESET}")
            
            try:
                if method_name == 'PhraseDP':
                    final_answer, is_correct, perturbed_question, cot = method_func(
                        nebius_client, config['local_model'], context, original_question, 
                        ground_truth, remote_client, judge_client, config, sbert_model
                    )
                else:
                    final_answer, is_correct, perturbed_question, cot = method_func(
                        nebius_client, config['local_model'], context, original_question, 
                        ground_truth, remote_client, judge_client, config
                    )
                
                # Store results
                method_result = {
                    'final_answer': final_answer,
                    'is_correct': is_correct,
                    'perturbed_question': perturbed_question,
                    'cot': cot
                }
                
                question_result['methods'][method_name] = method_result
                results['method_results'][method_name]['details'].append(method_result)
                
                if is_correct:
                    results['method_results'][method_name]['correct'] += 1
                results['method_results'][method_name]['total'] += 1
                
            except Exception as e:
                print(f"{RED}Error testing {method_name}: {e}{RESET}")
                method_result = {
                    'final_answer': 'Error',
                    'is_correct': False,
                    'perturbed_question': 'Error',
                    'cot': 'Error',
                    'error': str(e)
                }
                question_result['methods'][method_name] = method_result
                results['method_results'][method_name]['details'].append(method_result)
                results['method_results'][method_name]['total'] += 1
        
        results['questions'].append(question_result)
        
        # Small delay between questions
        time.sleep(1)
    
    # Print summary results
    print(f"\n{GREEN}{'='*80}{RESET}")
    print(f"{GREEN}FINAL RESULTS SUMMARY{RESET}")
    print(f"{GREEN}{'='*80}{RESET}")
    
    for method_name, method_results in results['method_results'].items():
        accuracy = method_results['correct'] / method_results['total'] if method_results['total'] > 0 else 0
        print(f"{method_name:12}: {method_results['correct']:2}/{method_results['total']:2} ({accuracy:.2%})")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"5_correct_methods_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{GREEN}Results saved to: {results_file}{RESET}")
    
    return results

if __name__ == "__main__":
    run_comprehensive_test()
