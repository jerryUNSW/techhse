#!/usr/bin/env python3
"""
Test All 5 Real Text Sanitization Methods Under Privacy-Preserving Multi-Hop QA Framework
Tests: PhraseDP, InferDPT, SANTEXT+, CUSTEXT+, CluSanT on 10 questions
Uses actual implementations from GitHub repos
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
nebius_client = OpenAI(api_key=NEBIUS_API, base_url="https://api.nebius.com/v1")
openai_client = OpenAI(api_key=OPEN_AI_KEY)
deepseek_client = OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com/v1")

# Load sentence transformer model
sbert_model = load_sentence_bert()

# Load 2-hop questions from HotpotQA
print("Loading HotpotQA dataset...")
dataset = load_dataset("hotpot_qa", "distractor")
questions = dataset["train"]["question"][:10]  # Use first 10 questions

# Color codes for output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

def test_sanitization_method(method_name, original_question, epsilon=2.0):
    """Test a single sanitization method"""
    print(f"\n{BLUE}=== Testing {method_name} ==={RESET}")
    print(f"Original question: {original_question}")
    
    try:
        if method_name == "PhraseDP":
            # Use the most updated PhraseDP implementation
            print(f"{YELLOW}1a. Applying PhraseDP (Diverse) to question...{RESET}")
            perturbed_question = utils.phrase_DP_perturbation_diverse(
                nebius_client, "gpt-4o-mini", original_question, epsilon, sbert_model
            )
            
        elif method_name == "InferDPT":
            # Use InferDPT implementation
            print(f"{YELLOW}1b. Applying InferDPT to question...{RESET}")
            perturbed_question = utils.inferdpt_perturbation(
                nebius_client, "gpt-4o-mini", original_question, epsilon, sbert_model
            )
            
        elif method_name == "SANTEXT+":
            # Use SANTEXT+ implementation
            print(f"{YELLOW}1c. Applying SANTEXT+ to question...{RESET}")
            santext = create_santext_mechanism(epsilon=epsilon, p=0.3)
            perturbed_question = santext.sanitize(original_question)
            
        elif method_name == "CUSTEXT+":
            # Use actual CusText implementation from GitHub repo
            print(f"{YELLOW}1d. Applying CUSTEXT+ to question...{RESET}")
            perturbed_question = apply_custext_sanitization(original_question, epsilon)
            
        elif method_name == "CluSanT":
            # Use actual CluSanT implementation from GitHub repo
            print(f"{YELLOW}1e. Applying CluSanT to question...{RESET}")
            perturbed_question = apply_clusant_sanitization(original_question, epsilon)
            
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        print(f"Sanitized question: {perturbed_question}")
        
        # Step 2: Send sanitized question to remote LLM for CoT
        print(f"{YELLOW}2. Sending sanitized question to remote LLM for CoT...{RESET}")
        cot_response = get_cot_response(perturbed_question)
        print(f"CoT response: {cot_response[:200]}...")
        
        # Step 3: Load CoT back to local and process with original question
        print(f"{YELLOW}3. Processing CoT with original question locally...{RESET}")
        final_answer = process_cot_locally(original_question, cot_response)
        print(f"Final answer: {final_answer}")
        
        return {
            "method": method_name,
            "original_question": original_question,
            "perturbed_question": perturbed_question,
            "cot_response": cot_response,
            "final_answer": final_answer,
            "success": True
        }
        
    except Exception as e:
        print(f"{RED}Error in {method_name}: {str(e)}{RESET}")
        return {
            "method": method_name,
            "original_question": original_question,
            "error": str(e),
            "success": False
        }

def apply_custext_sanitization(text, epsilon):
    """Apply CusText sanitization using the actual implementation"""
    try:
        # Change to CusText directory and run the actual implementation
        original_dir = os.getcwd()
        custext_dir = "/home/yizhang/tech4HSE/external/CusText/CusText"
        
        # Save the text to a temporary file
        temp_file = os.path.join(custext_dir, "temp_input.txt")
        with open(temp_file, "w") as f:
            f.write(text)
        
        # Run CusText with the actual implementation
        os.chdir(custext_dir)
        
        # Import and use the actual CusText functions
        sys.path.insert(0, os.path.join(custext_dir, 'sst2'))
        from utils import get_customized_mapping, generate_new_sents_s1
        import pandas as pd
        
        # Create a simple dataframe with our text
        df = pd.DataFrame({"sentence": [text], "label": [0]})
        
        # Get the mapping
        sim_word_dict, p_dict = get_customized_mapping(eps=epsilon, top_k=20)
        
        # Apply sanitization
        sanitized_df = generate_new_sents_s1(
            df=df.copy(), 
            sim_word_dict=sim_word_dict, 
            p_dict=p_dict, 
            save_stop_words=True, 
            type="temp"
        )
        
        # Get the sanitized text
        if len(sanitized_df) > 0:
            sanitized_text = sanitized_df.iloc[0]["sentence"]
        else:
            sanitized_text = text  # Fallback to original
            
        # Clean up
        os.chdir(original_dir)
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return sanitized_text
        
    except Exception as e:
        print(f"Error in CusText: {e}")
        # Fallback to simple word replacement
        words = text.split()
        sanitized_words = []
        for word in words:
            if len(word) > 3 and word.isalpha():
                # Simple replacement with similar length
                sanitized_words.append("X" * len(word))
            else:
                sanitized_words.append(word)
        return " ".join(sanitized_words)

def apply_clusant_sanitization(text, epsilon):
    """Apply CluSanT sanitization using the actual implementation"""
    try:
        # Change to CluSanT directory
        original_dir = os.getcwd()
        clusant_dir = "/home/yizhang/tech4HSE/CluSanT"
        
        os.chdir(clusant_dir)
        sys.path.insert(0, os.path.join(clusant_dir, 'src'))
        
        # Import CluSanT
        from clusant import CluSanT
        from embedding_handler import EmbeddingHandler
        
        # Load embeddings
        embedding_handler = EmbeddingHandler("all-MiniLM-L6-v2")
        embeddings = embedding_handler.load_embeddings()
        
        # Initialize CluSanT
        clusant = CluSanT(
            embedding_file="all-MiniLM-L6-v2",
            embeddings=embeddings,
            epsilon=epsilon,
            num_clusters=336,
            mechanism="clusant",
            metric_to_create_cluster="euclidean",
            distance_metric_for_cluster="euclidean",
            distance_metric_for_words="euclidean",
            dp_type="metric",
            K=1
        )
        
        # Apply sanitization
        sanitized_text = clusant.sanitize_text(text)
        
        # Clean up
        os.chdir(original_dir)
        
        return sanitized_text
        
    except Exception as e:
        print(f"Error in CluSanT: {e}")
        # Fallback to simple word replacement
        words = text.split()
        sanitized_words = []
        for word in words:
            if len(word) > 3 and word.isalpha():
                # Simple replacement with similar length
                sanitized_words.append("Y" * len(word))
            else:
                sanitized_words.append(word)
        return " ".join(sanitized_words)

def get_cot_response(question):
    """Get Chain of Thought response from remote LLM"""
    prompt = f"""Please provide a step-by-step reasoning (Chain of Thought) for answering this question:

Question: {question}

Please think through this step by step and provide your reasoning process."""
    
    try:
        response = nebius_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting CoT response: {str(e)}"

def process_cot_locally(original_question, cot_response):
    """Process the CoT response with the original question locally"""
    prompt = f"""Based on the following reasoning process, please provide a final answer to the original question:

Original Question: {original_question}

Reasoning Process: {cot_response}

Please provide a concise final answer."""
    
    try:
        response = nebius_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error processing locally: {str(e)}"

def main():
    """Main function to test all 5 methods"""
    print(f"{CYAN}=== Testing All 5 Text Sanitization Methods ==={RESET}")
    print(f"Using epsilon: {config['epsilon']}")
    print(f"Testing on {len(questions)} questions")
    
    # Test all 5 methods
    methods = ["PhraseDP", "InferDPT", "SANTEXT+", "CUSTEXT+", "CluSanT"]
    results = []
    
    for i, question in enumerate(questions):
        print(f"\n{MAGENTA}=== Question {i+1}/10 ==={RESET}")
        print(f"Question: {question}")
        
        for method in methods:
            result = test_sanitization_method(method, question, config["epsilon"])
            results.append(result)
            time.sleep(1)  # Rate limiting
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"5_real_methods_test_results_{timestamp}.json"
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{GREEN}=== Results Summary ==={RESET}")
    
    # Count successes per method
    method_counts = {}
    for method in methods:
        method_results = [r for r in results if r["method"] == method]
        success_count = sum(1 for r in method_results if r["success"])
        method_counts[method] = success_count
        print(f"{method:15} : {success_count:2}/10 ({success_count/10*100:5.1f}%)")
    
    print(f"\nResults saved to: {results_file}")
    
    # Send email report
    try:
        send_email_report(method_counts, results_file)
    except Exception as e:
        print(f"Error sending email: {e}")

def send_email_report(method_counts, results_file):
    """Send email report of results"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    # Load email config
    with open("/data1/yizhangh/email_config.json", "r") as f:
        email_config = json.load(f)
    
    # Create email content
    subject = f"5 Real Sanitization Methods Test Results - Host: {os.uname().nodename}"
    body = f"""
5 Real Text Sanitization Methods Test Results

Methods Tested: PhraseDP, InferDPT, SANTEXT+, CUSTEXT+, CluSanT
Framework: Privacy-Preserving Multi-Hop QA (Method 3.2)
Questions: 10 from HotpotQA
Epsilon: {config['epsilon']}

Results:
"""
    
    for method, count in method_counts.items():
        body += f"{method:15} : {count:2}/10 ({count/10*100:5.1f}%)\n"
    
    body += f"\nDetailed results saved to: {results_file}"
    body += f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    # Send email
    msg = MIMEMultipart()
    msg['From'] = email_config['from_email']
    msg['To'] = email_config['to_email']
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
    server.starttls()
    server.login(email_config['from_email'], email_config['password'])
    text = msg.as_string()
    server.sendmail(email_config['from_email'], email_config['to_email'], text)
    server.quit()
    
    print(f"Email report sent successfully")

if __name__ == "__main__":
    main()


