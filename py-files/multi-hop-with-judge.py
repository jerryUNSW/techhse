import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import openai
from dotenv import load_dotenv

# --- 1. Configuration ---
# Load environment variables from the .env file
load_dotenv()

# Set the model names for local and remote LLMs
LOCAL_MODEL_NAME = "microsoft/phi-2" # 2.7B
REMOTE_LLM_PROVIDER = "deepseek" # Changed to deepseek

# Remote model for direct answer generation (if used)
REMOTE_LLM_MODEL = "deepseek-chat" # Changed to a DeepSeek model
# Remote model for generating the Chain-of-Thought (CoT)
REMOTE_COT_MODEL = "deepseek-chat" # Already set to DeepSeek
# Remote model for the LLM Judge
REMOTE_JUDGE_MODEL = "deepseek-chat" # Already set to DeepSeek

# Set the dataset and split
DATASET_NAME = "hotpotqa/hotpot_qa" 
DATASET_SPLIT = "validation" 
NUM_SAMPLES_TO_TEST = 50 

# Get API keys from environment variables
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY")
HUGGING_FACE_API = os.getenv("HUGGING_FACE")

# ANSI color codes for better console output
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


# --- 2. Helper Functions ---
def extract_final_answer_from_cot(text):
    """
    Extracts the final, concise answer from a Chain-of-Thought (CoT) response.
    """
    final_answer_marker = "Final Answer:"
    if final_answer_marker in text:
        return text.split(final_answer_marker)[-1].strip()
    sentences = text.split('.')
    if len(sentences) > 1:
        return sentences[-2].strip() if sentences[-2].strip() else text.strip()
    return text.strip()

def llm_judge_answer(client, question, ground_truth, model_answer):
    """
    Uses an LLM as a judge to determine if the model's answer is correct.
    """
    judge_prompt = (
        f"You are an expert evaluator. Your task is to determine if a model's "
        f"answer to a question is semantically correct, based on a given ground truth. "
        f"You must respond with only 'Correct' or 'Incorrect'.\n\n"
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Model Answer: {model_answer}\n\n"
        f"Is the model's answer semantically correct? Your answer must be a single word: Correct or Incorrect."
    )
    
    try:
        response = client.chat.completions.create(
            model=REMOTE_JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": judge_prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        verdict = response.choices[0].message.content.strip().lower()
        return verdict == "correct"
    except Exception as e:
        print(f"{RED}Error with Judge LLM API: {e}{RESET}")
        return False


# --- 3. Model Loading and Setup ---
def load_local_model(model_name):
    """Load a local Hugging Face model and tokenizer."""
    print(f"{CYAN}Loading local model: {model_name}...{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    print(f"{GREEN}Local model loaded successfully!{RESET}")
    return model, tokenizer

def get_remote_llm_client(provider):
    """Get the appropriate client for the remote LLM API."""
    if provider == "openai":
        if not OPEN_AI_KEY:
            raise ValueError("OPEN_AI_KEY not found. Please set it in your .env file.")
        return openai.OpenAI(api_key=OPEN_AI_KEY)
    elif provider == "deepseek":
        if not DEEP_SEEK_KEY:
            raise ValueError("DEEP_SEEK_KEY not found. Please set it in your .env file.")
        # DeepSeek's API is compatible with the OpenAI SDK but requires a custom base_url
        return openai.OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unsupported remote LLM provider: {provider}")


# --- 4. Main Experiment Logic ---
def get_answer_from_local_model(model, tokenizer, context, question):
    """
    Generates a response from the local model using a Chain-of-Thought prompt.
    """
    prompt_template = (
        "Context: {context}\n"
        "Question: {question}\n"
        "Please think step-by-step and provide your final, concise answer at the end.\n"
        "Answer:"
    )
    prompt = prompt_template.format(context=context, question=question)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response_text

def get_answer_from_remote_llm(client, model_name, context, question):
    """
    Generates a response from a powerful remote LLM using a CoT prompt.
    """
    prompt_template = (
        "Context: {context}\n"
        "Question: {question}\n"
        "Please think step-by-step and provide your final, concise answer at the end.\n"
        "Answer:"
    )
    prompt = prompt_template.format(context=context, question=question)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"{RED}Error with remote LLM API: {e}{RESET}")
        return "API Error"

def run_experiment():
    """
    Main function to run the multi-hop reasoning experiment.
    """
    print(f"{CYAN}Loading dataset: {DATASET_NAME}...{RESET}")
    dataset = load_dataset(DATASET_NAME, "distractor", split=DATASET_SPLIT)
    
    multi_hop_questions = [
        q for q in dataset if len(q['supporting_facts']) > 1
    ]

    local_model, local_tokenizer = load_local_model(LOCAL_MODEL_NAME)
    
    try:
        remote_client = get_remote_llm_client(REMOTE_LLM_PROVIDER)
        judge_client = get_remote_llm_client(REMOTE_LLM_PROVIDER) # Using a new client for the judge
    except ValueError as e:
        print(f"{RED}{e}{RESET}")
        print(f"{YELLOW}Skipping remote LLM inference and judge evaluation.{RESET}")
        remote_client = None
        judge_client = None

    local_correct = 0
    remote_correct = 0

    for i, item in enumerate(multi_hop_questions[:NUM_SAMPLES_TO_TEST]):
        print(f"\n{YELLOW}--- Question {i+1}/{NUM_SAMPLES_TO_TEST} ---{RESET}")
        
        question = item['question']
        ground_truth = item['answer']
        # context = " ".join([p[1] for p in item['context']])
        all_sentences = [sentence for sublist in item['context']['sentences'] for sentence in sublist]
        context = " ".join(all_sentences)
        
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
        # --- Local Model Inference ---
        print(f"{BLUE}Running Local Model...{RESET}")
        try:
            local_response_cot = get_answer_from_local_model(
                local_model, local_tokenizer, context, question
            )
            local_answer = extract_final_answer_from_cot(local_response_cot)
            
            if judge_client:
                is_correct_local = llm_judge_answer(judge_client, question, ground_truth, local_answer)
                if is_correct_local:
                    local_correct += 1
            else:
                is_correct_local = False
            
            print(f"Local Answer: {local_answer}")
            print(f"Result (LLM Judge): {'Correct' if is_correct_local else 'Incorrect'}")
        except Exception as e:
            print(f"{RED}Error during local model inference: {e}{RESET}")
            

        # --- Remote LLM Inference ---
        if remote_client and judge_client:
            print(f"{GREEN}Running Remote LLM...{RESET}")
            try:
                remote_response_cot = get_answer_from_remote_llm(
                    remote_client, REMOTE_LLM_MODEL, context, question
                )
                remote_answer = extract_final_answer_from_cot(remote_response_cot)
                
                is_correct_remote = llm_judge_answer(judge_client, question, ground_truth, remote_answer)
                if is_correct_remote:
                    remote_correct += 1
                
                print(f"Remote Answer: {remote_answer}")
                print(f"Result (LLM Judge): {'Correct' if is_correct_remote else 'Incorrect'}")

            except Exception as e:
                print(f"{RED}Error during remote LLM inference: {e}{RESET}")

    # 4. Final Results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    local_accuracy = (local_correct / NUM_SAMPLES_TO_TEST) * 100
    print(f"Local Model ({LOCAL_MODEL_NAME}) Accuracy: {local_accuracy:.2f}% (LLM Judge)")
    if remote_client:
        remote_accuracy = (remote_correct / NUM_SAMPLES_TO_TEST) * 100
        print(f"Remote LLM ({REMOTE_LLM_MODEL}) Accuracy: {remote_accuracy:.2f}% (LLM Judge)")
        print(f"Performance Gap: {remote_accuracy - local_accuracy:.2f}%")
    else:
        print("Remote LLM test was skipped due to missing API key.")

if __name__ == '__main__':
    run_experiment()