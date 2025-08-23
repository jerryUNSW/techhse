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
LOCAL_MODEL_NAME = "microsoft/phi-2"
# REMOTE_LLM_PROVIDER = "openai" 
REMOTE_LLM_PROVIDER = "deepseek" 
# # Remote model for generating the Chain-of-Thought (CoT)
# REMOTE_COT_MODEL = "gpt-4o"
# # Remote model for the LLM Judge
# REMOTE_JUDGE_MODEL = "gpt-4o-mini"

# Remote model for direct answer generation (if used)
REMOTE_LLM_MODEL = "deepseek-chat" # Changed to a DeepSeek model
# Remote model for generating the Chain-of-Thought (CoT)
REMOTE_COT_MODEL = "deepseek-chat" # Already set to DeepSeek
# Remote model for the LLM Judge
REMOTE_JUDGE_MODEL = "deepseek-chat" # Already set to DeepSeek

# Set the dataset and split
DATASET_NAME = "hotpotqa/hotpot_qa" 
DATASET_SPLIT = "validation" 
NUM_SAMPLES_TO_TEST = 10 

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
    else:
        raise ValueError(f"Unsupported remote LLM provider: {provider}")

# --- 4. Main Experiment Logic ---
def get_cot_from_remote_llm(client, model_name, question):
    """
    Generates a Chain-of-Thought (CoT) using a powerful remote LLM.
    """
    cot_prompt = (
        # f"Context: {context}\n"
        f"Question: {question}\n"
        f"Please think step-by-step to arrive at the answer, but DO NOT provide the final answer yet. "
        f"Only output your thought process.\n\n"
        f"Thought:"
    )
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": cot_prompt}
            ],
            max_tokens=256,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error with CoT LLM API: {e}{RESET}")
        return "API Error"

def get_answer_from_local_model_with_cot(model, tokenizer, context, question, cot):
    """
    Generates a response from the local model, guided by a provided CoT.
    """
    prompt_template = (
        "Context: {context}\n"
        "Question: {question}\n"
        "You have been provided with a Chain-of-Thought to help you solve this question. "
        "Follow these steps to derive your final answer.\n\n"
        "Chain-of-Thought: {cot}\n\n"
        "Based on the context and the Chain-of-Thought, please provide your final, concise answer.\n"
        "Answer:"
    )
    prompt = prompt_template.format(context=context, question=question, cot=cot)

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

def get_answer_from_local_model_alone(model, tokenizer, context, question):
    """
    Generates a response from the local model without any CoT.
    """
    prompt_template = (
        "Context: {context}\n"
        "Question: {question}\n"
        "Please provide your final, concise answer.\n"
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


def run_experiment():
    """
    Main function to run the multi-hop reasoning experiment comparing local model performance
    with and without a remote LLM-generated CoT.
    """
    print(f"{CYAN}Loading dataset: {DATASET_NAME}...{RESET}")
    dataset = load_dataset(DATASET_NAME, "distractor", split=DATASET_SPLIT)
    
    multi_hop_questions = [
        q for q in dataset if len(q['supporting_facts']) > 1
    ]

    local_model, local_tokenizer = load_local_model(LOCAL_MODEL_NAME)
    
    try:
        remote_client = get_remote_llm_client(REMOTE_LLM_PROVIDER)
        judge_client = get_remote_llm_client(REMOTE_LLM_PROVIDER)
    except ValueError as e:
        print(f"{RED}{e}{RESET}")
        print(f"{YELLOW}Skipping remote LLM CoT generation and judge evaluation.{RESET}")
        remote_client = None
        judge_client = None

    local_alone_correct = 0
    local_cot_aided_correct = 0

    for i, item in enumerate(multi_hop_questions[:NUM_SAMPLES_TO_TEST]):
        print(f"\n{YELLOW}--- Question {i+1}/{NUM_SAMPLES_TO_TEST} ---{RESET}")
        
        question = item['question']
        ground_truth = item['answer']

        all_sentences = [sentence for sublist in item['context']['sentences'] for sentence in sublist]
        context = " ".join(all_sentences)
        
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")

        # --- Local Model Alone Inference ---
        print(f"\n{BLUE}--- Local Model Alone ---{RESET}")
        try:
            local_response_alone = get_answer_from_local_model_alone(
                local_model, local_tokenizer, context, question
            )
            local_answer_alone = extract_final_answer_from_cot(local_response_alone)
            
            is_correct_alone = False
            if judge_client:
                is_correct_alone = llm_judge_answer(judge_client, question, ground_truth, local_answer_alone)
                if is_correct_alone:
                    local_alone_correct += 1
            
            print(f"Local Answer: {local_answer_alone}")
            print(f"Result (LLM Judge): {'Correct' if is_correct_alone else 'Incorrect'}")

        except Exception as e:
            print(f"{RED}Error during local model (alone) inference: {e}{RESET}")


        # --- CoT-Aided Local Model Inference ---
        print(f"\n{BLUE}--- Local Model with CoT-Aiding ---{RESET}")
        if remote_client and judge_client:
            print(f"{GREEN}Generating CoT with Remote LLM ({REMOTE_COT_MODEL})...{RESET}")
            cot = get_cot_from_remote_llm(remote_client, REMOTE_COT_MODEL, question)
            print(f"{CYAN}Generated Chain-of-Thought:{RESET}\n{cot}\n")
            print(f"{BLUE}Running Local Model with CoT...{RESET}")
            
            try:
                local_response_cot_aided = get_answer_from_local_model_with_cot(
                    local_model, local_tokenizer, context, question, cot
                )
                local_cot_aided_answer = extract_final_answer_from_cot(local_response_cot_aided)
                
                is_correct_cot = llm_judge_answer(judge_client, question, ground_truth, local_cot_aided_answer)
                if is_correct_cot:
                    local_cot_aided_correct += 1
                
                print(f"Local Answer (CoT-Aided): {local_cot_aided_answer}")
                print(f"Result (LLM Judge): {'Correct' if is_correct_cot else 'Incorrect'}")

            except Exception as e:
                print(f"{RED}Error during local model (CoT-aided) inference: {e}{RESET}")
        else:
            print(f"{YELLOW}Skipping CoT-aided local model inference due to missing API key.{RESET}")

    # 4. Final Results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    local_alone_accuracy = (local_alone_correct / NUM_SAMPLES_TO_TEST) * 100
    print(f"Local Model ({LOCAL_MODEL_NAME}) Alone Accuracy: {local_alone_accuracy:.2f}% (LLM Judge)")

    if remote_client and judge_client:
        local_cot_accuracy = (local_cot_aided_correct / NUM_SAMPLES_TO_TEST) * 100
        print(f"Local Model ({LOCAL_MODEL_NAME}) with CoT-Aiding Accuracy: {local_cot_accuracy:.2f}% (LLM Judge)")
        print(f"Performance Gain with CoT-Aiding: {local_cot_accuracy - local_alone_accuracy:.2f}%")
    else:
        print("CoT-aided experiment was skipped due to missing API key.")

if __name__ == '__main__':
    run_experiment()