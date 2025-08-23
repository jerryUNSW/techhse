import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import openai
from dotenv import load_dotenv
from rapidfuzz import fuzz # New import for fuzzy matching

# --- 1. Configuration ---
# Load environment variables from the .env file
load_dotenv()

# Set the model names for local and remote LLMs
# You can choose a different local model, but make sure it's manageable on your hardware.
LOCAL_MODEL_NAME = "microsoft/phi-2"
# Use a powerful remote model for the oracle.
REMOTE_LLM_PROVIDER = "openai" # You could also use 'anthropic' or 'google'

# REMOTE_LLM_MODEL = "gpt-4o-mini" # A powerful and cost-effective choice for this task
REMOTE_LLM_MODEL = "gpt-4o" # A powerful and cost-effective choice for this task

# Set the dataset and split
# DATASET_NAME = "hotpotqa"
DATASET_NAME = "hotpotqa/hotpot_qa" # <-- Corrected line

DATASET_SPLIT = "validation" # Or "train"
NUM_SAMPLES_TO_TEST = 50 # Keep this small for initial testing due to API costs and time
FUZZY_MATCH_THRESHOLD = 85 # New constant for fuzzy matching score threshold

# Get API keys from environment variables
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY") # You can add this as another remote option
HUGGING_FACE_API = os.getenv("HUGGING_FACE") # Required for some Hugging Face models/datasets

# ANSI color codes for better console output
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# --- 2. Helper Functions ---
def normalize_answer(ans):
    """Normalize the answer for robust comparison."""
    return ans.lower().strip()

def fuzzy_match(pred, gold, threshold=FUZZY_MATCH_THRESHOLD):
    """Check for fuzzy match between predicted and ground truth answers."""
    normalized_pred = normalize_answer(pred)
    normalized_gold = normalize_answer(gold)
    
    # Calculate the fuzzy ratio and compare to the threshold
    score = fuzz.ratio(normalized_pred, normalized_gold)
    return score >= threshold

def extract_final_answer_from_cot(text):
    """
    Extracts the final, concise answer from a Chain-of-Thought (CoT) response.
    This is a critical and potentially tricky part. The quality of this function
    will impact the measured accuracy.
    """
    # A simple approach: find the last occurrence of "Final Answer:" or "Answer:"
    # This can be brittle and may need to be refined for a real research paper.
    final_answer_marker = "Final Answer:"
    if final_answer_marker in text:
        return text.split(final_answer_marker)[-1].strip()

    # Another common pattern is a direct answer at the end of the text
    sentences = text.split('.')
    if len(sentences) > 1:
        # Take the last meaningful sentence
        return sentences[-2].strip()
    return text.strip()


# --- 3. Model Loading and Setup ---
def load_local_model(model_name):
    """Load a local Hugging Face model and tokenizer."""
    print(f"{CYAN}Loading local model: {model_name}...{RESET}")
    # Use 'device_map="auto"' to automatically handle model placement on GPU/CPU
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,  # Use float16 for reduced memory usage
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
    # Add other providers if needed
    # elif provider == "deepseek":
    #     if not DEEP_SEEK_KEY:
    #         raise ValueError("DEEP_SEEK_KEY not found. Please set it in your .env file.")
    #     return DeepSeekClient(api_key=DEEP_SEEK_KEY)
    else:
        raise ValueError(f"Unsupported remote LLM provider: {provider}")


# --- 4. Main Experiment Logic ---
def get_answer_from_local_model(model, tokenizer, context, question):
    """
    Generates a response from the local model using a Chain-of-Thought prompt.
    """
    # Craft a CoT prompt
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
            do_sample=False,  # Use greedy decoding for reproducibility
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
            temperature=0.0 # Use low temp for reproducibility
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"{RED}Error with remote LLM API: {e}{RESET}")
        return "API Error"


def run_experiment():
    """
    Main function to run the multi-hop reasoning experiment.
    """
    # # 1. Load Dataset
    # print(f"{CYAN}Loading dataset: {DATASET_NAME}...{RESET}")
    # dataset = load_dataset(DATASET_NAME, "distractor", split=DATASET_SPLIT)
    # multi_hop_questions = [
    #     q for q in dataset if q['type'] == 'multi-hop'
    # ]
    # print(f"Found {len(multi_hop_questions)} multi-hop questions.")
    # 1. Load Dataset
    print(f"{CYAN}Loading dataset: {DATASET_NAME}...{RESET}")
    # The 'distractor' configuration is implicitly part of the dataset, so you can remove the second argument
    # Or, to be explicit, you can specify it. 
    dataset = load_dataset(DATASET_NAME, "distractor", split=DATASET_SPLIT)
    
    # A question is considered multi-hop if it has more than one supporting fact.
    multi_hop_questions = [
        q for q in dataset if len(q['supporting_facts']) > 1
    ]
    # print(f"\nTotal questions in the '{DATASET_SPLIT}' split: {len(dataset)}")
    # print(f"Found {len(multi_hop_questions)} multi-hop questions.")

    
    # 2. Load Models
    local_model, local_tokenizer = load_local_model(LOCAL_MODEL_NAME)
    
    # The remote client will now check for the key from the loaded .env file
    try:
        remote_client = get_remote_llm_client(REMOTE_LLM_PROVIDER)
    except ValueError as e:
        print(f"{RED}{e}{RESET}")
        print(f"{YELLOW}Skipping remote LLM inference.{RESET}")
        remote_client = None

    local_correct = 0
    remote_correct = 0

    # 3. Iterate through a sample of questions
    for i, item in enumerate(multi_hop_questions[:NUM_SAMPLES_TO_TEST]):
        print(f"\n{YELLOW}--- Question {i+1}/{NUM_SAMPLES_TO_TEST} ---{RESET}")
        
        question = item['question']
        ground_truth = item['answer']
        # The context in HotpotQA is a list of paragraphs
        context = " ".join([p[1] for p in item['context']])
        
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
        # --- Local Model Inference ---
        print(f"{BLUE}Running Local Model...{RESET}")
        try:
            local_response_cot = get_answer_from_local_model(
                local_model, local_tokenizer, context, question
            )
            local_answer = extract_final_answer_from_cot(local_response_cot)
            is_correct_local = fuzzy_match(local_answer, ground_truth)
            if is_correct_local:
                local_correct += 1
            
            print(f"Local Answer: {local_answer}")
            print(f"Result (Fuzzy Match): {'Correct' if is_correct_local else 'Incorrect'}")

        except Exception as e:
            print(f"{RED}Error during local model inference: {e}{RESET}")
            
        # --- Remote LLM Inference ---
        if remote_client:
            print(f"{GREEN}Running Remote LLM...{RESET}")
            try:
                remote_response_cot = get_answer_from_remote_llm(
                    remote_client, REMOTE_LLM_MODEL, context, question
                )
                remote_answer = extract_final_answer_from_cot(remote_response_cot)
                is_correct_remote = fuzzy_match(remote_answer, ground_truth)
                if is_correct_remote:
                    remote_correct += 1
                
                print(f"Remote Answer: {remote_answer}")
                print(f"Result (Fuzzy Match): {'Correct' if is_correct_remote else 'Incorrect'}")

            except Exception as e:
                print(f"{RED}Error during remote LLM inference: {e}{RESET}")

    # 4. Final Results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    local_accuracy = (local_correct / NUM_SAMPLES_TO_TEST) * 100
    print(f"Local Model ({LOCAL_MODEL_NAME}) Accuracy: {local_accuracy:.2f}%")
    if remote_client:
        remote_accuracy = (remote_correct / NUM_SAMPLES_TO_TEST) * 100
        print(f"Remote LLM ({REMOTE_LLM_MODEL}) Accuracy: {remote_accuracy:.2f}%")
        print(f"Performance Gap: {remote_accuracy - local_accuracy:.2f}%")
    else:
        print("Remote LLM test was skipped due to missing API key.")

if __name__ == '__main__':
    run_experiment()