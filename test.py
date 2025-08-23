import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # AutoTokenizer and AutoModelForCausalLM are technically not used for local inference anymore, but kept for potential future local model use cases.
import openai
from dotenv import load_dotenv
from difflib import SequenceMatcher # Needed for fuzzy_match, if you re-introduce it

# --- Imports for Differential Privacy ---
# Assuming these modules are available in your environment
from inferdpt import * # Contains perturb_sentence, if used.
from dp_sanitizer import get_embedding, load_sentence_bert, compute_similarity, differentially_private_replacement
from openai import OpenAI # For Nebius client

# --- 1. Configuration ---
# Load environment variables from the .env file
load_dotenv()

# Set the model names for local and remote LLMs
# LOCAL_MODEL_NAME now refers to the model used via Nebius for "local" tasks
# LOCAL_MODEL_NAME = "microsoft/phi-4" 
# LOCAL_MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct" 
# google/gemma-2-2b-it
LOCAL_MODEL_NAME = "google/gemma-2-2b-it"
# LOCAL_MODEL_NAME = "Qwen/Qwen3-4B-fast"

local_models = ["microsoft/phi-4", "google/gemma-2-9b-it-fast", 
    "google/gemma-2-2b-it", "Qwen/Qwen2.5-Coder-7B", 
    "mistralai/Mistral-Nemo-Instruct-2407", "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-4B-fast", "Qwen/Qwen3-14B"
]


# model_name = 'models/gemini-2.5-pro-preview-05-06'


REMOTE_LLM_PROVIDER = "deepseek" 

# Remote model for direct answer generation (for purely remote model scenario)
REMOTE_LLM_MODEL = "deepseek-chat" 
# Remote model for generating the Chain-of-Thought (CoT)
REMOTE_COT_MODEL = "deepseek-chat" 
# Remote model for the LLM Judge
REMOTE_JUDGE_MODEL = "deepseek-chat" 

# Set the dataset and split
DATASET_NAME = "hotpotqa/hotpot_qa" 
DATASET_SPLIT = "validation" 

# train = 90447, validation = 7045
NUM_SAMPLES_TO_TEST = 100 # Reduced for quicker testing, adjust as needed

# Differential Privacy Epsilon
EPSILON = 1.0 

# Get API keys from environment variables
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY")
HUGGING_FACE_API = os.getenv("HUGGING_FACE")
NEBIUS_API = os.getenv("NEBIUS") 

# ANSI color codes for better console output
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Initialize Nebius client once for DP perturbation and all "local" model calls
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=NEBIUS_API,
)


# --- 2. Helper Functions ---
def extract_final_answer_from_cot(text):
    """
    Extracts the final, concise answer from a Chain-of-Thought (CoT) response.
    """
    final_answer_marker = "Final Answer:"
    if final_answer_marker in text:
        return text.split(final_answer_marker)[-1].strip()
    sentences = text.split('.')
    # This logic assumes the final answer is in the second to last sentence if no marker
    # This might be brittle; a more robust approach might involve a regex or LLM extraction.
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
# This function is now simplified as LOCAL_MODEL_NAME is accessed via Nebius API.
def load_local_model(model_name):
    """
    Placeholder function. In this setup, the "local" model is accessed via Nebius API,
    not loaded locally from Hugging Face.
    """
    print(f"{CYAN}The model '{model_name}' will be accessed via Nebius API for local operations.{RESET}")
    # Return dummy values as actual model/tokenizer objects are not loaded locally
    return None, None 

def get_remote_llm_client(provider):
    """Get the appropriate client for the remote LLM API."""
    if provider == "openai":
        if not OPEN_AI_KEY:
            raise ValueError("OPEN_AI_KEY not found. Please set it in your .env file.")
        return openai.OpenAI(api_key=OPEN_AI_KEY)
    elif provider == "deepseek": 
        if not DEEP_SEEK_KEY:
            raise ValueError("DEEP_SEEK_KEY not found. Please set it in your .env file.")
        return openai.OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unsupported remote LLM provider: {provider}")

# --- DP Perturbation Functions ---
def generate_sentence_replacements_with_nebius(nebius_client, nebius_model_name, input_sentence, num_return_sequences=100, max_tokens=150):
    """
    Generates diverse, generalized, and anonymized paraphrases of an input sentence
    using a Nebius LLM via the Nebius API.
    """
    system_prompt = """
    You are a highly skilled AI assistant specializing in natural language processing.
    Your primary goal is to rephrase user questions in a generalized manner,
    preserving their core logical structure and intent, but explicitly removing
    or generalizing any sensitive personal identifying information (PII) or
    unique details that could lead to re-identification.

    Focus on transforming specific entities (like names, exact locations, unique events,
    or precise dates) into generic descriptions or placeholders.
    The output should be a semantically diverse and coherent paraphrase.
    """

    user_prompt = f"""
    Generalize and paraphrase the following sentence.
    Ensure the logical structure and core meaning are maintained, but remove or generalize
    any sensitive personal information (like names of people, specific unique places,
    exact dates, or highly specific event details).

    ### Output Rules:
    - Output ONLY the paraphrased and generalized sentence.
    - Do NOT repeat the original sentence or use near-identical phrasing.
    - No numbering, bullet points, or commentary.
    - Do NOT include introductory or explanatory text.
    - Ensure the sentence is grammatically correct and semantically coherent.

    ### Generalization and Anonymization Examples:
    Original: "Was John Smith, born on October 26, 1970, in London, the first CEO of ExampleCorp?"
    Generalized: "Was an individual, born in a specific city, the first CEO of a certain corporation?"

    Original: "Did Sarah visit the Eiffel Tower on her trip to Paris last summer?"
    Generalized: "Did a person visit a famous landmark during their trip to a major European city recently?"

    Original: "Were Scott Derrickson and Ed Wood of the same nationality?"
    Generalized: "Did the two filmmakers share the same nationality?"

    ### Task:
    Original: {input_sentence}
    Paraphrase:
    """
    
    try:
        response = nebius_client.chat.completions.create(
            model=nebius_model_name, # Use the specified Nebius model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.98,
            n=num_return_sequences
        )

        paraphrases = set()
        for choice in response.choices:
            clean_line = choice.message.content.strip()
            if clean_line and clean_line.lower() != input_sentence.lower():
                # Add additional filtering if the Nebius model tends to repeat the prompt or add boilerplate
                if not (clean_line.startswith("Generalize and paraphrase") or "Output ONLY the paraphrased" in clean_line):
                    paraphrases.add(clean_line)

        return list(paraphrases)

    except Exception as e:
        print(f"{RED}Error with Nebius API for paraphrase generation: {e}{RESET}")
        return []

def phrase_DP_perturbation(nebius_client, nebius_model_name, cnn_dm_prompt, epsilon, sbert_model):
    """
    Applies differential privacy perturbation to the question using the user's provided logic.
    This function now uses the Nebius API for paraphrase generation.
    """

    print(f"{GREEN}Applying differential privacy perturbation with epsilon={epsilon}...{RESET}")

    # Step 1: Generate candidate sentence-level replacements using the Nebius model
    candidate_sentences = generate_sentence_replacements_with_nebius( 
        nebius_client, # Pass nebius_client
        nebius_model_name, # Pass nebius_model_name
        input_sentence=cnn_dm_prompt,
        num_return_sequences=10, 
    )

    # If no candidate sentences are generated, raise an error to prevent the next one
    if not candidate_sentences:
        raise ValueError("No candidate sentences were generated. Check the Nebius API call for paraphrase generation.")

    # Step 2: Precompute embeddings
    candidate_embeddings = {
        sent: get_embedding(sbert_model, sent).cpu().numpy()
        for sent in candidate_sentences
    }

    # Step 3: Select a replacement using exponential mechanism
    dp_replacement = differentially_private_replacement(
        target_phrase=cnn_dm_prompt,
        epsilon=epsilon,
        candidate_phrases=candidate_sentences,
        candidate_embeddings=candidate_embeddings,
        sbert_model=sbert_model
    )

    print("DP replacement selected:", dp_replacement)

    return dp_replacement

# --- 4. Main Experiment Logic (Updated) ---
def get_cot_from_remote_llm(client, model_name, question): # Context parameter removed
    """
    Generates a Chain-of-Thought (CoT) using a powerful remote LLM,
    based ONLY on the question (no context).
    """
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
            messages=[
                {"role": "user", "content": cot_prompt}
            ],
            max_tokens=256,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error with CoT LLM API ({model_name}): {e}{RESET}")
        return "API Error"

def get_answer_from_local_model_with_non_private_cot(client, model_name, context, original_question, cot):
    """
    Generates a response from the "local" model (accessed via Nebius), guided by a CoT from the original question.
    Scenario 2: Non-private Local Model + CoT.
    """
    prompt_template = (
        "Context: {context}\n"
        "Question: {original_question}.\n"
        "You have been provided with a Chain-of-Thought to help you solve this question. "
        "Follow these steps to derive your final answer.\n\n"
        "Chain-of-Thought: {cot}\n\n"
        "Based on the context and the Chain-of-Thought, please provide your final, concise answer.\n"
        "Final Answer:"
    )
    prompt = prompt_template.format(
        context=context,
        original_question=original_question,
        cot=cot
    )

    try:
        response = client.chat.completions.create(
            model=model_name, # Use the Nebius model, which is LOCAL_MODEL_NAME
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error with Nebius API for non-private CoT-aided response: {e}{RESET}")
        return "API Error"


def get_answer_from_nebius_with_cot_and_dp(client, context, original_question, perturbed_question, cot):
    """
    Generates a response from the Nebius model, guided by a provided CoT from a perturbed question.
    This function is specifically for Scenario 3's final answer generation.
    """
    prompt_template = (
        "Context: {context}\n"
        "This is the original question: {original_question}.\n"
        "This is the perturbed question: {perturbed_question}.\n"
        "This is the chain of thought given to solve the perturbed_question. Apply the same logic to answer the original question please.\n"
        "Final Answer:"
    )
    prompt = prompt_template.format(
        context=context,
        original_question=original_question,
        perturbed_question=perturbed_question,
        cot=cot
    )

    try:
        response = client.chat.completions.create(
            model=LOCAL_MODEL_NAME, # Use the Nebius model, which is LOCAL_MODEL_NAME
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error with Nebius API: {e}{RESET}")
        return "API Error"


def get_answer_from_local_model_alone(client, model_name, context, question):
    """
    Generates a response from the "local" model (accessed via Nebius) without any CoT.
    """
    prompt_template = (
        "Context: {context}\n"
        "Question: {question}\n"
        "Please provide your final, concise answer.\n"
        "Answer:"
    )
    prompt = prompt_template.format(context=context, question=question)

    try:
        response = client.chat.completions.create(
            model=model_name, # Use the Nebius model, which is LOCAL_MODEL_NAME
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"{RED}Error with Nebius API for local alone response: {e}{RESET}")
        return "API Error"

def get_answer_from_purely_remote_llm(client, model_name, context, question):
    """
    Generates a response from a powerful remote LLM with full context access.
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
        print(f"{RED}Error with purely remote LLM API: {e}{RESET}")
        return "API Error"


def run_experiment():
    """
    Main function to run the multi-hop reasoning experiment comparing various LLM setups.
    """
    print(f"{CYAN}Loading dataset: {DATASET_NAME}...{RESET}")
    dataset = load_dataset(DATASET_NAME, "distractor", split=DATASET_SPLIT)
    
    multi_hop_questions = [
        q for q in dataset if len(q['supporting_facts']) > 1
    ]

    # Load local model (now just a placeholder for the Nebius-accessed model)
    # The actual Nebius client is initialized globally as 'nebius_client'
    _ , _ = load_local_model(LOCAL_MODEL_NAME) # Keeping the call for consistency in print output

    try:
        remote_client = get_remote_llm_client(REMOTE_LLM_PROVIDER)
        judge_client = get_remote_llm_client(REMOTE_LLM_PROVIDER)
    except ValueError as e:
        print(f"{RED}{e}{RESET}")
        print(f"{YELLOW}Skipping remote LLM interactions due to missing API key.{RESET}")
        remote_client = None
        judge_client = None

    # Counters for each scenario
    local_alone_correct = 0
    non_private_local_cot_correct = 0
    private_local_cot_correct = 0
    purely_remote_correct = 0

    # Initialize SBERT model for DP perturbation
    sbert_model = load_sentence_bert()

    for i, item in enumerate(multi_hop_questions[:NUM_SAMPLES_TO_TEST]):
        print(f"\n{YELLOW}--- Question {i+1}/{NUM_SAMPLES_TO_TEST} ---{RESET}")
        
        original_question = item['question'] 
        ground_truth = item['answer']

        all_sentences = [sentence for sublist in item['context']['sentences'] for sentence in sublist]
        context = " ".join(all_sentences)
        
        print(f"Original Question: {original_question}")
        print(f"Ground Truth: {ground_truth}")

        # --- Scenario 1: Purely Local Model (Baseline) ---
        # print(f"\n{BLUE}--- Scenario 1: Purely Local Model (Baseline) ---{RESET}")
        # try:
        #     # Call Nebius client for "local" model operations
        #     local_response_alone = get_answer_from_local_model_alone(
        #         nebius_client, LOCAL_MODEL_NAME, context, original_question
        #     )
        #     local_answer_alone = extract_final_answer_from_cot(local_response_alone)
            
        #     is_correct_alone = False
        #     if judge_client:
        #         is_correct_alone = llm_judge_answer(judge_client, original_question, ground_truth, local_answer_alone)
        #         if is_correct_alone:
        #             local_alone_correct += 1
            
        #     print(f"Local Answer (Alone): {local_answer_alone}")
        #     print(f"Result (LLM Judge): {'Correct' if is_correct_alone else 'Incorrect'}")

        # except Exception as e:
        #     print(f"{RED}Error during purely local model inference: {e}{RESET}")


        # --- Scenario 2: Non-Private Local Model + CoT ---
        print(f"\n{BLUE}--- Scenario 2: Non-Private Local Model + CoT ---{RESET}")
        if remote_client and judge_client:
            try:
                # Generate CoT from the ORIGINAL Question with Remote LLM (NO CONTEXT)
                print(f"{GREEN}2a. Generating CoT from ORIGINAL Question with REMOTE LLM ({REMOTE_COT_MODEL}) (Context NOT sent)...{RESET}")
                cot_non_private = get_cot_from_remote_llm(remote_client, REMOTE_COT_MODEL, original_question)
                print(f"{CYAN}Generated Chain-of-Thought (Remote, Non-Private):{RESET}\n{cot_non_private}\n")
                
                # Local Model answers using original question, context, and remote CoT (no perturbed question involved)
                print(f"{BLUE}2b. Running Local Model with Non-Private CoT...{RESET}")
                non_private_local_response = get_answer_from_local_model_with_non_private_cot( 
                    nebius_client, LOCAL_MODEL_NAME, context, original_question, cot_non_private 
                )
                non_private_local_answer = extract_final_answer_from_cot(non_private_local_response)
                
                is_correct_non_private = llm_judge_answer(judge_client, original_question, ground_truth, non_private_local_answer)
                if is_correct_non_private:
                    non_private_local_cot_correct += 1
                
                print(f"Local Answer (Non-Private CoT-Aided): {non_private_local_answer}")
                print(f"Result (LLM Judge): {'Correct' if is_correct_non_private else 'Incorrect'}")

            except Exception as e:
                print(f"{RED}Error during non-private CoT-aided inference: {e}{RESET}") 
        else:
            print(f"{YELLOW}Skipping non-private CoT-aided local model inference due to missing API key.{RESET}")



        # # --- Scenario 3: Private Local Model + CoT (phrase DP + remote CoT) ---
        # print(f"\n{BLUE}--- Scenario 3: Private Local Model + CoT ---{RESET}")
        # if remote_client and judge_client:
        #     try:
        #         # 3a. Apply Differential Privacy to the original question
        #         print(f"{GREEN}3a. Applying Differential Privacy to the question...{RESET}")
        #         # Pass nebius_client and LOCAL_MODEL_NAME (now used for DP via Nebius) to phrase_DP_perturbation
        #         perturbed_question = phrase_DP_perturbation(nebius_client, LOCAL_MODEL_NAME, original_question, EPSILON, sbert_model)
        #         print(f"Perturbed Question: {perturbed_question}")

        #         # 3b. Generate CoT from the PERTURBED Question with Remote LLM (NO CONTEXT)
        #         print(f"{GREEN}3b. Generating CoT from Perturbed Question with REMOTE LLM ({REMOTE_COT_MODEL}) (Context NOT sent)...{RESET}")
        #         cot_private = get_cot_from_remote_llm(remote_client, REMOTE_COT_MODEL, perturbed_question)
        #         print(f"{CYAN}Generated Chain-of-Thought (Remote, Private):{RESET}\n{cot_private}\n")
                
        #         # 3c. Local Model answers using original question, perturbed question, context, and remote CoT
        #         print(f"{BLUE}3c. Running Local Model with Private CoT...{RESET}")
        #         # Use get_answer_from_nebius_with_cot_and_dp as requested
        #         private_local_response = get_answer_from_nebius_with_cot_and_dp( 
        #             nebius_client, context, original_question, perturbed_question, cot_private 
        #         )
        #         private_local_answer = extract_final_answer_from_cot(private_local_response)
                
        #         is_correct_private = llm_judge_answer(judge_client, original_question, ground_truth, private_local_answer)
        #         if is_correct_private:
        #             private_local_cot_correct += 1
                
        #         print(f"Local Answer (Private CoT-Aided): {private_local_answer}")
        #         print(f"Result (LLM Judge): {'Correct' if is_correct_private else 'Incorrect'}")

        #     except Exception as e:
        #         print(f"{RED}Error during private CoT-aided inference: {e}{RESET}")
        # else:
        #     print(f"{YELLOW}Skipping private CoT-aided local model inference due to missing API key.{RESET}")


        # --- Scenario 4: Purely Remote Model ---
        # print(f"\n{BLUE}--- Scenario 4: Purely Remote Model ---{RESET}")
        # if remote_client and judge_client:
        #     try:
        #         # Remote LLM answers using full context and original question
        #         print(f"{GREEN}4a. Running Purely Remote LLM ({REMOTE_LLM_MODEL}) with full context...{RESET}")
        #         purely_remote_response = get_answer_from_purely_remote_llm(
        #             remote_client, REMOTE_LLM_MODEL, context, original_question
        #         )
        #         purely_remote_answer = extract_final_answer_from_cot(purely_remote_response) # Reusing extract_final_answer_from_cot
                
        #         is_correct_purely_remote = llm_judge_answer(judge_client, original_question, ground_truth, purely_remote_answer)
        #         if is_correct_purely_remote:
        #             purely_remote_correct += 1
                
        #         print(f"Purely Remote Answer: {purely_remote_answer}")
        #         print(f"Result (LLM Judge): {'Correct' if is_correct_purely_remote else 'Incorrect'}")

        #     except Exception as e:
        #         print(f"{RED}Error during purely remote model inference: {e}{RESET}")
        # else:
        #     print(f"{YELLOW}Skipping purely remote model inference due to missing API key.{RESET}")


    # 4. Final Results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    # Calculate and print accuracy for each scenario
    if NUM_SAMPLES_TO_TEST > 0:
        local_alone_accuracy = (local_alone_correct / NUM_SAMPLES_TO_TEST) * 100
        print(f"1. Purely Local Model ({LOCAL_MODEL_NAME}) Accuracy: {local_alone_accuracy:.2f}% (LLM Judge)")

        if remote_client and judge_client:
            non_private_local_cot_accuracy = (non_private_local_cot_correct / NUM_SAMPLES_TO_TEST) * 100
            print(f"2. Non-Private Local Model ({LOCAL_MODEL_NAME}) + CoT Accuracy: {non_private_local_cot_accuracy:.2f}% (LLM Judge)")
            
            private_local_cot_accuracy = (private_local_cot_correct / NUM_SAMPLES_TO_TEST) * 100
            print(f"3. Private Local Model ({LOCAL_MODEL_NAME}) + CoT Accuracy: {private_local_cot_accuracy:.2f}% (LLM Judge)")
            
            purely_remote_accuracy = (purely_remote_correct / NUM_SAMPLES_TO_TEST) * 100
            print(f"4. Purely Remote Model ({REMOTE_LLM_MODEL}) Accuracy: {purely_remote_accuracy:.2f}% (LLM Judge)")
            
            print("\n--- Performance Comparisons ---")
            print(f"Performance Gain (Non-Private CoT-Aiding vs. Local Alone): {non_private_local_cot_accuracy - local_alone_accuracy:.2f}%")
            print(f"Performance Change (Private CoT-Aiding vs. Non-Private CoT-Aiding): {private_local_cot_accuracy - non_private_local_cot_accuracy:.2f}%")
            print(f"Performance Gap (Purely Remote vs. Private CoT-Aiding): {purely_remote_accuracy - private_local_cot_accuracy:.2f}%")
        else:
            print("Remote LLM-dependent scenarios were skipped due to missing API key.")
    else:
        print("No samples were tested.")

if __name__ == '__main__':
    run_experiment()
