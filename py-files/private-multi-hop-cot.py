import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import openai
from dotenv import load_dotenv
from difflib import SequenceMatcher
import spacy

# The following imports are from the user's provided code block
from inferdpt import * # Assuming this module contains perturb_sentence
from dp_sanitizer import get_embedding, load_sentence_bert, compute_similarity, differentially_private_replacement
from openai import OpenAI

# --- 1. Configuration ---
# Load environment variables from the .env file
load_dotenv()

# Set the model names for local and remote LLMs
# LOCAL_MODEL_NAME = "microsoft/phi-2" # The local model that will produce the final answer
LOCAL_MODEL_NAME = "microsoft/phi-4" # The local model that will produce the final answer
REMOTE_LLM_PROVIDER = "openai" 

# Remote model for generating the Chain-of-Thought (CoT) from the perturbed question
REMOTE_COT_MODEL = "gpt-4o-mini"
# Remote model for the LLM Judge
REMOTE_JUDGE_MODEL = "gpt-4o-mini"

# Set the dataset and split
DATASET_NAME = "hotpotqa/hotpot_qa" 
DATASET_SPLIT = "validation" 
NUM_SAMPLES_TO_TEST = 10 

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


# --- 2. Helper Functions (Adapted from user's code) ---
def normalize_answer(ans):
    return ans.lower().strip()

def exact_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)

def fuzzy_match(pred, gold):
    return SequenceMatcher(None, normalize_answer(pred), normalize_answer(gold)).ratio()

def extract_final_answer(text):
    """
    Extracts the final, concise answer from a model's response.
    """
    final_answer_marker = "Final Answer:"
    if final_answer_marker in text:
        return text.split(final_answer_marker)[-1].strip()
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
        return openai.OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unsupported remote LLM provider: {provider}")

def generate_sentence_replacements_with_nebius(
    local_model,
    input_sentence,
    num_return_sequences=100,
    max_tokens=150
):
    system_prompt = """
    You are a creative assistant specializing in generating paraphrases with diverse semantic interpretations. Your goal is to rephrase the input sentence in varied ways, altering structure, word choice, and meaning while preserving the core intent. Avoid literal rephrasing; instead, explore different perspectives, contexts, or expressions.
    """

    user_prompt = f"""
    Your task is to generate a diverse paraphrase of the sentence below, emphasizing varied semantic meanings.

    ### Output Rules:
    - Output ONLY the paraphrased sentence.
    - Do NOT repeat the original sentence or use near-identical phrasing.
    - No numbering, bullet points, or commentary.
    - Do NOT include introductory or explanatory text.
    - Ensure the sentence is grammatically correct and semantically coherent.

    ### Diversity Requirements:
    - Vary sentence structure, vocabulary, and perspective.
    - Explore alternative contexts or interpretations of the original meaning.
    - Avoid minor word substitutions; aim for creative re-expressions.

    ### Example:
    Original: The quick brown fox jumps over the lazy dog.
    Paraphrases:
    - A swift fox vaults over a resting hound.
    - The energetic brown fox hops past the idle dog.
    - A nimble fox clears the lounging canine in a bound.

    ### Task:
    Original: {input_sentence}
    Paraphrase:
    """

    try:
        # Initialize Nebius client here
        nebius_client = OpenAI(
            base_url="https://api.studio.nebius.ai/v1/",
            api_key=NEBIUS_API,
        )

        response = nebius_client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=1.7,
            top_p=0.98,
            n=num_return_sequences
        )

        paraphrases = set()
        for choice in response.choices:
            clean_line = choice.message.content.strip()
            if clean_line and clean_line.lower() != input_sentence.lower():
                paraphrases.add(clean_line)

        # Basic semantic diversity filter: remove paraphrases too similar to each other
        filtered_paraphrases = []
        for paraphrase in paraphrases:
            if not any(
                len(set(paraphrase.lower().split()) & set(existing.lower().split())) > 0.7 * len(paraphrase.lower().split())
                for existing in filtered_paraphrases
            ):
                filtered_paraphrases.append(paraphrase)

        return filtered_paraphrases[:num_return_sequences]

    except Exception as e:
        print(f"Error with Nebius API: {e}")
        return []

def phrase_DP_perturbation(local_model_name_str, cnn_dm_prompt, epsilon, sbert_model):
    """
    Applies differential privacy perturbation to the question using the user's provided logic.
    """
    print(f"{GREEN}Applying differential privacy perturbation with epsilon={epsilon}...{RESET}")

    # Step 1: Generate candidate sentence-level replacements
    candidate_sentences = generate_sentence_replacements_with_nebius(
        local_model_name_str, 
        input_sentence=cnn_dm_prompt,
        num_return_sequences=10,
    )

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

def get_cot_from_remote_llm(client, model_name, context, question):
    """
    Generates a Chain-of-Thought (CoT) using a powerful remote LLM.
    """
    cot_prompt = (
        f"Context: {context}\n"
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

def get_answer_from_local_model_with_cot_and_dp(model, tokenizer, context, original_question, perturbed_question, cot):
    """
    Generates a response from the local model, guided by a provided CoT from a perturbed question.
    """
    prompt_template = (
        "Context: {context}\n"
        "This is the original question: {original_question}.\n"
        "This is a related, but perturbed question: {perturbed_question}.\n"
        "This is a chain of thought given to solve the perturbed question: {cot}\n"
        "Apply the same logic from the chain of thought to answer the original question.\n"
        "Final Answer:"
    )
    prompt = prompt_template.format(
        context=context,
        original_question=original_question,
        perturbed_question=perturbed_question,
        cot=cot
    )

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

def run_experiment(sbert_model):
    """
    Main function to run the privacy-preserving multi-hop reasoning experiment.
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
        print(f"{YELLOW}Skipping remote LLM CoT generation and judge evaluation due to missing API key.{RESET}")
        remote_client = None
        judge_client = None
        return

    privacy_preserving_correct = 0

    for i, item in enumerate(multi_hop_questions[:NUM_SAMPLES_TO_TEST]):
        print(f"\n{YELLOW}--- Question {i+1}/{NUM_SAMPLES_TO_TEST} ---{RESET}")
        
        original_question = item['question']
        ground_truth = item['answer']
        context = " ".join([p[1] for p in item['context']])
        
        print(f"Original Question: {original_question}")
        print(f"Ground Truth: {ground_truth}")
        
        # --- Privacy-Preserving Workflow ---
        # print(f"{GREEN}1. Applying Differential Privacy to the question...{RESET}")
        # perturbed_question = phrase_DP_perturbation(local_model, original_question, EPSILON, sbert_model)
        # print(f"Perturbed Question: {perturbed_question}")

        print(f"{GREEN}1. Applying Differential Privacy to the question...{RESET}")
        perturbed_question = phrase_DP_perturbation(LOCAL_MODEL_NAME, original_question, EPSILON, sbert_model)
        print(f"Perturbed Question: {perturbed_question}")


        print(f"{GREEN}2. Generating CoT from Perturbed Question with Remote LLM ({REMOTE_COT_MODEL})...{RESET}")
        # Note: We need to pass the context for the CoT to be useful, as the LLM doesn't have a knowledge base
        cot = get_cot_from_remote_llm(remote_client, REMOTE_COT_MODEL, context, perturbed_question)
        print(f"{CYAN}Generated Chain-of-Thought:{RESET}\n{cot}\n")

        print(f"{BLUE}3. Running Local Model with Original Question, Perturbed Question, and CoT...{RESET}")
        
        try:
            local_response = get_answer_from_local_model_with_cot_and_dp(
                local_model, local_tokenizer, context, original_question, perturbed_question, cot
            )
            final_answer = extract_final_answer(local_response)
            
            is_correct = llm_judge_answer(judge_client, original_question, ground_truth, final_answer)
            if is_correct:
                privacy_preserving_correct += 1
            
            print(f"Final Answer: {final_answer}")
            print(f"Result (LLM Judge): {'Correct' if is_correct else 'Incorrect'}")

        except Exception as e:
            print(f"{RED}Error during local model inference: {e}{RESET}")
            

    # 4. Final Results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    if NUM_SAMPLES_TO_TEST > 0:
        privacy_preserving_accuracy = (privacy_preserving_correct / NUM_SAMPLES_TO_TEST) * 100
        print(f"Local Model with Privacy-Preserving CoT Accuracy: {privacy_preserving_accuracy:.2f}% (LLM Judge)")
    else:
        print("No samples were tested.")

if __name__ == '__main__':
    # Initialize sentence-BERT and spaCy for the DP function
    sbert_model = load_sentence_bert()
    # nlp = spacy.load("en_core_web_sm")
    run_experiment(sbert_model)