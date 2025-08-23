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
LOCAL_MODEL_NAME = "microsoft/phi-4" # The local model that will produce the final answer
# why cannot i use phi-2

# REMOTE_LLM_PROVIDER = "openai" 
REMOTE_LLM_PROVIDER = "deepseek" 

# Remote model for generating the Chain-of-Thought (CoT) from the perturbed question
# REMOTE_COT_MODEL = "gpt-4o-mini"
REMOTE_COT_MODEL = "deepseek-chat" 

# Remote model for the LLM Judge
# REMOTE_JUDGE_MODEL = "gpt-4o-mini"
REMOTE_JUDGE_MODEL = "deepseek-chat" 

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

# Initialize Nebius client once
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=NEBIUS_API,
)


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
# This function is now removed as per user request to not load local models.
# def load_local_model(model_name):
#     ...

# remote LLM does not involve nebius 
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
    max_tokens=150):
    """
    Generates diverse, generalized, and anonymized paraphrases of an input sentence
    using a local Nebius LLM.

    The prompt instructs the LLM to:
    1. Preserve the logical structure of the sentence.
    2. Exclude sensitive information (like PII) by generalizing it.
    3. Generate diverse semantic interpretations.

    Args:
        local_model (str): The name or identifier of the local Nebius LLM to use.
        input_sentence (str): The original sentence to paraphrase and anonymize.
        num_return_sequences (int): The number of diverse paraphrases to attempt to generate.
                                    The actual number returned might be less due to filtering.
        max_tokens (int): The maximum number of tokens for each generated paraphrase.

    Returns:
        list: A list of unique, generalized, and diverse paraphrased sentences.
              Returns an empty list if an error occurs or no valid paraphrases are generated.
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
        # Assuming nebius_client is globally available or passed in the context
        # For demonstration, a placeholder for the client call:
        # In a real scenario, you'd use your actual Nebius client.
        # Example: response = nebius_client.chat.completions.create(...)

        # Placeholder for actual Nebius API call
        # Mock response for testing without a live Nebius client:
        # Replace this with your actual nebius_client.chat.completions.create call
        # once you have it configured.
        class MockChoice:
            def __init__(self, content):
                self.message = type('obj', (object,), {'content': content})()

        class MockResponse:
            def __init__(self, contents):
                self.choices = [MockChoice(c) for c in contents]

        # Example mock responses for testing the prompt logic:
        mock_responses = [
            "Did the two individuals share a common country of origin?",
            "Were the two directors from the same nation?",
            "What was the nationality shared by the two artists?",
            "Is the country of birth identical for these two people?",
            "Did the pair have the same citizenship?",
            input_sentence, # Include original to test filtering
            "A swift fox vaults over a resting hound." # Irrelevant to test filtering
        ]
        response = MockResponse(mock_responses * (num_return_sequences // len(mock_responses) + 1))


        paraphrases = set()
        for choice in response.choices:
            clean_line = choice.message.content.strip()
            # Filter out the original sentence and empty/near-empty strings
            # and irrelevant paraphrases (if any from mock)
            if clean_line and \
               clean_line.lower() != input_sentence.lower() and \
               "swift fox" not in clean_line.lower(): # Specific filter for mock example
                paraphrases.add(clean_line)

        # You can keep the basic semantic diversity filter if you want to further
        # reduce redundancy beyond what the LLM's 'n' parameter might provide,
        # but the prompt already encourages diversity.
        # For now, returning unique paraphrases.
        return list(paraphrases)

    except Exception as e:
        print(f"{RED}Error with Nebius API: {e}{RESET}")
        return []


# def generate_sentence_replacements_with_nebius(
#     local_model,
#     input_sentence,
#     num_return_sequences=100,
#     max_tokens=150):
#     system_prompt = """
#     You are a creative assistant specializing in generating paraphrases with diverse semantic interpretations. Your goal is to rephrase the input sentence in varied ways, altering structure, word choice, and meaning while preserving the core intent. Avoid literal rephrasing; instead, explore different perspectives, contexts, or expressions.
#     """

#     user_prompt = f"""
#     Your task is to generate a diverse paraphrase of the sentence below, emphasizing varied semantic meanings.

#     ### Output Rules:
#     - Output ONLY the paraphrased sentence.
#     - Do NOT repeat the original sentence or use near-identical phrasing.
#     - No numbering, bullet points, or commentary.
#     - Do NOT include introductory or explanatory text.
#     - Ensure the sentence is grammatically correct and semantically coherent.

#     ### Diversity Requirements:
#     - Vary sentence structure, vocabulary, and perspective.
#     - Explore alternative contexts or interpretations of the original meaning.
#     - Avoid minor word substitutions; aim for creative re-expressions.

#     ### Example:
#     Original: The quick brown fox jumps over the lazy dog.
#     Paraphrases:
#     - A swift fox vaults over a resting hound.
#     - The energetic brown fox hops past the idle dog.
#     - A nimble fox clears the lounging canine in a bound.

#     ### Task:
#     Original: {input_sentence}
#     Paraphrase:
#     """

#     try:
#         response = nebius_client.chat.completions.create(
#             model=local_model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             # max_tokens=max_tokens,
#             # temperature=0.7,
#             # top_p=0.98,
#             n=num_return_sequences
#         )

#         paraphrases = set()
#         for choice in response.choices:
#             clean_line = choice.message.content.strip()
#             if clean_line and clean_line.lower() != input_sentence.lower():
#                 paraphrases.add(clean_line)

#         return [i for i in paraphrases]

#     except Exception as e:
#         print(f"{RED}Error with Nebius API: {e}{RESET}")
#         return []

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

    # If no candidate sentences are generated, raise an error to prevent the next one
    if not candidate_sentences:
        raise ValueError("No candidate sentences were generated. Check the Nebius API call and model name.")

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

    # exit(0)

    return dp_replacement

### only send question to remote LLM
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

def get_answer_from_nebius_with_cot_and_dp(client, context, original_question, perturbed_question, cot):
    """
    Generates a response from the Nebius model, guided by a provided CoT from a perturbed question.
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

    try:
        response = client.chat.completions.create(
            model=LOCAL_MODEL_NAME, # Use the Nebius model
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

def run_experiment(sbert_model):
    """
    Main function to run the privacy-preserving multi-hop reasoning experiment.
    """
    print(f"{CYAN}Loading dataset: {DATASET_NAME}...{RESET}")
    dataset = load_dataset(DATASET_NAME, "distractor", split=DATASET_SPLIT)
    
    multi_hop_questions = [
        q for q in dataset if len(q['supporting_facts']) > 1
    ]

    # No local model is loaded, we will use the Nebius client instead.
    # The `nebius_client` is initialized globally.
    
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

        # context = " ".join(item['context']['sentences'])
        all_sentences = [sentence for sublist in item['context']['sentences'] for sentence in sublist]
        context = " ".join(all_sentences)
                
        print(f"Original Question: {original_question}")
        # print("context: ", context)
        print(f"Ground Truth: {ground_truth}")

        # continue


        ### note that we should not send any context to remote 
        
        # --- Privacy-Preserving Workflow ---
        try:
            print(f"{GREEN}1. Applying Differential Privacy to the question...{RESET}")
            perturbed_question = phrase_DP_perturbation(LOCAL_MODEL_NAME, original_question, EPSILON, sbert_model)
            print(f"Perturbed Question: {perturbed_question}")

            print(f"{GREEN}2. Generating CoT from Perturbed Question with Remote LLM ({REMOTE_COT_MODEL})...{RESET}")

            # Note: We need to pass the context for the CoT to be useful, as the LLM doesn't have a knowledge base
            cot = get_cot_from_remote_llm(remote_client, REMOTE_COT_MODEL, context, perturbed_question)
            print(f"{CYAN}Generated Chain-of-Thought:{RESET}\n{cot}\n")

            # exit(0)
            print(f"{BLUE}3. Running Nebius LLM with Original Question, Perturbed Question, and CoT...{RESET}")
            
            # This call is now updated to use the Nebius client directly
            local_response = get_answer_from_nebius_with_cot_and_dp(
                nebius_client, context, original_question, perturbed_question, cot
            )
            final_answer = extract_final_answer(local_response)
            
            is_correct = llm_judge_answer(judge_client, original_question, ground_truth, final_answer)
            if is_correct:
                privacy_preserving_correct += 1
            
            print(f"Final Answer: {final_answer}")
            print(f"Result (LLM Judge): {'Correct' if is_correct else 'Incorrect'}")

        except Exception as e:
            print(f"{RED}Error during privacy-preserving workflow: {e}{RESET}")
            

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