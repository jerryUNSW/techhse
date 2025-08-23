import os
import yaml
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
# from inferdpt import phrase_DP_perturbation  # Placeholder for your DP module
from dp_sanitizer import load_sentence_bert
import utils

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

# Initialize Nebius client
nebius_client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=NEBIUS_API)

def get_cot_from_remote_llm(client, model_name, question):
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
            messages=[{"role": "user", "content": cot_prompt}],
            max_tokens=256,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\033[91mError with CoT LLM API ({model_name}): {e}\033[0m")
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
    prompt = prompt_template.format(context=context, original_question=original_question, cot=cot)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\033[91mError with Nebius API for non-private CoT-aided response: {e}\033[0m")
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
    prompt = prompt_template.format(context=context, original_question=original_question, perturbed_question=perturbed_question, cot=cot)

    try:
        response = client.chat.completions.create(
            model=config["local_model"],  # Use the default local model from config
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\033[91mError with Nebius API: {e}\033[0m")
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
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\033[91mError with Nebius API for local alone response: {e}\033[0m")
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
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"\033[91mError with purely remote LLM API: {e}\033[0m")
        return "API Error"

def run_experiment_for_model(model_name):
    """
    Run the multi-hop reasoning experiment for a given local model.
    """
    print(f"\033[96mLoading dataset: {config['dataset']['name']}...\033[0m")
    dataset = load_dataset(config["dataset"]["name"], "distractor", split=config["dataset"]["split"])
    
    multi_hop_questions = [q for q in dataset if len(q["supporting_facts"]) > 1]
    multi_hop_questions = multi_hop_questions[:config["dataset"]["num_samples"]]

    try:
        remote_client = utils.get_remote_llm_client(config["remote_llm_provider"])
        judge_client = utils.get_remote_llm_client(config["remote_llm_provider"])
    except ValueError as e:
        print(f"\033[91m{e}\033[0m")
        print(f"\033[93mSkipping remote LLM interactions due to missing API key.\033[0m")
        remote_client = None
        judge_client = None

    local_alone_correct = 0
    non_private_local_cot_correct = 0
    private_local_cot_correct = 0
    purely_remote_correct = 0

    sbert_model = load_sentence_bert()

    for i, item in enumerate(multi_hop_questions):
        print(f"\n\033[93m--- Question {i+1}/{config['dataset']['num_samples']} ---\033[0m")
        
        original_question = item["question"]
        ground_truth = item["answer"]
        all_sentences = [sentence for sublist in item["context"]["sentences"] for sentence in sublist]
        context = " ".join(all_sentences)
        
        print(f"Original Question: {original_question}")
        print(f"Ground Truth: {ground_truth}")

        # --- Scenario 1: Purely Local Model (Baseline) ---
        print(f"\n\033[94m--- Scenario 1: Purely Local Model (Baseline) ---\033[0m")
        try:
            local_response_alone = get_answer_from_local_model_alone(
                nebius_client, model_name, context, original_question
            )
            local_answer_alone = utils.extract_final_answer_from_cot(local_response_alone)
            
            is_correct_alone = False
            if judge_client:
                is_correct_alone = utils.llm_judge_answer(judge_client, original_question, ground_truth, local_answer_alone)
                if is_correct_alone:
                    local_alone_correct += 1
            
            print(f"Local Answer (Alone): {local_answer_alone}")
            print(f"Result (LLM Judge): {'Correct' if is_correct_alone else 'Incorrect'}")

        except Exception as e:
            print(f"\033[91mError during purely local model inference: {e}\033[0m")

        # --- Scenario 2: Non-Private Local Model + CoT ---
        print(f"\n\033[94m--- Scenario 2: Non-Private Local Model + CoT ---\033[0m")
        if remote_client and judge_client:
            try:
                print(f"\033[92m2a. Generating CoT from ORIGINAL Question with REMOTE LLM ({config['remote_models']['cot_model']}) (Context NOT sent)...\033[0m")
                cot_non_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], original_question)
                print(f"\033[96mGenerated Chain-of-Thought (Remote, Non-Private):\033[0m\n{cot_non_private}\n")
                
                print(f"\033[94m2b. Running Local Model with Non-Private CoT...\033[0m")
                non_private_local_response = get_answer_from_local_model_with_non_private_cot(
                    nebius_client, model_name, context, original_question, cot_non_private
                )
                non_private_local_answer = utils.extract_final_answer_from_cot(non_private_local_response)
                
                is_correct_non_private = utils.llm_judge_answer(judge_client, original_question, ground_truth, non_private_local_answer)
                if is_correct_non_private:
                    non_private_local_cot_correct += 1
                
                print(f"Local Answer (Non-Private CoT-Aided): {non_private_local_answer}")
                print(f"Result (LLM Judge): {'Correct' if is_correct_non_private else 'Incorrect'}")

            except Exception as e:
                print(f"\033[91mError during non-private CoT-aided inference: {e}\033[0m")
        else:
            print(f"\033[93mSkipping non-private CoT-aided local model inference due to missing API key.\033[0m")

        # --- Scenario 3: Private Local Model + CoT (phrase DP + remote CoT) ---
        print(f"\n\033[94m--- Scenario 3: Private Local Model + CoT ---\033[0m")
        if remote_client and judge_client:
            try:
                print(f"\033[92m3a. Applying Differential Privacy to the question...\033[0m")
                perturbed_question = utils.phrase_DP_perturbation(nebius_client, model_name, original_question, config["epsilon"], sbert_model)
                print(f"Perturbed Question: {perturbed_question}")

                print(f"\033[92m3b. Generating CoT from Perturbed Question with REMOTE LLM ({config['remote_models']['cot_model']}) (Context NOT sent)...\033[0m")
                cot_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], perturbed_question)
                print(f"\033[96mGenerated Chain-of-Thought (Remote, Private):\033[0m\n{cot_private}\n")
                
                print(f"\033[94m3c. Running Local Model with Private CoT...\033[0m")
                private_local_response = get_answer_from_nebius_with_cot_and_dp(
                    nebius_client, context, original_question, perturbed_question, cot_private
                )
                private_local_answer = utils.extract_final_answer_from_cot(private_local_response)
                
                is_correct_private = utils.llm_judge_answer(judge_client, original_question, ground_truth, private_local_answer)
                if is_correct_private:
                    private_local_cot_correct += 1
                
                print(f"Local Answer (Private CoT-Aided): {private_local_answer}")
                print(f"Result (LLM Judge): {'Correct' if is_correct_private else 'Incorrect'}")

            except Exception as e:
                print(f"\033[91mError during private CoT-aided inference: {e}\033[0m")
        else:
            print(f"\033[93mSkipping private CoT-aided local model inference due to missing API key.\033[0m")

        # --- Scenario 4: Purely Remote Model ---
        print(f"\n\033[94m--- Scenario 4: Purely Remote Model ---\033[0m")
        if remote_client and judge_client:
            try:
                print(f"\033[92m4a. Running Purely Remote LLM ({config['remote_models']['llm_model']}) with full context...\033[0m")
                purely_remote_response = get_answer_from_purely_remote_llm(
                    remote_client, config["remote_models"]["llm_model"], context, original_question
                )
                purely_remote_answer = utils.extract_final_answer_from_cot(purely_remote_response)
                
                is_correct_purely_remote = utils.llm_judge_answer(judge_client, original_question, ground_truth, purely_remote_answer)
                if is_correct_purely_remote:
                    purely_remote_correct += 1
                
                print(f"Purely Remote Answer: {purely_remote_answer}")
                print(f"Result (LLM Judge): {'Correct' if is_correct_purely_remote else 'Incorrect'}")

            except Exception as e:
                print(f"\033[91mError during purely remote model inference: {e}\033[0m")
        else:
            print(f"\033[93mSkipping purely remote model inference due to missing API key.\033[0m")

    # Final Results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    if config["dataset"]["num_samples"] > 0:
        local_alone_accuracy = (local_alone_correct / config["dataset"]["num_samples"]) * 100
        print(f"1. Purely Local Model ({model_name}) Accuracy: {local_alone_accuracy:.2f}% (LLM Judge)")

        if remote_client and judge_client:
            non_private_local_cot_accuracy = (non_private_local_cot_correct / config["dataset"]["num_samples"]) * 100
            print(f"2. Non-Private Local Model ({model_name}) + CoT Accuracy: {non_private_local_cot_accuracy:.2f}% (LLM Judge)")
            
            private_local_cot_accuracy = (private_local_cot_correct / config["dataset"]["num_samples"]) * 100
            print(f"3. Private Local Model ({model_name}) + CoT Accuracy: {private_local_cot_accuracy:.2f}% (LLM Judge)")
            
            purely_remote_accuracy = (purely_remote_correct / config["dataset"]["num_samples"]) * 100
            print(f"4. Purely Remote Model ({config['remote_models']['llm_model']}) Accuracy: {purely_remote_accuracy:.2f}% (LLM Judge)")
            
            print("\n--- Performance Comparisons ---")
            print(f"Performance Gain (Non-Private CoT-Aiding vs. Local Alone): {non_private_local_cot_accuracy - local_alone_accuracy:.2f}%")
            print(f"Performance Change (Private CoT-Aiding vs. Non-Private CoT-Aiding): {private_local_cot_accuracy - non_private_local_cot_accuracy:.2f}%")
            print(f"Performance Gap (Purely Remote vs. Private CoT-Aiding): {purely_remote_accuracy - private_local_cot_accuracy:.2f}%")
        else:
            print("Remote LLM-dependent scenarios were skipped due to missing API key.")
    else:
        print("No samples were tested.")

if __name__ == "__main__":
    for model_name in config["local_models"]:
        run_experiment_for_model(model_name)