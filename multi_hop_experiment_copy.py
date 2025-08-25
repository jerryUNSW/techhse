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

def create_completion_with_model_support(client, model_name, messages, max_tokens=256, temperature=0.0):
    """
    Create a chat completion with proper parameter support for different models.
    GPT-5 uses max_completion_tokens and doesn't support temperature=0.0, others use max_tokens.
    """
    if "gpt-5" in model_name:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens
            # GPT-5 doesn't support temperature parameter
        )
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )

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
    response = create_completion_with_model_support(
        client, model_name, 
        [{"role": "user", "content": cot_prompt}],
        max_tokens=256, temperature=0.0
    )
    return response.choices[0].message.content.strip()

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

    response = create_completion_with_model_support(client, model_name, 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

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

    response = create_completion_with_model_support(client, config["local_model"], 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256, temperature=0.0
    )
    return response.choices[0].message.content.strip()

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

    response = create_completion_with_model_support(client, model_name, 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

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

    response = create_completion_with_model_support(client, model_name, 
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def get_cot_from_local_model(client, model_name, question):
    """
    Generates a Chain-of-Thought (CoT) using a local model (via Nebius),
    based ONLY on the question (no context).
    """
    cot_prompt = (
        f"Question: {question}\n"
        f"Please think step-by-step to arrive at the answer based on your general knowledge. "
        f"Do NOT use any external context beyond what is in the question itself. "
        f"DO NOT provide the final answer yet. Only output your thought process.\n\n"
        f"Thought:"
    )
    response = create_completion_with_model_support(client, model_name, 
        messages=[{"role": "user", "content": cot_prompt}],
        max_tokens=256,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def run_scenario_1_purely_local(nebius_client, model_name, context, original_question, ground_truth, judge_client):
    """
    Scenario 1: Purely Local Model (Baseline)
    Returns: (local_answer, is_correct)
    """
    print(f"\n\033[94m--- Scenario 1: Purely Local Model (Baseline) ---\033[0m")
    local_response_alone = get_answer_from_local_model_alone(
        nebius_client, model_name, context, original_question
    )
    local_answer_alone = utils.extract_final_answer_from_cot(local_response_alone)
    
    is_correct_alone = utils.llm_judge_answer(judge_client, original_question, ground_truth, local_answer_alone)
    
    print(f"Local Answer (Alone): {local_answer_alone}")
    print(f"Result (LLM Judge): {'Correct' if is_correct_alone else 'Incorrect'}")
    
    return local_answer_alone, is_correct_alone

def run_scenario_2_non_private_remote_cot(nebius_client, model_name, context, original_question, ground_truth, remote_client, judge_client, config):
    """
    Scenario 2: Non-Private Local Model + Remote CoT
    Returns: (local_answer, is_correct)
    """
    print(f"\n\033[94m--- Scenario 2: Non-Private Local Model + CoT ---\033[0m")
    print(f"\033[92m2a. Generating CoT from ORIGINAL Question with REMOTE LLM ({config['remote_models']['cot_model']}) (Context NOT sent)...\033[0m")
    cot_non_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], original_question)
    print(f"\033[96mGenerated Chain-of-Thought (Remote, Non-Private):\033[0m\n{cot_non_private}\n")
    
    print(f"\033[94m2b. Running Local Model with Non-Private CoT...\033[0m")
    non_private_local_response = get_answer_from_local_model_with_non_private_cot(
        nebius_client, model_name, context, original_question, cot_non_private
    )
    non_private_local_answer = utils.extract_final_answer_from_cot(non_private_local_response)
    
    is_correct_non_private = utils.llm_judge_answer(judge_client, original_question, ground_truth, non_private_local_answer)
    
    print(f"Local Answer (Non-Private CoT-Aided): {non_private_local_answer}")
    print(f"Result (LLM Judge): {'Correct' if is_correct_non_private else 'Incorrect'}")
    
    return non_private_local_answer, is_correct_non_private

def run_scenario_2_5_non_private_local_cot(nebius_client, model_name, context, original_question, ground_truth, judge_client):
    """
    Scenario 2.5: Non-Private Local Model + Local CoT
    Returns: (local_answer, is_correct)
    """
    print(f"\n\033[94m--- Scenario 2.5: Non-Private Local Model + Local CoT ---\033[0m")
    cot = get_cot_from_local_model(nebius_client, model_name, original_question)
    print(f"\033[96mNon-Private Local CoT: {cot}\033[0m")
    
    non_private_local_cot_local_answer = get_answer_from_local_model_with_non_private_cot(
        nebius_client, model_name, context, original_question, cot
    )
    print(f"\033[94mNon-Private Local + Local CoT Answer: {non_private_local_cot_local_answer}\033[0m")
    
    is_correct = utils.llm_judge_answer(judge_client, original_question, ground_truth, non_private_local_cot_local_answer)
    print(f"\033[92mNon-Private Local + Local CoT Correct: {is_correct}\033[0m")
    
    return non_private_local_cot_local_answer, is_correct

def run_scenario_3_1_phrase_dp_local_cot(nebius_client, model_name, context, original_question, ground_truth, remote_client, judge_client, config, sbert_model):
    """
    Scenario 3.1: Private Local Model + CoT (Phrase DP + remote CoT)
    Returns: (private_answer, is_correct)
    """
    print(f"\n\033[94m--- Scenario 3.1: Private Local Model + CoT (Phrase DP) ---\033[0m")
    try:
        print(f"\033[92m3.1a. Applying Phrase-Level Differential Privacy to the question...\033[0m")
        perturbed_question = utils.phrase_DP_perturbation(nebius_client, model_name, original_question, config["epsilon"], sbert_model)
        print(f"Perturbed Question: {perturbed_question}")

        print(f"\033[92m3.1b. Generating CoT from Perturbed Question with REMOTE LLM ({config['remote_models']['cot_model']}) (Context NOT sent)...\033[0m")
        cot_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], perturbed_question)
        print(f"\033[96mGenerated Chain-of-Thought (Remote, Private):\033[0m\n{cot_private}\n")
        
        print(f"\033[94m3.1c. Running Local Model with Private CoT...\033[0m")
        private_local_response = get_answer_from_nebius_with_cot_and_dp(
            nebius_client, context, original_question, perturbed_question, cot_private
        )
        private_local_answer = utils.extract_final_answer_from_cot(private_local_response)
        
        is_correct_private = utils.llm_judge_answer(judge_client, original_question, ground_truth, private_local_answer)
        
        print(f"Local Answer (Private CoT-Aided): {private_local_answer}")
        print(f"Result (LLM Judge): {'Correct' if is_correct_private else 'Incorrect'}")
        
        return private_local_answer, is_correct_private
        
    except Exception as e:
        print(f"\033[91mError during private CoT-aided inference: {e}\033[0m")
        return "Error", False

def run_scenario_3_2_inferdpt_local_cot(nebius_client, model_name, context, original_question, ground_truth, remote_client, judge_client, config):
    """
    Scenario 3.2: Private Local Model + CoT (InferDPT + remote CoT)
    Returns: (private_answer, is_correct)
    """
    print(f"\n\033[94m--- Scenario 3.2: Private Local Model + CoT (InferDPT) ---\033[0m")
    try:
        print(f"\033[92m3.2a. Applying InferDPT Differential Privacy to the question...\033[0m")
        # Import InferDPT perturbation function inside function to avoid argument conflicts
        import sys
        original_argv = sys.argv.copy()
        sys.argv = [sys.argv[0]]  # Reset to just the script name to avoid argument conflicts
        from inferdpt import perturb_sentence
        sys.argv = original_argv  # Restore original arguments
        perturbed_question = perturb_sentence(original_question, config["epsilon"])
        print(f"Perturbed Question: {perturbed_question}")

        print(f"\033[92m3.2b. Generating CoT from Perturbed Question with REMOTE LLM ({config['remote_models']['cot_model']}) (Context NOT sent)...\033[0m")
        cot_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], perturbed_question)
        print(f"\033[96mGenerated Chain-of-Thought (Remote, Private):\033[0m\n{cot_private}\n")
        
        print(f"\033[94m3.2c. Running Local Model with Private CoT...\033[0m")
        # Use the same pattern as Scenario 2: local model with CoT guidance
        private_local_response = get_answer_from_local_model_with_non_private_cot(
            nebius_client, model_name, context, original_question, cot_private
        )
        private_local_answer = utils.extract_final_answer_from_cot(private_local_response)
        
        is_correct_private = utils.llm_judge_answer(judge_client, original_question, ground_truth, private_local_answer)
        
        print(f"Local Answer (Private CoT-Aided): {private_local_answer}")
        print(f"Result (LLM Judge): {'Correct' if is_correct_private else 'Incorrect'}")
        
        return private_local_answer, is_correct_private
        
    except Exception as e:
        print(f"\033[91mError during InferDPT private CoT-aided inference: {e}\033[0m")
        return "Error", False

def run_scenario_4_purely_remote(remote_client, context, original_question, ground_truth, judge_client, config):
    """
    Scenario 4: Purely Remote Model
    Returns: (remote_answer, is_correct)
    """
    print(f"\n\033[94m--- Scenario 4: Purely Remote Model ---\033[0m")
    print(f"\033[92m4a. Running Purely Remote LLM ({config['remote_models']['llm_model']}) with full context...\033[0m")
    purely_remote_response = get_answer_from_purely_remote_llm(
        remote_client, config["remote_models"]["llm_model"], context, original_question
    )
    purely_remote_answer = utils.extract_final_answer_from_cot(purely_remote_response)
    
    is_correct_purely_remote = utils.llm_judge_answer(judge_client, original_question, ground_truth, purely_remote_answer)
    
    print(f"Purely Remote Answer: {purely_remote_answer}")
    print(f"Result (LLM Judge): {'Correct' if is_correct_purely_remote else 'Incorrect'}")
    
    return purely_remote_answer, is_correct_purely_remote

class ExperimentResults:
    """
    Class to track experiment results across all scenarios
    """
    def __init__(self):
        self.local_alone_correct = 0
        self.non_private_local_cot_correct = 0
        self.non_private_local_cot_local_correct = 0
        self.phrase_dp_local_cot_correct = 0
        self.inferdpt_local_cot_correct = 0
        self.purely_remote_correct = 0
    
    def print_final_results(self, model_name, config):
        """Print the final accuracy results"""
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        num_samples = config["dataset"]["num_samples"]
        
        local_alone_accuracy = (self.local_alone_correct / num_samples) * 100
        print(f"1. Purely Local Model ({model_name}) Accuracy: {local_alone_accuracy:.2f}% (LLM Judge)")

        non_private_local_cot_accuracy = (self.non_private_local_cot_correct / num_samples) * 100
        print(f"2. Non-Private Local Model ({model_name}) + Remote CoT Accuracy: {non_private_local_cot_accuracy:.2f}% (LLM Judge)")
        
        non_private_local_cot_local_accuracy = (self.non_private_local_cot_local_correct / num_samples) * 100
        print(f"2.5. Non-Private Local Model ({model_name}) + Local CoT Accuracy: {non_private_local_cot_local_accuracy:.2f}% (LLM Judge)")
        
        phrase_dp_local_cot_accuracy = (self.phrase_dp_local_cot_correct / num_samples) * 100
        print(f"3.1. Private Local Model ({model_name}) + CoT (Phrase DP) Accuracy: {phrase_dp_local_cot_accuracy:.2f}% (LLM Judge)")
        
        inferdpt_local_cot_accuracy = (self.inferdpt_local_cot_correct / num_samples) * 100
        print(f"3.2. Private Local Model ({model_name}) + CoT (InferDPT) Accuracy: {inferdpt_local_cot_accuracy:.2f}% (LLM Judge)")
        
        purely_remote_accuracy = (self.purely_remote_correct / num_samples) * 100
        print(f"4. Purely Remote Model ({config['remote_models']['llm_model']}) Accuracy: {purely_remote_accuracy:.2f}% (LLM Judge)")

def run_experiment_for_model(model_name):
    """
    Run the multi-hop reasoning experiment for a given local model.
    """
    print(f"\033[96mLoading dataset: {config['dataset']['name']}...\033[0m")
    dataset = load_dataset(config["dataset"]["name"], "distractor", split=config["dataset"]["split"])
    
    multi_hop_questions = [q for q in dataset if len(q["supporting_facts"]) > 1]
    multi_hop_questions = multi_hop_questions[:config["dataset"]["num_samples"]]

    remote_client = utils.get_remote_llm_client(config["remote_llm_provider"])
    judge_client = utils.get_remote_llm_client(config["remote_llm_provider"])
    
    # Initialize results tracker
    results = ExperimentResults()
    
    sbert_model = load_sentence_bert()

    for i, item in enumerate(multi_hop_questions):
        print(f"\n\033[93m--- Question {i+1}/{config['dataset']['num_samples']} ---\033[0m")
        
        original_question = item["question"]
        ground_truth = item["answer"]
        all_sentences = [sentence for sublist in item["context"]["sentences"] for sentence in sublist]
        context = " ".join(all_sentences)
        
        print(f"Original Question: {original_question}")
        print(f"Ground Truth: {ground_truth}")

        # Run Scenario 1: Purely Local Model (Baseline)
        _, is_correct_alone = run_scenario_1_purely_local(
            nebius_client, model_name, context, original_question, ground_truth, judge_client
        )
        if is_correct_alone:
            results.local_alone_correct += 1

        # Run Scenario 2: Non-Private Local Model + Remote CoT
        _, is_correct_non_private = run_scenario_2_non_private_remote_cot(
            nebius_client, model_name, context, original_question, ground_truth, 
            remote_client, judge_client, config
        )
        if is_correct_non_private:
            results.non_private_local_cot_correct += 1

        # Run Scenario 2.5: Non-Private Local Model + Local CoT
        _, is_correct_local_cot = run_scenario_2_5_non_private_local_cot(
            nebius_client, model_name, context, original_question, ground_truth, judge_client
        )
        if is_correct_local_cot:
            results.non_private_local_cot_local_correct += 1

        # Run Scenario 3.1: Phrase DP + Local Model + Remote CoT
        _, is_correct_phrase_dp = run_scenario_3_1_phrase_dp_local_cot(
            nebius_client, model_name, context, original_question, ground_truth,
            remote_client, judge_client, config, sbert_model
        )
        if is_correct_phrase_dp:
            results.phrase_dp_local_cot_correct += 1

        # Run Scenario 3.2: InferDPT + Local Model + Remote CoT
        _, is_correct_inferdpt = run_scenario_3_2_inferdpt_local_cot(
            nebius_client, model_name, context, original_question, ground_truth,
            remote_client, judge_client, config
        )
        if is_correct_inferdpt:
            results.inferdpt_local_cot_correct += 1

        # Run Scenario 4: Purely Remote Model
        _, is_correct_purely_remote = run_scenario_4_purely_remote(
            remote_client, context, original_question, ground_truth, judge_client, config
        )
        if is_correct_purely_remote:
            results.purely_remote_correct += 1

    # Print final results
    results.print_final_results(model_name, config)

def test_single_question_from_file(file_path):
    """
    Test a single question loaded from an external file.
    The file should contain: question, answer, context (one per line or JSON format)
    Args:
        file_path: Path to the file containing the QA data
    """
    print(f"\033[96mLoading QA data from file: {file_path}\033[0m")
    
    try:
        # Try to load as JSON first
        import json
        with open(file_path, 'r') as f:
            data = json.load(f)
            question = data.get('question', '')
            answer = data.get('answer', '')
            context = data.get('context', '')
    except (json.JSONDecodeError, FileNotFoundError):
        # If JSON fails, try to load as simple text file (one item per line)
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 3:
                    question = lines[0].strip()
                    answer = lines[1].strip()
                    context = lines[2].strip()
                else:
                    print(f"\033[91mError: File must contain at least 3 lines (question, answer, context)\033[0m")
                    return
        except FileNotFoundError:
            print(f"\033[91mError: File {file_path} not found\033[0m")
            return
    
    model_name = config["local_model"]
    
    print(f"\033[93m--- Testing Single Question from File ---\033[0m")
    print(f"Question: {question}")
    print(f"Ground Truth: {answer}")
    print(f"Context Length: {len(context)} characters")

    remote_client = utils.get_remote_llm_client(config["remote_llm_provider"])
    judge_client = utils.get_remote_llm_client(config["remote_llm_provider"])
    sbert_model = load_sentence_bert()

    # Run Scenario 1: Purely Local Model (Baseline)
    print(f"\n\033[94m--- Scenario 1: Purely Local Model ---\033[0m")
    _, is_correct_alone = run_scenario_1_purely_local(
        nebius_client, model_name, context, question, answer, judge_client
    )

    # Run Scenario 2: Non-Private Local Model + Remote CoT
    print(f"\n\033[94m--- Scenario 2: Non-Private Local Model + Remote CoT ---\033[0m")
    _, is_correct_non_private = run_scenario_2_non_private_remote_cot(
        nebius_client, model_name, context, question, answer, 
        remote_client, judge_client, config
    )

    # Run Scenario 2.5: Non-Private Local Model + Local CoT
    print(f"\n\033[94m--- Scenario 2.5: Non-Private Local Model + Local CoT ---\033[0m")
    _, is_correct_local_cot = run_scenario_2_5_non_private_local_cot(
        nebius_client, model_name, context, question, answer, judge_client
    )

    # Run Scenario 3.1: Phrase DP + Local Model + Remote CoT
    print(f"\n\033[94m--- Scenario 3.1: Private Local Model + CoT (Phrase DP) ---\033[0m")
    _, is_correct_phrase_dp = run_scenario_3_1_phrase_dp_local_cot(
        nebius_client, model_name, context, question, answer,
        remote_client, judge_client, config, sbert_model
    )

    # Run Scenario 3.2: InferDPT + Local Model + Remote CoT
    print(f"\n\033[94m--- Scenario 3.2: Private Local Model + CoT (InferDPT) ---\033[0m")
    _, is_correct_inferdpt = run_scenario_3_2_inferdpt_local_cot(
        nebius_client, model_name, context, question, answer,
        remote_client, judge_client, config
    )

    # Run Scenario 4: Purely Remote Model
    print(f"\n\033[94m--- Scenario 4: Purely Remote Model ---\033[0m")
    _, is_correct_purely_remote = run_scenario_4_purely_remote(
        remote_client, context, question, answer, judge_client, config
    )

    # Print summary
    print(f"\n\033[93m--- Single Question Results Summary ---\033[0m")
    print(f"1. Purely Local Model: {'✅ Correct' if is_correct_alone else '❌ Incorrect'}")
    print(f"2. Non-Private Local + Remote CoT: {'✅ Correct' if is_correct_non_private else '❌ Incorrect'}")
    print(f"2.5. Non-Private Local + Local CoT: {'✅ Correct' if is_correct_local_cot else '❌ Incorrect'}")
    print(f"3.1. Phrase DP + Local + Remote CoT: {'✅ Correct' if is_correct_phrase_dp else '❌ Incorrect'}")
    print(f"3.2. InferDPT + Local + Remote CoT: {'✅ Correct' if is_correct_inferdpt else '❌ Incorrect'}")
    print(f"4. Purely Remote Model: {'✅ Correct' if is_correct_purely_remote else '❌ Incorrect'}")

def test_single_question(question_index=0):
    """
    Test a single question with all scenarios.
    Args:
        question_index: Index of the question to test (0-based)
    """
    print(f"\033[96mLoading dataset: {config['dataset']['name']}...\033[0m")
    dataset = load_dataset(config["dataset"]["name"], "distractor", split=config["dataset"]["split"])
    
    multi_hop_questions = [q for q in dataset if len(q["supporting_facts"]) > 1]
    
    if question_index >= len(multi_hop_questions):
        print(f"\033[91mError: Question index {question_index} is out of range. Dataset has {len(multi_hop_questions)} questions.\033[0m")
        return
    
    item = multi_hop_questions[question_index]
    model_name = config["local_model"]
    
    print(f"\033[93m--- Testing Single Question (Index {question_index}) ---\033[0m")
    
    original_question = item["question"]
    ground_truth = item["answer"]
    all_sentences = [sentence for sublist in item["context"]["sentences"] for sentence in sublist]
    context = " ".join(all_sentences)
    
    print(f"Original Question: {original_question}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Context Length: {len(context)} characters")

    remote_client = utils.get_remote_llm_client(config["remote_llm_provider"])
    judge_client = utils.get_remote_llm_client(config["remote_llm_provider"])
    sbert_model = load_sentence_bert()

    # Run Scenario 1: Purely Local Model (Baseline)
    print(f"\n\033[94m--- Scenario 1: Purely Local Model ---\033[0m")
    _, is_correct_alone = run_scenario_1_purely_local(
        nebius_client, model_name, context, original_question, ground_truth, judge_client
    )

    # Run Scenario 2: Non-Private Local Model + Remote CoT
    print(f"\n\033[94m--- Scenario 2: Non-Private Local Model + Remote CoT ---\033[0m")
    _, is_correct_non_private = run_scenario_2_non_private_remote_cot(
        nebius_client, model_name, context, original_question, ground_truth, 
        remote_client, judge_client, config
    )

    # Run Scenario 2.5: Non-Private Local Model + Local CoT
    print(f"\n\033[94m--- Scenario 2.5: Non-Private Local Model + Local CoT ---\033[0m")
    _, is_correct_local_cot = run_scenario_2_5_non_private_local_cot(
        nebius_client, model_name, context, original_question, ground_truth, judge_client
    )

    # Run Scenario 3.1: Phrase DP + Local Model + Remote CoT
    print(f"\n\033[94m--- Scenario 3.1: Private Local Model + CoT (Phrase DP) ---\033[0m")
    _, is_correct_phrase_dp = run_scenario_3_1_phrase_dp_local_cot(
        nebius_client, model_name, context, original_question, ground_truth,
        remote_client, judge_client, config, sbert_model
    )

    # Run Scenario 3.2: InferDPT + Local Model + Remote CoT
    print(f"\n\033[94m--- Scenario 3.2: Private Local Model + CoT (InferDPT) ---\033[0m")
    _, is_correct_inferdpt = run_scenario_3_2_inferdpt_local_cot(
        nebius_client, model_name, context, original_question, ground_truth,
        remote_client, judge_client, config
    )

    # Run Scenario 4: Purely Remote Model
    print(f"\n\033[94m--- Scenario 4: Purely Remote Model ---\033[0m")
    _, is_correct_purely_remote = run_scenario_4_purely_remote(
        remote_client, context, original_question, ground_truth, judge_client, config
    )

    # Print summary
    print(f"\n\033[93m--- Single Question Results Summary ---\033[0m")
    print(f"1. Purely Local Model: {'✅ Correct' if is_correct_alone else '❌ Incorrect'}")
    print(f"2. Non-Private Local + Remote CoT: {'✅ Correct' if is_correct_non_private else '❌ Incorrect'}")
    print(f"2.5. Non-Private Local + Local CoT: {'✅ Correct' if is_correct_local_cot else '❌ Incorrect'}")
    print(f"3.1. Phrase DP + Local + Remote CoT: {'✅ Correct' if is_correct_phrase_dp else '❌ Incorrect'}")
    print(f"3.2. InferDPT + Local + Remote CoT: {'✅ Correct' if is_correct_inferdpt else '❌ Incorrect'}")
    print(f"4. Purely Remote Model: {'✅ Correct' if is_correct_purely_remote else '❌ Incorrect'}")

if __name__ == "__main__":
    import sys
    
    model_name = config["local_model"]
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--file" and len(sys.argv) > 2:
            # Test single question from file
            file_path = sys.argv[2]
            test_single_question_from_file(file_path)
        elif sys.argv[1] == "--index" and len(sys.argv) > 2:
            # Test single question from dataset by index
            try:
                question_index = int(sys.argv[2])
                test_single_question(question_index)
            except ValueError:
                print("Error: Question index must be a number")
        else:
            print("Usage:")
            print("  python multi_hop_experiment_copy.py                    # Run full experiment")
            print("  python multi_hop_experiment_copy.py --file <file_path> # Test from file")
            print("  python multi_hop_experiment_copy.py --index <number>   # Test from dataset")
    else:
        # Run the full experiment
        run_experiment_for_model(model_name)