import os
import yaml
from datasets import load_dataset
from openai import OpenAI
from dotenv import load_dotenv
from inferdpt import perturb_sentence
import utils

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
NEBIUS_API = os.getenv("NEBIUS")

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize clients
nebius_client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=NEBIUS_API)
remote_client = utils.get_remote_llm_client(config["remote_llm_provider"])
judge_client = utils.get_remote_llm_client(config["remote_llm_provider"])

def get_cot_from_remote_llm(client, model_name, question):
    """Generate CoT from remote LLM"""
    cot_prompt = (
        f"Question: {question}\n"
        f"Please think step-by-step to arrive at the answer based on your general knowledge. "
        f"Do NOT use any external context beyond what is in the question itself. "
        f"DO NOT provide the final answer yet. Only output your thought process.\n\n"
        f"Thought:"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": cot_prompt}],
        max_tokens=256,
        temperature=0.0
    )
    return response.choices[0].message.content.strip()

def get_answer_from_local_model_with_cot(nebius_client, model_name, context, original_question, cot):
    """Generate answer using local model with CoT guidance"""
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

    response = nebius_client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=256,
        temperature=0.0,
    )
    return response.choices[0].message.content.strip()

def run_scenario_3_2_inferdpt(nebius_client, model_name, context, original_question, ground_truth, remote_client, judge_client, config):
    """Scenario 3.2: Private Local Model + CoT (InferDPT + remote CoT)"""
    print(f"\n\033[94m--- Scenario 3.2: Private Local Model + CoT (InferDPT) ---\033[0m")
    try:
        print(f"\033[92m3.2a. Applying InferDPT Differential Privacy to the question...\033[0m")
        perturbed_question = perturb_sentence(original_question, config["epsilon"])
        print(f"Perturbed Question: {perturbed_question}")

        print(f"\033[92m3.2b. Generating CoT from Perturbed Question with REMOTE LLM ({config['remote_models']['cot_model']}) (Context NOT sent)...\033[0m")
        cot_private = get_cot_from_remote_llm(remote_client, config["remote_models"]["cot_model"], perturbed_question)
        print(f"\033[96mGenerated Chain-of-Thought (Remote, Private):\033[0m\n{cot_private}\n")
        
        print(f"\033[94m3.2c. Running Local Model with Private CoT...\033[0m")
        private_local_response = get_answer_from_local_model_with_cot(
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

def test_scenario_3_2_only():
    """Test only Scenario 3.2 (InferDPT) on a few questions"""
    print("="*80)
    print("TESTING SCENARIO 3.2 (InferDPT) ONLY")
    print("="*80)
    
    # Load dataset
    print(f"Loading dataset: {config['dataset']['name']}...")
    dataset = load_dataset(config["dataset"]["name"], "distractor", split=config["dataset"]["split"])
    
    # Get multi-hop questions
    multi_hop_questions = [q for q in dataset if len(q["supporting_facts"]) > 1]
    test_questions = multi_hop_questions[:3]  # Test only 3 questions
    
    model_name = config["local_model"]
    print(f"Using model: {model_name}")
    print(f"Testing {len(test_questions)} questions with epsilon = {config['epsilon']}")
    print("="*80)
    
    correct_count = 0
    
    for i, item in enumerate(test_questions):
        print(f"\n\033[93m--- Question {i+1}/{len(test_questions)} ---\033[0m")
        
        original_question = item["question"]
        ground_truth = item["answer"]
        all_sentences = [sentence for sublist in item["context"]["sentences"] for sentence in sublist]
        context = " ".join(all_sentences)
        
        print(f"Original Question: {original_question}")
        print(f"Ground Truth: {ground_truth}")

        # Run Scenario 3.2: InferDPT + Local Model + Remote CoT
        _, is_correct = run_scenario_3_2_inferdpt(
            nebius_client, model_name, context, original_question, ground_truth,
            remote_client, judge_client, config
        )
        
        if is_correct:
            correct_count += 1
    
    # Print final results
    print("\n" + "="*50)
    print("FINAL RESULTS - SCENARIO 3.2 (InferDPT)")
    print("="*50)
    accuracy = (correct_count / len(test_questions)) * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{len(test_questions)})")

if __name__ == "__main__":
    test_scenario_3_2_only()
