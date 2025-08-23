import os
import yaml
from openai import OpenAI
from dotenv import load_dotenv
from dp_sanitizer import load_sentence_bert
import utils

# Load environment variables
load_dotenv()

# Get API keys from environment variables
NEBIUS_API = os.getenv("NEBIUS")

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Nebius client
nebius_client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=NEBIUS_API)

def test_phrase_dp_perturbation():
    """
    Test the phrase_DP_perturbation function with actual HotpotQA questions.
    """
    print("="*80)
    print("TESTING phrase_DP_perturbation FUNCTION")
    print("="*80)
    
    # Load the sentence BERT model
    print("Loading sentence BERT model...")
    sbert_model = load_sentence_bert()
    print("✓ Sentence BERT model loaded successfully")
    
    # Load HotpotQA dataset
    print(f"Loading dataset: {config['dataset']['name']}...")
    from datasets import load_dataset
    dataset = load_dataset(config["dataset"]["name"], "distractor", split=config["dataset"]["split"])
    
    # Get multi-hop questions (questions with more than 1 supporting fact)
    multi_hop_questions = [q for q in dataset if len(q["supporting_facts"]) > 1]
    multi_hop_questions = multi_hop_questions[:config["dataset"]["num_samples"]]
    
    # Extract just the questions for testing
    test_questions = [item["question"] for item in multi_hop_questions]
    
    # Test only with epsilon = 1.0
    epsilon = 1.0
    
    model_name = config["local_model"]
    
    print(f"\nUsing model: {model_name}")
    print(f"Testing {len(test_questions)} questions from HotpotQA with epsilon = {epsilon}")
    print("="*80)
    
    results = []
    
    # Open file to write results
    with open("testing-phraseDP.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("TESTING phrase_DP_perturbation FUNCTION\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Number of questions: {len(test_questions)}\n")
        f.write("="*80 + "\n\n")
        
        for i, question in enumerate(test_questions):
            print(f"\n{'='*60}")
            print(f"QUESTION {i+1}/{len(test_questions)}")
            print(f"{'='*60}")
            print(f"Original Question: {question}")
            
            f.write(f"QUESTION {i+1}/{len(test_questions)}\n")
            f.write("="*60 + "\n")
            f.write(f"Original Question: {question}\n")
            f.write("\nCandidates and similarities:\n")
            
            try:
                # Apply phrase DP perturbation and get candidates
                perturbed_question, candidate_sentences = utils.phrase_DP_perturbation_with_candidates(
                    nebius_client, 
                    model_name, 
                    question, 
                    epsilon, 
                    sbert_model
                )
                
                print(f"Perturbed: {perturbed_question}")
                f.write(f"Perturbed: {perturbed_question}\n")
                
                # Write all candidates and their similarities
                f.write("\nAll Generated Candidates:\n")
                f.write("-" * 40 + "\n")
                for j, candidate in enumerate(candidate_sentences, 1):
                    f.write(f"{j:3d}. {candidate}\n")
                f.write("-" * 40 + "\n")
                
                results.append({
                    "original": question,
                    "perturbed": perturbed_question,
                    "epsilon": epsilon
                })
                
            except Exception as e:
                error_msg = f"ERROR: {e}"
                print(f"✗ {error_msg}")
                f.write(f"ERROR: {e}\n")
                results.append({
                    "original": question,
                    "perturbed": error_msg,
                    "epsilon": epsilon
                })
            
            f.write("\n")
            print()
        
        # Write summary
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY OF RESULTS\n")
        f.write("="*80 + "\n")
        
        for i, result in enumerate(results):
            f.write(f"\nQuestion {i+1}: {result['original']}\n")
            f.write(f"Perturbed: {result['perturbed']}\n")
    
    print(f"\nResults saved to testing-phraseDP.txt")
    
    return results

def test_single_question(question, epsilon=1.0):
    """
    Test phrase_DP_perturbation with a single question.
    """
    print(f"\n{'='*60}")
    print(f"TESTING SINGLE QUESTION")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"Epsilon: {epsilon}")
    
    # Load the sentence BERT model
    sbert_model = load_sentence_bert()
    model_name = config["local_model"]
    
    try:
        perturbed_question = utils.phrase_DP_perturbation(
            nebius_client, 
            model_name, 
            question, 
            epsilon, 
            sbert_model
        )
        
        print(f"\nOriginal: {question}")
        print(f"Perturbed: {perturbed_question}")
        
        return perturbed_question
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Test with a single question first
    print("Testing single question...")
    test_single_question("What is the capital of France?", epsilon=1.0)
    
    # Run full test
    print("\nRunning full test...")
    test_phrase_dp_perturbation()
