import os
import yaml
import json
from openai import OpenAI
from dotenv import load_dotenv
from dp_sanitizer import load_sentence_bert
import utils
from datasets import load_dataset

# Load environment variables
load_dotenv()

# Get API keys from environment variables
NEBIUS_API = os.getenv("NEBIUS")

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Nebius client
nebius_client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=NEBIUS_API)

def explore_emrqa_msquad_dataset():
    """
    Load and explore the Eladio/emrqa-msquad dataset from Hugging Face.
    """
    print("="*80)
    print("EXPLORING Eladio/emrqa-msquad DATASET")
    print("="*80)
    
    try:
        # Load the dataset
        print("Loading Eladio/emrqa-msquad dataset...")
        dataset = load_dataset("Eladio/emrqa-msquad")
        
        print(f"✓ Successfully loaded dataset")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Show dataset info for each split
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()} split:")
            print(f"  Total samples: {len(split_data)}")
            
            if len(split_data) > 0:
                print(f"  Available fields: {list(split_data[0].keys())}")
        
        # Show first 10 examples from the first available split
        first_split = list(dataset.keys())[0]
        split_data = dataset[first_split]
        
        print(f"\n{'='*80}")
        print(f"FIRST 10 EXAMPLES FROM {first_split.upper()} SPLIT")
        print(f"{'='*80}")
        
        for i in range(min(10, len(split_data))):
            example = split_data[i]
            print(f"\n--- Example {i+1} ---")
            
            # Print all fields
            for key, value in example.items():
                if isinstance(value, str):
                    if len(value) > 200:
                        print(f"{key}: {value[:200]}...")
                    else:
                        print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
            
            print("-" * 50)
        
        return dataset
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None

def test_medical_phrase_dp_with_emrqa():
    """
    Test the phrase_DP_perturbation function with actual emrqa-msquad questions.
    """
    print("="*80)
    print("TESTING phrase_DP_perturbation ON EMRQA-MSQUAD QUESTIONS")
    print("="*80)
    
    # Load the sentence BERT model
    print("Loading sentence BERT model...")
    sbert_model = load_sentence_bert()
    print("✓ Sentence BERT model loaded successfully")
    
    # Load emrqa-msquad dataset
    print("Loading Eladio/emrqa-msquad dataset...")
    try:
        dataset = load_dataset("Eladio/emrqa-msquad")
        first_split = list(dataset.keys())[0]
        split_data = dataset[first_split]
        
        # Get first 10 examples for testing (including context)
        medical_examples = []
        for i in range(min(10, len(split_data))):
            example = split_data[i]
            if 'question' in example and 'context' in example:
                medical_examples.append({
                    'question': example['question'],
                    'context': example['context'],
                    'answers': example.get('answers', {})
                })
        
        print(f"✓ Loaded {len(medical_examples)} examples from {first_split} split")
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("Falling back to sample questions...")
        medical_examples = [
            {
                'question': "What medications is the patient currently taking?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            },
            {
                'question': "What is the patient's primary diagnosis?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            },
            {
                'question': "What are the patient's vital signs?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            },
            {
                'question': "What procedures has the patient undergone?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            },
            {
                'question': "What is the patient's smoking status?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            },
            {
                'question': "What is the patient's body mass index?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            },
            {
                'question': "What allergies does the patient have?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            },
            {
                'question': "What is the patient's heart disease risk?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            },
            {
                'question': "What is the patient's blood pressure reading?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            },
            {
                'question': "What laboratory results are available for the patient?",
                'context': "Sample medical context for testing purposes.",
                'answers': {}
            }
        ]
    
    # Test only with epsilon = 1.0
    epsilon = 1.0
    
    model_name = config["local_model"]
    
    print(f"\nUsing model: {model_name}")
    print(f"Testing {len(medical_examples)} medical examples with epsilon = {epsilon}")
    print("="*80)
    
    results = []
    
    # Open file to write results
    with open("testing-emrqa-msquad-phraseDP.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("TESTING phrase_DP_perturbation ON EMRQA-MSQUAD QUESTIONS\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Number of examples: {len(medical_examples)}\n")
        f.write(f"Dataset: Eladio/emrqa-msquad\n")
        f.write("="*80 + "\n\n")
        
        for i, example in enumerate(medical_examples):
            print(f"\n{'='*60}")
            print(f"MEDICAL EXAMPLE {i+1}/{len(medical_examples)}")
            print(f"{'='*60}")
            print(f"Context: {example['context'][:200]}...")
            print(f"Original Question: {example['question']}")
            
            f.write(f"MEDICAL EXAMPLE {i+1}/{len(medical_examples)}\n")
            f.write("="*60 + "\n")
            f.write(f"Context: {example['context']}\n")
            f.write(f"Original Question: {example['question']}\n")
            
            try:
                # Apply phrase DP perturbation
                perturbed_question = utils.phrase_DP_perturbation(
                    nebius_client, 
                    model_name, 
                    example['question'], 
                    epsilon, 
                    sbert_model
                )
                
                print(f"Perturbed: {perturbed_question}")
                f.write(f"Perturbed: {perturbed_question}\n")
                
                results.append({
                    "context": example['context'],
                    "original": example['question'],
                    "perturbed": perturbed_question,
                    "epsilon": epsilon
                })
                
            except Exception as e:
                error_msg = f"ERROR: {e}"
                print(f"✗ {error_msg}")
                f.write(f"ERROR: {e}\n")
                results.append({
                    "context": example['context'],
                    "original": example['question'],
                    "perturbed": error_msg,
                    "epsilon": epsilon
                })
            
            f.write("\n")
            print()
        
        # Write detailed results to file
        with open("testing-emrqa-msquad-phraseDP.txt", "w") as f:
            f.write("="*80 + "\n")
            f.write("TESTING phrase_DP_perturbation ON EMRQA-MSQUAD QUESTIONS\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results):
                f.write(f"Medical Example {i+1}:\n")
                f.write(f"Context: {result['context']}\n")
                f.write(f"Original Question: {result['original']}\n")
                f.write(f"Perturbed Question: {result['perturbed']}\n")
                
                # Add the real answer from the dataset
                if 'answers' in medical_examples[i]:
                    answers = medical_examples[i]['answers']
                    if 'text' in answers:
                        f.write(f"Real Answer: {answers['text']}\n")
                    if 'answer_start' in answers:
                        f.write(f"Answer Start Position: {answers['answer_start']}\n")
                
                f.write("-" * 80 + "\n\n")
            
            # Write summary
            f.write("\n" + "="*80 + "\n")
            f.write("SUMMARY OF EMRQA-MSQUAD RESULTS\n")
            f.write("="*80 + "\n")
            
            for i, result in enumerate(results):
                f.write(f"\nMedical Example {i+1}:\n")
                f.write(f"Context: {result['context']}\n")
                f.write(f"Original Question: {result['original']}\n")
                f.write(f"Perturbed: {result['perturbed']}\n")
                if 'answers' in medical_examples[i]:
                    answers = medical_examples[i]['answers']
                    if 'text' in answers:
                        f.write(f"Real Answer: {answers['text']}\n")
                f.write("-" * 80 + "\n")
    
    print(f"\nResults saved to testing-emrqa-msquad-phraseDP.txt")
    
    return results

if __name__ == "__main__":
    # First explore the dataset structure and show first 10 examples
    dataset = explore_emrqa_msquad_dataset()
    
    # Then run the medical phrase DP test
    print("\n" + "="*80)
    test_medical_phrase_dp_with_emrqa()
