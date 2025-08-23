import os
import yaml
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from dp_sanitizer import load_sentence_bert
import utils
from prompt_loader import load_system_prompt, load_user_prompt_template, format_user_prompt

def test_epsilon_range(question, epsilon_range=(1, 5), num_steps=5):
    """
    Test how different epsilon values affect DP replacement selection.
    Generate candidates once, then test different epsilon values.
    """
    print(f"\n{'='*80}")
    print(f"EPSILON EXPERIMENT")
    print(f"{'='*80}")
    print(f"Original Question: {question}")
    print(f"Epsilon Range: {epsilon_range[0]} to {epsilon_range[1]} (steps: {num_steps})")
    
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables
    load_dotenv()
    
    # Initialize clients and models
    nebius_client = OpenAI(
        base_url='https://api.studio.nebius.ai/v1/',
        api_key=os.getenv('NEBIUS')
    )
    model_name = 'microsoft/phi-4'  # Use the model directly
    sbert_model = load_sentence_bert()
    
    # Generate candidates once (using epsilon=1 for generation)
    print(f"\nGenerating candidates with epsilon=1...")
    try:
        # Generate candidates with 5 API calls, 5 candidates each
        candidate_sentences = utils.generate_sentence_replacements_with_nebius(
            nebius_client,
            model_name,
            question,
            num_return_sequences=5,
            num_api_calls=5
        )
        
        print(f"Generated {len(candidate_sentences)} candidates")
        
        # Calculate similarities for all candidates
        from dp_sanitizer import get_embedding, differentially_private_replacement
        candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in candidate_sentences}
        original_embedding = get_embedding(sbert_model, question).cpu().numpy()
        
        # Calculate similarities
        similarities = {}
        for candidate in candidate_sentences:
            candidate_emb = candidate_embeddings[candidate]
            similarity = np.dot(original_embedding, candidate_emb) / (np.linalg.norm(original_embedding) * np.linalg.norm(candidate_emb))
            similarities[candidate] = similarity
        
        # Test different epsilon values
        epsilon_values = np.linspace(epsilon_range[0], epsilon_range[1], num_steps)
        results = []
        
        print(f"\nTesting epsilon values: {epsilon_values}")
        print(f"{'Epsilon':<8} {'Similarity':<12} {'Selected Replacement'}")
        print("-" * 120)
        
        for epsilon in epsilon_values:
            # Select replacement using different epsilon
            selected_replacement = differentially_private_replacement(
                target_phrase=question,
                epsilon=epsilon,
                candidate_phrases=candidate_sentences,
                candidate_embeddings=candidate_embeddings,
                sbert_model=sbert_model
            )
            
            similarity = similarities[selected_replacement]
            results.append({
                'epsilon': epsilon,
                'replacement': selected_replacement,
                'similarity': similarity
            })
            
            # Show full replacement without truncation
            print(f"{epsilon:<8.2f} {similarity:<12.4f} {selected_replacement}")
        
        return results, candidate_sentences, similarities
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

def run_epsilon_experiment():
    """Run epsilon experiment on multiple questions and save results."""
    
    # Test questions
    test_questions = [
        "Were Scott Derrickson and Ed Wood of the same nationality?",
        "Is Annie Morton older than Terry Richardson?",
        "Are Local H and For Against both from the United States?"
    ]
    
    all_results = {}
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'='*80}")
        print(f"QUESTION {i}/{len(test_questions)}")
        print(f"{'='*80}")
        
        results, candidates, similarities = test_epsilon_range(question, epsilon_range=(1, 5), num_steps=9)
        
        if results:
            all_results[question] = {
                'results': results,
                'candidates': candidates,
                'similarities': similarities
            }
    
    # Save results to file
    save_epsilon_results(all_results)
    
    return all_results

def save_epsilon_results(all_results):
    """Save epsilon experiment results to a file."""
    filename = "epsilon_experiment_results.txt"
    
    with open(filename, 'w') as f:
        f.write("EPSILON EXPERIMENT RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        for question, data in all_results.items():
            f.write(f"ORIGINAL QUESTION: {question}\n")
            f.write("-" * 80 + "\n")
            
            # Write epsilon results
            f.write(f"{'Epsilon':<8} {'Similarity':<12} {'Selected Replacement'}\n")
            f.write("-" * 120 + "\n")
            
            for result in data['results']:
                epsilon = result['epsilon']
                replacement = result['replacement']
                similarity = result['similarity']
                
                # Show full replacement without truncation
                f.write(f"{epsilon:<8.2f} {similarity:<12.4f} {replacement}\n")
            
            f.write("\n")
            
            # Write all candidates with similarities
            f.write("ALL CANDIDATES WITH SIMILARITIES:\n")
            f.write("-" * 60 + "\n")
            
            # Sort candidates by similarity
            sorted_candidates = sorted(data['similarities'].items(), key=lambda x: x[1], reverse=True)
            
            for j, (candidate, similarity) in enumerate(sorted_candidates, 1):
                f.write(f"{j:3d}. {candidate}\n")
                f.write(f"     Similarity: {similarity:.4f}\n")
            
            f.write("-" * 60 + "\n\n")
    
    print(f"\nResults saved to {filename}")

if __name__ == "__main__":
    run_epsilon_experiment()
