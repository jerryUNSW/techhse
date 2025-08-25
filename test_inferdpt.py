import os
import yaml
import numpy as np
from dotenv import load_dotenv
from inferdpt import perturb_sentence

# Load environment variables
load_dotenv()

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def test_inferdpt_perturbation():
    """
    Test the InferDPT perturb_sentence function with varying epsilon values on a single question.
    """
    print("="*80)
    print("TESTING InferDPT perturb_sentence FUNCTION")
    print("="*80)
    
    # Test question
    test_question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    
    # Epsilon values to test
    epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    print(f"Test Question: {test_question}")
    print(f"Epsilon values to test: {epsilon_values}")
    print("="*80)
    
    results = []
    
    # Open file to write results
    with open("testing-inferdpt.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("TESTING InferDPT perturb_sentence FUNCTION\n")
        f.write("="*80 + "\n")
        f.write(f"Test Question: {test_question}\n")
        f.write(f"Epsilon values: {epsilon_values}\n")
        f.write("="*80 + "\n\n")
        
        for epsilon in epsilon_values:
            print(f"\n{'='*60}")
            print(f"TESTING WITH EPSILON = {epsilon}")
            print(f"{'='*60}")
            print(f"Original Question: {test_question}")
            
            f.write(f"EPSILON = {epsilon}\n")
            f.write("="*60 + "\n")
            f.write(f"Original Question: {test_question}\n")
            
            try:
                # Apply InferDPT perturbation
                perturbed_question = perturb_sentence(test_question, epsilon)
                
                print(f"Perturbed Question: {perturbed_question}")
                f.write(f"Perturbed Question: {perturbed_question}\n")
                
                # Calculate similarity using sentence embeddings (optional)
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer('all-MiniLM-L6-v2')
                    
                    # Get embeddings
                    original_embedding = model.encode([test_question])[0]
                    perturbed_embedding = model.encode([perturbed_question])[0]
                    
                    # Calculate cosine similarity
                    similarity = np.dot(original_embedding, perturbed_embedding) / (
                        np.linalg.norm(original_embedding) * np.linalg.norm(perturbed_embedding)
                    )
                    
                    print(f"Similarity: {similarity:.4f}")
                    f.write(f"Similarity: {similarity:.4f}\n")
                    
                except Exception as e:
                    print(f"Could not calculate similarity: {e}")
                    f.write(f"Similarity calculation failed: {e}\n")
                
                results.append({
                    "epsilon": epsilon,
                    "original": test_question,
                    "perturbed": perturbed_question,
                    "similarity": similarity if 'similarity' in locals() else None
                })
                
            except Exception as e:
                error_msg = f"ERROR: {e}"
                print(f"✗ {error_msg}")
                f.write(f"ERROR: {e}\n")
                results.append({
                    "epsilon": epsilon,
                    "original": test_question,
                    "perturbed": error_msg,
                    "similarity": None
                })
            
            f.write("\n")
            print()
        
        # Write summary
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY OF RESULTS\n")
        f.write("="*80 + "\n")
        
        for result in results:
            f.write(f"\nEpsilon: {result['epsilon']}\n")
            f.write(f"Original: {result['original']}\n")
            f.write(f"Perturbed: {result['perturbed']}\n")
            if result['similarity'] is not None:
                f.write(f"Similarity: {result['similarity']:.4f}\n")
            f.write("-" * 40 + "\n")
        
        # Write analysis
        f.write("\n" + "="*80 + "\n")
        f.write("ANALYSIS\n")
        f.write("="*80 + "\n")
        
        successful_results = [r for r in results if not r['perturbed'].startswith('ERROR')]
        
        if successful_results:
            f.write(f"Successful perturbations: {len(successful_results)}/{len(results)}\n")
            
            # Analyze similarity trends
            similarities = [r['similarity'] for r in successful_results if r['similarity'] is not None]
            if similarities:
                f.write(f"Average similarity: {np.mean(similarities):.4f}\n")
                f.write(f"Min similarity: {np.min(similarities):.4f}\n")
                f.write(f"Max similarity: {np.max(similarities):.4f}\n")
                
                # Analyze epsilon vs similarity relationship
                f.write("\nEpsilon vs Similarity Analysis:\n")
                for result in successful_results:
                    if result['similarity'] is not None:
                        f.write(f"  ε={result['epsilon']}: similarity={result['similarity']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF TEST\n")
        f.write("="*80 + "\n")
    
    print(f"\nResults saved to testing-inferdpt.txt")
    
    return results

def test_single_epsilon(question, epsilon=1.0):
    """
    Test InferDPT perturbation with a single question and epsilon value.
    """
    print(f"\n{'='*60}")
    print(f"TESTING SINGLE EPSILON")
    print(f"{'='*60}")
    print(f"Question: {question}")
    print(f"Epsilon: {epsilon}")
    
    try:
        perturbed_question = perturb_sentence(question, epsilon)
        
        print(f"\nOriginal: {question}")
        print(f"Perturbed: {perturbed_question}")
        
        return perturbed_question
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Test with a single epsilon first
    print("Testing single epsilon...")
    test_single_epsilon("Were Scott Derrickson and Ed Wood of the same nationality?", epsilon=1.0)
    
    # Run full test with varying epsilon
    print("\nRunning full test with varying epsilon...")
    test_inferdpt_perturbation()
