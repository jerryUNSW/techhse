import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from dotenv import load_dotenv
from dp_sanitizer import load_sentence_bert, get_embedding
import utils
import json
from datetime import datetime

# Load environment variables
load_dotenv()

# Get API keys from environment variables
NEBIUS_API = os.getenv("NEBIUS")

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Initialize Nebius client
nebius_client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=NEBIUS_API)

def calculate_similarity_distribution(original_question, candidate_sentences, sbert_model):
    """
    Calculate similarity scores for all candidates against the original question.
    """
    similarities = []
    
    # Get original embedding
    original_embedding = get_embedding(sbert_model, original_question).cpu().numpy()
    
    # Calculate similarities for all candidates
    for candidate in candidate_sentences:
        candidate_emb = get_embedding(sbert_model, candidate).cpu().numpy()
        similarity = np.dot(original_embedding, candidate_emb) / (
            np.linalg.norm(original_embedding) * np.linalg.norm(candidate_emb)
        )
        similarities.append(similarity)
    
    return similarities

def test_phrase_dp_comparison():
    """
    Compare old and new Phrase DP implementations and generate similarity distributions.
    """
    print("="*80)
    print("COMPARING OLD vs NEW PHRASE DP IMPLEMENTATIONS")
    print("="*80)
    
    # Load the sentence BERT model
    print("Loading sentence BERT model...")
    sbert_model = load_sentence_bert()
    print("✓ Sentence BERT model loaded successfully")
    
    # Test questions (mix of different types)
    test_questions = [
        "What is the capital of France?",
        "A 45-year-old patient presents to Memorial Hospital with chest pain. What is the most likely diagnosis?",
        "Which company was founded by Steve Jobs in 1976?",
        "A 30-year-old individual visits a medical facility with fever and cough. What treatment should be considered?",
        "What is the largest planet in our solar system?",
        "Dr. Smith at Johns Hopkins University conducted a study on diabetes. What were the main findings?",
        "A person in New York City experiences shortness of breath. What could be the cause?",
        "What is the chemical symbol for gold?",
        "A patient at General Hospital has been diagnosed with hypertension. What lifestyle changes are recommended?",
        "Which programming language was created by Guido van Rossum?"
    ]
    
    epsilon = 1.0
    model_name = config["local_model"]
    
    print(f"\nUsing model: {model_name}")
    print(f"Testing {len(test_questions)} questions with epsilon = {epsilon}")
    print("="*80)
    
    # Results storage
    old_results = []
    new_results = []
    
    # Open file to write detailed results
    with open("phrase_dp_comparison_results.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("COMPARISON: OLD vs NEW PHRASE DP IMPLEMENTATIONS\n")
        f.write("="*80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Epsilon: {epsilon}\n")
        f.write(f"Number of questions: {len(test_questions)}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for i, question in enumerate(test_questions):
            print(f"\n{'='*60}")
            print(f"QUESTION {i+1}/{len(test_questions)}")
            print(f"{'='*60}")
            print(f"Original Question: {question}")
            
            f.write(f"QUESTION {i+1}/{len(test_questions)}\n")
            f.write("="*60 + "\n")
            f.write(f"Original Question: {question}\n\n")
            
            # Test OLD implementation
            print("\n--- Testing OLD Phrase DP ---")
            f.write("OLD PHRASE DP IMPLEMENTATION:\n")
            f.write("-" * 40 + "\n")
            
            try:
                # Get candidates from old implementation
                old_perturbed, old_candidates = utils.phrase_DP_perturbation_with_candidates(
                    nebius_client, 
                    model_name, 
                    question, 
                    epsilon, 
                    sbert_model
                )
                
                # Calculate similarities for old implementation
                old_similarities = calculate_similarity_distribution(question, old_candidates, sbert_model)
                
                print(f"Old Perturbed: {old_perturbed}")
                print(f"Old Similarities: {[f'{s:.3f}' for s in old_similarities]}")
                print(f"Old Similarity Range: {min(old_similarities):.3f} - {max(old_similarities):.3f}")
                print(f"Old Similarity Mean: {np.mean(old_similarities):.3f}")
                print(f"Old Similarity Std: {np.std(old_similarities):.3f}")
                
                f.write(f"Perturbed: {old_perturbed}\n")
                f.write(f"Number of candidates: {len(old_candidates)}\n")
                f.write(f"Similarity range: {min(old_similarities):.3f} - {max(old_similarities):.3f}\n")
                f.write(f"Similarity mean: {np.mean(old_similarities):.3f}\n")
                f.write(f"Similarity std: {np.std(old_similarities):.3f}\n")
                f.write("All candidates with similarities:\n")
                
                for j, (candidate, sim) in enumerate(zip(old_candidates, old_similarities), 1):
                    f.write(f"  {j:2d}. [{sim:.3f}] {candidate}\n")
                
                old_results.append({
                    "question": question,
                    "perturbed": old_perturbed,
                    "candidates": old_candidates,
                    "similarities": old_similarities,
                    "similarity_range": max(old_similarities) - min(old_similarities),
                    "similarity_mean": np.mean(old_similarities),
                    "similarity_std": np.std(old_similarities)
                })
                
            except Exception as e:
                error_msg = f"ERROR: {e}"
                print(f"✗ {error_msg}")
                f.write(f"ERROR: {e}\n")
                old_results.append({
                    "question": question,
                    "perturbed": error_msg,
                    "candidates": [],
                    "similarities": [],
                    "similarity_range": 0,
                    "similarity_mean": 0,
                    "similarity_std": 0
                })
            
            # Test NEW implementation
            print("\n--- Testing NEW Diverse Phrase DP ---")
            f.write("\nNEW DIVERSE PHRASE DP IMPLEMENTATION:\n")
            f.write("-" * 40 + "\n")
            
            try:
                # Get candidates from new implementation
                new_perturbed, new_candidates = utils.phrase_DP_perturbation_with_candidates_diverse(
                    nebius_client, 
                    model_name, 
                    question, 
                    epsilon, 
                    sbert_model
                )
                
                # Calculate similarities for new implementation
                new_similarities = calculate_similarity_distribution(question, new_candidates, sbert_model)
                
                print(f"New Perturbed: {new_perturbed}")
                print(f"New Similarities: {[f'{s:.3f}' for s in new_similarities]}")
                print(f"New Similarity Range: {min(new_similarities):.3f} - {max(new_similarities):.3f}")
                print(f"New Similarity Mean: {np.mean(new_similarities):.3f}")
                print(f"New Similarity Std: {np.std(new_similarities):.3f}")
                
                f.write(f"Perturbed: {new_perturbed}\n")
                f.write(f"Number of candidates: {len(new_candidates)}\n")
                f.write(f"Similarity range: {min(new_similarities):.3f} - {max(new_similarities):.3f}\n")
                f.write(f"Similarity mean: {np.mean(new_similarities):.3f}\n")
                f.write(f"Similarity std: {np.std(new_similarities):.3f}\n")
                f.write("All candidates with similarities:\n")
                
                for j, (candidate, sim) in enumerate(zip(new_candidates, new_similarities), 1):
                    f.write(f"  {j:2d}. [{sim:.3f}] {candidate}\n")
                
                new_results.append({
                    "question": question,
                    "perturbed": new_perturbed,
                    "candidates": new_candidates,
                    "similarities": new_similarities,
                    "similarity_range": max(new_similarities) - min(new_similarities),
                    "similarity_mean": np.mean(new_similarities),
                    "similarity_std": np.std(new_similarities)
                })
                
            except Exception as e:
                error_msg = f"ERROR: {e}"
                print(f"✗ {error_msg}")
                f.write(f"ERROR: {e}\n")
                new_results.append({
                    "question": question,
                    "perturbed": error_msg,
                    "candidates": [],
                    "similarities": [],
                    "similarity_range": 0,
                    "similarity_mean": 0,
                    "similarity_std": 0
                })
            
            f.write("\n" + "="*60 + "\n\n")
            print()
        
        # Write summary comparison
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY COMPARISON\n")
        f.write("="*80 + "\n")
        
        # Calculate overall statistics
        old_ranges = [r["similarity_range"] for r in old_results if r["similarity_range"] > 0]
        new_ranges = [r["similarity_range"] for r in new_results if r["similarity_range"] > 0]
        
        old_means = [r["similarity_mean"] for r in old_results if r["similarity_mean"] > 0]
        new_means = [r["similarity_mean"] for r in new_results if r["similarity_mean"] > 0]
        
        old_stds = [r["similarity_std"] for r in old_results if r["similarity_std"] > 0]
        new_stds = [r["similarity_std"] for r in new_results if r["similarity_std"] > 0]
        
        f.write(f"OLD Implementation:\n")
        f.write(f"  Average similarity range: {np.mean(old_ranges):.3f} ± {np.std(old_ranges):.3f}\n")
        f.write(f"  Average similarity mean: {np.mean(old_means):.3f} ± {np.std(old_means):.3f}\n")
        f.write(f"  Average similarity std: {np.mean(old_stds):.3f} ± {np.std(old_stds):.3f}\n")
        
        f.write(f"\nNEW Implementation:\n")
        f.write(f"  Average similarity range: {np.mean(new_ranges):.3f} ± {np.std(new_ranges):.3f}\n")
        f.write(f"  Average similarity mean: {np.mean(new_means):.3f} ± {np.std(new_means):.3f}\n")
        f.write(f"  Average similarity std: {np.mean(new_stds):.3f} ± {np.std(new_stds):.3f}\n")
        
        f.write(f"\nIMPROVEMENT:\n")
        f.write(f"  Range improvement: {((np.mean(new_ranges) - np.mean(old_ranges)) / np.mean(old_ranges) * 100):+.1f}%\n")
        f.write(f"  Std improvement: {((np.mean(new_stds) - np.mean(old_stds)) / np.mean(old_stds) * 100):+.1f}%\n")
    
    # Generate similarity distribution plots
    generate_similarity_plots(old_results, new_results)
    
    # Save results to JSON for further analysis
    save_results_to_json(old_results, new_results)
    
    print(f"\nResults saved to phrase_dp_comparison_results.txt")
    print(f"Plots saved to similarity_distributions.png")
    print(f"JSON results saved to phrase_dp_comparison.json")
    
    return old_results, new_results

def generate_similarity_plots(old_results, new_results):
    """
    Generate plots comparing similarity distributions between old and new implementations.
    """
    # Collect all similarities
    old_similarities = []
    new_similarities = []
    
    for result in old_results:
        if result["similarities"]:
            old_similarities.extend(result["similarities"])
    
    for result in new_results:
        if result["similarities"]:
            new_similarities.extend(result["similarities"])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Phrase DP: Old vs New Implementation Comparison', fontsize=16, fontweight='bold')
    
    # Plot 1: Histogram comparison
    axes[0, 0].hist(old_similarities, bins=20, alpha=0.7, label='Old Implementation', color='red', density=True)
    axes[0, 0].hist(new_similarities, bins=20, alpha=0.7, label='New Implementation', color='blue', density=True)
    axes[0, 0].set_xlabel('Similarity Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Similarity Score Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Box plot comparison
    data_to_plot = [old_similarities, new_similarities]
    box_plot = axes[0, 1].boxplot(data_to_plot, labels=['Old', 'New'], patch_artist=True)
    box_plot['boxes'][0].set_facecolor('red')
    box_plot['boxes'][1].set_facecolor('blue')
    axes[0, 1].set_ylabel('Similarity Score')
    axes[0, 1].set_title('Similarity Score Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Range comparison per question
    old_ranges = [r["similarity_range"] for r in old_results if r["similarity_range"] > 0]
    new_ranges = [r["similarity_range"] for r in new_results if r["similarity_range"] > 0]
    
    x_pos = range(len(old_ranges))
    axes[1, 0].bar([x - 0.2 for x in x_pos], old_ranges, 0.4, label='Old Implementation', color='red', alpha=0.7)
    axes[1, 0].bar([x + 0.2 for x in x_pos], new_ranges, 0.4, label='New Implementation', color='blue', alpha=0.7)
    axes[1, 0].set_xlabel('Question Index')
    axes[1, 0].set_ylabel('Similarity Range')
    axes[1, 0].set_title('Similarity Range per Question')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Statistics comparison
    stats_data = {
        'Metric': ['Mean Range', 'Mean Std', 'Overall Mean', 'Overall Std'],
        'Old': [
            np.mean(old_ranges),
            np.mean([r["similarity_std"] for r in old_results if r["similarity_std"] > 0]),
            np.mean(old_similarities),
            np.std(old_similarities)
        ],
        'New': [
            np.mean(new_ranges),
            np.mean([r["similarity_std"] for r in new_results if r["similarity_std"] > 0]),
            np.mean(new_similarities),
            np.std(new_similarities)
        ]
    }
    
    x_pos = range(len(stats_data['Metric']))
    axes[1, 1].bar([x - 0.2 for x in x_pos], stats_data['Old'], 0.4, label='Old Implementation', color='red', alpha=0.7)
    axes[1, 1].bar([x + 0.2 for x in x_pos], stats_data['New'], 0.4, label='New Implementation', color='blue', alpha=0.7)
    axes[1, 1].set_xlabel('Statistics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Overall Statistics Comparison')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(stats_data['Metric'], rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('similarity_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_json(old_results, new_results):
    """
    Save results to JSON file for further analysis.
    """
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        return obj
    
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "old_implementation": [],
        "new_implementation": []
    }
    
    for result in old_results:
        converted_result = {}
        for key, value in result.items():
            converted_result[key] = convert_numpy(value)
        results_data["old_implementation"].append(converted_result)
    
    for result in new_results:
        converted_result = {}
        for key, value in result.items():
            converted_result[key] = convert_numpy(value)
        results_data["new_implementation"].append(converted_result)
    
    with open("phrase_dp_comparison.json", "w") as f:
        json.dump(results_data, f, indent=2)

if __name__ == "__main__":
    print("Starting Phrase DP comparison test...")
    old_results, new_results = test_phrase_dp_comparison()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    
    # Print summary statistics
    old_ranges = [r["similarity_range"] for r in old_results if r["similarity_range"] > 0]
    new_ranges = [r["similarity_range"] for r in new_results if r["similarity_range"] > 0]
    
    print(f"OLD Implementation - Average similarity range: {np.mean(old_ranges):.3f}")
    print(f"NEW Implementation - Average similarity range: {np.mean(new_ranges):.3f}")
    print(f"Improvement: {((np.mean(new_ranges) - np.mean(old_ranges)) / np.mean(old_ranges) * 100):+.1f}%")
