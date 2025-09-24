#!/usr/bin/env python3
"""
Test script to compare old vs new Phrase DP across different epsilon values.
Tests epsilon values: 0.5, 1.0, 1.5, 2.0
Analyzes how the exponential mechanism selects different candidates based on privacy budget.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import utils
from dp_sanitizer import get_embedding
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def load_test_questions():
    """Load a subset of test questions for epsilon comparison."""
    questions = [
        "What is the capital of France?",
        "Who wrote the novel '1984'?",
        "What is the chemical symbol for gold?",
        "In which year did World War II end?",
        "What is the largest planet in our solar system?"
    ]
    return questions

def calculate_similarity(original, candidate, sbert_model):
    """Calculate cosine similarity between original and candidate."""
    original_embedding = get_embedding(sbert_model, original).cpu().numpy()
    candidate_embedding = get_embedding(sbert_model, candidate).cpu().numpy()
    similarity = np.dot(original_embedding, candidate_embedding) / (
        np.linalg.norm(original_embedding) * np.linalg.norm(candidate_embedding)
    )
    return float(similarity)

def test_epsilon_comparison():
    """Test both methods across different epsilon values."""
    print("🔬 Testing Epsilon Comparison: Old vs New Phrase DP")
    print("=" * 60)
    
    # Setup
    load_dotenv()
    nebius_api_key = os.getenv("NEBIUS")
    if not nebius_api_key:
        raise ValueError("NEBIUS API key not found in environment variables")
    
    nebius_client = OpenAI(api_key=nebius_api_key, base_url="https://api.studio.nebius.ai/v1/")
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    
    # Load SBERT model
    print("Loading SBERT model...")
    from sentence_transformers import SentenceTransformer
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test questions
    test_questions = load_test_questions()
    epsilon_values = [0.5, 1.0, 1.5, 2.0]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'epsilon_values': epsilon_values,
        'questions': []
    }
    
    print(f"Testing {len(test_questions)} questions across {len(epsilon_values)} epsilon values")
    print(f"Total tests: {len(test_questions) * len(epsilon_values) * 2} (old + new methods)")
    print()
    
    for q_idx, question in enumerate(test_questions):
        print(f"📝 Question {q_idx + 1}/{len(test_questions)}: {question}")
        
        question_results = {
            'question_text': question,
            'question_index': q_idx,
            'epsilon_tests': []
        }
        
        for eps_idx, epsilon in enumerate(epsilon_values):
            print(f"  🔄 Epsilon {epsilon} ({eps_idx + 1}/{len(epsilon_values)})")
            
            epsilon_results = {
                'epsilon': epsilon,
                'old_method': {},
                'new_method': {}
            }
            
            try:
                # --- Test OLD Method ---
                print(f"    🔴 Testing OLD method...")
                old_perturbed, old_candidates = utils.phrase_DP_perturbation_with_candidates(
                    nebius_client, model_name, question, epsilon, sbert_model
                )
                old_similarity = calculate_similarity(question, old_perturbed, sbert_model)
                
                epsilon_results['old_method'] = {
                    'perturbed_question': old_perturbed,
                    'similarity_to_original': old_similarity,
                    'num_candidates': len(old_candidates),
                    'candidate_similarities': [calculate_similarity(question, c, sbert_model) for c in old_candidates]
                }
                
                # --- Test NEW Method ---
                print(f"    🔵 Testing NEW method...")
                new_perturbed, new_candidates = utils.phrase_DP_perturbation_with_candidates_diverse(
                    nebius_client, model_name, question, epsilon, sbert_model
                )
                new_similarity = calculate_similarity(question, new_perturbed, sbert_model)
                
                epsilon_results['new_method'] = {
                    'perturbed_question': new_perturbed,
                    'similarity_to_original': new_similarity,
                    'num_candidates': len(new_candidates),
                    'candidate_similarities': [calculate_similarity(question, c, sbert_model) for c in new_candidates]
                }
                
                # Print results for this epsilon
                print(f"      Old: similarity={old_similarity:.3f}, candidates={len(old_candidates)}")
                print(f"      New: similarity={new_similarity:.3f}, candidates={len(new_candidates)}")
                
            except Exception as e:
                print(f"    ❌ Error testing epsilon {epsilon}: {e}")
                epsilon_results['error'] = str(e)
            
            question_results['epsilon_tests'].append(epsilon_results)
            print()
        
        results['questions'].append(question_results)
        print("-" * 60)
    
    # Save results
    save_epsilon_results(results)
    
    # Generate analysis plots
    generate_epsilon_analysis_plots(results)
    
    return results

def save_epsilon_results(results):
    """Save epsilon comparison results to JSON file."""
    def convert_numpy(obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.float64):
            return float(obj)
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.int64):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Convert numpy types
    results_json = json.loads(json.dumps(results, default=convert_numpy))
    
    filename = f"epsilon_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {filename}")

def generate_epsilon_analysis_plots(results):
    """Generate comprehensive analysis plots for epsilon comparison."""
    print("📊 Generating epsilon analysis plots...")
    
    # Extract data for plotting
    epsilon_values = results['epsilon_values']
    questions = results['questions']
    
    # Prepare data arrays
    old_similarities = []
    new_similarities = []
    old_candidate_ranges = []
    new_candidate_ranges = []
    
    for question_data in questions:
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                # Selected similarities
                old_sim = eps_test['old_method']['similarity_to_original']
                new_sim = eps_test['new_method']['similarity_to_original']
                old_similarities.append(old_sim)
                new_similarities.append(new_sim)
                
                # Candidate diversity (range)
                old_candidates = eps_test['old_method']['candidate_similarities']
                new_candidates = eps_test['new_method']['candidate_similarities']
                old_candidate_ranges.append(max(old_candidates) - min(old_candidates) if old_candidates else 0)
                new_candidate_ranges.append(max(new_candidates) - min(new_candidates) if new_candidates else 0)
    
    # Create comprehensive analysis plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Epsilon Comparison: Old vs New Phrase DP Methods', fontsize=16, fontweight='bold')
    
    # Plot 1: Selected Similarity vs Epsilon
    axes[0, 0].scatter(epsilon_values * len(questions), old_similarities, 
                      color='red', alpha=0.7, label='Old Method', s=60)
    axes[0, 0].scatter(epsilon_values * len(questions), new_similarities, 
                      color='blue', alpha=0.7, label='New Method', s=60)
    axes[0, 0].set_xlabel('Epsilon Value')
    axes[0, 0].set_ylabel('Selected Similarity')
    axes[0, 0].set_title('Selected Similarity vs Epsilon')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Candidate Diversity vs Epsilon
    axes[0, 1].scatter(epsilon_values * len(questions), old_candidate_ranges, 
                      color='red', alpha=0.7, label='Old Method', s=60)
    axes[0, 1].scatter(epsilon_values * len(questions), new_candidate_ranges, 
                      color='blue', alpha=0.7, label='New Method', s=60)
    axes[0, 1].set_xlabel('Epsilon Value')
    axes[0, 1].set_ylabel('Candidate Similarity Range')
    axes[0, 1].set_title('Candidate Diversity vs Epsilon')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Similarity Distribution Comparison
    axes[1, 0].hist(old_similarities, bins=15, alpha=0.6, color='red', label='Old Method', density=True)
    axes[1, 0].hist(new_similarities, bins=15, alpha=0.6, color='blue', label='New Method', density=True)
    axes[1, 0].set_xlabel('Selected Similarity')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution of Selected Similarities')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Epsilon Sensitivity Analysis
    # Calculate mean similarity for each epsilon
    old_eps_means = []
    new_eps_means = []
    
    for eps in epsilon_values:
        old_eps_sims = [old_similarities[i] for i in range(len(epsilon_values) * len(questions)) 
                       if (i // len(questions)) == epsilon_values.index(eps)]
        new_eps_sims = [new_similarities[i] for i in range(len(epsilon_values) * len(questions)) 
                       if (i // len(questions)) == epsilon_values.index(eps)]
        
        old_eps_means.append(np.mean(old_eps_sims) if old_eps_sims else 0)
        new_eps_means.append(np.mean(new_eps_sims) if new_eps_sims else 0)
    
    axes[1, 1].plot(epsilon_values, old_eps_means, 'ro-', label='Old Method', linewidth=2, markersize=8)
    axes[1, 1].plot(epsilon_values, new_eps_means, 'bo-', label='New Method', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Epsilon Value')
    axes[1, 1].set_ylabel('Mean Selected Similarity')
    axes[1, 1].set_title('Epsilon Sensitivity: Mean Similarity')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('epsilon_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate detailed per-question analysis
    generate_per_question_epsilon_plots(results)
    
    print("📊 Analysis plots generated:")
    print("  - epsilon_comparison_analysis.png (overall analysis)")
    print("  - question_*_epsilon_analysis.png (per-question analysis)")

def generate_per_question_epsilon_plots(results):
    """Generate individual plots for each question showing epsilon behavior."""
    epsilon_values = results['epsilon_values']
    
    for q_idx, question_data in enumerate(results['questions']):
        question_text = question_data['question_text']
        
        # Extract data for this question
        epsilons = []
        old_similarities = []
        new_similarities = []
        old_candidate_ranges = []
        new_candidate_ranges = []
        
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                epsilons.append(eps_test['epsilon'])
                old_similarities.append(eps_test['old_method']['similarity_to_original'])
                new_similarities.append(eps_test['new_method']['similarity_to_original'])
                
                old_candidates = eps_test['old_method']['candidate_similarities']
                new_candidates = eps_test['new_method']['candidate_similarities']
                old_candidate_ranges.append(max(old_candidates) - min(old_candidates) if old_candidates else 0)
                new_candidate_ranges.append(max(new_candidates) - min(new_candidates) if new_candidates else 0)
        
        # Create per-question plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Question {q_idx + 1}: Epsilon Analysis\n"{question_text}"', fontsize=12, fontweight='bold')
        
        # Plot 1: Selected Similarity vs Epsilon
        axes[0].plot(epsilons, old_similarities, 'ro-', label='Old Method', linewidth=2, markersize=8)
        axes[0].plot(epsilons, new_similarities, 'bo-', label='New Method', linewidth=2, markersize=8)
        axes[0].set_xlabel('Epsilon Value')
        axes[0].set_ylabel('Selected Similarity')
        axes[0].set_title('Selected Similarity vs Epsilon')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1)
        
        # Plot 2: Candidate Diversity vs Epsilon
        axes[1].plot(epsilons, old_candidate_ranges, 'ro-', label='Old Method', linewidth=2, markersize=8)
        axes[1].plot(epsilons, new_candidate_ranges, 'bo-', label='New Method', linewidth=2, markersize=8)
        axes[1].set_xlabel('Epsilon Value')
        axes[1].set_ylabel('Candidate Similarity Range')
        axes[1].set_title('Candidate Diversity vs Epsilon')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'question_{q_idx + 1:02d}_epsilon_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    try:
        results = test_epsilon_comparison()
        print("\n✅ Epsilon comparison test completed successfully!")
        print(f"📊 Generated analysis plots for {len(results['questions'])} questions")
        print(f"🔬 Tested {len(results['epsilon_values'])} epsilon values: {results['epsilon_values']}")
        
    except Exception as e:
        print(f"\n❌ Error during epsilon comparison test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
