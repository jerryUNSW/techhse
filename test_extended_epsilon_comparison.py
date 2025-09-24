#!/usr/bin/env python3
"""
Extended epsilon comparison test with additional epsilon values (2.5, 3.0).
Building on the scaled results to complete the epsilon sensitivity analysis.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import utils
from dp_sanitizer import get_embedding, differentially_private_replacement
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

def send_completion_email(results_file, plot_file, test_duration):
    """Send email notification when the extended epsilon test is completed."""
    try:
        # Load email configuration
        with open('email_config.json', 'r') as f:
            email_config = json.load(f)
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['from_email']
        msg['To'] = email_config['to_email']
        msg['Subject'] = "🔬 Extended Epsilon Comparison Test Completed"
        
        # Email body
        body = f"""
Extended Epsilon Comparison Test Completed Successfully!

📊 Test Results:
- Test Duration: {test_duration}
- Results File: {results_file}
- Analysis Plot: {plot_file}
- Epsilon Values: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
- Questions Tested: 10
- Total Tests: 120 (10 questions × 6 epsilons × 2 methods)

🎯 Key Findings:
- Extended epsilon range to include 2.5 and 3.0
- Comprehensive 9-panel analysis generated
- Both old and new Phrase DP methods tested
- 100 candidates per method per test

📈 Next Steps:
1. Review the comprehensive_epsilon_analysis.png plot
2. Analyze epsilon sensitivity across the full range
3. Check if higher epsilon values show better sensitivity
4. Compare with previous results

The test has completed and all results are saved. You can now analyze the extended epsilon behavior.

Best regards,
PrivGuard-QA System
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach the analysis plot if it exists
        if os.path.exists(plot_file):
            with open(plot_file, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {plot_file}'
                )
                msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['from_email'], email_config['password'])
        text = msg.as_string()
        server.sendmail(email_config['from_email'], email_config['to_email'], text)
        server.quit()
        
        print(f"📧 Completion email sent to {email_config['to_email']}")
        
    except Exception as e:
        print(f"❌ Failed to send completion email: {e}")

def load_test_questions():
    """Load 10 diverse test questions for extended epsilon comparison."""
    questions = [
        "What is the capital of France?",
        "What is the largest country in the world?",
        "Which ocean is the largest?",
        "What is the longest river in the world?",
        "In which year did World War II end?",
        "Who was the first president of the United States?",
        "When did the Berlin Wall fall?",
        "What year did the Titanic sink?",
        "What is the chemical symbol for gold?",
        "What is the speed of light?"
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

def test_extended_epsilon_comparison():
    """Test both methods across extended epsilon values including 2.5 and 3.0."""
    start_time = datetime.now()
    print("🔬 Testing Extended Epsilon Comparison: Old vs New Phrase DP")
    print("=" * 80)
    
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
    
    # Test questions and extended epsilon values
    test_questions = load_test_questions()
    epsilon_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'epsilon_values': epsilon_values,
        'test_parameters': {
            'old_method': {'api_calls': 10, 'candidates_per_call': 10, 'total_candidates': 100},
            'new_method': {'api_calls': 5, 'candidates_per_call': 20, 'total_candidates': 100},
            'total_questions': len(test_questions)
        },
        'questions': []
    }
    
    print(f"Testing {len(test_questions)} questions across {len(epsilon_values)} epsilon values")
    print(f"Epsilon values: {epsilon_values}")
    print(f"Old method: {results['test_parameters']['old_method']['api_calls']} API calls × {results['test_parameters']['old_method']['candidates_per_call']} candidates = {results['test_parameters']['old_method']['total_candidates']} total candidates")
    print(f"New method: {results['test_parameters']['new_method']['api_calls']} API calls × {results['test_parameters']['new_method']['candidates_per_call']} candidates = {results['test_parameters']['new_method']['total_candidates']} total candidates")
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
                # --- Test OLD Method (10 API calls × 10 candidates) ---
                print(f"    🔴 Testing OLD method (10×10=100 candidates)...")
                # Generate candidates with 10 API calls
                old_candidates = utils.generate_sentence_replacements_with_nebius(
                    nebius_client, model_name, question, num_return_sequences=10, num_api_calls=10
                )
                # Apply exponential mechanism
                candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in old_candidates}
                old_perturbed = differentially_private_replacement(
                    target_phrase=question,
                    epsilon=epsilon,
                    candidate_phrases=old_candidates,
                    candidate_embeddings=candidate_embeddings,
                    sbert_model=sbert_model
                )
                old_similarity = calculate_similarity(question, old_perturbed, sbert_model)
                
                epsilon_results['old_method'] = {
                    'perturbed_question': old_perturbed,
                    'similarity_to_original': old_similarity,
                    'num_candidates': len(old_candidates),
                    'candidate_similarities': [calculate_similarity(question, c, sbert_model) for c in old_candidates]
                }
                
                # --- Test NEW Method (5 API calls × 20 candidates) ---
                print(f"    🔵 Testing NEW method (5×20=100 candidates)...")
                # Generate candidates with 20 per API call
                new_candidates = utils.generate_sentence_replacements_with_nebius_diverse(
                    nebius_client, model_name, question, num_return_sequences=20, num_api_calls=5
                )
                # Apply exponential mechanism
                candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in new_candidates}
                new_perturbed = differentially_private_replacement(
                    target_phrase=question,
                    epsilon=epsilon,
                    candidate_phrases=new_candidates,
                    candidate_embeddings=candidate_embeddings,
                    sbert_model=sbert_model
                )
                new_similarity = calculate_similarity(question, new_perturbed, sbert_model)
                
                epsilon_results['new_method'] = {
                    'perturbed_question': new_perturbed,
                    'similarity_to_original': new_similarity,
                    'num_candidates': len(new_candidates),
                    'candidate_similarities': [calculate_similarity(question, c, sbert_model) for c in new_candidates]
                }
                
                # Print results for this epsilon
                old_range = max(epsilon_results['old_method']['candidate_similarities']) - min(epsilon_results['old_method']['candidate_similarities'])
                new_range = max(epsilon_results['new_method']['candidate_similarities']) - min(epsilon_results['new_method']['candidate_similarities'])
                
                print(f"      Old: similarity={old_similarity:.3f}, candidates={len(old_candidates)}, range={old_range:.3f}")
                print(f"      New: similarity={new_similarity:.3f}, candidates={len(new_candidates)}, range={new_range:.3f}")
                
            except Exception as e:
                print(f"    ❌ Error testing epsilon {epsilon}: {e}")
                epsilon_results['error'] = str(e)
            
            question_results['epsilon_tests'].append(epsilon_results)
            print()
        
        results['questions'].append(question_results)
        print("-" * 80)
    
    # Save results
    results_file = save_extended_epsilon_results(results)
    
    # Generate comprehensive analysis plots
    plot_file = generate_comprehensive_epsilon_analysis_plots(results)
    
    # Calculate test duration
    end_time = datetime.now()
    test_duration = str(end_time - start_time)
    
    # Send completion email
    send_completion_email(results_file, plot_file, test_duration)
    
    return results

def save_extended_epsilon_results(results):
    """Save extended epsilon comparison results to JSON file."""
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
    
    filename = f"extended_epsilon_comparison_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {filename}")
    return filename

def generate_comprehensive_epsilon_analysis_plots(results):
    """Generate comprehensive analysis plots for extended epsilon comparison."""
    print("📊 Generating comprehensive epsilon analysis plots...")
    
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
    
    # Create comprehensive 9-panel analysis plot
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Comprehensive Epsilon Analysis: Old vs New Phrase DP Methods\n(Extended Epsilon Range: 0.5-3.0)', fontsize=16, fontweight='bold')
    
    # Plot 1: Selected Similarity vs Epsilon (Scatter)
    axes[0, 0].scatter(epsilon_values * len(questions), old_similarities, 
                      color='red', alpha=0.7, label='Old Method', s=60)
    axes[0, 0].scatter(epsilon_values * len(questions), new_similarities, 
                      color='blue', alpha=0.7, label='New Method', s=60)
    axes[0, 0].set_xlabel('Epsilon Value')
    axes[0, 0].set_ylabel('Selected Similarity')
    axes[0, 0].set_title('Selected Similarity vs Epsilon (Scatter)')
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
    axes[0, 2].hist(old_similarities, bins=20, alpha=0.6, color='red', label='Old Method', density=True)
    axes[0, 2].hist(new_similarities, bins=20, alpha=0.6, color='blue', label='New Method', density=True)
    axes[0, 2].set_xlabel('Selected Similarity')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Distribution of Selected Similarities')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Epsilon Sensitivity Analysis (Mean Similarity)
    old_eps_means = []
    new_eps_means = []
    
    for eps in epsilon_values:
        old_eps_sims = [old_similarities[i] for i in range(len(epsilon_values) * len(questions)) 
                       if (i // len(questions)) == epsilon_values.index(eps)]
        new_eps_sims = [new_similarities[i] for i in range(len(epsilon_values) * len(questions)) 
                       if (i // len(questions)) == epsilon_values.index(eps)]
        
        old_eps_means.append(np.mean(old_eps_sims) if old_eps_sims else 0)
        new_eps_means.append(np.mean(new_eps_sims) if new_eps_sims else 0)
    
    axes[1, 0].plot(epsilon_values, old_eps_means, 'ro-', label='Old Method', linewidth=2, markersize=8)
    axes[1, 0].plot(epsilon_values, new_eps_means, 'bo-', label='New Method', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Epsilon Value')
    axes[1, 0].set_ylabel('Mean Selected Similarity')
    axes[1, 0].set_title('Epsilon Sensitivity: Mean Similarity')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Epsilon Sensitivity Analysis (Std Similarity)
    old_eps_stds = []
    new_eps_stds = []
    
    for eps in epsilon_values:
        old_eps_sims = [old_similarities[i] for i in range(len(epsilon_values) * len(questions)) 
                       if (i // len(questions)) == epsilon_values.index(eps)]
        new_eps_sims = [new_similarities[i] for i in range(len(epsilon_values) * len(questions)) 
                       if (i // len(questions)) == epsilon_values.index(eps)]
        
        old_eps_stds.append(np.std(old_eps_sims) if old_eps_sims else 0)
        new_eps_stds.append(np.std(new_eps_sims) if new_eps_sims else 0)
    
    axes[1, 1].plot(epsilon_values, old_eps_stds, 'ro-', label='Old Method', linewidth=2, markersize=8)
    axes[1, 1].plot(epsilon_values, new_eps_stds, 'bo-', label='New Method', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Epsilon Value')
    axes[1, 1].set_ylabel('Std Dev of Selected Similarity')
    axes[1, 1].set_title('Epsilon Sensitivity: Similarity Variability')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Candidate Range vs Epsilon
    old_range_means = []
    new_range_means = []
    
    for eps in epsilon_values:
        old_eps_ranges = [old_candidate_ranges[i] for i in range(len(epsilon_values) * len(questions)) 
                         if (i // len(questions)) == epsilon_values.index(eps)]
        new_eps_ranges = [new_candidate_ranges[i] for i in range(len(epsilon_values) * len(questions)) 
                         if (i // len(questions)) == epsilon_values.index(eps)]
        
        old_range_means.append(np.mean(old_eps_ranges) if old_eps_ranges else 0)
        new_range_means.append(np.mean(new_eps_ranges) if new_eps_ranges else 0)
    
    axes[1, 2].plot(epsilon_values, old_range_means, 'ro-', label='Old Method', linewidth=2, markersize=8)
    axes[1, 2].plot(epsilon_values, new_range_means, 'bo-', label='New Method', linewidth=2, markersize=8)
    axes[1, 2].set_xlabel('Epsilon Value')
    axes[1, 2].set_ylabel('Mean Candidate Range')
    axes[1, 2].set_title('Candidate Diversity vs Epsilon')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Plot 7: Correlation Analysis
    eps_array = []
    old_sim_array = []
    new_sim_array = []
    
    for question_data in questions:
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                eps_array.append(eps_test['epsilon'])
                old_sim_array.append(eps_test['old_method']['similarity_to_original'])
                new_sim_array.append(eps_test['new_method']['similarity_to_original'])
    
    old_corr = np.corrcoef(eps_array, old_sim_array)[0, 1]
    new_corr = np.corrcoef(eps_array, new_sim_array)[0, 1]
    
    axes[2, 0].bar(['Old Method', 'New Method'], [old_corr, new_corr], 
                   color=['red', 'blue'], alpha=0.7)
    axes[2, 0].set_ylabel('Epsilon-Similarity Correlation')
    axes[2, 0].set_title('Epsilon Sensitivity: Correlation Strength')
    axes[2, 0].grid(True, alpha=0.3)
    axes[2, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 8: Expected vs Actual Similarity Ranges
    expected_ranges = {
        0.5: (0.2, 0.4),
        1.0: (0.4, 0.6),
        1.5: (0.6, 0.8),
        2.0: (0.8, 0.9),
        2.5: (0.85, 0.95),
        3.0: (0.9, 0.98)
    }
    
    expected_lows = [expected_ranges[eps][0] for eps in epsilon_values]
    expected_highs = [expected_ranges[eps][1] for eps in epsilon_values]
    
    axes[2, 1].fill_between(epsilon_values, expected_lows, expected_highs, 
                           alpha=0.3, color='green', label='Expected Range')
    axes[2, 1].plot(epsilon_values, old_eps_means, 'ro-', label='Old Actual', linewidth=2, markersize=8)
    axes[2, 1].plot(epsilon_values, new_eps_means, 'bo-', label='New Actual', linewidth=2, markersize=8)
    axes[2, 1].set_xlabel('Epsilon Value')
    axes[2, 1].set_ylabel('Similarity')
    axes[2, 1].set_title('Expected vs Actual Similarity Ranges')
    axes[2, 1].legend()
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 9: Summary Statistics
    axes[2, 2].axis('off')
    summary_text = f"""
SUMMARY STATISTICS

Dataset:
• Questions: {len(questions)}
• Epsilon Values: {epsilon_values}
• Total Tests: {len(questions) * len(epsilon_values) * 2}

Selected Similarity:
• Old Method: {np.mean(old_similarities):.3f} ± {np.std(old_similarities):.3f}
• New Method: {np.mean(new_similarities):.3f} ± {np.std(new_similarities):.3f}

Candidate Diversity:
• Old Method: {np.mean(old_candidate_ranges):.3f} ± {np.std(old_candidate_ranges):.3f}
• New Method: {np.mean(new_candidate_ranges):.3f} ± {np.std(new_candidate_ranges):.3f}

Epsilon Sensitivity:
• Old Correlation: {old_corr:.3f}
• New Correlation: {new_corr:.3f}
• Expected: >0.5 (strong positive)

Diversity Improvement:
• {((np.mean(new_candidate_ranges) - np.mean(old_candidate_ranges)) / np.mean(old_candidate_ranges) * 100):+.1f}%
    """
    axes[2, 2].text(0.05, 0.95, summary_text, transform=axes[2, 2].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plot_filename = 'comprehensive_epsilon_analysis.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("📊 Comprehensive analysis plots generated:")
    print(f"  - {plot_filename} (9-panel analysis)")
    return plot_filename

if __name__ == "__main__":
    try:
        results = test_extended_epsilon_comparison()
        print("\n✅ Extended epsilon comparison test completed successfully!")
        print(f"📊 Generated comprehensive analysis plots for {len(results['questions'])} questions")
        print(f"🔬 Tested {len(results['epsilon_values'])} epsilon values: {results['epsilon_values']}")
        print(f"📈 Total candidates per test: {results['test_parameters']['old_method']['total_candidates']} (old) vs {results['test_parameters']['new_method']['total_candidates']} (new)")
        
    except Exception as e:
        print(f"\n❌ Error during extended epsilon comparison test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
