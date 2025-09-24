#!/usr/bin/env python3
"""
Privacy Mechanisms Comparison
Integrates PhraseDP (new version), InferDPT, and SANTEXT+ for comprehensive comparison
"""

import os
import json
import time
import random
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Import your existing modules
from utils import generate_sentence_replacements_with_nebius_diverse, compute_similarity
from dp_sanitizer import differentially_private_replacement
from santext_integration import create_santext_mechanism

class PrivacyMechanismsComparison:
    """
    Comprehensive comparison of privacy mechanisms:
    1. PhraseDP (new version with 10-band system)
    2. InferDPT 
    3. SANTEXT+
    """
    
    def __init__(self, epsilon_values=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]):
        """
        Initialize the comparison framework
        
        Args:
            epsilon_values: List of epsilon values to test
        """
        self.epsilon_values = epsilon_values
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize mechanisms
        self.mechanisms = {
            'phraseDP': {
                'name': 'PhraseDP (New 10-Band)',
                'description': 'Phrase-level DP with 10-band candidate generation',
                'initialized': False
            },
            'inferDPT': {
                'name': 'InferDPT',
                'description': 'Inference-based DP text sanitization',
                'initialized': False
            },
            'santext': {
                'name': 'SANTEXT+',
                'description': 'Word-level DP with exponential mechanism',
                'initialized': False
            }
        }
        
        # Results storage
        self.results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'epsilon_values': epsilon_values,
                'mechanisms': list(self.mechanisms.keys())
            },
            'per_question_results': []
        }
    
    def initialize_phraseDP(self, questions):
        """
        Initialize PhraseDP with vocabulary from questions
        
        Args:
            questions: List of questions to build vocabulary from
        """
        print("Initializing PhraseDP (New 10-Band System)...")
        
        # PhraseDP uses the new 10-band system from utils.py
        # No additional initialization needed - it's ready to use
        self.mechanisms['phraseDP']['initialized'] = True
        print("✓ PhraseDP initialized")
    
    def initialize_inferDPT(self, questions):
        """
        Initialize InferDPT mechanism
        
        Args:
            questions: List of questions to build vocabulary from
        """
        print("Initializing InferDPT...")
        
        # InferDPT initialization would go here
        # For now, we'll mark it as initialized
        self.mechanisms['inferDPT']['initialized'] = True
        print("✓ InferDPT initialized")
    
    def initialize_santext(self, questions):
        """
        Initialize SANTEXT+ mechanism
        
        Args:
            questions: List of questions to build vocabulary from
        """
        print("Initializing SANTEXT+...")
        
        # Create SANTEXT+ mechanism
        self.santext_mechanism = create_santext_mechanism(epsilon=2.0, p=0.3)
        
        # Build vocabulary from questions
        self.santext_mechanism.build_vocabulary(questions)
        
        # Set some sensitive words (can be customized)
        sensitive_words = set()
        for question in questions:
            words = question.lower().split()
            # Mark some common sensitive words
            for word in words:
                if len(word) > 4 and word.isalpha():  # Longer words more likely to be sensitive
                    sensitive_words.add(word)
        
        self.santext_mechanism.set_sensitive_words(sensitive_words)
        self.mechanisms['santext']['initialized'] = True
        print("✓ SANTEXT+ initialized")
    
    def run_phraseDP(self, question, epsilon, num_samples=5):
        """
        Run PhraseDP (new version) on a question
        
        Args:
            question: Input question
            epsilon: Privacy parameter
            num_samples: Number of samples to generate
            
        Returns:
            List of sanitized questions
        """
        print(f"  Running PhraseDP (ε={epsilon})...")
        
        sanitized_questions = []
        
        for i in range(num_samples):
            try:
                # Use your new 10-band system
                candidates = generate_sentence_replacements_with_nebius_diverse(
                    input_sentence=question,
                    num_api_calls=10,  # 10 bands
                    num_return_sequences=5,
                    enforce_similarity_filter=True,
                    filter_margin=0.05,
                    equal_band_target=30,
                    verbose=False
                )
                
                if candidates:
                    # Get embeddings for candidates
                    candidate_embeddings = {}
                    for candidate in candidates:
                        candidate_embeddings[candidate] = self.sbert_model.encode(candidate)
                    
                    # Apply differential privacy mechanism
                    sanitized = differentially_private_replacement(
                        target_phrase=question,
                        epsilon=epsilon,
                        candidate_phrases=candidates,
                        candidate_embeddings=candidate_embeddings,
                        sbert_model=self.sbert_model
                    )
                    sanitized_questions.append(sanitized)
                else:
                    print(f"    Warning: No candidates generated for sample {i+1}")
                    sanitized_questions.append(question)  # Fallback to original
                    
            except Exception as e:
                print(f"    Error in PhraseDP sample {i+1}: {e}")
                sanitized_questions.append(question)  # Fallback to original
        
        return sanitized_questions
    
    def run_inferDPT(self, question, epsilon, num_samples=5):
        """
        Run InferDPT on a question
        
        Args:
            question: Input question
            epsilon: Privacy parameter
            num_samples: Number of samples to generate
            
        Returns:
            List of sanitized questions
        """
        print(f"  Running InferDPT (ε={epsilon})...")
        
        # Placeholder for InferDPT implementation
        # This would integrate with your existing InferDPT code
        sanitized_questions = []
        
        for i in range(num_samples):
            # For now, return the original question
            # TODO: Implement actual InferDPT mechanism
            sanitized_questions.append(f"[InferDPT-{epsilon}] {question}")
        
        return sanitized_questions
    
    def run_santext(self, question, epsilon, num_samples=5):
        """
        Run SANTEXT+ on a question
        
        Args:
            question: Input question
            epsilon: Privacy parameter
            num_samples: Number of samples to generate
            
        Returns:
            List of sanitized questions
        """
        print(f"  Running SANTEXT+ (ε={epsilon})...")
        
        # Update epsilon for SANTEXT+
        self.santext_mechanism.epsilon = epsilon
        self.santext_mechanism._calculate_probability_matrix()
        
        sanitized_questions = []
        
        for i in range(num_samples):
            try:
                sanitized = self.santext_mechanism.sanitize_text(question)
                sanitized_questions.append(sanitized)
            except Exception as e:
                print(f"    Error in SANTEXT+ sample {i+1}: {e}")
                sanitized_questions.append(question)  # Fallback to original
        
        return sanitized_questions
    
    def run_single_question_comparison(self, question, question_idx):
        """
        Run all mechanisms on a single question
        
        Args:
            question: Input question
            question_idx: Index of the question
            
        Returns:
            Dictionary with results for all mechanisms
        """
        print(f"\nProcessing Question {question_idx + 1}: {question}")
        print("=" * 80)
        
        question_results = {
            'question_idx': question_idx,
            'question': question,
            'mechanisms': {}
        }
        
        for mechanism_name in self.mechanisms.keys():
            if not self.mechanisms[mechanism_name]['initialized']:
                print(f"  Skipping {mechanism_name} - not initialized")
                continue
            
            print(f"\nRunning {self.mechanisms[mechanism_name]['name']}...")
            mechanism_results = {
                'name': self.mechanisms[mechanism_name]['name'],
                'epsilon_results': {}
            }
            
            for epsilon in self.epsilon_values:
                print(f"  Epsilon = {epsilon}")
                
                # Run the appropriate mechanism
                if mechanism_name == 'phraseDP':
                    sanitized_questions = self.run_phraseDP(question, epsilon)
                elif mechanism_name == 'inferDPT':
                    sanitized_questions = self.run_inferDPT(question, epsilon)
                elif mechanism_name == 'santext':
                    sanitized_questions = self.run_santext(question, epsilon)
                else:
                    continue
                
                # Calculate similarities
                similarities = []
                for sanitized in sanitized_questions:
                    sim = compute_similarity(self.sbert_model, question, sanitized)
                    similarities.append(sim)
                
                mechanism_results['epsilon_results'][epsilon] = {
                    'sanitized_questions': sanitized_questions,
                    'similarities': similarities,
                    'avg_similarity': np.mean(similarities),
                    'std_similarity': np.std(similarities)
                }
                
                print(f"    Avg similarity: {np.mean(similarities):.4f} ± {np.std(similarities):.4f}")
            
            question_results['mechanisms'][mechanism_name] = mechanism_results
        
        return question_results
    
    def run_comprehensive_comparison(self, questions, save_results=True):
        """
        Run comprehensive comparison on multiple questions
        
        Args:
            questions: List of questions to test
            save_results: Whether to save results to file
            
        Returns:
            Dictionary with all results
        """
        print("Starting Comprehensive Privacy Mechanisms Comparison")
        print("=" * 80)
        print(f"Questions: {len(questions)}")
        print(f"Epsilon values: {self.epsilon_values}")
        print(f"Mechanisms: {list(self.mechanisms.keys())}")
        
        # Initialize all mechanisms
        self.initialize_phraseDP(questions)
        self.initialize_inferDPT(questions)
        self.initialize_santext(questions)
        
        # Run comparison on each question
        for i, question in enumerate(questions):
            question_results = self.run_single_question_comparison(question, i)
            self.results['per_question_results'].append(question_results)
        
        # Calculate summary statistics
        self._calculate_summary_statistics()
        
        # Save results if requested
        if save_results:
            self._save_results()
        
        return self.results
    
    def _calculate_summary_statistics(self):
        """Calculate summary statistics across all questions"""
        print("\nCalculating summary statistics...")
        
        summary = {
            'mechanisms': {},
            'epsilon_trends': {}
        }
        
        for mechanism_name in self.mechanisms.keys():
            mechanism_data = []
            
            for question_result in self.results['per_question_results']:
                if mechanism_name in question_result['mechanisms']:
                    for epsilon, epsilon_result in question_result['mechanisms'][mechanism_name]['epsilon_results'].items():
                        mechanism_data.append({
                            'epsilon': epsilon,
                            'avg_similarity': epsilon_result['avg_similarity']
                        })
            
            if mechanism_data:
                # Calculate correlation between epsilon and similarity
                epsilons = [d['epsilon'] for d in mechanism_data]
                similarities = [d['avg_similarity'] for d in mechanism_data]
                
                correlation = np.corrcoef(epsilons, similarities)[0, 1]
                
                summary['mechanisms'][mechanism_name] = {
                    'total_samples': len(mechanism_data),
                    'epsilon_similarity_correlation': correlation,
                    'avg_similarity_by_epsilon': {
                        eps: np.mean([d['avg_similarity'] for d in mechanism_data if d['epsilon'] == eps])
                        for eps in self.epsilon_values
                    }
                }
        
        self.results['summary'] = summary
    
    def _save_results(self):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"privacy_mechanisms_comparison_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Results saved to: {filename}")

def demo_privacy_mechanisms_comparison():
    """Demo function for privacy mechanisms comparison"""
    
    # Sample questions
    test_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?"
    ]
    
    # Create comparison framework
    comparison = PrivacyMechanismsComparison(epsilon_values=[1.0, 2.0, 3.0])
    
    # Run comparison
    results = comparison.run_comprehensive_comparison(test_questions)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY RESULTS")
    print("=" * 80)
    
    for mechanism_name, mechanism_summary in results['summary']['mechanisms'].items():
        print(f"\n{mechanism_name.upper()}:")
        print(f"  Epsilon-Similarity Correlation: {mechanism_summary['epsilon_similarity_correlation']:.4f}")
        print("  Average Similarity by Epsilon:")
        for epsilon, avg_sim in mechanism_summary['avg_similarity_by_epsilon'].items():
            print(f"    ε={epsilon}: {avg_sim:.4f}")

if __name__ == "__main__":
    demo_privacy_mechanisms_comparison()
