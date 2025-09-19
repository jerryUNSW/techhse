#!/usr/bin/env python3
"""
Unified Sentence-Level GPT Inference Attack for Phrase DP vs InferDPT
====================================================================

This script implements a unified sentence-level GPT inference attack
to compare privacy protection between Phrase DP and InferDPT methods.
Both methods are evaluated using the same process for fair comparison.

Author: Tech4HSE Team
Date: 2025-08-26
"""

import re
import json
import numpy as np
from datetime import datetime
import openai
from dotenv import load_dotenv
import time
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# Load environment variables
load_dotenv()

class UnifiedGPTInferenceAttack:
    """Unified GPT inference attack evaluator for both Phrase DP and InferDPT."""
    
    def __init__(self, results_file="complete-550.txt"):
        self.results_file = results_file
        self.original_questions = {}
        self.perturbed_questions = {
            'phrase_dp': {},
            'inferdpt': {}
        }
        
        # Initialize models
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # OpenAI client
        api_key = os.getenv('OPEN_AI_KEY')
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        else:
            self.client = None
            print("Warning: No OpenAI API key found, GPT inference attack will be skipped")
        
        # Results storage
        self.results = {
            'phrase_dp': {'similarities': [], 'privacy_level': 0.0},
            'inferdpt': {'similarities': [], 'privacy_level': 0.0}
        }
    
    def extract_questions_and_perturbations(self):
        """Extract original questions and their perturbed versions from results file."""
        print("Extracting original questions and perturbations...")
        
        current_question = None
        current_scenario = None
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Extract question number
                question_match = re.search(r'Question (\d+)/500', line)
                if question_match:
                    current_question = int(question_match.group(1))
                    continue
                
                # Extract scenario information
                if 'Scenario 3.1:' in line:
                    current_scenario = 'phrase_dp'
                elif 'Scenario 3.2:' in line:
                    current_scenario = 'inferdpt'
                elif 'Scenario 1:' in line or 'Scenario 2:' in line or 'Scenario 4:' in line:
                    current_scenario = None
                
                # Extract original question text
                if line.startswith('Question:') and current_question:
                    current_question_text = line.replace('Question:', '').strip()
                    self.original_questions[current_question] = current_question_text
                
                # Extract perturbed question text
                if current_scenario and 'Perturbed Question:' in line:
                    current_perturbed_text = line.replace('Perturbed Question:', '').strip()
                    if current_question and current_perturbed_text:
                        self.perturbed_questions[current_scenario][current_question] = current_perturbed_text
        
        print(f"Extracted {len(self.original_questions)} original questions")
        print(f"Extracted {len(self.perturbed_questions['phrase_dp'])} Phrase DP perturbations")
        print(f"Extracted {len(self.perturbed_questions['inferdpt'])} InferDPT perturbations")
    
    def gpt_sentence_recovery(self, original_question, perturbed_question, method_name):
        """Use GPT to recover original question from perturbed version."""
        if not self.client:
            print("Skipping GPT recovery (no API key available)")
            return original_question  # Return original as fallback
            
        if method_name == "phrase_dp":
            prompt = f"""Given a perturbed medical question that has been paraphrased while preserving medical meaning, your task is to recover the most likely original question.

The perturbed question maintains the same medical meaning but may have different wording, anonymized details, or paraphrased phrases.

Perturbed Question: {perturbed_question}

Recovered Original Question:"""
        
        elif method_name == "inferdpt":
            prompt = f"""Given a perturbed medical question where some tokens have been replaced with random alternatives, your task is to recover the most likely original question.

The perturbed question may contain some nonsensical or random tokens that need to be corrected.

Perturbed Question: {perturbed_question}

Recovered Original Question:"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical privacy researcher analyzing question recovery."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            recovered_question = response.choices[0].message.content.strip()
            return recovered_question
            
        except Exception as e:
            print(f"Error in GPT recovery: {e}")
            return original_question  # Return original as fallback
    
    def calculate_sentence_similarity(self, original, recovered):
        """Calculate BERT embedding similarity between original and recovered questions."""
        original_embedding = self.bert_model.encode([original])[0]
        recovered_embedding = self.bert_model.encode([recovered])[0]
        
        similarity = cosine_similarity([original_embedding], [recovered_embedding])[0][0]
        return similarity
    
    def run_unified_attack(self, max_questions=10):
        """Run the unified sentence-level GPT inference attack on both methods."""
        print(f"\n{'='*60}")
        print("UNIFIED SENTENCE-LEVEL GPT INFERENCE ATTACK")
        print(f"{'='*60}")
        
        # Extract questions and perturbations
        self.extract_questions_and_perturbations()
        
        # Get questions to evaluate
        available_questions = set(self.original_questions.keys())
        phrase_dp_questions = set(self.perturbed_questions['phrase_dp'].keys())
        inferdpt_questions = set(self.perturbed_questions['inferdpt'].keys())
        
        common_questions = available_questions.intersection(phrase_dp_questions).intersection(inferdpt_questions)
        questions_to_evaluate = sorted(list(common_questions))[:max_questions]
        
        print(f"Evaluating {len(questions_to_evaluate)} questions...")
        
        for i, question_id in enumerate(questions_to_evaluate):
            original = self.original_questions[question_id]
            phrase_dp_perturbed = self.perturbed_questions['phrase_dp'][question_id]
            inferdpt_perturbed = self.perturbed_questions['inferdpt'][question_id]
            
            print(f"\nQuestion {question_id} ({i+1}/{len(questions_to_evaluate)}):")
            print(f"Original: {original[:100]}...")
            
            # Test Phrase DP
            print("Testing Phrase DP...")
            phrase_dp_recovered = self.gpt_sentence_recovery(original, phrase_dp_perturbed, "phrase_dp")
            phrase_dp_similarity = self.calculate_sentence_similarity(original, phrase_dp_recovered)
            self.results['phrase_dp']['similarities'].append(phrase_dp_similarity)
            
            print(f"  Phrase DP Recovered: {phrase_dp_recovered[:100]}...")
            print(f"  Phrase DP Similarity: {phrase_dp_similarity:.3f}")
            
            # Test InferDPT
            print("Testing InferDPT...")
            inferdpt_recovered = self.gpt_sentence_recovery(original, inferdpt_perturbed, "inferdpt")
            inferdpt_similarity = self.calculate_sentence_similarity(original, inferdpt_recovered)
            self.results['inferdpt']['similarities'].append(inferdpt_similarity)
            
            print(f"  InferDPT Recovered: {inferdpt_recovered[:100]}...")
            print(f"  InferDPT Similarity: {inferdpt_similarity:.3f}")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        # Calculate privacy levels
        self.results['phrase_dp']['privacy_level'] = 1 - np.mean(self.results['phrase_dp']['similarities'])
        self.results['inferdpt']['privacy_level'] = 1 - np.mean(self.results['inferdpt']['similarities'])
        
        return self.results
    
    def print_results(self):
        """Print comprehensive results analysis."""
        print(f"\n{'='*60}")
        print("UNIFIED GPT INFERENCE ATTACK RESULTS")
        print(f"{'='*60}")
        
        print("Attack Results (Higher Privacy Level = Better Privacy Protection):")
        print(f"  Phrase DP Privacy Level: {self.results['phrase_dp']['privacy_level']:.3f}")
        print(f"  InferDPT Privacy Level: {self.results['inferdpt']['privacy_level']:.3f}")
        
        print(f"\nDetailed Similarities:")
        print(f"  Phrase DP Similarities: {[f'{s:.3f}' for s in self.results['phrase_dp']['similarities']]}")
        print(f"  InferDPT Similarities: {[f'{s:.3f}' for s in self.results['inferdpt']['similarities']]}")
        
        # Comparison
        if self.results['phrase_dp']['privacy_level'] > self.results['inferdpt']['privacy_level']:
            print(f"\nPhrase DP provides BETTER privacy protection")
            improvement = ((self.results['phrase_dp']['privacy_level'] - self.results['inferdpt']['privacy_level']) / self.results['inferdpt']['privacy_level']) * 100
            print(f"Improvement: {improvement:.1f}%")
        else:
            print(f"\nInferDPT provides BETTER privacy protection")
            improvement = ((self.results['inferdpt']['privacy_level'] - self.results['phrase_dp']['privacy_level']) / self.results['phrase_dp']['privacy_level']) * 100
            print(f"Improvement: {improvement:.1f}%")
        
        # Privacy rating
        for method, results in self.results.items():
            privacy_level = results['privacy_level']
            if privacy_level >= 0.8:
                rating = "EXCELLENT"
            elif privacy_level >= 0.6:
                rating = "GOOD"
            elif privacy_level >= 0.4:
                rating = "MODERATE"
            else:
                rating = "POOR"
            print(f"  {method.upper()} Privacy Rating: {rating}")
    
    def save_results(self):
        """Save results to JSON file."""
        output_data = {
            'evaluation_date': datetime.now().isoformat(),
            'evaluation_method': 'Unified Sentence-Level GPT Inference Attack',
            'methods_evaluated': ['phrase_dp', 'inferdpt'],
            'results': {
                'phrase_dp': {
                    'similarities': [float(s) for s in self.results['phrase_dp']['similarities']],
                    'privacy_level': float(self.results['phrase_dp']['privacy_level'])
                },
                'inferdpt': {
                    'similarities': [float(s) for s in self.results['inferdpt']['similarities']],
                    'privacy_level': float(self.results['inferdpt']['privacy_level'])
                }
            },
            'comparison': {
                'phrase_dp_better': self.results['phrase_dp']['privacy_level'] > self.results['inferdpt']['privacy_level'],
                'improvement_percentage': abs(self.results['phrase_dp']['privacy_level'] - self.results['inferdpt']['privacy_level']) / min(self.results['phrase_dp']['privacy_level'], self.results['inferdpt']['privacy_level']) * 100
            }
        }
        
        with open('unified_gpt_inference_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved as: unified_gpt_inference_results.json")

def main():
    """Main function to run the unified GPT inference attack."""
    evaluator = UnifiedGPTInferenceAttack()
    results = evaluator.run_unified_attack(max_questions=10)
    evaluator.print_results()
    evaluator.save_results()
    
    print(f"\n{'='*60}")
    print("UNIFIED GPT INFERENCE ATTACK COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()




