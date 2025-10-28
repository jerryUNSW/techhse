#!/usr/bin/env python3
"""
Privacy Evaluation for MedQA Experiment
======================================

This script evaluates the privacy strength of Phrase DP and InferDPT methods
by testing whether an adversary can recover original questions from perturbed ones.

Evaluation Method:
1. Extract original questions and their perturbed versions
2. Feed perturbed questions to an LLM to recover original
3. Compare similarity between recovered and actual original questions
4. Analyze privacy strength based on recovery accuracy

Author: Tech4HSE Team
Date: 2025-08-26
"""

import re
import json
import numpy as np
from datetime import datetime
from collections import defaultdict
import openai
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class PrivacyEvaluator:
    """Evaluator for privacy strength of perturbation methods."""
    
    def __init__(self, results_file="xxx-500-qa"):
        self.results_file = results_file
        self.original_questions = {}
        self.perturbed_questions = {
            'phrase_dp': {},
            'inferdpt': {}
        }
        self.recovery_results = {
            'phrase_dp': {},
            'inferdpt': {}
        }
        
        # Initialize OpenAI client
        self.client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def extract_questions_and_perturbations(self):
        """Extract original questions and their perturbed versions from results file."""
        print("Extracting original questions and perturbations...")
        
        current_question = None
        current_scenario = None
        current_question_text = ""
        current_perturbed_text = ""
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Extract question number
                question_match = re.search(r'Question (\d+)/500', line)
                if question_match:
                    current_question = int(question_match.group(1))
                    current_question_text = ""
                    current_perturbed_text = ""
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
    
    def create_recovery_prompt(self, perturbed_question):
        """Create a prompt for the LLM to recover the original question."""
        prompt = f"""You are a privacy researcher trying to recover the original question from a perturbed version. 
The perturbed question has been modified to protect privacy while maintaining semantic meaning.

Perturbed Question: {perturbed_question}

Your task is to recover the original question as accurately as possible. 
Consider the medical context and try to reconstruct what the original question might have been.

Recovered Original Question:"""
        
        return prompt
    
    def recover_original_question(self, perturbed_question, method_name):
        """Use LLM to recover original question from perturbed version."""
        try:
            prompt = self.create_recovery_prompt(perturbed_question)
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Using a strong model for recovery
                messages=[
                    {"role": "system", "content": "You are a privacy researcher analyzing question recovery."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1  # Low temperature for consistent recovery
            )
            
            recovered_question = response.choices[0].message.content.strip()
            return recovered_question
            
        except Exception as e:
            print(f"Error recovering question for {method_name}: {e}")
            return None
    
    def calculate_similarity(self, text1, text2):
        """Calculate similarity between two texts using simple metrics."""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Word overlap percentage
        overlap_percentage = intersection / max(len(words1), len(words2)) if max(len(words1), len(words2)) > 0 else 0.0
        
        return {
            'jaccard': jaccard_similarity,
            'overlap_percentage': overlap_percentage,
            'intersection_size': intersection,
            'union_size': union
        }
    
    def evaluate_privacy(self, method_name, max_questions=50):
        """Evaluate privacy strength for a specific method."""
        print(f"\nEvaluating privacy for {method_name.upper()}...")
        
        perturbed_questions = self.perturbed_questions[method_name]
        similarities = []
        recovery_results = []
        
        # Limit the number of questions to evaluate (for cost and time efficiency)
        questions_to_evaluate = list(perturbed_questions.keys())[:max_questions]
        
        for i, question_id in enumerate(questions_to_evaluate):
            if question_id not in self.original_questions:
                continue
                
            original_question = self.original_questions[question_id]
            perturbed_question = perturbed_questions[question_id]
            
            print(f"Processing question {question_id} ({i+1}/{len(questions_to_evaluate)})...")
            
            # Recover original question
            recovered_question = self.recover_original_question(perturbed_question, method_name)
            
            if recovered_question:
                # Calculate similarity
                similarity = self.calculate_similarity(original_question, recovered_question)
                similarities.append(similarity)
                
                recovery_results.append({
                    'question_id': question_id,
                    'original': original_question,
                    'perturbed': perturbed_question,
                    'recovered': recovered_question,
                    'similarity': similarity
                })
                
                print(f"  Jaccard Similarity: {similarity['jaccard']:.3f}")
                print(f"  Overlap Percentage: {similarity['overlap_percentage']:.3f}")
            
            # Add delay to avoid rate limiting
            time.sleep(1)
        
        return recovery_results, similarities
    
    def analyze_privacy_results(self, recovery_results, similarities, method_name):
        """Analyze privacy evaluation results."""
        if not similarities:
            print(f"No results to analyze for {method_name}")
            return
        
        # Calculate statistics
        jaccard_scores = [s['jaccard'] for s in similarities]
        overlap_scores = [s['overlap_percentage'] for s in similarities]
        
        analysis = {
            'method': method_name,
            'total_questions': len(similarities),
            'jaccard_similarity': {
                'mean': np.mean(jaccard_scores),
                'std': np.std(jaccard_scores),
                'min': np.min(jaccard_scores),
                'max': np.max(jaccard_scores),
                'median': np.median(jaccard_scores)
            },
            'overlap_percentage': {
                'mean': np.mean(overlap_scores),
                'std': np.std(overlap_scores),
                'min': np.min(overlap_scores),
                'max': np.max(overlap_scores),
                'median': np.median(overlap_scores)
            },
            'privacy_strength': {
                'high_privacy_threshold': 0.3,  # Below this is considered good privacy
                'medium_privacy_threshold': 0.6,  # Below this is considered moderate privacy
                'high_privacy_count': sum(1 for s in jaccard_scores if s < 0.3),
                'medium_privacy_count': sum(1 for s in jaccard_scores if 0.3 <= s < 0.6),
                'low_privacy_count': sum(1 for s in jaccard_scores if s >= 0.6)
            }
        }
        
        return analysis
    
    def print_privacy_analysis(self, analysis):
        """Print privacy analysis results."""
        print(f"\n{'='*60}")
        print(f"PRIVACY ANALYSIS: {analysis['method'].upper()}")
        print(f"{'='*60}")
        print(f"Total Questions Evaluated: {analysis['total_questions']}")
        print()
        
        print("Jaccard Similarity Statistics:")
        print(f"  Mean: {analysis['jaccard_similarity']['mean']:.3f}")
        print(f"  Std:  {analysis['jaccard_similarity']['std']:.3f}")
        print(f"  Min:  {analysis['jaccard_similarity']['min']:.3f}")
        print(f"  Max:  {analysis['jaccard_similarity']['max']:.3f}")
        print(f"  Median: {analysis['jaccard_similarity']['median']:.3f}")
        print()
        
        print("Overlap Percentage Statistics:")
        print(f"  Mean: {analysis['overlap_percentage']['mean']:.3f}")
        print(f"  Std:  {analysis['overlap_percentage']['std']:.3f}")
        print(f"  Min:  {analysis['overlap_percentage']['min']:.3f}")
        print(f"  Max:  {analysis['overlap_percentage']['max']:.3f}")
        print(f"  Median: {analysis['overlap_percentage']['median']:.3f}")
        print()
        
        print("Privacy Strength Assessment:")
        print(f"  High Privacy (< 0.3): {analysis['privacy_strength']['high_privacy_count']} questions")
        print(f"  Medium Privacy (0.3-0.6): {analysis['privacy_strength']['medium_privacy_count']} questions")
        print(f"  Low Privacy (> 0.6): {analysis['privacy_strength']['low_privacy_count']} questions")
        
        # Privacy strength rating
        high_privacy_ratio = analysis['privacy_strength']['high_privacy_count'] / analysis['total_questions']
        if high_privacy_ratio >= 0.7:
            privacy_rating = "EXCELLENT"
        elif high_privacy_ratio >= 0.5:
            privacy_rating = "GOOD"
        elif high_privacy_ratio >= 0.3:
            privacy_rating = "MODERATE"
        else:
            privacy_rating = "POOR"
        
        print(f"\nOverall Privacy Rating: {privacy_rating}")
        print(f"High Privacy Ratio: {high_privacy_ratio:.1%}")
    
    def save_results(self, phrase_dp_results, inferdpt_results, phrase_dp_analysis, inferdpt_analysis):
        """Save privacy evaluation results to JSON file."""
        output_data = {
            'evaluation_date': datetime.now().isoformat(),
            'methods_evaluated': ['phrase_dp', 'inferdpt'],
            'phrase_dp': {
                'analysis': phrase_dp_analysis,
                'recovery_results': phrase_dp_results
            },
            'inferdpt': {
                'analysis': inferdpt_analysis,
                'recovery_results': inferdpt_results
            },
            'comparison': {
                'phrase_dp_mean_jaccard': phrase_dp_analysis['jaccard_similarity']['mean'],
                'inferdpt_mean_jaccard': inferdpt_analysis['jaccard_similarity']['mean'],
                'privacy_difference': phrase_dp_analysis['jaccard_similarity']['mean'] - inferdpt_analysis['jaccard_similarity']['mean']
            }
        }
        
        with open('privacy_evaluation_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved as: privacy_evaluation_results.json")
    
    def run_evaluation(self, max_questions_per_method=30):
        """Run the complete privacy evaluation."""
        print("Starting Privacy Evaluation for MedQA Experiment...")
        
        # Extract questions and perturbations
        self.extract_questions_and_perturbations()
        
        # Evaluate Phrase DP privacy
        phrase_dp_results, phrase_dp_similarities = self.evaluate_privacy('phrase_dp', max_questions_per_method)
        phrase_dp_analysis = self.analyze_privacy_results(phrase_dp_results, phrase_dp_similarities, 'phrase_dp')
        self.print_privacy_analysis(phrase_dp_analysis)
        
        # Evaluate InferDPT privacy
        inferdpt_results, inferdpt_similarities = self.evaluate_privacy('inferdpt', max_questions_per_method)
        inferdpt_analysis = self.analyze_privacy_results(inferdpt_results, inferdpt_similarities, 'inferdpt')
        self.print_privacy_analysis(inferdpt_analysis)
        
        # Compare methods
        print(f"\n{'='*60}")
        print("METHOD COMPARISON")
        print(f"{'='*60}")
        print(f"Phrase DP Mean Jaccard: {phrase_dp_analysis['jaccard_similarity']['mean']:.3f}")
        print(f"InferDPT Mean Jaccard: {inferdpt_analysis['jaccard_similarity']['mean']:.3f}")
        
        if phrase_dp_analysis['jaccard_similarity']['mean'] < inferdpt_analysis['jaccard_similarity']['mean']:
            print("Phrase DP provides BETTER privacy (lower recovery accuracy)")
        else:
            print("InferDPT provides BETTER privacy (lower recovery accuracy)")
        
        # Save results
        self.save_results(phrase_dp_results, inferdpt_results, phrase_dp_analysis, inferdpt_analysis)
        
        return phrase_dp_analysis, inferdpt_analysis

def main():
    """Main function to run the privacy evaluation."""
    import os
    
    # Check if OpenAI API key is available
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file")
        return
    
    evaluator = PrivacyEvaluator()
    phrase_dp_analysis, inferdpt_analysis = evaluator.run_evaluation()
    
    print(f"\n{'='*60}")
    print("PRIVACY EVALUATION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
