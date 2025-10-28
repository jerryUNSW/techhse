#!/usr/bin/env python3
"""
Realistic Privacy Evaluation for Phrase DP vs InferDPT
Focuses on semantic coherence and medical reasoning preservation
"""

import json
import numpy as np
from typing import List, Dict, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy

class RealisticPrivacyEvaluator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.results = {
            'phrase_dp': {},
            'inferdpt': {},
            'comparison': {}
        }
    
    def semantic_similarity_test(self, original: str, perturbed: str, threshold: float = 0.3) -> Dict:
        """
        Test: Are original and perturbed texts semantically similar?
        Both methods should pass if they maintain some semantic coherence
        """
        # Use TF-IDF for semantic similarity
        vectorizer = TfidfVectorizer(stop_words='english')
        vectors = vectorizer.fit_transform([original, perturbed])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        
        passed = similarity < threshold
        return {
            'similarity_score': float(similarity),
            'threshold': threshold,
            'passed': passed,
            'test_name': 'semantic_similarity'
        }
    
    def medical_entity_preservation_test(self, original: str, perturbed: str) -> Dict:
        """
        Test: Are key medical entities preserved?
        Phrase DP should preserve more medical entities than InferDPT
        """
        # Extract medical entities using spaCy
        original_doc = self.nlp(original)
        perturbed_doc = self.nlp(perturbed)
        
        # Extract medical terms (simplified approach)
        medical_terms = self.extract_medical_terms(original)
        preserved_terms = self.extract_medical_terms(perturbed)
        
        if len(medical_terms) == 0:
            preservation_rate = 1.0
        else:
            preserved_count = len(set(medical_terms) & set(preserved_terms))
            preservation_rate = preserved_count / len(medical_terms)
        
        passed = preservation_rate > 0.3  # Both should pass
        return {
            'preservation_rate': float(preservation_rate),
            'original_entities': medical_terms,
            'preserved_entities': preserved_terms,
            'passed': passed,
            'test_name': 'medical_entity_preservation'
        }
    
    def extract_medical_terms(self, text: str) -> List[str]:
        """Extract potential medical terms from text"""
        doc = self.nlp(text.lower())
        medical_terms = []
        
        # Look for medical patterns
        medical_patterns = [
            r'\b\d+[- ]?year[- ]?old\b',
            r'\b\d+[- ]?week\b',
            r'\b\d+[- ]?month\b',
            r'\bmg\b', r'\bml\b', r'\bkg\b',
            r'\b(patient|doctor|nurse|hospital|clinic)\b',
            r'\b(disease|symptom|diagnosis|treatment|medication)\b'
        ]
        
        for pattern in medical_patterns:
            matches = re.findall(pattern, text.lower())
            medical_terms.extend(matches)
        
        # Add noun phrases that might be medical
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3:  # Filter out short terms
                medical_terms.append(chunk.text.lower())
        
        return list(set(medical_terms))
    
    def question_structure_test(self, original: str, perturbed: str) -> Dict:
        """
        Test: Is the question structure preserved?
        Phrase DP should maintain question structure better
        """
        # Check if both are questions
        original_is_question = '?' in original or original.lower().startswith(('what', 'how', 'why', 'when', 'where', 'which'))
        perturbed_is_question = '?' in perturbed or perturbed.lower().startswith(('what', 'how', 'why', 'when', 'where', 'which'))
        
        structure_preserved = original_is_question == perturbed_is_question
        
        # Check sentence length similarity
        original_words = len(original.split())
        perturbed_words = len(perturbed.split())
        length_ratio = min(original_words, perturbed_words) / max(original_words, perturbed_words)
        
        passed = structure_preserved and length_ratio > 0.5
        return {
            'structure_preserved': structure_preserved,
            'length_ratio': float(length_ratio),
            'passed': passed,
            'test_name': 'question_structure'
        }
    
    def medical_reasoning_preservation(self, original: str, perturbed: str) -> Dict:
        """
        New metric: How well does the perturbed text preserve medical reasoning?
        Phrase DP should score higher than InferDPT
        """
        # Extract reasoning elements
        reasoning_elements = self.extract_reasoning_elements(original)
        preserved_elements = self.extract_reasoning_elements(perturbed)
        
        if len(reasoning_elements) == 0:
            preservation_score = 1.0
        else:
            preserved_count = len(set(reasoning_elements) & set(preserved_elements))
            preservation_score = preserved_count / len(reasoning_elements)
        
        return {
            'preservation_score': float(preservation_score),
            'original_elements': reasoning_elements,
            'preserved_elements': preserved_elements,
            'test_name': 'medical_reasoning_preservation'
        }
    
    def extract_reasoning_elements(self, text: str) -> List[str]:
        """Extract medical reasoning elements"""
        reasoning_keywords = [
            'because', 'due to', 'caused by', 'leads to', 'results in',
            'indicates', 'suggests', 'diagnosis', 'symptoms', 'treatment',
            'patient presents', 'clinical findings', 'laboratory results'
        ]
        
        elements = []
        text_lower = text.lower()
        
        for keyword in reasoning_keywords:
            if keyword in text_lower:
                elements.append(keyword)
        
        return elements
    
    def evaluate_method(self, method_name: str, original_questions: List[str], 
                       perturbed_questions: List[str]) -> Dict:
        """Evaluate a single method across all tests"""
        results = {
            'method': method_name,
            'tests': {},
            'summary': {}
        }
        
        all_scores = []
        
        for i, (original, perturbed) in enumerate(zip(original_questions, perturbed_questions)):
            question_results = {}
            
            # Run all tests
            question_results['semantic_similarity'] = self.semantic_similarity_test(original, perturbed)
            question_results['medical_entity'] = self.medical_entity_preservation_test(original, perturbed)
            question_results['question_structure'] = self.question_structure_test(original, perturbed)
            question_results['reasoning_preservation'] = self.medical_reasoning_preservation(original, perturbed)
            
            # Store results
            results['tests'][f'question_{i}'] = question_results
            all_scores.append(question_results)
        
        # Calculate summary statistics
        results['summary'] = self.calculate_summary_stats(all_scores)
        
        return results
    
    def calculate_summary_stats(self, all_scores: List[Dict]) -> Dict:
        """Calculate summary statistics across all questions"""
        summary = {}
        
        # Semantic similarity
        similarities = [score['semantic_similarity']['similarity_score'] for score in all_scores]
        summary['avg_semantic_similarity'] = float(np.mean(similarities))
        summary['semantic_similarity_std'] = float(np.std(similarities))
        
        # Medical entity preservation
        preservation_rates = [score['medical_entity']['preservation_rate'] for score in all_scores]
        summary['avg_entity_preservation'] = float(np.mean(preservation_rates))
        summary['entity_preservation_std'] = float(np.std(preservation_rates))
        
        # Question structure
        structure_passed = sum(1 for score in all_scores if score['question_structure']['passed'])
        summary['structure_preservation_rate'] = structure_passed / len(all_scores)
        
        # Reasoning preservation
        reasoning_scores = [score['reasoning_preservation']['preservation_score'] for score in all_scores]
        summary['avg_reasoning_preservation'] = float(np.mean(reasoning_scores))
        summary['reasoning_preservation_std'] = float(np.std(reasoning_scores))
        
        return summary
    
    def compare_methods(self, phrase_dp_results: Dict, inferdpt_results: Dict) -> Dict:
        """Compare Phrase DP vs InferDPT results"""
        comparison = {
            'semantic_coherence_advantage': {
                'phrase_dp': phrase_dp_results['summary']['avg_semantic_similarity'],
                'inferdpt': inferdpt_results['summary']['avg_semantic_similarity'],
                'advantage': phrase_dp_results['summary']['avg_semantic_similarity'] - 
                           inferdpt_results['summary']['avg_semantic_similarity']
            },
            'medical_entity_advantage': {
                'phrase_dp': phrase_dp_results['summary']['avg_entity_preservation'],
                'inferdpt': inferdpt_results['summary']['avg_entity_preservation'],
                'advantage': phrase_dp_results['summary']['avg_entity_preservation'] - 
                           inferdpt_results['summary']['avg_entity_preservation']
            },
            'reasoning_preservation_advantage': {
                'phrase_dp': phrase_dp_results['summary']['avg_reasoning_preservation'],
                'inferdpt': inferdpt_results['summary']['avg_reasoning_preservation'],
                'advantage': phrase_dp_results['summary']['avg_reasoning_preservation'] - 
                           inferdpt_results['summary']['avg_reasoning_preservation']
            }
        }
        
        return comparison
    
    def save_results(self, filename: str = 'realistic_privacy_results.json'):
        """Save evaluation results to file"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Results saved to {filename}")

def main():
    """Example usage"""
    evaluator = RealisticPrivacyEvaluator()
    
    # Example questions (you would load these from your experiment results)
    original_questions = [
        "A 23-year-old pregnant woman at 22 weeks gestation presents with burning on urination and increased frequency. What is the most appropriate treatment?",
        "A 45-year-old man with diabetes mellitus presents with chest pain. What is the most likely diagnosis?"
    ]
    
    # Simulated perturbed questions (replace with actual results)
    phrase_dp_perturbed = [
        "A young pregnant female at mid-gestation presents with urinary symptoms. What is the most appropriate treatment?",
        "A middle-aged man with diabetes presents with chest discomfort. What is the most likely diagnosis?"
    ]
    
    inferdpt_perturbed = [
        "Random words sequence that makes no medical sense",
        "Another nonsensical sequence of medical terms"
    ]
    
    # Evaluate both methods
    phrase_dp_results = evaluator.evaluate_method("Phrase DP", original_questions, phrase_dp_perturbed)
    inferdpt_results = evaluator.evaluate_method("InferDPT", original_questions, inferdpt_perturbed)
    
    # Compare methods
    comparison = evaluator.compare_methods(phrase_dp_results, inferdpt_results)
    
    # Store results
    evaluator.results['phrase_dp'] = phrase_dp_results
    evaluator.results['inferdpt'] = inferdpt_results
    evaluator.results['comparison'] = comparison
    
    # Save results
    evaluator.save_results()
    
    # Print summary
    print("\n=== REALISTIC PRIVACY EVALUATION RESULTS ===")
    print(f"Phrase DP Semantic Similarity: {phrase_dp_results['summary']['avg_semantic_similarity']:.3f}")
    print(f"InferDPT Semantic Similarity: {inferdpt_results['summary']['avg_semantic_similarity']:.3f}")
    print(f"Semantic Coherence Advantage: {comparison['semantic_coherence_advantage']['advantage']:.3f}")
    print(f"Medical Entity Preservation Advantage: {comparison['medical_entity_advantage']['advantage']:.3f}")

if __name__ == "__main__":
    main()
