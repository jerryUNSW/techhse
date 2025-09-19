#!/usr/bin/env python3
"""
NER-Based PII Privacy Evaluation for Phrase DP vs InferDPT
=========================================================

This script uses spaCy NER to extract sensitive entities from original and perturbed questions,
then evaluates privacy protection by measuring what PII is preserved vs removed.

Author: Tech4HSE Team
Date: 2025-08-26
"""

import re
import json
import numpy as np
from datetime import datetime
import spacy
from collections import defaultdict

class NERPIIPrivacyEvaluator:
    """NER-based PII privacy evaluator for both Phrase DP and InferDPT."""
    
    def __init__(self, results_file="complete-550.txt"):
        self.results_file = results_file
        self.original_questions = {}
        self.perturbed_questions = {
            'phrase_dp': {},
            'inferdpt': {}
        }
        
        # Load spaCy NER model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Loaded spaCy NER model: en_core_web_sm")
        except OSError:
            print("❌ spaCy NER model not found. Please install with: python -m spacy download en_core_web_sm")
            raise
        
        # Results storage
        self.results = {
            'phrase_dp': {'privacy_levels': [], 'entity_analysis': {}},
            'inferdpt': {'privacy_levels': [], 'entity_analysis': {}}
        }
        
        # Entity categories we care about for privacy
        self.sensitive_entity_types = [
            'PERSON', 'GPE', 'ORG', 'DATE', 'CARDINAL', 'ORDINAL', 
            'MONEY', 'PERCENT', 'QUANTITY', 'FAC', 'EVENT', 'LAW', 
            'LANGUAGE', 'LOC', 'NORP', 'PRODUCT', 'WORK_OF_ART', 'TIME'
        ]
    
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
    
    def extract_pii_entities(self, text):
        """Extract PII entities from text using spaCy NER."""
        doc = self.nlp(text)
        entities = defaultdict(list)
        
        for ent in doc.ents:
            if ent.label_ in self.sensitive_entity_types:
                entities[ent.label_].append(ent.text.strip())
        
        return dict(entities)
    
    def calculate_entity_overlap(self, original_entities, perturbed_entities):
        """Calculate how many original entities are preserved in perturbed text."""
        preserved_entities = 0
        total_original_entities = 0
        
        for entity_type, original_items in original_entities.items():
            perturbed_items = perturbed_entities.get(entity_type, [])
            total_original_entities += len(original_items)
            
            for original_item in original_items:
                # Check if any part of the original entity appears in perturbed entities
                original_lower = original_item.lower()
                for perturbed_item in perturbed_items:
                    if original_lower in perturbed_item.lower() or perturbed_item.lower() in original_lower:
                        preserved_entities += 1
                        break
        
        return preserved_entities, total_original_entities
    
    def calculate_privacy_level(self, original_entities, perturbed_entities):
        """Calculate privacy protection level based on entity preservation."""
        preserved, total = self.calculate_entity_overlap(original_entities, perturbed_entities)
        
        if total == 0:
            return 1.0  # No entities to protect, perfect privacy
        
        privacy_level = 1 - (preserved / total)
        return privacy_level
    
    def analyze_entity_categories(self, original_entities, perturbed_entities):
        """Analyze privacy protection by entity category."""
        category_analysis = {}
        
        for entity_type in self.sensitive_entity_types:
            original_items = original_entities.get(entity_type, [])
            perturbed_items = perturbed_entities.get(entity_type, [])
            
            if not original_items:
                continue
            
            preserved = 0
            for original_item in original_items:
                original_lower = original_item.lower()
                for perturbed_item in perturbed_items:
                    if original_lower in perturbed_item.lower() or perturbed_item.lower() in original_lower:
                        preserved += 1
                        break
            
            protection_level = 1 - (preserved / len(original_items)) if original_items else 1.0
            category_analysis[entity_type] = {
                'original_count': len(original_items),
                'preserved_count': preserved,
                'protection_level': protection_level
            }
        
        return category_analysis
    
    def run_ner_privacy_evaluation(self, max_questions=100):
        """Run NER-based PII privacy evaluation on both methods."""
        print(f"\n{'='*60}")
        print("NER-BASED PII PRIVACY EVALUATION")
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
        
        # Initialize category analysis
        phrase_dp_category_analysis = defaultdict(lambda: {'original_count': 0, 'preserved_count': 0})
        inferdpt_category_analysis = defaultdict(lambda: {'original_count': 0, 'preserved_count': 0})
        
        for i, question_id in enumerate(questions_to_evaluate):
            original = self.original_questions[question_id]
            phrase_dp_perturbed = self.perturbed_questions['phrase_dp'][question_id]
            inferdpt_perturbed = self.perturbed_questions['inferdpt'][question_id]
            
            print(f"\nQuestion {question_id} ({i+1}/{len(questions_to_evaluate)}):")
            print(f"Original: {original[:100]}...")
            
            # Extract entities
            original_entities = self.extract_pii_entities(original)
            phrase_dp_entities = self.extract_pii_entities(phrase_dp_perturbed)
            inferdpt_entities = self.extract_pii_entities(inferdpt_perturbed)
            
            # Calculate privacy levels
            phrase_dp_privacy = self.calculate_privacy_level(original_entities, phrase_dp_entities)
            inferdpt_privacy = self.calculate_privacy_level(original_entities, inferdpt_entities)
            
            self.results['phrase_dp']['privacy_levels'].append(phrase_dp_privacy)
            self.results['inferdpt']['privacy_levels'].append(inferdpt_privacy)
            
            # Analyze categories
            phrase_dp_categories = self.analyze_entity_categories(original_entities, phrase_dp_entities)
            inferdpt_categories = self.analyze_entity_categories(original_entities, inferdpt_entities)
            
            # Aggregate category analysis
            for entity_type, analysis in phrase_dp_categories.items():
                phrase_dp_category_analysis[entity_type]['original_count'] += analysis['original_count']
                phrase_dp_category_analysis[entity_type]['preserved_count'] += analysis['preserved_count']
            
            for entity_type, analysis in inferdpt_categories.items():
                inferdpt_category_analysis[entity_type]['original_count'] += analysis['original_count']
                inferdpt_category_analysis[entity_type]['preserved_count'] += analysis['preserved_count']
            
            # Print detailed results
            print(f"  Original entities: {dict(original_entities)}")
            print(f"  Phrase DP entities: {dict(phrase_dp_entities)}")
            print(f"  InferDPT entities: {dict(inferdpt_entities)}")
            print(f"  Phrase DP Privacy Level: {phrase_dp_privacy:.3f}")
            print(f"  InferDPT Privacy Level: {inferdpt_privacy:.3f}")
        
        # Calculate final category analysis
        for entity_type in phrase_dp_category_analysis:
            total_original = phrase_dp_category_analysis[entity_type]['original_count']
            total_preserved = phrase_dp_category_analysis[entity_type]['preserved_count']
            protection_level = 1 - (total_preserved / total_original) if total_original > 0 else 1.0
            phrase_dp_category_analysis[entity_type]['protection_level'] = protection_level
        
        for entity_type in inferdpt_category_analysis:
            total_original = inferdpt_category_analysis[entity_type]['original_count']
            total_preserved = inferdpt_category_analysis[entity_type]['preserved_count']
            protection_level = 1 - (total_preserved / total_original) if total_original > 0 else 1.0
            inferdpt_category_analysis[entity_type]['protection_level'] = protection_level
        
        self.results['phrase_dp']['entity_analysis'] = dict(phrase_dp_category_analysis)
        self.results['inferdpt']['entity_analysis'] = dict(inferdpt_category_analysis)
        
        return self.results
    
    def print_results(self):
        """Print comprehensive results analysis."""
        print(f"\n{'='*60}")
        print("NER-BASED PII PRIVACY EVALUATION RESULTS")
        print(f"{'='*60}")
        
        # Calculate mean privacy levels
        phrase_dp_mean = np.mean(self.results['phrase_dp']['privacy_levels'])
        inferdpt_mean = np.mean(self.results['inferdpt']['privacy_levels'])
        
        print("Overall Privacy Protection (Higher = Better Privacy):")
        print(f"  Phrase DP Privacy Level: {phrase_dp_mean:.3f}")
        print(f"  InferDPT Privacy Level: {inferdpt_mean:.3f}")
        
        print(f"\nDetailed Privacy Levels:")
        print(f"  Phrase DP: {[f'{p:.3f}' for p in self.results['phrase_dp']['privacy_levels']]}")
        print(f"  InferDPT: {[f'{p:.3f}' for p in self.results['inferdpt']['privacy_levels']]}")
        
        # Comparison
        if phrase_dp_mean > inferdpt_mean:
            print(f"\nPhrase DP provides BETTER privacy protection")
            improvement = ((phrase_dp_mean - inferdpt_mean) / inferdpt_mean) * 100
            print(f"Improvement: {improvement:.1f}%")
        else:
            print(f"\nInferDPT provides BETTER privacy protection")
            improvement = ((inferdpt_mean - phrase_dp_mean) / phrase_dp_mean) * 100
            print(f"Improvement: {improvement:.1f}%")
        
        # Privacy ratings
        for method, mean_privacy in [('phrase_dp', phrase_dp_mean), ('inferdpt', inferdpt_mean)]:
            if mean_privacy >= 0.9:
                rating = "EXCELLENT"
            elif mean_privacy >= 0.7:
                rating = "GOOD"
            elif mean_privacy >= 0.5:
                rating = "MODERATE"
            else:
                rating = "POOR"
            print(f"  {method.upper()} Privacy Rating: {rating}")
        
        # Category analysis
        print(f"\nEntity Category Analysis:")
        all_categories = set(self.results['phrase_dp']['entity_analysis'].keys()) | set(self.results['inferdpt']['entity_analysis'].keys())
        
        for category in sorted(all_categories):
            phrase_dp_analysis = self.results['phrase_dp']['entity_analysis'].get(category, {})
            inferdpt_analysis = self.results['inferdpt']['entity_analysis'].get(category, {})
            
            phrase_dp_protection = phrase_dp_analysis.get('protection_level', 0.0)
            inferdpt_protection = inferdpt_analysis.get('protection_level', 0.0)
            
            phrase_dp_count = phrase_dp_analysis.get('original_count', 0)
            inferdpt_count = inferdpt_analysis.get('original_count', 0)
            
            if phrase_dp_count > 0 or inferdpt_count > 0:
                print(f"  {category}:")
                print(f"    Phrase DP: {phrase_dp_protection:.3f} ({phrase_dp_count} entities)")
                print(f"    InferDPT: {inferdpt_protection:.3f} ({inferdpt_count} entities)")
    
    def save_results(self):
        """Save results to JSON file."""
        # Convert defaultdict to regular dict for JSON serialization
        phrase_dp_analysis = {k: dict(v) for k, v in self.results['phrase_dp']['entity_analysis'].items()}
        inferdpt_analysis = {k: dict(v) for k, v in self.results['inferdpt']['entity_analysis'].items()}
        
        output_data = {
            'evaluation_date': datetime.now().isoformat(),
            'evaluation_method': 'NER-Based PII Privacy Evaluation',
            'methods_evaluated': ['phrase_dp', 'inferdpt'],
            'results': {
                'phrase_dp': {
                    'privacy_levels': [float(p) for p in self.results['phrase_dp']['privacy_levels']],
                    'mean_privacy_level': float(np.mean(self.results['phrase_dp']['privacy_levels'])),
                    'entity_analysis': phrase_dp_analysis
                },
                'inferdpt': {
                    'privacy_levels': [float(p) for p in self.results['inferdpt']['privacy_levels']],
                    'mean_privacy_level': float(np.mean(self.results['inferdpt']['privacy_levels'])),
                    'entity_analysis': inferdpt_analysis
                }
            },
            'comparison': {
                'phrase_dp_better': bool(np.mean(self.results['phrase_dp']['privacy_levels']) > np.mean(self.results['inferdpt']['privacy_levels'])),
                'improvement_percentage': float(abs(np.mean(self.results['phrase_dp']['privacy_levels']) - np.mean(self.results['inferdpt']['privacy_levels'])) / min(np.mean(self.results['phrase_dp']['privacy_levels']), np.mean(self.results['inferdpt']['privacy_levels'])) * 100)
            }
        }
        
        with open('ner_pii_privacy_results.json', 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved as: ner_pii_privacy_results.json")

def main():
    """Main function to run the NER-based PII privacy evaluation."""
    evaluator = NERPIIPrivacyEvaluator()
    results = evaluator.run_ner_privacy_evaluation(max_questions=100)
    evaluator.print_results()
    evaluator.save_results()
    
    print(f"\n{'='*60}")
    print("NER-BASED PII PRIVACY EVALUATION COMPLETE")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
