#!/usr/bin/env python3
"""
BERT-based Semantic Similarity Evaluation for Phrase DP vs InferDPT
Measures semantic similarity between original and perturbed questions using sentence embeddings.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BERTSimilarityEvaluator:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        """Initialize the BERT similarity evaluator."""
        logger.info(f"Loading BERT model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Model loaded successfully")
    
    def extract_question_text(self, question_data):
        """Extract the main question text from the data structure."""
        if isinstance(question_data, dict):
            # Handle different possible structures
            if 'question' in question_data:
                return question_data['question']
            elif 'text' in question_data:
                return question_data['text']
            elif 'stem' in question_data:
                return question_data['stem']
            else:
                # If it's a dict, try to get the first string value
                for key, value in question_data.items():
                    if isinstance(value, str) and len(value) > 10:
                        return value
        elif isinstance(question_data, str):
            return question_data
        
        return str(question_data)
    
    def calculate_similarity(self, original_questions, perturbed_questions):
        """Calculate BERT-based semantic similarity between original and perturbed questions."""
        logger.info(f"Calculating similarity for {len(original_questions)} question pairs")
        
        similarities = []
        
        for i, (orig, pert) in enumerate(zip(original_questions, perturbed_questions)):
            try:
                # Extract question text
                orig_text = self.extract_question_text(orig)
                pert_text = self.extract_question_text(pert)
                
                # Generate embeddings
                orig_embedding = self.model.encode([orig_text])
                pert_embedding = self.model.encode([pert_text])
                
                # Calculate cosine similarity
                similarity = cosine_similarity(orig_embedding, pert_embedding)[0][0]
                similarities.append(similarity)
                
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(original_questions)} questions")
                    
            except Exception as e:
                logger.error(f"Error processing question {i}: {e}")
                similarities.append(0.0)  # Default to 0 similarity on error
        
        return similarities
    
    def evaluate_method(self, method_name, original_questions, perturbed_questions):
        """Evaluate a specific perturbation method."""
        logger.info(f"Evaluating {method_name}")
        
        similarities = self.calculate_similarity(original_questions, perturbed_questions)
        
        # Calculate statistics
        mean_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        
        # Count questions by similarity ranges
        high_similarity = sum(1 for s in similarities if s > 0.7)
        medium_similarity = sum(1 for s in similarities if 0.4 <= s <= 0.7)
        low_similarity = sum(1 for s in similarities if s < 0.4)
        
        results = {
            'method': method_name,
            'total_questions': len(similarities),
            'mean_similarity': float(mean_similarity),
            'std_similarity': float(std_similarity),
            'min_similarity': float(min_similarity),
            'max_similarity': float(max_similarity),
            'high_similarity_count': high_similarity,
            'medium_similarity_count': medium_similarity,
            'low_similarity_count': low_similarity,
            'high_similarity_percentage': float(high_similarity / len(similarities) * 100),
            'medium_similarity_percentage': float(medium_similarity / len(similarities) * 100),
            'low_similarity_percentage': float(low_similarity / len(similarities) * 100),
            'individual_similarities': similarities
        }
        
        logger.info(f"{method_name} Results:")
        logger.info(f"  Mean Similarity: {mean_similarity:.4f}")
        logger.info(f"  High Similarity (>0.7): {high_similarity} ({high_similarity/len(similarities)*100:.1f}%)")
        logger.info(f"  Medium Similarity (0.4-0.7): {medium_similarity} ({medium_similarity/len(similarities)*100:.1f}%)")
        logger.info(f"  Low Similarity (<0.4): {low_similarity} ({low_similarity/len(similarities)*100:.1f}%)")
        
        return results
    
    def save_results(self, results, output_file):
        """Save results to JSON file."""
        # Remove individual similarities for cleaner output (too large)
        clean_results = {}
        for method, data in results.items():
            clean_data = data.copy()
            clean_data.pop('individual_similarities', None)
            clean_results[method] = clean_data
        
        with open(output_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")

def load_questions_from_experiment_log(log_file, num_questions=100):
    """Load questions from the experiment log file."""
    logger.info(f"Loading questions from {log_file}")
    
    original_questions = []
    phrase_dp_questions = []
    inferdpt_questions = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by question blocks using the actual separator (with ANSI codes)
        question_blocks = content.split('--- Question ')
        logger.info(f"Found {len(question_blocks)} question blocks")
        
        for block_idx, block in enumerate(question_blocks[1:]):  # Skip the first empty block
            lines = block.strip().split('\n')
            
            # Extract original question (starts with "Question:")
            orig_question = None
            for line in lines:
                if line.startswith('Question:'):
                    orig_question = line.replace('Question:', '').strip()
                    break
            
            if not orig_question:
                continue
            
            # Extract Phrase DP perturbed question
            phrase_dp_question = None
            for i, line in enumerate(lines):
                if 'Scenario 3.1' in line and 'Phrase DP' in line:
                    # Look for "Perturbed Question:" in the next few lines
                    for j in range(i, min(i+100, len(lines))):
                        if 'Perturbed Question:' in lines[j]:
                            phrase_dp_question = lines[j].replace('Perturbed Question:', '').strip()
                            break
                    break
            
            # Extract InferDPT perturbed question
            inferdpt_question = None
            for i, line in enumerate(lines):
                if 'Scenario 3.2' in line and 'InferDPT' in line:
                    # Look for "Perturbed Question:" in the next few lines
                    for j in range(i, min(i+100, len(lines))):
                        if 'Perturbed Question:' in lines[j]:
                            inferdpt_question = lines[j].replace('Perturbed Question:', '').strip()
                            break
                    break
            
            # Add questions if we found them
            if orig_question and phrase_dp_question and inferdpt_question:
                original_questions.append(orig_question)
                phrase_dp_questions.append(phrase_dp_question)
                inferdpt_questions.append(inferdpt_question)
                logger.info(f"Successfully loaded question {len(original_questions)}")
                
                # Stop when we have enough questions
                if len(original_questions) >= num_questions:
                    break
            else:
                if block_idx < 3:  # Only log first few failures
                    logger.info(f"Block {block_idx}: orig={bool(orig_question)}, phrase_dp={bool(phrase_dp_question)}, inferdpt={bool(inferdpt_question)}")
                    if block_idx == 0:
                        logger.info(f"First few lines of block 0: {lines[:5]}")
        
        # Trim to requested number
        original_questions = original_questions[:num_questions]
        phrase_dp_questions = phrase_dp_questions[:num_questions]
        inferdpt_questions = inferdpt_questions[:num_questions]
        
        logger.info(f"Loaded {len(original_questions)} original questions")
        logger.info(f"Loaded {len(phrase_dp_questions)} Phrase DP questions")
        logger.info(f"Loaded {len(inferdpt_questions)} InferDPT questions")
        
        return original_questions, phrase_dp_questions, inferdpt_questions
        
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return [], [], []

def main():
    parser = argparse.ArgumentParser(description='BERT-based semantic similarity evaluation')
    parser.add_argument('--log_file', default='complete-550.txt', 
                       help='Path to experiment log file')
    parser.add_argument('--num_questions', type=int, default=100,
                       help='Number of questions to evaluate')
    parser.add_argument('--output', default='bert_similarity_results.json',
                       help='Output file for results')
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2',
                       help='BERT model to use for embeddings')
    
    args = parser.parse_args()
    
    # Load questions
    original_questions, phrase_dp_questions, inferdpt_questions = load_questions_from_experiment_log(
        args.log_file, args.num_questions
    )
    
    if not original_questions:
        logger.error("No questions loaded. Exiting.")
        return
    
    # Initialize evaluator
    evaluator = BERTSimilarityEvaluator(args.model)
    
    # Evaluate both methods
    results = {}
    
    if phrase_dp_questions:
        results['phrase_dp'] = evaluator.evaluate_method(
            'Phrase DP', original_questions, phrase_dp_questions
        )
    
    if inferdpt_questions:
        results['inferdpt'] = evaluator.evaluate_method(
            'InferDPT', original_questions, inferdpt_questions
        )
    
    # Save results
    evaluator.save_results(results, args.output)
    
    # Print summary comparison
    if 'phrase_dp' in results and 'inferdpt' in results:
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        print(f"Phrase DP Mean Similarity: {results['phrase_dp']['mean_similarity']:.4f}")
        print(f"InferDPT Mean Similarity: {results['inferdpt']['mean_similarity']:.4f}")
        print(f"Difference: {results['phrase_dp']['mean_similarity'] - results['inferdpt']['mean_similarity']:.4f}")
        print(f"Phrase DP High Similarity: {results['phrase_dp']['high_similarity_percentage']:.1f}%")
        print(f"InferDPT High Similarity: {results['inferdpt']['high_similarity_percentage']:.1f}%")

if __name__ == "__main__":
    main()
