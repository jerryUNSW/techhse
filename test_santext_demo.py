#!/usr/bin/env python3
"""
Simplified SANTEXT+ Demo
Demonstrates the core SANTEXT+ sanitization mechanism on a single question
with different epsilon values (1, 2, 3).
"""

import numpy as np
import random
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
import torch

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def cal_probability(word_embed_1, word_embed_2, epsilon=2.0):
    """Calculate probability matrix for SANTEXT mechanism"""
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix

def santext_plus_sanitize(text, word_embeddings, word_list, epsilon=2.0, p=0.3, sensitive_words=None):
    """
    Simplified SANTEXT+ sanitization
    
    Args:
        text: Input text to sanitize
        word_embeddings: Matrix of word embeddings
        word_list: List of words corresponding to embeddings
        epsilon: Privacy parameter
        p: Probability of sanitizing non-sensitive words
        sensitive_words: Set of sensitive words (if None, treats all as sensitive)
    """
    if sensitive_words is None:
        sensitive_words = set(word_list)  # Treat all words as sensitive for demo
    
    # Create word to index mapping
    word2id = {word: i for i, word in enumerate(word_list)}
    
    # Calculate probability matrix
    prob_matrix = cal_probability(word_embeddings, word_embeddings, epsilon=epsilon)
    
    # Tokenize text (simple split for demo)
    tokens = text.lower().split()
    sanitized_tokens = []
    
    for token in tokens:
        # Clean token (remove punctuation for demo)
        clean_token = ''.join(c for c in token if c.isalnum())
        
        if clean_token in word2id:
            # In vocabulary
            if clean_token in sensitive_words:
                # Sensitive word - always sanitize
                word_idx = word2id[clean_token]
                sampling_prob = prob_matrix[word_idx]
                sampling_idx = np.random.choice(len(sampling_prob), 1, p=sampling_prob)[0]
                sanitized_tokens.append(word_list[sampling_idx])
            else:
                # Non-sensitive word - sanitize with probability p
                if random.random() <= p:
                    word_idx = word2id[clean_token]
                    sampling_prob = prob_matrix[word_idx]
                    sampling_idx = np.random.choice(len(sampling_prob), 1, p=sampling_prob)[0]
                    sanitized_tokens.append(word_list[sampling_idx])
                else:
                    # Keep original
                    sanitized_tokens.append(clean_token)
        else:
            # Out of vocabulary - random replacement
            random_word = random.choice(word_list)
            sanitized_tokens.append(random_word)
    
    return ' '.join(sanitized_tokens)

def main():
    # Test question
    test_question = "What is the capital of France?"
    print(f"Original question: {test_question}")
    print("=" * 60)
    
    # Load a small sentence transformer model for word embeddings
    print("Loading sentence transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create a vocabulary from the test question and some common words
    vocab_words = test_question.lower().split() + [
        'what', 'is', 'the', 'capital', 'of', 'france', 'paris', 'city', 'country',
        'where', 'located', 'europe', 'government', 'seat', 'administrative',
        'center', 'major', 'population', 'largest', 'famous', 'known', 'called'
    ]
    
    # Remove duplicates and get embeddings
    vocab_words = list(set(vocab_words))
    print(f"Vocabulary size: {len(vocab_words)}")
    
    # Get embeddings for vocabulary
    word_embeddings = model.encode(vocab_words)
    print(f"Embedding shape: {word_embeddings.shape}")
    
    # Define some sensitive words (for demo purposes)
    sensitive_words = {'france', 'paris', 'capital'}
    
    # Test different epsilon values
    epsilon_values = [1.0, 2.0, 3.0]
    
    print("\nSANTEXT+ Sanitization Results:")
    print("=" * 60)
    
    for epsilon in epsilon_values:
        print(f"\nEpsilon = {epsilon}:")
        print("-" * 30)
        
        # Generate 3 sanitized versions for each epsilon
        for i in range(3):
            sanitized = santext_plus_sanitize(
                test_question, 
                word_embeddings, 
                vocab_words, 
                epsilon=epsilon,
                p=0.3,
                sensitive_words=sensitive_words
            )
            print(f"  Version {i+1}: {sanitized}")
    
    print("\n" + "=" * 60)
    print("Analysis:")
    print("- Lower epsilon (1.0): More privacy, more distortion")
    print("- Higher epsilon (3.0): Less privacy, less distortion")
    print("- Sensitive words ('france', 'paris', 'capital') are always sanitized")
    print("- Non-sensitive words are sanitized with probability p=0.3")

if __name__ == "__main__":
    main()
