#!/usr/bin/env python3
"""
Explain SANTEXT+ Randomness
Shows why multiple versions are generated and how the mechanism works
"""

import numpy as np
import random
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances

def demonstrate_santext_randomness():
    """Demonstrate why SANTEXT+ produces different results each time"""
    
    print("SANTEXT+ Randomness Explanation")
    print("=" * 50)
    
    # Example: word "capital" with 3 possible replacements
    original_word = "capital"
    candidate_words = ["capital", "seat", "center", "major", "largest"]
    
    # Simulate embeddings (simplified)
    # In reality, these would be from GloVe/BERT embeddings
    embeddings = np.array([
        [1.0, 0.0, 0.0],  # capital
        [0.8, 0.2, 0.0],  # seat  
        [0.7, 0.3, 0.0],  # center
        [0.3, 0.7, 0.0],  # major
        [0.1, 0.9, 0.0]   # largest
    ])
    
    print(f"Original word: '{original_word}'")
    print(f"Candidate replacements: {candidate_words}")
    print()
    
    # Show probability distributions for different epsilon values
    for epsilon in [1.0, 2.0, 3.0]:
        print(f"Epsilon = {epsilon}:")
        print("-" * 20)
        
        # Calculate probabilities using exponential mechanism
        original_embedding = embeddings[0:1]  # "capital"
        distances = euclidean_distances(original_embedding, embeddings)
        similarities = -distances
        probabilities = softmax(epsilon * similarities / 2, axis=1)[0]
        
        print("Probability distribution:")
        for word, prob in zip(candidate_words, probabilities):
            print(f"  {word:8s}: {prob:.4f} ({prob*100:.1f}%)")
        
        print()
        
        # Show 5 random samples for this epsilon
        print("5 random samples:")
        for i in range(5):
            # Set different random seed for each sample
            np.random.seed(42 + i)
            random.seed(42 + i)
            
            # Sample according to probabilities
            chosen_idx = np.random.choice(len(probabilities), p=probabilities)
            chosen_word = candidate_words[chosen_idx]
            print(f"  Sample {i+1}: {chosen_word}")
        
        print()
    
    print("Key Points:")
    print("1. SANTEXT+ uses PROBABILITY SAMPLING, not deterministic replacement")
    print("2. Each word has a probability distribution over all possible replacements")
    print("3. Higher epsilon = more likely to keep similar words")
    print("4. Lower epsilon = more uniform distribution (more privacy)")
    print("5. Multiple runs = multiple samples from the same probability distribution")

def show_deterministic_vs_random():
    """Compare deterministic vs random approaches"""
    
    print("\n" + "=" * 50)
    print("Deterministic vs Random Approaches")
    print("=" * 50)
    
    print("DETERMINISTIC (what you might expect):")
    print("  Input: 'What is the capital of France?'")
    print("  Output: 'What is the seat of Europe?' (always the same)")
    print()
    
    print("RANDOM (SANTEXT+ approach):")
    print("  Input: 'What is the capital of France?'")
    print("  Run 1: 'What is the seat of Europe?'")
    print("  Run 2: 'What is the center of Europe?'") 
    print("  Run 3: 'What is the major of Europe?'")
    print("  (Different each time due to probability sampling)")
    print()
    
    print("Why Random?")
    print("- Differential Privacy requires randomness")
    print("- Prevents deterministic attacks")
    print("- Provides provable privacy guarantees")
    print("- Each output is a valid sample from the mechanism")

if __name__ == "__main__":
    demonstrate_santext_randomness()
    show_deterministic_vs_random()
