#!/usr/bin/env python3
"""
Analyze the actual utility function implementation in dp_sanitizer.py
to understand how similarity is converted to utility.
"""

import numpy as np

def analyze_utility_function():
    """Analyze the utility function implementation in dp_sanitizer.py."""
    print("🔍 ANALYZING UTILITY FUNCTION IMPLEMENTATION")
    print("=" * 80)
    
    print("📋 CODE ANALYSIS FROM dp_sanitizer.py:")
    print("-" * 50)
    print("Line 41: similarities = cosine_similarity(target_embedding, candidate_embeddings_matrix)[0]")
    print("Line 55: distances = 1 - similarities")
    print("Line 58: p_unnorm = np.exp(-epsilon * distances)")
    print("Line 59: p_norm = p_unnorm / np.sum(p_unnorm)")
    print("Line 62: return np.random.choice(candidate_phrases, p=p_norm)")
    print()
    
    print("🔬 STEP-BY-STEP ANALYSIS:")
    print("-" * 50)
    
    # Simulate the process with example values
    similarities = np.array([0.2, 0.4, 0.6, 0.8, 0.9])  # Example similarity scores
    epsilon = 0.5
    
    print("Step 1: Compute similarities")
    print(f"  similarities = {similarities}")
    print()
    
    print("Step 2: Convert similarity to distance")
    distances = 1 - similarities
    print(f"  distances = 1 - similarities = {distances}")
    print("  Note: distance = 1 - similarity")
    print("  Higher similarity → Lower distance")
    print("  Lower similarity → Higher distance")
    print()
    
    print("Step 3: Apply exponential mechanism")
    print(f"  p_unnorm = exp(-epsilon * distances)")
    print(f"  p_unnorm = exp(-{epsilon} * {distances})")
    p_unnorm = np.exp(-epsilon * distances)
    print(f"  p_unnorm = {p_unnorm}")
    print()
    
    print("Step 4: Normalize probabilities")
    p_norm = p_unnorm / np.sum(p_unnorm)
    print(f"  p_norm = {p_norm}")
    print()
    
    print("🎯 KEY INSIGHT:")
    print("-" * 50)
    print("The utility function is actually:")
    print("  utility = -distance = -(1 - similarity) = similarity - 1")
    print("  But since we use exp(-epsilon * distance), it becomes:")
    print("  P(candidate) ∝ exp(-epsilon * (1 - similarity))")
    print("  P(candidate) ∝ exp(-epsilon + epsilon * similarity)")
    print("  P(candidate) ∝ exp(epsilon * similarity) * exp(-epsilon)")
    print("  P(candidate) ∝ exp(epsilon * similarity)")
    print()
    
    print("So the effective utility function is:")
    print("  utility(candidate) = similarity(candidate, original)")
    print()
    
    print("🔍 VERIFICATION WITH DIFFERENT EPSILON VALUES:")
    print("-" * 50)
    
    for eps in [0.5, 1.0, 2.0]:
        print(f"Epsilon = {eps}:")
        p_unnorm = np.exp(-eps * distances)
        p_norm = p_unnorm / np.sum(p_unnorm)
        
        print(f"  Probabilities: {p_norm}")
        print(f"  Highest prob: {np.max(p_norm):.3f} (similarity = {similarities[np.argmax(p_norm)]:.1f})")
        print(f"  Lowest prob:  {np.min(p_norm):.3f} (similarity = {similarities[np.argmin(p_norm)]:.1f})")
        print()
    
    print("🎯 EXPECTED BEHAVIOR:")
    print("-" * 50)
    print("With utility = similarity:")
    print("  ε = 0.5: Should favor HIGH similarity (low privacy)")
    print("  ε = 2.0: Should favor LOW similarity (high privacy)")
    print()
    print("But we WANT:")
    print("  ε = 0.5: Should favor LOW similarity (high privacy)")
    print("  ε = 2.0: Should favor HIGH similarity (low privacy)")
    print()
    
    print("🚨 THE PROBLEM:")
    print("-" * 50)
    print("The current implementation uses:")
    print("  P(candidate) ∝ exp(epsilon * similarity)")
    print()
    print("This means higher epsilon favors HIGHER similarity, which is BACKWARDS!")
    print("For privacy, higher epsilon should favor LOWER similarity.")
    print()
    
    print("💡 THE FIX:")
    print("-" * 50)
    print("Change line 58 in dp_sanitizer.py from:")
    print("  p_unnorm = np.exp(-epsilon * distances)")
    print("to:")
    print("  p_unnorm = np.exp(epsilon * distances)")
    print()
    print("This gives us:")
    print("  P(candidate) ∝ exp(epsilon * distance)")
    print("  P(candidate) ∝ exp(epsilon * (1 - similarity))")
    print()
    print("Now:")
    print("  ε = 0.5: Favors HIGH distance (LOW similarity) → HIGH privacy ✅")
    print("  ε = 2.0: Favors LOW distance (HIGH similarity) → LOW privacy ✅")
    print()
    
    print("🔧 DEMONSTRATING THE FIX:")
    print("-" * 50)
    
    for eps in [0.5, 1.0, 2.0]:
        print(f"Epsilon = {eps} (FIXED):")
        p_unnorm_fixed = np.exp(eps * distances)  # Fixed version
        p_norm_fixed = p_unnorm_fixed / np.sum(p_unnorm_fixed)
        
        print(f"  Probabilities: {p_norm_fixed}")
        print(f"  Highest prob: {np.max(p_norm_fixed):.3f} (similarity = {similarities[np.argmax(p_norm_fixed)]:.1f})")
        print(f"  Lowest prob:  {np.min(p_norm_fixed):.3f} (similarity = {similarities[np.argmin(p_norm_fixed)]:.1f})")
        print()
    
    print("📋 SUMMARY:")
    print("-" * 50)
    print("1. ✅ Current code: P(candidate) ∝ exp(-epsilon * distance)")
    print("2. ❌ This is equivalent to: P(candidate) ∝ exp(epsilon * similarity)")
    print("3. ❌ Higher epsilon → Higher similarity → LOWER privacy (backwards!)")
    print("4. 🔧 Fix: Change to P(candidate) ∝ exp(epsilon * distance)")
    print("5. ✅ Fixed: Higher epsilon → Lower similarity → HIGHER privacy")
    print()
    print("The issue is a simple sign error in the exponential mechanism!")

if __name__ == "__main__":
    analyze_utility_function()

