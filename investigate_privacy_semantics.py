#!/usr/bin/env python3
"""
Investigate the privacy semantics and why epsilon=0.5 isn't selecting less similar candidates.
"""

import json
import numpy as np

def investigate_privacy_semantics():
    """Investigate why epsilon=0.5 isn't providing strong privacy (low similarity)."""
    print("🔍 INVESTIGATING PRIVACY SEMANTICS")
    print("=" * 80)
    
    print("📋 YOUR PRIVACY REQUIREMENTS (CORRECT):")
    print("-" * 50)
    print("ε = 0.5 (small): Strong privacy → Less similar candidates")
    print("ε = 2.0 (large): Weak privacy → More similar candidates")
    print()
    
    print("🔬 CURRENT IMPLEMENTATION ANALYSIS:")
    print("-" * 50)
    print("From dp_sanitizer.py:")
    print("  Line 55: distances = 1 - similarities")
    print("  Line 58: p_unnorm = np.exp(-epsilon * distances)")
    print()
    print("This gives us:")
    print("  P(candidate) ∝ exp(-epsilon * distance)")
    print("  P(candidate) ∝ exp(-epsilon * (1 - similarity))")
    print("  P(candidate) ∝ exp(-epsilon + epsilon * similarity)")
    print("  P(candidate) ∝ exp(epsilon * similarity) * exp(-epsilon)")
    print("  P(candidate) ∝ exp(epsilon * similarity)")
    print()
    
    print("🎯 CURRENT BEHAVIOR:")
    print("-" * 50)
    print("P(candidate) ∝ exp(epsilon * similarity)")
    print("  ε = 0.5: Favors HIGH similarity → WEAK privacy ❌")
    print("  ε = 2.0: Favors HIGH similarity → WEAK privacy ❌")
    print()
    print("This is BACKWARDS from what you want!")
    print()
    
    print("💡 THE CORRECT IMPLEMENTATION:")
    print("-" * 50)
    print("For your privacy requirements, we need:")
    print("  P(candidate) ∝ exp(-epsilon * similarity)")
    print()
    print("This gives us:")
    print("  ε = 0.5: Favors LOW similarity → STRONG privacy ✅")
    print("  ε = 2.0: Favors HIGH similarity → WEAK privacy ✅")
    print()
    
    print("🔧 THE FIX:")
    print("-" * 50)
    print("Change line 58 in dp_sanitizer.py from:")
    print("  p_unnorm = np.exp(-epsilon * distances)")
    print("to:")
    print("  p_unnorm = np.exp(-epsilon * similarities)")
    print()
    print("This gives us:")
    print("  P(candidate) ∝ exp(-epsilon * similarity)")
    print()
    
    print("🔬 DEMONSTRATION:")
    print("-" * 50)
    
    # Example similarities
    similarities = np.array([0.2, 0.4, 0.6, 0.8, 0.9])
    
    print("Example candidates with similarities:", similarities)
    print()
    
    for eps in [0.5, 1.0, 2.0]:
        print(f"Epsilon = {eps}:")
        
        # Current (wrong) implementation
        distances = 1 - similarities
        p_unnorm_current = np.exp(-eps * distances)
        p_norm_current = p_unnorm_current / np.sum(p_unnorm_current)
        
        # Correct implementation
        p_unnorm_correct = np.exp(-eps * similarities)
        p_norm_correct = p_unnorm_correct / np.sum(p_unnorm_correct)
        
        print(f"  Current (wrong):")
        print(f"    Highest prob: {np.max(p_norm_current):.3f} (similarity = {similarities[np.argmax(p_norm_current)]:.1f})")
        print(f"    Lowest prob:  {np.min(p_norm_current):.3f} (similarity = {similarities[np.argmin(p_norm_current)]:.1f})")
        
        print(f"  Correct:")
        print(f"    Highest prob: {np.max(p_norm_correct):.3f} (similarity = {similarities[np.argmax(p_norm_correct)]:.1f})")
        print(f"    Lowest prob:  {np.min(p_norm_correct):.3f} (similarity = {similarities[np.argmin(p_norm_correct)]:.1f})")
        print()
    
    print("🎯 VERIFICATION:")
    print("-" * 50)
    print("With the correct implementation:")
    print("  ε = 0.5: Should favor similarity=0.2 (low similarity = strong privacy)")
    print("  ε = 2.0: Should favor similarity=0.9 (high similarity = weak privacy)")
    print()
    
    print("📊 WHY CURRENT IMPLEMENTATION FAILS:")
    print("-" * 50)
    print("The current implementation uses distance instead of similarity:")
    print("  distance = 1 - similarity")
    print("  P(candidate) ∝ exp(-epsilon * distance)")
    print("  P(candidate) ∝ exp(-epsilon * (1 - similarity))")
    print("  P(candidate) ∝ exp(epsilon * similarity)")
    print()
    print("This makes higher epsilon favor HIGHER similarity, which is backwards!")
    print()
    
    print("🔧 ALTERNATIVE FIX:")
    print("-" * 50)
    print("Instead of changing the distance calculation, we can change the sign:")
    print("  Change: p_unnorm = np.exp(-epsilon * distances)")
    print("  To:     p_unnorm = np.exp(epsilon * distances)")
    print()
    print("This gives us:")
    print("  P(candidate) ∝ exp(epsilon * distance)")
    print("  P(candidate) ∝ exp(epsilon * (1 - similarity))")
    print()
    print("Now:")
    print("  ε = 0.5: Favors HIGH distance (LOW similarity) → STRONG privacy ✅")
    print("  ε = 2.0: Favors LOW distance (HIGH similarity) → WEAK privacy ✅")
    print()
    
    print("📋 SUMMARY:")
    print("-" * 50)
    print("1. ✅ Your privacy requirements are correct")
    print("2. ❌ Current implementation is backwards")
    print("3. 🔧 Fix: Change sign in exponential mechanism")
    print("4. 🎯 Result: Epsilon will work as expected for privacy")

if __name__ == "__main__":
    investigate_privacy_semantics()

