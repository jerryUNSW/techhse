#!/usr/bin/env python3
"""
Efficient Epsilon Sensitivity Test
==================================

Tests sanitization methods across epsilon values while reusing initialized mechanisms.
"""

import sys
from sanitization_methods import (
    inferdpt_sanitize_text, 
    santext_sanitize_text,
    reset_global_instances
)

def efficient_epsilon_test():
    """Test methods efficiently by reusing initialized mechanisms."""
    
    print("="*60, flush=True)
    print("EFFICIENT EPSILON SENSITIVITY TEST", flush=True)
    print("="*60, flush=True)
    
    sample_text = "A 65-year-old male is treated for anal carcinoma with therapy including external beam radiation."
    print(f"Sample: {sample_text}", flush=True)
    print("="*60, flush=True)
    
    epsilon_values = [1.0, 2.0, 3.0]
    
    # Test InferDPT (should reuse embeddings)
    print("\n--- InferDPT (Reusing Embeddings) ---", flush=True)
    for epsilon in epsilon_values:
        print(f"\nε={epsilon}:", flush=True)
        try:
            result = inferdpt_sanitize_text(sample_text, epsilon=epsilon)
            print(f"Result: {result[:60]}...", flush=True)
        except Exception as e:
            print(f"Error: {e}", flush=True)
    
    # Test SANTEXT+ (should reuse model and vocabulary)
    print("\n--- SANTEXT+ (Reusing Model & Vocabulary) ---", flush=True)
    
    # Initialize once with first text
    print("Initializing SANTEXT+ mechanism...", flush=True)
    try:
        # Build vocabulary once
        first_result = santext_sanitize_text(sample_text, epsilon=1.0)
        print(f"Initial result (ε=1.0): {first_result[:60]}...", flush=True)
        
        # Now test other epsilon values (should reuse vocabulary)
        for epsilon in epsilon_values[1:]:
            print(f"\nε={epsilon}:", flush=True)
            result = santext_sanitize_text(sample_text, epsilon=epsilon)
            print(f"Result: {result[:60]}...", flush=True)
            
    except Exception as e:
        print(f"Error: {e}", flush=True)
    
    print("\n" + "="*60, flush=True)
    print("EFFICIENT TEST COMPLETED", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    efficient_epsilon_test()
