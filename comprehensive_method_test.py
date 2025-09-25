#!/usr/bin/env python3
"""
Comprehensive Method Test
========================

Test all 5 sanitization methods to see which ones work and which have issues.
"""

import sys

def test_all_methods():
    """Test all 5 sanitization methods."""
    
    print("="*70, flush=True)
    print("COMPREHENSIVE SANITIZATION METHODS TEST", flush=True)
    print("="*70, flush=True)
    
    sample_text = "A 65-year-old male is treated for anal carcinoma with therapy including external beam radiation."
    print(f"Sample: {sample_text}", flush=True)
    print("="*70, flush=True)
    
    epsilon = 1.0
    
    # Test 1: PhraseDP
    print("\n1. PHRASEDP TEST", flush=True)
    print("-" * 30, flush=True)
    try:
        from sanitization_methods import phrasedp_sanitize_text
        print("✅ PhraseDP function imported successfully", flush=True)
        # This will fail without Nebius client, but let's see the error
        result = phrasedp_sanitize_text(sample_text, epsilon=epsilon)
        print(f"✅ PhraseDP result: {result[:50]}...", flush=True)
    except Exception as e:
        print(f"❌ PhraseDP error: {e}", flush=True)
    
    # Test 2: InferDPT
    print("\n2. INFERDPT TEST", flush=True)
    print("-" * 30, flush=True)
    try:
        from sanitization_methods import inferdpt_sanitize_text
        print("✅ InferDPT function imported successfully", flush=True)
        result = inferdpt_sanitize_text(sample_text, epsilon=epsilon)
        print(f"✅ InferDPT result: {result[:50]}...", flush=True)
    except Exception as e:
        print(f"❌ InferDPT error: {e}", flush=True)
    
    # Test 3: SANTEXT+
    print("\n3. SANTEXT+ TEST", flush=True)
    print("-" * 30, flush=True)
    try:
        from sanitization_methods import santext_sanitize_text
        print("✅ SANTEXT+ function imported successfully", flush=True)
        result = santext_sanitize_text(sample_text, epsilon=epsilon)
        print(f"✅ SANTEXT+ result: {result[:50]}...", flush=True)
    except Exception as e:
        print(f"❌ SANTEXT+ error: {e}", flush=True)
    
    # Test 4: CUSTEXT+
    print("\n4. CUSTEXT+ TEST", flush=True)
    print("-" * 30, flush=True)
    try:
        from sanitization_methods import custext_sanitize_text
        print("✅ CUSTEXT+ function imported successfully", flush=True)
        result = custext_sanitize_text(sample_text, epsilon=epsilon)
        print(f"✅ CUSTEXT+ result: {result[:50]}...", flush=True)
    except Exception as e:
        print(f"❌ CUSTEXT+ error: {e}", flush=True)
    
    # Test 5: CluSanT
    print("\n5. CLUSANT TEST", flush=True)
    print("-" * 30, flush=True)
    try:
        from sanitization_methods import clusant_sanitize_text
        print("✅ CluSanT function imported successfully", flush=True)
        result = clusant_sanitize_text(sample_text, epsilon=epsilon)
        print(f"✅ CluSanT result: {result[:50]}...", flush=True)
    except Exception as e:
        print(f"❌ CluSanT error: {e}", flush=True)
    
    # Summary
    print("\n" + "="*70, flush=True)
    print("SUMMARY", flush=True)
    print("="*70, flush=True)
    print("This test shows which methods are working and which need fixes.", flush=True)
    print("Methods that show ❌ need to be debugged and fixed.", flush=True)
    print("="*70, flush=True)

if __name__ == "__main__":
    test_all_methods()

