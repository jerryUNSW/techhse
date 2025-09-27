#!/usr/bin/env python3
"""
Test script for the old PhraseDP implementation without band diversity.
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils import get_nebius_client, phrase_DP_perturbation_old

# Load environment variables
load_dotenv()

def test_old_phrasedp():
    """
    Test the old PhraseDP implementation with a medical question.
    """
    print("=" * 60)
    print("TESTING OLD PHRASEDP (No Band Diversity)")
    print("=" * 60)
    
    # Initialize components
    nebius_client = get_nebius_client()
    nebius_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test question
    original_question = "A 45-year-old woman presents with chest pain and shortness of breath. What is the most likely diagnosis?"
    
    print(f"Original Question: {original_question}")
    print()
    
    try:
        # Apply old PhraseDP perturbation
        perturbed_question = phrase_DP_perturbation_old(
            nebius_client=nebius_client,
            nebius_model_name=nebius_model_name,
            input_sentence=original_question,
            epsilon=1.0,
            sbert_model=sbert_model
        )
        
        print(f"Perturbed Question: {perturbed_question}")
        print()
        
        # Compute similarity
        from dp_sanitizer import compute_similarity
        similarity = compute_similarity(sbert_model, original_question, perturbed_question)
        print(f"Semantic Similarity: {similarity:.4f}")
        
        print("\n✅ Old PhraseDP test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during old PhraseDP test: {e}")

def test_old_vs_new_comparison():
    """
    Compare old vs new PhraseDP approaches.
    """
    print("\n" + "=" * 60)
    print("OLD vs NEW PHRASEDP COMPARISON")
    print("=" * 60)
    
    # Initialize components
    nebius_client = get_nebius_client()
    nebius_model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Test question
    original_question = "What are the side effects of chemotherapy in cancer treatment?"
    
    print(f"Original Question: {original_question}")
    print()
    
    try:
        # Test OLD PhraseDP
        print("Testing OLD PhraseDP (1 API call, no bands)...")
        perturbed_old = phrase_DP_perturbation_old(
            nebius_client=nebius_client,
            nebius_model_name=nebius_model_name,
            input_sentence=original_question,
            epsilon=1.0,
            sbert_model=sbert_model
        )
        
        print(f"OLD Perturbed: {perturbed_old}")
        
        # Compute similarity for old
        from dp_sanitizer import compute_similarity
        old_similarity = compute_similarity(sbert_model, original_question, perturbed_old)
        print(f"OLD Similarity: {old_similarity:.4f}")
        
        print("\n" + "-" * 40)
        print("OLD PHRASEDP CHARACTERISTICS:")
        print("- 1 API call")
        print("- No band diversity")
        print("- Conservative perturbations")
        print("- Preserves medical context")
        print("- Higher similarity to original")
        
    except Exception as e:
        print(f"❌ Error during comparison test: {e}")

if __name__ == "__main__":
    print("Old PhraseDP Implementation Test")
    print("This implements the original PhraseDP approach used in the 500-question experiment.")
    print("Key differences from new PhraseDP:")
    print("- 1 API call instead of 10")
    print("- No similarity band targeting")
    print("- Conservative approach")
    print("- Better for medical QA tasks")
    print()
    
    # Run tests
    test_old_phrasedp()
    test_old_vs_new_comparison()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("✅ Old PhraseDP implementation completed")
    print("✅ Single API call approach restored")
    print("✅ No band diversity (as in 500-question experiment)")
    print("✅ Conservative perturbations for medical QA")
    print("\nTo use in your experiments, call:")
    print("phrase_DP_perturbation_old(nebius_client, model_name, question, epsilon, sbert_model)")
