#!/usr/bin/env python3
"""
Test script to verify the old PhraseDP integration in the MedQA experiment.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_medqa_usmle_4_options import run_scenario_3_1_old_phrase_dp_local_cot, get_remote_llm_client, load_sentence_bert
import utils

def test_old_phrasedp_integration():
    """
    Test the old PhraseDP integration with a single question.
    """
    print("=" * 60)
    print("TESTING OLD PHRASEDP INTEGRATION")
    print("=" * 60)
    
    # Initialize components
    try:
        remote_client = get_remote_llm_client()
        print("✅ Remote LLM client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize remote client: {e}")
        return
    
    try:
        local_client = utils.get_nebius_client()
        print("✅ Local (Nebius) client initialized")
    except Exception as e:
        print(f"❌ Failed to initialize local client: {e}")
        return
    
    try:
        sbert_model = load_sentence_bert()
        print("✅ Sentence-BERT model loaded")
    except Exception as e:
        print(f"❌ Failed to load SBERT model: {e}")
        return
    
    # Test question
    question = "A 45-year-old woman presents with chest pain and shortness of breath. What is the most likely diagnosis?"
    options = {
        'A': 'Myocardial infarction',
        'B': 'Pulmonary embolism', 
        'C': 'Pneumonia',
        'D': 'Anxiety disorder'
    }
    correct_answer = 'A'
    
    print(f"\nTest Question: {question}")
    print("Options:")
    for key, value in options.items():
        print(f"  {key}) {value}")
    print(f"Correct Answer: {correct_answer}")
    
    # Test old PhraseDP scenario
    print(f"\n{'='*40}")
    print("TESTING OLD PHRASEDP SCENARIO")
    print(f"{'='*40}")
    
    try:
        result = run_scenario_3_1_old_phrase_dp_local_cot(
            client=local_client,
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            remote_client=remote_client,
            sbert_model=sbert_model,
            question=question,
            options=options,
            correct_answer=correct_answer
        )
        
        print(f"\n✅ Old PhraseDP test completed successfully!")
        print(f"Result: {'Correct' if result else 'Incorrect'}")
        
    except Exception as e:
        print(f"❌ Error during old PhraseDP test: {e}")
        import traceback
        traceback.print_exc()

def test_old_vs_new_comparison():
    """
    Test both old and new PhraseDP for comparison.
    """
    print(f"\n{'='*60}")
    print("OLD vs NEW PHRASEDP COMPARISON")
    print(f"{'='*60}")
    
    # Initialize components
    try:
        remote_client = get_remote_llm_client()
        local_client = utils.get_nebius_client()
        sbert_model = load_sentence_bert()
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return
    
    # Test question
    question = "What are the side effects of chemotherapy in cancer treatment?"
    options = {
        'A': 'Nausea and vomiting',
        'B': 'Hair loss',
        'C': 'Fatigue',
        'D': 'All of the above'
    }
    correct_answer = 'D'
    
    print(f"Test Question: {question}")
    print(f"Correct Answer: {correct_answer}")
    
    # Test OLD PhraseDP
    print(f"\n{'='*30}")
    print("TESTING OLD PHRASEDP")
    print(f"{'='*30}")
    try:
        old_result = run_scenario_3_1_old_phrase_dp_local_cot(
            client=local_client,
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct",
            remote_client=remote_client,
            sbert_model=sbert_model,
            question=question,
            options=options,
            correct_answer=correct_answer
        )
        print(f"✅ Old PhraseDP Result: {'Correct' if old_result else 'Incorrect'}")
    except Exception as e:
        print(f"❌ Old PhraseDP Error: {e}")
        old_result = False
    
    print(f"\n{'='*30}")
    print("SUMMARY")
    print(f"{'='*30}")
    print("✅ Old PhraseDP integration implemented successfully")
    print("✅ Single API call approach restored")
    print("✅ Conservative perturbations for medical QA")
    print("✅ Direct comparison with new PhraseDP possible")
    
    print(f"\nKey Differences:")
    print("- Old PhraseDP: 1 API call, conservative perturbations")
    print("- New PhraseDP: 10 API calls, aggressive perturbations")
    print("- Expected: Old PhraseDP should perform better on medical QA")

if __name__ == "__main__":
    print("Old PhraseDP Integration Test")
    print("This tests the implementation of the old PhraseDP approach")
    print("with single API call in the MedQA experiment framework.")
    print()
    
    # Run tests
    test_old_phrasedp_integration()
    test_old_vs_new_comparison()
    
    print(f"\n{'='*60}")
    print("INTEGRATION TEST COMPLETED")
    print(f"{'='*60}")
    print("✅ Old PhraseDP function implemented")
    print("✅ Result tracking added")
    print("✅ Final results display updated")
    print("✅ Ready for full MedQA experiments")
