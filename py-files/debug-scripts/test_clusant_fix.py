#!/usr/bin/env python3
"""
Quick CluSanT Test Script
Test if CluSanT is working correctly after the empty string fix
"""

import sys
import os
import re

# Add CluSanT src to path
sys.path.append('/home/yizhang/tech4HSE/CluSanT/src')
from embedding_handler import EmbeddingHandler
from clusant import CluSanT

def test_clusant():
    """Test CluSanT with the fix applied"""
    print("=== CluSanT Fix Test ===")
    
    # Test text with PII
    test_text = "My name is John Smith, and my email is john.smith@example.com"
    print(f"Original text: {test_text}")
    print()
    
    # Setup CluSanT
    original_cwd = os.getcwd()
    clusant_root = '/home/yizhang/tech4HSE/CluSanT'
    os.chdir(clusant_root)
    
    try:
        # Load embeddings
        emb_dir = os.path.join(clusant_root, 'embeddings')
        emb_path = os.path.join(emb_dir, 'all-MiniLM-L6-v2.txt')
        handler = EmbeddingHandler(model_name='all-MiniLM-L6-v2')
        
        if not os.path.exists(emb_path):
            print("Generating embeddings...")
            handler.generate_and_save_embeddings([
                os.path.join(clusant_root, 'clusters/gpt-4/LOC.json'),
                os.path.join(clusant_root, 'clusters/gpt-4/ORG.json'),
            ], emb_dir)
        
        embeddings = handler.load_embeddings(emb_path)
        print(f"Loaded {len(embeddings)} embeddings")
        
        # Test different epsilon values
        for epsilon in [1.0, 2.0, 3.0]:
            print(f"\n--- Testing Epsilon {epsilon} ---")
            
            # Initialize CluSanT
            clus = CluSanT(
                embedding_file='all-MiniLM-L6-v2',
                embeddings=embeddings,
                epsilon=epsilon,
                num_clusters=336,
                mechanism='clusant',
                metric_to_create_cluster='euclidean',
                distance_metric_for_cluster='euclidean',
                distance_metric_for_words='euclidean',
                dp_type='metric',
                K=16,
            )
            
            # Apply FIXED target detection
            sanitized_text = test_text
            targets_present = []
            
            # Multi-word targets first - WITH FIX
            for w in embeddings.keys():
                if w.strip() and ' ' in w and re.search(rf"\b{re.escape(w)}\b", sanitized_text, flags=re.IGNORECASE):
                    targets_present.append(w)
            
            # Single-word targets - WITH FIX
            for w in embeddings.keys():
                if w.strip() and ' ' not in w and re.search(rf"\b{re.escape(w)}\b", sanitized_text, flags=re.IGNORECASE):
                    targets_present.append(w)
            
            targets_present = sorted(set(targets_present), key=lambda x: (-len(x), x))
            
            print(f"Targets found: {targets_present}")
            
            # Apply replacements
            for t in targets_present:
                new = clus.replace_word(t)
                if not new:
                    continue
                pattern = re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE)
                if pattern.search(sanitized_text):
                    sanitized_text = pattern.sub(new, sanitized_text)
            
            print(f"Sanitized text: {sanitized_text}")
            
            # Check if output is reasonable (not full of random words)
            word_count = len(sanitized_text.split())
            random_word_indicators = ['mauritanian', 'government', 'change', 'hearing', 'list']
            has_random_words = any(indicator in sanitized_text.lower() for indicator in random_word_indicators)
            
            if has_random_words:
                print("❌ FAILED: Still contains random word insertions")
            elif word_count > len(test_text.split()) * 2:
                print("❌ FAILED: Output has too many words (likely random insertions)")
            elif sanitized_text == test_text:
                print("⚠️  WARNING: No changes made (may be expected if no targets found)")
            else:
                print("✅ SUCCESS: CluSanT working correctly")
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    test_clusant()
