#!/usr/bin/env python3
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

import utils

def sbert_sim(m, a, b):
    ea = m.encode(a, convert_to_tensor=False).astype(np.float32)
    eb = m.encode(b, convert_to_tensor=False).astype(np.float32)
    ea /= np.linalg.norm(ea) + 1e-12
    eb /= np.linalg.norm(eb) + 1e-12
    return float(np.dot(ea, eb))

def sample_candidates(candidates, sims, eps, k):
    """Sample k candidates and return their indices and similarities"""
    logits = eps * sims
    m = logits.max()
    w = np.exp(logits - m)
    p = w / w.sum()
    idx = np.arange(len(sims))
    picks = np.random.choice(idx, size=k, replace=True, p=p)
    return picks, sims[picks], [candidates[i] for i in picks]

def main():
    load_dotenv()
    api = os.getenv('NEBIUS')
    if not api:
        raise RuntimeError('NEBIUS API key not found in env')

    client = OpenAI(api_key=api, base_url='https://api.studio.nebius.ai/v1/')
    model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    question = 'What is the capital of France?'
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Generating balanced candidates...")
    
    # Generate balanced candidates using the actual diverse method
    candidates = utils.generate_sentence_replacements_with_nebius_diverse(
        client, model, question,
        num_return_sequences=20, num_api_calls=10,
        enforce_similarity_filter=True, filter_margin=0.05,
        low_band_quota_boost=True, refill_underfilled_bands=True,
        max_refill_retries=3, equal_band_target=30,
        verbose=False  # Don't show detailed output
    )
    
    # Calculate similarities
    sims = np.array([sbert_sim(sbert, question, c) for c in candidates], dtype=float)
    
    print(f"Question: {question}")
    print(f"Generated {len(candidates)} candidates")
    print(f"Similarity range: {sims.min():.3f} to {sims.max():.3f}")
    print(f"Mean similarity: {sims.mean():.3f}")
    print()
    
    # Show all candidates with their similarities
    print("All Generated Candidates:")
    for i, (candidate, sim) in enumerate(zip(candidates, sims)):
        print(f"  {i:2d}. [{sim:.3f}] {candidate}")
    print()
    
    # Create balanced subset (take minimum count per band)
    bands = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
    
    def assign_band(val):
        for i,(lo,hi) in enumerate(bands):
            if lo-0.05 <= val <= hi+0.05:
                return i
        return None
    
    idx_by_band = {i: [] for i in range(len(bands))}
    for i, s in enumerate(sims):
        b = assign_band(s)
        if b is not None:
            idx_by_band[b].append(i)
    
    raw_counts = [len(idx_by_band[i]) for i in range(len(bands))]
    print("Band counts:", raw_counts)
    
    # Balance by taking minimum count
    target = min(raw_counts) if min(raw_counts) > 0 else 0
    balanced_idx = []
    if target > 0:
        rng = np.random.default_rng(42)
        for i in range(len(bands)):
            if len(idx_by_band[i]) >= target:
                sel = rng.choice(idx_by_band[i], size=target, replace=False)
                balanced_idx.extend(sel.tolist())
        balanced_idx = sorted(set(balanced_idx))
    
    balanced_candidates = [candidates[i] for i in balanced_idx]
    balanced_sims = sims[balanced_idx]
    
    print(f"Balanced subset: {len(balanced_candidates)} candidates ({target} per band)")
    print()
    
    # Show balanced candidates
    print("Balanced Candidates (used for selection):")
    for i, (candidate, sim) in enumerate(zip(balanced_candidates, balanced_sims)):
        band_idx = assign_band(sim)
        band = bands[band_idx] if band_idx is not None else "Unknown"
        print(f"  {i:2d}. [{sim:.3f}] Band {band_idx} ({band[0]:.1f}-{band[1]:.1f}) {candidate}")
    print()
    
    # Epsilon sweep
    EPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    K = 5  # Show 5 samples per epsilon for readability
    
    for eps in EPS:
        print(f"ε = {eps:.1f} (K={K} samples):")
        picks, selected_sims, selected_candidates = sample_candidates(balanced_candidates, balanced_sims, eps, K)
        
        # Show selected candidates
        for i, (pick_idx, sim, candidate) in enumerate(zip(picks, selected_sims, selected_candidates)):
            band_idx = assign_band(sim)
            band = bands[band_idx] if band_idx is not None else "Unknown"
            print(f"  Sample {i+1}: [{sim:.3f}] Band {band_idx} ({band[0]:.1f}-{band[1]:.1f}) {candidate}")
        
        # Calculate statistics
        mean_sim = selected_sims.mean()
        std_sim = selected_sims.std()
        print(f"  Mean similarity: {mean_sim:.3f} ± {std_sim:.3f}")
        print()

if __name__ == '__main__':
    main()
