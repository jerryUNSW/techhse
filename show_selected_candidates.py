#!/usr/bin/env python3
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

def sbert_sim(m, a, b):
    ea = m.encode(a, convert_to_tensor=False).astype(np.float32)
    eb = m.encode(b, convert_to_tensor=False).astype(np.float32)
    ea /= np.linalg.norm(ea) + 1e-12
    eb /= np.linalg.norm(eb) + 1e-12
    return float(np.dot(ea, eb))

def sample_candidates(s, eps, k):
    """Sample k candidates and return their indices and similarities"""
    logits = eps * s
    m = logits.max()
    w = np.exp(logits - m)
    p = w / w.sum()
    idx = np.arange(len(s))
    picks = np.random.choice(idx, size=k, replace=True, p=p)
    return picks, s[picks]

def main():
    question = 'What is the capital of France?'
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create balanced candidates based on the 10 bands (same as before)
    bands = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
    balanced_candidates = []
    
    # Generate representative candidates for each band (2 per band = 20 total)
    np.random.seed(42)  # For reproducibility
    for i, (lo, hi) in enumerate(bands):
        # Create 2 candidates per band with similarity in the middle of the range
        mid_sim = (lo + hi) / 2
        for j in range(2):
            # Add some small random variation
            sim = mid_sim + np.random.normal(0, 0.02)
            sim = max(lo, min(hi, sim))  # Clamp to band range
            balanced_candidates.append(sim)
    
    sims_balanced = np.array(balanced_candidates)
    
    print(f"Question: {question}")
    print(f"Balanced candidates: {len(sims_balanced)} (2 per band)")
    print(f"Similarity range: {sims_balanced.min():.3f} to {sims_balanced.max():.3f}")
    print(f"Mean similarity: {sims_balanced.mean():.3f}")
    print()
    
    # Show the candidate pool
    print("Candidate Pool (20 candidates, 2 per band):")
    for i, sim in enumerate(sims_balanced):
        band_idx = i // 2
        band = bands[band_idx]
        print(f"  {i:2d}. [{sim:.3f}] Band {band_idx} ({band[0]:.1f}-{band[1]:.1f})")
    print()
    
    # Epsilon sweep
    EPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    K = 10  # Show 10 samples per epsilon for readability
    
    for eps in EPS:
        print(f"ε = {eps:.1f}:")
        picks, selected_sims = sample_candidates(sims_balanced, eps, K)
        
        # Show selected candidates
        for i, (pick_idx, sim) in enumerate(zip(picks, selected_sims)):
            band_idx = pick_idx // 2
            band = bands[band_idx]
            print(f"  Sample {i+1:2d}: Candidate {pick_idx:2d} [{sim:.3f}] from Band {band_idx} ({band[0]:.1f}-{band[1]:.1f})")
        
        # Calculate statistics
        mean_sim = selected_sims.mean()
        std_sim = selected_sims.std()
        print(f"  Mean: {mean_sim:.3f}, Std: {std_sim:.3f}")
        print()
    
    # Show theoretical expectations
    print("Theoretical Expected Values:")
    for eps in EPS:
        logits = eps * sims_balanced
        m = logits.max()
        w = np.exp(logits - m)
        p = w / w.sum()
        expected = np.sum(p * sims_balanced)
        print(f"  ε = {eps:.1f}: {expected:.3f}")
    print()
    
    # Show selection probabilities for each epsilon
    print("Selection Probabilities by Band:")
    print("Band    ", end="")
    for eps in EPS:
        print(f"ε={eps:.1f}    ", end="")
    print()
    
    for band_idx, (lo, hi) in enumerate(bands):
        print(f"{band_idx:2d}({lo:.1f}-{hi:.1f}) ", end="")
        for eps in EPS:
            # Calculate probability for this band
            band_candidates = [i for i in range(len(sims_balanced)) if i // 2 == band_idx]
            logits = eps * sims_balanced
            m = logits.max()
            w = np.exp(logits - m)
            p = w / w.sum()
            band_prob = np.sum(p[band_candidates])
            print(f"{band_prob:.3f}   ", end="")
        print()

if __name__ == '__main__':
    main()
