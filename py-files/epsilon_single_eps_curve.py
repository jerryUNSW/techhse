#!/usr/bin/env python3
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import utils
from dp_sanitizer import get_embedding, compute_similarity, differentially_private_replacement
from sentence_transformers import SentenceTransformer

QUESTION = "What is the capital of France?"
EPSILONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.studio.nebius.ai/v1/"
ROUNDS = 5


def main():
    load_dotenv()
    api_key = os.getenv("NEBIUS")
    if not api_key:
        raise RuntimeError("NEBIUS API key not found in env")

    print("🔬 Epsilon curve demo (current mechanism)")
    print(f"Question: {QUESTION}")
    print(f"Epsilons: {EPSILONS}")
    print(f"Rounds per epsilon: {ROUNDS}")
    print("=" * 80)

    # Init clients/models
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate candidates ONCE (use diverse method)
    print("Generating candidates (5 calls × 20 = 100)...")
    candidates = utils.generate_sentence_replacements_with_nebius_diverse(
        client, MODEL_NAME, QUESTION, num_return_sequences=20, num_api_calls=5
    )
    print(f"Generated {len(candidates)} candidates\n")

    # Compute candidate similarities to report pool stats
    sims = [float(compute_similarity(sbert, QUESTION, c)) for c in candidates]
    print(f"Candidate pool similarity stats: min={min(sims):.3f}, max={max(sims):.3f}, mean={np.mean(sims):.3f}, n={len(sims)}")

    # Precompute embeddings for current mechanism
    candidate_embeddings = {sent: get_embedding(sbert, sent).cpu().numpy() for sent in candidates}

    # For each epsilon, sample ROUNDS and compute mean similarity
    results = {}
    for eps in EPSILONS:
        chosen_sims = []
        for _ in range(ROUNDS):
            chosen = differentially_private_replacement(
                target_phrase=QUESTION,
                epsilon=eps,
                candidate_phrases=candidates,
                candidate_embeddings=candidate_embeddings,
                sbert_model=sbert,
            )
            sim = float(compute_similarity(sbert, QUESTION, chosen))
            chosen_sims.append(sim)
        results[eps] = {
            'mean': float(np.mean(chosen_sims)),
            'std': float(np.std(chosen_sims)),
            'vals': chosen_sims,
        }

    print("\nAverage selected similarity by epsilon (current mechanism):")
    for eps in EPSILONS:
        r = results[eps]
        print(f"  ε={eps:.1f}: mean={r['mean']:.4f}, std={r['std']:.4f}, vals={[round(v,4) for v in r['vals']]}")

    # Simple monotonicity hint
    means = [results[eps]['mean'] for eps in EPSILONS]
    trend = "increasing" if all(means[i] <= means[i+1] for i in range(len(means)-1)) else "non-monotonic"
    print(f"\nTrend of means: {trend} -> {means}")

if __name__ == "__main__":
    main()
