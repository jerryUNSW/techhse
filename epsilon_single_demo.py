#!/usr/bin/env python3
import os
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime

import utils
from dp_sanitizer import get_embedding, compute_similarity, differentially_private_replacement
from sentence_transformers import SentenceTransformer

# Config
QUESTION = "What is the capital of France?"
EPSILONS = [0.5, 1.0, 1.5, 2.0]
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.studio.nebius.ai/v1/"


def main():
    np.random.seed(42)
    load_dotenv()
    api_key = os.getenv("NEBIUS")
    if not api_key:
        raise RuntimeError("NEBIUS API key not found in env")

    print("🔬 Single-Question Epsilon Demo (using current mechanism)")
    print("Question:", QUESTION)
    print("Epsilons:", EPSILONS)
    print("=" * 80)

    # Init clients/models
    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate candidates ONCE (use diverse method to get a broad spectrum)
    print("Generating candidates (5 calls × 20 = 100)...")
    candidates = utils.generate_sentence_replacements_with_nebius_diverse(
        client, MODEL_NAME, QUESTION, num_return_sequences=20, num_api_calls=5
    )
    print(f"Generated {len(candidates)} candidates\n")

    # Compute similarities for all candidates
    print("Candidates and similarities (to original):")
    candidate_sims = []
    for idx, cand in enumerate(candidates, start=1):
        sim = compute_similarity(sbert, QUESTION, cand)
        candidate_sims.append((cand, float(sim)))
    # Sort by similarity ascending for readability
    candidate_sims.sort(key=lambda x: x[1])

    for cand, sim in candidate_sims:
        print(f"  - {cand} \t similarity={sim:.4f}")
    print("\n" + ("-" * 80))

    # Precompute embeddings dict for current mechanism
    candidate_embeddings = {sent: get_embedding(sbert, sent).cpu().numpy() for sent, _ in candidate_sims}

    # Run selection for each epsilon using the existing mechanism
    print("Selections by epsilon (using current dp_sanitizer implementation):")
    for eps in EPSILONS:
        chosen = differentially_private_replacement(
            target_phrase=QUESTION,
            epsilon=eps,
            candidate_phrases=[c for c, _ in candidate_sims],
            candidate_embeddings=candidate_embeddings,
            sbert_model=sbert,
        )
        chosen_sim = compute_similarity(sbert, QUESTION, chosen)
        print(f"  ε={eps}: chosen='{chosen}' (similarity={chosen_sim:.4f})")

    print("=" * 80)
    print("Done.")

if __name__ == "__main__":
    main()
