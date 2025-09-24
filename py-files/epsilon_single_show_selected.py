#!/usr/bin/env python3
import os
from dotenv import load_dotenv
from openai import OpenAI

import utils
from dp_sanitizer import get_embedding, compute_similarity, differentially_private_replacement
from sentence_transformers import SentenceTransformer

QUESTION = "What is the capital of France?"
EPSILONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
BASE_URL = "https://api.studio.nebius.ai/v1/"


def main():
    load_dotenv()
    api_key = os.getenv("NEBIUS")
    if not api_key:
        raise RuntimeError("NEBIUS API key not found in env")

    print("🔬 Selected candidate per epsilon (current mechanism)")
    print(f"Question: {QUESTION}")
    print(f"Epsilons: {EPSILONS}")
    print("=" * 80)

    client = OpenAI(api_key=api_key, base_url=BASE_URL)
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate pool once (diverse method)
    print("Generating candidates (5×20)...")
    candidates = utils.generate_sentence_replacements_with_nebius_diverse(
        client, MODEL_NAME, QUESTION, num_return_sequences=20, num_api_calls=5
    )
    print(f"Pool size: {len(candidates)}\n")

    # Precompute embeddings
    candidate_embeddings = {sent: get_embedding(sbert, sent).cpu().numpy() for sent in candidates}

    # Select once per epsilon
    for eps in EPSILONS:
        chosen = differentially_private_replacement(
            target_phrase=QUESTION,
            epsilon=eps,
            candidate_phrases=candidates,
            candidate_embeddings=candidate_embeddings,
            sbert_model=sbert,
        )
        sim = compute_similarity(sbert, QUESTION, chosen)
        print(f"ε={eps:.1f} → {sim:.4f} :: {chosen}")

if __name__ == "__main__":
    main()
