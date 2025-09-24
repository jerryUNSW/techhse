#!/usr/bin/env python3
import os
import math
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import utils


def sbert_sim(m, a, b):
    ea = m.encode(a, convert_to_tensor=False).astype(np.float32)
    eb = m.encode(b, convert_to_tensor=False).astype(np.float32)
    ea /= np.linalg.norm(ea) + 1e-12
    eb /= np.linalg.norm(eb) + 1e-12
    return float(np.dot(ea, eb))


def softmax_expectation(s, eps):
    logits = eps * s
    m = logits.max()
    w = np.exp(logits - m)
    w /= w.sum()
    return float((w * s).sum())


def sample_mean(s, eps, k):
    logits = eps * s
    m = logits.max()
    w = np.exp(logits - m)
    p = w / w.sum()
    idx = np.arange(len(s))
    picks = np.random.choice(idx, size=k, replace=True, p=p)
    vals = s[picks]
    return float(vals.mean()), float(vals.std() / math.sqrt(k))


def main():
    load_dotenv()
    api = os.getenv('NEBIUS')
    if not api:
        raise RuntimeError('NEBIUS API key not found in env')

    client = OpenAI(api_key=api, base_url='https://api.studio.nebius.ai/v1/')
    model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    question = 'What is the capital of France?'
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    print('Question:', question)
    print('Generating candidates (low-band quota boosted)...')
    candidates = utils.generate_sentence_replacements_with_nebius_diverse(
        client, model, question,
        num_return_sequences=20, num_api_calls=5,
        enforce_similarity_filter=True, filter_margin=0.05,
        low_band_quota_boost=True,
    )

    sims = np.array([sbert_sim(sbert, question, c) for c in candidates], dtype=float)
    print(f"Pool: n={len(candidates)}, min={sims.min():.3f}, max={sims.max():.3f}, mean={sims.mean():.3f}")

    epsilons = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    K = 30
    print('\nSimilarity vs epsilon (mean ± SEM) and expected:')
    for eps in epsilons:
        m, sem = sample_mean(sims, eps, K)
        exp = softmax_expectation(sims, eps)
        print(f"  ε={eps:.1f}: {m:.3f} ± {sem:.3f}   (expected {exp:.3f})")


if __name__ == '__main__':
    main()



