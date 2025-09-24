#!/usr/bin/env python3
"""
Regenerate candidates (old and new methods) for 10 questions using the updated
diverse generator with numeric targets + SBERT filtering, then run fixed-pool
epsilon sensitivity with K=30 draws per epsilon, producing per-question plots
and a summary plot.

Outputs:
- results JSON: regenerated_fixed_pool_results_{timestamp}.json
- per-question plots: plots/per_question_trends_fixed_pool_regen/question_{idx:02d}_epsilon_trend_fixed.png
- summary plot: plots/summary_fixed_pool_regen.png
"""

import os
import json
import math
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from openai import OpenAI

import utils
from dp_sanitizer import get_embedding, compute_similarity
from sentence_transformers import SentenceTransformer


K_DRAWS = 30
EPSILONS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


def load_test_questions() -> List[str]:
    return [
        "What is the capital of France?",
        "What is the largest country in the world?",
        "Which ocean is the largest?",
        "What is the longest river in the world?",
        "In which year did World War II end?",
        "Who was the first president of the United States?",
        "When did the Berlin Wall fall?",
        "What year did the Titanic sink?",
        "What is the chemical symbol for gold?",
        "What is the speed of light?",
    ]


def _softmax_expectation(sims: np.ndarray, epsilon: float) -> float:
    logits = epsilon * sims
    m = np.max(logits)
    w = np.exp(logits - m)
    w /= np.sum(w)
    return float(np.sum(w * sims))


def _sample_means(sims: np.ndarray, epsilon: float, k: int) -> Tuple[float, float]:
    logits = epsilon * sims
    m = np.max(logits)
    w = np.exp(logits - m)
    p = w / np.sum(w)
    idx = np.arange(len(sims))
    picks = np.random.choice(idx, size=k, replace=True, p=p)
    vals = sims[picks]
    mean = float(np.mean(vals))
    sem = float(np.std(vals) / math.sqrt(k))
    return mean, sem


def regenerate_pools(client: OpenAI, model_name: str, sbert, questions: List[str]) -> Dict:
    results: Dict = {
        'timestamp': datetime.now().isoformat(),
        'epsilon_values': EPSILONS,
        'questions': []
    }
    for qidx, question in enumerate(questions):
        print(f"\n=== Q{qidx}: {question}")
        qres = {
            'question_text': question,
            'question_index': qidx,
            'pools': {
                'old': {},
                'new': {},
            }
        }

        # OLD: 10 calls × 10 returns (target ~100)
        print("  Generating OLD candidates (10×10)…")
        old_cands = utils.generate_sentence_replacements_with_nebius(
            client, model_name, question, num_return_sequences=10, num_api_calls=10
        )
        # NEW: 5 calls × 20 returns with filtering
        print("  Generating NEW candidates (5×20, filtered)…")
        new_cands = utils.generate_sentence_replacements_with_nebius_diverse(
            client, model_name, question, num_return_sequences=20, num_api_calls=5,
            enforce_similarity_filter=True, filter_margin=0.05
        )

        # Compute similarities
        def sims_for(cands: List[str]) -> List[float]:
            return [float(compute_similarity(sbert, question, c)) for c in cands]

        qres['pools']['old'] = {
            'candidates': old_cands,
            'similarities': sims_for(old_cands)
        }
        qres['pools']['new'] = {
            'candidates': new_cands,
            'similarities': sims_for(new_cands)
        }

        print(f"    OLD: n={len(old_cands)}, min={min(qres['pools']['old']['similarities']) if old_cands else float('nan'):.3f}, max={max(qres['pools']['old']['similarities']) if old_cands else float('nan'):.3f}")
        print(f"    NEW: n={len(new_cands)}, min={min(qres['pools']['new']['similarities']) if new_cands else float('nan'):.3f}, max={max(qres['pools']['new']['similarities']) if new_cands else float('nan'):.3f}")

        results['questions'].append(qres)
    return results


def analyze_and_plot(results: Dict) -> None:
    out_dir = os.path.join('plots', 'per_question_trends_fixed_pool_regen')
    os.makedirs(out_dir, exist_ok=True)
    sns.set(style='whitegrid')

    # For summary across questions
    old_all_means = {eps: [] for eps in EPSILONS}
    new_all_means = {eps: [] for eps in EPSILONS}

    for q in results['questions']:
        qidx = q['question_index']
        qtext = q['question_text']
        sims_old = np.asarray(q['pools']['old'].get('similarities', []), dtype=float)
        sims_new = np.asarray(q['pools']['new'].get('similarities', []), dtype=float)
        if sims_old.size == 0 or sims_new.size == 0:
            print(f"Skipping Q{qidx} due to empty pool(s)")
            continue

        x = EPSILONS
        old_exp, new_exp = [], []
        old_mean, old_sem = [], []
        new_mean, new_sem = [], []
        for eps in EPSILONS:
            old_exp.append(_softmax_expectation(sims_old, eps))
            new_exp.append(_softmax_expectation(sims_new, eps))
            m, s = _sample_means(sims_old, eps, K_DRAWS)
            old_mean.append(m); old_sem.append(s)
            m, s = _sample_means(sims_new, eps, K_DRAWS)
            new_mean.append(m); new_sem.append(s)
            old_all_means[eps].append(old_mean[-1])
            new_all_means[eps].append(new_mean[-1])

        plt.figure(figsize=(7.5, 4.8))
        plt.errorbar(x, old_mean, yerr=old_sem, fmt='-o', color='red', label=f'Old (mean±SEM, K={K_DRAWS})')
        plt.errorbar(x, new_mean, yerr=new_sem, fmt='-o', color='blue', label=f'New (mean±SEM, K={K_DRAWS})')
        plt.plot(x, old_exp, 'r--', alpha=0.7, label='Old expected')
        plt.plot(x, new_exp, 'b--', alpha=0.7, label='New expected')
        title_txt = (qtext[:80] + '…') if len(qtext) > 80 else qtext
        plt.title(f"Q{qidx}: ε vs similarity (fixed pool, K={K_DRAWS})\n{title_txt}")
        plt.xlabel('Epsilon')
        plt.ylabel('Selected similarity')
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"question_{qidx:02d}_epsilon_trend_fixed.png")
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()

    # Summary plot across questions
    def _mean_sem(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return (float('nan'), float('nan'))
        arr = np.asarray(vals, dtype=float)
        return float(np.mean(arr)), float(np.std(arr) / math.sqrt(len(arr)))

    x = EPSILONS
    old_means, old_sems, new_means, new_sems = [], [], [], []
    for eps in EPSILONS:
        m, s = _mean_sem(old_all_means[eps]); old_means.append(m); old_sems.append(s)
        m, s = _mean_sem(new_all_means[eps]); new_means.append(m); new_sems.append(s)

    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(7.5, 4.8))
    plt.errorbar(x, old_means, yerr=old_sems, fmt='-o', color='red', label='Old (mean±SEM across questions)')
    plt.errorbar(x, new_means, yerr=new_sems, fmt='-o', color='blue', label='New (mean±SEM across questions)')
    plt.title(f'Epsilon sensitivity summary (fixed pools, K={K_DRAWS})')
    plt.xlabel('Epsilon')
    plt.ylabel('Selected similarity')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    summary_path = os.path.join('plots', 'summary_fixed_pool_regen.png')
    plt.savefig(summary_path, dpi=220, bbox_inches='tight')
    plt.close()

    print(f"Saved per-question plots to {out_dir}")
    print(f"Saved summary plot to {summary_path}")


def main():
    load_dotenv()
    api_key = os.getenv('NEBIUS')
    if not api_key:
        raise RuntimeError('NEBIUS API key not found in env')
    client = OpenAI(api_key=api_key, base_url='https://api.studio.nebius.ai/v1/')
    model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    questions = load_test_questions()
    results = regenerate_pools(client, model_name, sbert, questions)

    # Save results JSON
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_json = f'regenerated_fixed_pool_results_{ts}.json'
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved results JSON: {out_json}")

    analyze_and_plot(results)


if __name__ == '__main__':
    main()



