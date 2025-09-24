#!/usr/bin/env python3
"""
Fixed-pool epsilon sensitivity analysis with repeated sampling.

For each question and method (old/new):
- Select a fixed candidate pool from existing results (prefer ε=1.5, else mid ε)
- For every ε, compute:
  * Theoretical expected mean similarity under P ∝ exp(ε·sim)
  * Sample K=30 draws from the distribution, compute mean ± SEM

Outputs:
- plots/per_question_trends_fixed_pool/question_{idx:02d}_epsilon_trend_fixed.png
- plots/summary_fixed_pool.png (across-question mean±SEM by ε)
"""

import os
import glob
import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


K_DRAWS = 30
PREFERRED_POOL_EPS = 1.5


def _find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


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


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _choose_fixed_pool(q: Dict, method_key: str, epsilons: List[float]) -> Optional[np.ndarray]:
    # Prefer ε=1.5 if present; else choose middle ε
    target_eps = PREFERRED_POOL_EPS if PREFERRED_POOL_EPS in epsilons else epsilons[len(epsilons)//2]
    for t in q["epsilon_tests"]:
        if "error" in t:
            continue
        if float(t["epsilon"]) == float(target_eps):
            sims = t[method_key].get("candidate_similarities", [])
            if sims:
                return np.asarray(sims, dtype=float)
    # Fallback: first available
    for t in q["epsilon_tests"]:
        if "error" in t:
            continue
        sims = t[method_key].get("candidate_similarities", [])
        if sims:
            return np.asarray(sims, dtype=float)
    return None


def run_fixed_pool_analysis():
    # Load results (prefer extended)
    results_path = _find_latest("extended_epsilon_comparison_results_*.json") or _find_latest("scaled_epsilon_comparison_results_*.json")
    if not results_path:
        raise FileNotFoundError("No results JSON found.")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    epsilons: List[float] = [float(e) for e in data["epsilon_values"]]
    questions = data["questions"]
    out_dir = os.path.join("plots", "per_question_trends_fixed_pool")
    _ensure_dir(out_dir)

    # For summary
    old_all_means = {eps: [] for eps in epsilons}
    new_all_means = {eps: [] for eps in epsilons}

    sns.set(style="whitegrid")

    for q in questions:
        qidx = q.get("question_index", 0)
        qtext = q.get("question_text", "")

        sims_old = _choose_fixed_pool(q, "old_method", epsilons)
        sims_new = _choose_fixed_pool(q, "new_method", epsilons)
        if sims_old is None or sims_new is None:
            continue

        x = epsilons
        # Compute curves
        old_exp = []
        new_exp = []
        old_mean = []
        old_sem = []
        new_mean = []
        new_sem = []

        for eps in epsilons:
            # Theoretical
            old_exp.append(_softmax_expectation(sims_old, eps))
            new_exp.append(_softmax_expectation(sims_new, eps))
            # Sampling
            m, s = _sample_means(sims_old, eps, K_DRAWS)
            old_mean.append(m); old_sem.append(s)
            m, s = _sample_means(sims_new, eps, K_DRAWS)
            new_mean.append(m); new_sem.append(s)
            # For summary
            old_all_means[eps].append(old_mean[-1])
            new_all_means[eps].append(new_mean[-1])

        # Plot per-question
        plt.figure(figsize=(7.5, 4.8))
        plt.errorbar(x, old_mean, yerr=old_sem, fmt="-o", color="red", label="Old (mean±SEM, K=30)")
        plt.errorbar(x, new_mean, yerr=new_sem, fmt="-o", color="blue", label="New (mean±SEM, K=30)")
        plt.plot(x, old_exp, "r--", alpha=0.7, label="Old expected")
        plt.plot(x, new_exp, "b--", alpha=0.7, label="New expected")
        title_txt = (qtext[:80] + "…") if len(qtext) > 80 else qtext
        plt.title(f"Q{qidx}: ε vs similarity (fixed pool, K={K_DRAWS})\n{title_txt}")
        plt.xlabel("Epsilon")
        plt.ylabel("Selected similarity")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"question_{qidx:02d}_epsilon_trend_fixed.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    # Summary (across questions)
    def _mean_sem(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return (float("nan"), float("nan"))
        arr = np.asarray(vals, dtype=float)
        return float(np.mean(arr)), float(np.std(arr) / math.sqrt(len(arr)))

    x = epsilons
    old_means = []
    old_sems = []
    new_means = []
    new_sems = []
    for eps in epsilons:
        m, s = _mean_sem(old_all_means[eps]); old_means.append(m); old_sems.append(s)
        m, s = _mean_sem(new_all_means[eps]); new_means.append(m); new_sems.append(s)

    _ensure_dir("plots")
    plt.figure(figsize=(7.5, 4.8))
    plt.errorbar(x, old_means, yerr=old_sems, fmt="-o", color="red", label="Old (mean±SEM across questions)")
    plt.errorbar(x, new_means, yerr=new_sems, fmt="-o", color="blue", label="New (mean±SEM across questions)")
    plt.title(f"Epsilon sensitivity summary (fixed pools, K={K_DRAWS})")
    plt.xlabel("Epsilon")
    plt.ylabel("Selected similarity")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    summary_path = os.path.join("plots", "summary_fixed_pool.png")
    plt.savefig(summary_path, dpi=220, bbox_inches="tight")
    plt.close()

    print(f"Saved per-question plots to {out_dir}")
    print(f"Saved summary plot to {summary_path}")


if __name__ == "__main__":
    run_fixed_pool_analysis()


