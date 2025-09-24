#!/usr/bin/env python3
"""
Per-question candidate similarity distributions (fixed pool per method).

For each question (old/new):
- Use the same fixed-pool choice as in fixed_pool_sampling_analysis (prefer ε=1.5)
- Summarize distribution stats and bucket counts
- Save histograms per question/method
- Write a concise markdown summary highlighting scarcity of very low similarities
"""

import os
import glob
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


PREFERRED_POOL_EPS = 1.5


def _find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _choose_fixed_pool(q: Dict, method_key: str, epsilons: List[float]) -> Optional[np.ndarray]:
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


def _bucket_counts(arr: np.ndarray, edges: List[float]) -> List[int]:
    counts = [0] * (len(edges) - 1)
    for v in arr:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i+1]:
                counts[i] += 1
                break
        else:
            if abs(v - edges[-1]) < 1e-9:  # include right edge if exactly 1.0
                counts[-1] += 1
    return counts


def run():
    # Load results (prefer regenerated pools, then extended/scaled)
    results_path = (
        _find_latest("regenerated_fixed_pool_results_*.json")
        or _find_latest("extended_epsilon_comparison_results_*.json")
        or _find_latest("scaled_epsilon_comparison_results_*.json")
    )
    if not results_path:
        raise FileNotFoundError("No results JSON found.")
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Two possible schemas:
    # A) extended/scaled: questions -> epsilon_tests -> method.candidate_similarities
    # B) regenerated: questions -> pools -> method.similarities
    questions = data["questions"]
    epsilons: List[float] = [float(e) for e in data["epsilon_values"]] if "epsilon_values" in data else []

    out_dir = os.path.join("plots", "candidate_distributions_fixed_pool")
    _ensure_dir(out_dir)

    edges = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    edge_labels = ["[0.0,0.2)", "[0.2,0.3)", "[0.3,0.4)", "[0.4,0.5)", "[0.5,0.6)", "[0.6,0.7)", "[0.7,0.8)", "[0.8,0.9)", "[0.9,1.0]"]

    lines = []
    lines.append("# Candidate Similarity Distributions (Fixed Pool)")
    lines.append("")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source: `{results_path}`")
    lines.append("")

    sns.set(style="whitegrid")

    for q in questions:
        qidx = q.get("question_index", 0)
        qtext = q.get("question_text", "")
        # Try regenerated schema first
        sims_old = None
        sims_new = None
        if "pools" in q and isinstance(q["pools"], dict):
            old_sims_list = q["pools"].get("old", {}).get("similarities", [])
            new_sims_list = q["pools"].get("new", {}).get("similarities", [])
            sims_old = np.asarray(old_sims_list, dtype=float) if old_sims_list else None
            sims_new = np.asarray(new_sims_list, dtype=float) if new_sims_list else None
        else:
            # Fallback to extended/scaled schema with epsilon_tests
            if epsilons:
                sims_old = _choose_fixed_pool(q, "old_method", epsilons)
                sims_new = _choose_fixed_pool(q, "new_method", epsilons)
        if sims_old is None or sims_new is None:
            continue

        # Stats and buckets
        def stats_and_plot(sims: np.ndarray, label: str, color: str) -> Tuple[str, List[int]]:
            counts = _bucket_counts(sims, edges)
            fig = plt.figure(figsize=(6.5, 3.8))
            sns.histplot(sims, bins=20, kde=False, color=color, alpha=0.85)
            plt.xlim(0, 1)
            plt.ylim(bottom=0)
            plt.xlabel("Similarity")
            plt.ylabel("Count")
            plt.title(f"Q{qidx} {label}: cand. distribution (n={len(sims)})")
            out_png = os.path.join(out_dir, f"question_{qidx:02d}_{label.lower()}_hist.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=200, bbox_inches="tight")
            plt.close(fig)
            return out_png, counts

        old_png, old_counts = stats_and_plot(sims_old, "Old", "red")
        new_png, new_counts = stats_and_plot(sims_new, "New", "blue")

        def describe(arr: np.ndarray) -> str:
            return (f"min={np.min(arr):.3f}, p5={np.percentile(arr,5):.3f}, p25={np.percentile(arr,25):.3f}, "
                    f"median={np.median(arr):.3f}, p75={np.percentile(arr,75):.3f}, p95={np.percentile(arr,95):.3f}, "
                    f"max={np.max(arr):.3f}, mean={np.mean(arr):.3f}")

        lines.append(f"## Q{qidx}")
        lines.append(qtext)
        lines.append(f"- Old stats: {describe(sims_old)}")
        lines.append(f"- New stats: {describe(sims_new)}")
        lines.append(f"- Old buckets: " + ", ".join([f"{edge_labels[i]}={old_counts[i]}" for i in range(len(edge_labels))]))
        lines.append(f"- New buckets: " + ", ".join([f"{edge_labels[i]}={new_counts[i]}" for i in range(len(edge_labels))]))
        lines.append(f"- Plots: `{old_png}`, `{new_png}`")
        lines.append("")

    out_md = "candidate_distributions_fixed_pool.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out_md} and histograms under {out_dir}")


if __name__ == "__main__":
    run()


