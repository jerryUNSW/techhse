#!/usr/bin/env python3
"""
Per-question epsilon vs selected-similarity trends (old vs new).
Outputs:
- plots/per_question_trends/question_{idx:02d}_epsilon_trend.png
- per_question_epsilon_trends.md (brief numeric summary)
"""

import os
import glob
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _spearman(x: List[float], y: List[float]) -> float:
    # Minimal Spearman via ranks
    def rankdata(a: List[float]) -> np.ndarray:
        arr = np.asarray(a)
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(arr))
        # ties -> average ranks
        uniq, idx, cnt = np.unique(arr, return_index=True, return_counts=True)
        for v, i, c in zip(uniq, idx, cnt):
            if c > 1:
                where = np.where(arr == v)[0]
                ranks[where] = np.mean(ranks[where])
        return ranks + 1.0

    if not x or not y or len(x) != len(y):
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def load_results() -> Tuple[Dict, str]:
    extended = _find_latest("extended_epsilon_comparison_results_*.json")
    scaled = _find_latest("scaled_epsilon_comparison_results_*.json")
    path = extended or scaled
    if not path:
        raise FileNotFoundError("No results JSON found.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data, path


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_per_question_trends(data: Dict) -> List[str]:
    out_dir = os.path.join("plots", "per_question_trends")
    ensure_dir(out_dir)

    epsilons: List[float] = data["epsilon_values"]
    questions = data["questions"]

    saved = []
    sns.set(style="whitegrid")

    for q in questions:
        qidx = q.get("question_index", 0)
        qtext = q.get("question_text", "")

        eps_vals = []
        old_sims = []
        new_sims = []
        for t in q["epsilon_tests"]:
            if "error" in t:
                continue
            eps_vals.append(float(t["epsilon"]))
            old_sims.append(float(t["old_method"]["similarity_to_original"]))
            new_sims.append(float(t["new_method"]["similarity_to_original"]))

        # sort by epsilon
        order = np.argsort(eps_vals)
        x = [eps_vals[i] for i in order]
        yo = [old_sims[i] for i in order]
        yn = [new_sims[i] for i in order]

        # correlations
        rho_old = _spearman(x, yo)
        rho_new = _spearman(x, yn)

        plt.figure(figsize=(7, 4.5))
        plt.plot(x, yo, "-o", color="red", label=f"Old (ρ={rho_old:.2f})")
        plt.plot(x, yn, "-o", color="blue", label=f"New (ρ={rho_new:.2f})")
        plt.xlabel("Epsilon")
        plt.ylabel("Selected similarity")
        title_txt = (qtext[:80] + "…") if len(qtext) > 80 else qtext
        plt.title(f"Q{qidx}: ε vs similarity\n{title_txt}")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"question_{qidx:02d}_epsilon_trend.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        saved.append(out_path)

    return saved


def write_summary_md(data: Dict, image_paths: List[str], source_file: str) -> str:
    epsilons = data["epsilon_values"]
    questions = data["questions"]
    lines = []
    lines.append("# Per-Question Epsilon Trend Summary")
    lines.append("")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Source results: `{source_file}`")
    lines.append(f"Epsilons: {epsilons}")
    lines.append("")
    for q in questions:
        qidx = q.get("question_index", 0)
        qtext = q.get("question_text", "")
        eps_vals = []
        old_sims = []
        new_sims = []
        for t in q["epsilon_tests"]:
            if "error" in t:
                continue
            eps_vals.append(float(t["epsilon"]))
            old_sims.append(float(t["old_method"]["similarity_to_original"]))
            new_sims.append(float(t["new_method"]["similarity_to_original"]))
        order = np.argsort(eps_vals)
        x = [eps_vals[i] for i in order]
        yo = [old_sims[i] for i in order]
        yn = [new_sims[i] for i in order]
        rho_old = _spearman(x, yo)
        rho_new = _spearman(x, yn)
        inc_old = all(yo[i] <= yo[i+1] for i in range(len(yo)-1))
        inc_new = all(yn[i] <= yn[i+1] for i in range(len(yn)-1))

        lines.append(f"## Q{qidx}")
        lines.append(qtext)
        lines.append(f"- Old: Spearman={rho_old:.2f}, monotonic_increase={'Yes' if inc_old else 'No'}")
        lines.append(f"- New: Spearman={rho_new:.2f}, monotonic_increase={'Yes' if inc_new else 'No'}")
        img_path = next((p for p in image_paths if f"question_{qidx:02d}_" in p), None)
        if img_path:
            lines.append(f"- Plot: `{img_path}`")
        lines.append("")

    out_md = "per_question_epsilon_trends.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out_md


def main():
    data, src = load_results()
    imgs = plot_per_question_trends(data)
    md = write_summary_md(data, imgs, src)
    print(f"Saved {len(imgs)} per-question plots under plots/per_question_trends/")
    print(f"Summary: {md}")


if __name__ == "__main__":
    main()


