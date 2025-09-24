#!/usr/bin/env python3
"""
Analyze per-question reasons for (non-)monotonic epsilon→similarity trends.
Outputs: per_question_epsilon_trend_diagnostics.md
"""

import os
import glob
import json
import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np


def _find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _spearman(x: List[float], y: List[float]) -> float:
    def rankdata(a: List[float]) -> np.ndarray:
        arr = np.asarray(a)
        order = np.argsort(arr)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(arr))
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
        return json.load(f), path


def diag_for_question(q: Dict) -> Dict:
    eps_vals = []
    old_sel = []
    new_sel = []
    old_min = []
    old_max = []
    old_mean = []
    new_min = []
    new_max = []
    new_mean = []
    old_range = []
    new_range = []
    for t in q["epsilon_tests"]:
        if "error" in t:
            continue
        eps = float(t["epsilon"]) 
        eps_vals.append(eps)
        o = t["old_method"]
        n = t["new_method"]
        old_sel.append(float(o["similarity_to_original"]))
        new_sel.append(float(n["similarity_to_original"]))
        oc = o.get("candidate_similarities", [])
        nc = n.get("candidate_similarities", [])
        if oc:
            old_min.append(float(np.min(oc)))
            old_max.append(float(np.max(oc)))
            old_mean.append(float(np.mean(oc)))
            old_range.append(float(np.max(oc) - np.min(oc)))
        if nc:
            new_min.append(float(np.min(nc)))
            new_max.append(float(np.max(nc)))
            new_mean.append(float(np.mean(nc)))
            new_range.append(float(np.max(nc) - np.min(nc)))

    # Correlations
    rho_old = _spearman(eps_vals, old_sel)
    rho_new = _spearman(eps_vals, new_sel)

    # Scarcity of low-sim: fraction of eps where min<0.4
    frac_old_low = float(np.mean([m < 0.4 for m in old_min])) if old_min else float("nan")
    frac_new_low = float(np.mean([m < 0.4 for m in new_min])) if new_min else float("nan")

    # Range magnitude
    mean_old_range = float(np.mean(old_range)) if old_range else float("nan")
    mean_new_range = float(np.mean(new_range)) if new_range else float("nan")

    # Pool drift across eps: std of pool mean across eps
    drift_old = float(np.std(old_mean)) if old_mean else float("nan")
    drift_new = float(np.std(new_mean)) if new_mean else float("nan")

    return {
        "rho_old": rho_old,
        "rho_new": rho_new,
        "frac_old_min_lt_0_4": frac_old_low,
        "frac_new_min_lt_0_4": frac_new_low,
        "mean_old_range": mean_old_range,
        "mean_new_range": mean_new_range,
        "drift_old": drift_old,
        "drift_new": drift_new,
    }


def main():
    data, src = load_results()
    eps = data["epsilon_values"]
    out_lines = []
    out_lines.append("# Per-Question Epsilon Trend Diagnostics")
    out_lines.append("")
    out_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out_lines.append(f"Source: `{src}`")
    out_lines.append(f"Epsilons: {eps}")
    out_lines.append("")

    for q in data["questions"]:
        qidx = q.get("question_index", 0)
        qtext = q.get("question_text", "")
        d = diag_for_question(q)

        def reason(method: str) -> str:
            rho = d[f"rho_{method}"]
            frac_low = d[f"frac_{method}_min_lt_0_4"]
            mean_rng = d[f"mean_{method}_range"]
            drift = d[f"drift_{method}"]
            msgs = []
            # Expected if rho >= 0.5
            if rho >= 0.5:
                msgs.append("Expected upward trend (ρ≥0.5)")
                return "; ".join(msgs)
            # Partial trend if 0<rho<0.5
            if 0.0 < rho < 0.5:
                msgs.append("Weak upward tendency (0<ρ<0.5)")
            elif rho <= 0.0:
                msgs.append("No/negative trend (ρ≤0)")
            # Diagnose causes
            if not math.isnan(frac_low) and frac_low < 0.3:
                msgs.append("scarce low-sim candidates (min<0.4 rare)")
            if not math.isnan(mean_rng) and mean_rng < 0.2:
                msgs.append("small candidate similarity range (Δ<0.2)")
            if not math.isnan(drift) and drift > 0.05:
                msgs.append("pool drift across ε (pool mean varies)")
            if len(msgs) == 0:
                msgs.append("insufficient evidence; likely sampling noise")
            return "; ".join(msgs)

        out_lines.append(f"## Q{qidx}")
        out_lines.append(qtext)
        out_lines.append(f"- Old: ρ={d['rho_old']:.2f}; {reason('old')}")
        out_lines.append(f"- New: ρ={d['rho_new']:.2f}; {reason('new')}")
        out_lines.append(f"  - Diagnostics: old[min<0.4%]={d['frac_old_min_lt_0_4']*100 if not math.isnan(d['frac_old_min_lt_0_4']) else float('nan'):.1f}%, new[min<0.4%]={d['frac_new_min_lt_0_4']*100 if not math.isnan(d['frac_new_min_lt_0_4']) else float('nan'):.1f}%")
        out_lines.append(f"                old[Δ̄]={d['mean_old_range']:.3f}, new[Δ̄]={d['mean_new_range']:.3f}, old[drift]={d['drift_old']:.3f}, new[drift]={d['drift_new']:.3f}")
        out_lines.append("")

    out_md = "per_question_epsilon_trend_diagnostics.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()


