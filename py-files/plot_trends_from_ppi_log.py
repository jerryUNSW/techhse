import re
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


LOG_PATH = "/home/yizhang/tech4HSE/ppi-protection-exp.txt"
OUT_DIR = "/home/yizhang/tech4HSE"


def parse_summary_from_log(log_path: str) -> Dict[str, Dict[float, Dict[str, float]]]:
    """
    Parse the mechanism performance summary from the end of the ppi-protection-exp.txt log.

    Returns a nested dict: { mechanism: { epsilon: {metric: value, ...} } }
    Metrics include: overall, emails, phones, addresses, names
    """
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Restrict to the SUMMARY block if present to avoid parsing intermediate lines
    summary_start = text.find("PII PROTECTION EXPERIMENT SUMMARY")
    if summary_start != -1:
        text = text[summary_start:]

    mechanism_order = ["PhraseDP", "InferDPT", "SANTEXT+"]
    data: Dict[str, Dict[float, Dict[str, float]]] = {m: {} for m in mechanism_order}

    mech_re = re.compile(r"^(PhraseDP|InferDPT|SANTEXT\+)\:\s*$", re.MULTILINE)
    eps_re = re.compile(r"^\s*Epsilon\s+([0-9]+\.[0-9]+)\:\s*$", re.MULTILINE)
    metric_re = re.compile(r"^\s*(Overall|Email|Phone|Address|Name)\s+Protection\:\s+([0-9]+\.[0-9]+)\s*$", re.MULTILINE)

    # Find blocks per mechanism
    mech_iter = list(mech_re.finditer(text))
    for i, m in enumerate(mech_iter):
        mech = m.group(1)
        start = m.end()
        end = mech_iter[i + 1].start() if i + 1 < len(mech_iter) else len(text)
        block = text[start:end]

        # For each epsilon in block
        eps_matches = list(eps_re.finditer(block))
        for j, em in enumerate(eps_matches):
            eps = float(em.group(1))
            bstart = em.end()
            bend = eps_matches[j + 1].start() if j + 1 < len(eps_matches) else len(block)
            eblock = block[bstart:bend]

            # Collect metrics within this epsilon block
            metrics: Dict[str, float] = {}
            for mm in metric_re.finditer(eblock):
                key = mm.group(1).lower()  # overall, email, phone, address, name
                val = float(mm.group(2))
                # Normalize keys to plural where used elsewhere
                if key == "email":
                    key = "emails"
                elif key == "phone":
                    key = "phones"
                elif key == "address":
                    key = "addresses"
                elif key == "name":
                    key = "names"
                metrics[key] = val

            if metrics:
                data[mech][eps] = metrics

    return data


def plot_trends(data: Dict[str, Dict[float, Dict[str, float]]], out_dir: str) -> List[str]:
    """
    Generate trend plots for Overall and per-PII-type metrics across epsilons for all mechanisms.
    Returns list of output file paths.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outputs: List[str] = []

    metrics = [
        ("overall", "Overall Protection"),
        ("emails", "Email Protection"),
        ("phones", "Phone Protection"),
        ("addresses", "Address Protection"),
        ("names", "Name Protection"),
    ]

    mechanisms = ["PhraseDP", "InferDPT", "SANTEXT+"]
    colors = {
        "PhraseDP": "#1f77b4",
        "InferDPT": "#2ca02c",
        "SANTEXT+": "#d62728",
    }

    # Build common epsilon axis from union of keys
    epsilons_sorted = sorted({e for mech in data.values() for e in mech.keys()})

    for metric_key, metric_title in metrics:
        plt.figure(figsize=(7, 4.5))
        for mech in mechanisms:
            mech_data = data.get(mech, {})
            y_vals: List[float] = []
            for eps in epsilons_sorted:
                y_vals.append(mech_data.get(eps, {}).get(metric_key, float("nan")))
            plt.plot(
                epsilons_sorted,
                y_vals,
                marker="o",
                linewidth=2,
                label=mech,
                color=colors.get(mech),
            )

        plt.title(f"{metric_title} vs Epsilon (Row-by-Row)")
        plt.xlabel("Epsilon")
        plt.ylabel(metric_title)
        plt.ylim(0.0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend(frameon=False)

        out_path = os.path.join(out_dir, f"ppi_trend_{metric_key}_{timestamp}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        outputs.append(out_path)

    # Combined 2x2 grid for per-PII metrics (emails, phones, addresses, names)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    grid_metrics: List[Tuple[str, str, Tuple[int, int]]] = [
        ("emails", "Email" , (0, 0)),
        ("phones", "Phone" , (0, 1)),
        ("addresses", "Address", (1, 0)),
        ("names", "Name" , (1, 1)),
    ]
    for metric_key, short_title, (r, c) in grid_metrics:
        ax = axes[r][c]
        for mech in mechanisms:
            mech_data = data.get(mech, {})
            y_vals = [mech_data.get(eps, {}).get(metric_key, float("nan")) for eps in epsilons_sorted]
            ax.plot(epsilons_sorted, y_vals, marker="o", linewidth=2, label=mech, color=colors.get(mech))
        ax.set_title(f"{short_title}")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Per-PII Protection vs Epsilon (Row-by-Row)", y=0.98)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    grid_out_path = os.path.join(out_dir, f"ppi_trend_pii_grid_{timestamp}.png")
    plt.savefig(grid_out_path, dpi=150)
    plt.close(fig)
    outputs.append(grid_out_path)

    # Save parsed data snapshot for reference
    snapshot_path = os.path.join(out_dir, f"ppi_trend_parsed_{timestamp}.json")
    with open(snapshot_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    outputs.append(snapshot_path)

    return outputs


def main():
    data = parse_summary_from_log(LOG_PATH)
    if not any(data.values()):
        raise RuntimeError("No summary data parsed from log. Ensure the log contains the experiment summary block.")
    outputs = plot_trends(data, OUT_DIR)
    print("Generated files:")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()


