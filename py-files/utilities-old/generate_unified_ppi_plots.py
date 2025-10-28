import os
import json
from glob import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt


RESULTS_DIR = "/home/yizhang/tech4HSE"
PLOTS_DIR = "/home/yizhang/tech4HSE/plots/ppi"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Global font sizing for high-readability figures (~20pt)
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 22,
    'axes.labelsize': 20,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 18,
})


def remove_old_plots():
    patterns = [
        os.path.join(PLOTS_DIR, 'overall_protection_vs_epsilon_5mech_*.png'),
        os.path.join(PLOTS_DIR, 'protection_radar_5mech_*_eps_*.png'),
    ]
    removed = []
    for pat in patterns:
        for p in glob(pat):
            try:
                os.remove(p)
                removed.append(p)
            except Exception:
                pass
    return removed


def load_latest_main_results():
    # Load the latest results produced by pii_protection_experiment.py
    paths = sorted(glob(os.path.join(RESULTS_DIR, "pii_protection_results_*.json")))
    if not paths:
        raise FileNotFoundError("No pii_protection_results_*.json found")
    with open(paths[-1], 'r') as f:
        data = json.load(f)
    # results might be directly the dict or nested under 'results'
    return data.get('results', data)


def load_clusant_results(path):
    with open(path, 'r') as f:
        d = json.load(f)
    # normalize to {'CluSanT': {eps: metrics}}
    if 'results' in d:
        r = d['results']
    else:
        r = d
    if 'CluSanT' in r:
        return r['CluSanT']
    return r

def load_custext_results():
    """Load CusText+ results from separate files for each epsilon."""
    custext_results = {}
    epsilons = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    for eps in epsilons:
        # Find the latest CusText+ file for this epsilon
        pattern = os.path.join(RESULTS_DIR, "results", f"custext_ppi_protection_eps*.json")
        files = glob(pattern)
        
        # Filter files that contain this epsilon
        matching_files = []
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if data.get('summary', {}).get('epsilon') == eps:
                        matching_files.append(file_path)
            except:
                continue
        
        if matching_files:
            # Use the latest file for this epsilon
            latest_file = sorted(matching_files)[-1]
            try:
                with open(latest_file, 'r') as f:
                    data = json.load(f)
                    summary = data.get('summary', {})
                    custext_results[eps] = {
                        'overall': summary.get('overall', 0.0),
                        'emails': summary.get('emails', 0.0),
                        'phones': summary.get('phones', 0.0),
                        'addresses': summary.get('addresses', 0.0),
                        'names': summary.get('names', 0.0)
                    }
            except:
                continue
    
    return custext_results


def parse_ppi_experiment_txt(path):
    """Parse ppi-protection-exp.txt to compute averaged protection by epsilon for
    PhraseDP, InferDPT, and SANTEXT+ across all data points present.
    Returns: dict(mech -> dict(eps -> metrics)) with float eps keys.
    """
    import re
    from collections import defaultdict
    mech = None
    eps = None
    mech_re = re.compile(r'^---\s*(PhraseDP|InferDPT|SANTEXT\+)\s*---')
    eps_re = re.compile(r'^\s*Epsilon\s+([0-9.]+):')
    prot_re = re.compile(r"Protection \(binary\): \{([^}]+)\}, overall=([0-9.]+)")
    store = defaultdict(lambda: defaultdict(list))
    try:
        with open(path, 'r', errors='ignore') as f:
            for line in f:
                m = mech_re.search(line)
                if m:
                    mech = m.group(1)
                    eps = None
                    continue
                m = eps_re.search(line)
                if m:
                    try:
                        eps = float(m.group(1))
                    except Exception:
                        eps = None
                    continue
                m = prot_re.search(line)
                if m and mech and eps is not None:
                    kv = m.group(1)
                    overall = float(m.group(2))
                    d = {'overall': overall}
                    for pair in kv.split(','):
                        if ':' not in pair:
                            continue
                        k, v = pair.split(':', 1)
                        k = k.strip().strip("'\"")
                        try:
                            v = float(v.strip())
                        except Exception:
                            continue
                        d[k] = v
                    store[mech][eps].append(d)
    except FileNotFoundError:
        return {}
    # average
    out = {}
    for mech, eps_map in store.items():
        out[mech] = {}
        for e, vals in eps_map.items():
            if not vals:
                continue
            keys = set().union(*[v.keys() for v in vals])
            avg = {k: sum(v.get(k, 0.0) for v in vals) / len(vals) for k in keys}
            out[mech][e] = avg
    return out


def merge_results(main_res, clusant_eps_map, custext_eps_map):
    merged = dict(main_res)
    
    # Add CluSanT results
    merged['CluSanT'] = {}
    for k, vals in clusant_eps_map.items():
        try:
            e = float(k)
        except Exception:
            continue
        merged['CluSanT'][e] = vals
    
    # Add CusText+ results
    merged['CusText+'] = custext_eps_map
    
    # ensure epsilon keys are float for others too
    for mech in list(merged.keys()):
        eps_map = merged[mech]
        new_map = {}
        for k, v in eps_map.items():
            try:
                e = float(k)
            except Exception:
                continue
            new_map[e] = v
        merged[mech] = new_map
    return merged


def protection_vs_epsilon_plot(merged, epsilons, out_path):
    colors = {
        "PhraseDP": "#1f77b4",   # blue
        "InferDPT": "#17becf",   # cyan (high-contrast vs red/orange/magenta)
        "SANTEXT+": "#2ca02c",   # green
        "CusText+": "#9467bd",   # purple
        "CluSanT": "#e377c2",   # magenta
    }
    linestyles = {
        "PhraseDP": "-",
        "InferDPT": "--",
        "SANTEXT+": "-.",
        "CusText+": ":",
        "CluSanT": (0, (3, 1, 1, 1)),
    }
    markers = {
        "PhraseDP": "o",
        "InferDPT": "s",
        "SANTEXT+": "^",
        "CusText+": "D",
        "CluSanT": "P",
    }
    plt.figure(figsize=(10, 6))
    for mech in ["PhraseDP", "InferDPT", "SANTEXT+", "CusText+", "CluSanT"]:
        if mech not in merged:
            continue
        y = [merged[mech].get(e, {}).get('overall', 0.0) for e in epsilons]
        plt.plot(
            epsilons,
            y,
            marker=markers.get(mech, 'o'),
            linestyle=linestyles.get(mech, '-'),
            linewidth=3.2,
            label=mech,
            color=colors.get(mech),
            markersize=7,
            markeredgecolor="#ffffff",
            markeredgewidth=1.5,
            zorder=3,
        )
    plt.title('Overall PII Protection Rate vs Epsilon (5 mechanisms)')
    plt.xlabel('Epsilon (ε)')
    plt.ylabel('Protection Rate')
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.25, zorder=0)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def radar_plots_per_epsilon(merged, epsilons, out_prefix):
    import math
    mechs = ["PhraseDP", "InferDPT", "SANTEXT+", "CusText+", "CluSanT"]
    metrics = ['overall', 'emails', 'phones', 'addresses', 'names']
    labels = ['Overall', 'Emails', 'Phones', 'Addresses', 'Names']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    colors = {
        "PhraseDP": "#1f77b4",
        "InferDPT": "#17becf",
        "SANTEXT+": "#2ca02c",
        "CusText+": "#9467bd",
        "CluSanT": "#e377c2",
    }
    linestyles = {
        "PhraseDP": "-",
        "InferDPT": "--",
        "SANTEXT+": "-.",
        "CusText+": ":",
        "CluSanT": (0, (3, 1, 1, 1)),
    }
    for e in epsilons:
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='polar'))
        for i, mech in enumerate(mechs):
            if mech not in merged:
                continue
            vals = [merged[mech].get(e, {}).get(m, 0.0) for m in metrics]
            # slight radial jitter to differentiate identical values across mechs
            jitter = 0.002 * i
            vals = [min(1.0, max(0.0, v + jitter)) for v in vals]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2.6, label=mech, color=colors.get(mech), linestyle=linestyles.get(mech, '-'), zorder=3)
            ax.fill(angles, vals, alpha=0.30, color=colors.get(mech), zorder=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title(f'PII Protection Radar (ε={e})', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.05), frameon=True)
        plt.tight_layout()
        out_path = f"{out_prefix}_eps_{str(e).replace('.', '_')}.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()


def main():
    main_res = load_latest_main_results()
    clusant_path = "/home/yizhang/tech4HSE/results/clusant_ppi_protection_20250924_220026.json"
    clusant = load_clusant_results(clusant_path)
    custext = load_custext_results()
    merged = merge_results(main_res, clusant, custext)
    # Override PhraseDP/InferDPT/SANTEXT+ with averages from ppi-protection-exp.txt if available
    txt_path = "/home/yizhang/tech4HSE/ppi-protection-exp.txt"
    txt_avgs = parse_ppi_experiment_txt(txt_path)
    for mech in ("PhraseDP", "InferDPT", "SANTEXT+"):
        if mech in txt_avgs and txt_avgs[mech]:
            merged[mech] = txt_avgs[mech]
    # Epsilon set
    eps = sorted({e for mech in merged for e in merged[mech].keys()})
    # Remove old plots before generating new
    remove_old_plots()
    # Plots
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    line_path = os.path.join(PLOTS_DIR, f'overall_protection_vs_epsilon_5mech_{ts}.png')
    protection_vs_epsilon_plot(merged, eps, line_path)
    radar_prefix = os.path.join(PLOTS_DIR, f'protection_radar_5mech_{ts}')
    radar_plots_per_epsilon(merged, eps, radar_prefix)
    print('Generated:')
    print(line_path)
    for e in eps:
        print(f"{radar_prefix}_eps_{str(e).replace('.', '_')}.png")


if __name__ == '__main__':
    main()


