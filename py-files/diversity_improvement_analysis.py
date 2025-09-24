#!/usr/bin/env python3
"""
Candidate Diversity Analysis and Recommendations
- Reads latest regenerated results (with filtering/refill if present)
- Quantifies coverage in target bands (0.1–0.3, 0.3–0.5, 0.5–0.7, 0.7–0.8, 0.8–0.9)
- Flags thin bands per question
- Produces a Markdown report and emails it with a couple of key plots
"""

import os
import glob
import json
import smtplib
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders


def _find_latest(pattern: str) -> Optional[str]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]


def _bucket_counts(arr: np.ndarray, edges: List[float]) -> List[int]:
    counts = [0] * (len(edges) - 1)
    for v in arr:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i+1]:
                counts[i] += 1
                break
        else:
            if abs(v - edges[-1]) < 1e-9:
                counts[-1] += 1
    return counts


def load_results() -> Dict:
    path = (
        _find_latest('regenerated_fixed_pool_results_*.json')
        or _find_latest('extended_epsilon_comparison_results_*.json')
        or _find_latest('scaled_epsilon_comparison_results_*.json')
    )
    if not path:
        raise FileNotFoundError('No results JSON found.')
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze(data: Dict) -> Dict:
    edges = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.01]
    labels = ['0.0–0.3', '0.3–0.5', '0.5–0.7', '0.7–0.8', '0.8–0.9', '0.9–1.0']

    summary = {
        'per_question': [],
        'global': { 'old': np.zeros(len(labels), dtype=int), 'new': np.zeros(len(labels), dtype=int) },
        'labels': labels,
        'edges': edges,
    }

    for q in data['questions']:
        qidx = q.get('question_index', 0)
        qtext = q.get('question_text', '')

        def get_sims(method: str) -> np.ndarray:
            if 'pools' in q:
                sims = q['pools'][method].get('similarities', [])
                return np.asarray(sims, dtype=float)
            # fallback to epsilon_tests aggregation
            sims_all = []
            for t in q['epsilon_tests']:
                if 'error' in t: continue
                sims_all.extend(t[method]['candidate_similarities'])
            return np.asarray(sims_all, dtype=float)

        sims_old = get_sims('old')
        sims_new = get_sims('new')

        old_counts = _bucket_counts(sims_old, edges) if sims_old.size else [0]*len(labels)
        new_counts = _bucket_counts(sims_new, edges) if sims_new.size else [0]*len(labels)
        summary['global']['old'] += np.asarray(old_counts)
        summary['global']['new'] += np.asarray(new_counts)

        summary['per_question'].append({
            'qidx': qidx,
            'qtext': qtext,
            'old': {
                'n': int(sims_old.size),
                'min': float(np.min(sims_old)) if sims_old.size else float('nan'),
                'max': float(np.max(sims_old)) if sims_old.size else float('nan'),
                'counts': old_counts,
            },
            'new': {
                'n': int(sims_new.size),
                'min': float(np.min(sims_new)) if sims_new.size else float('nan'),
                'max': float(np.max(sims_new)) if sims_new.size else float('nan'),
                'counts': new_counts,
            }
        })

    return summary


def plot_global(summary: Dict) -> str:
    labels = summary['labels']
    old_counts = summary['global']['old']
    new_counts = summary['global']['new']

    x = np.arange(len(labels))
    width = 0.38

    os.makedirs('plots', exist_ok=True)
    plt.figure(figsize=(9,4.5))
    plt.bar(x - width/2, old_counts, width, label='Old', color='salmon')
    plt.bar(x + width/2, new_counts, width, label='New', color='steelblue')
    plt.xticks(x, labels)
    plt.ylabel('Count')
    plt.title('Global candidate similarity coverage by band')
    plt.legend()
    plt.tight_layout()
    out = 'plots/global_candidate_band_coverage.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    return out


def write_report(summary: Dict, plot_path: str) -> str:
    lines = []
    lines.append('# Candidate Diversity Analysis')
    lines.append('')
    lines.append(f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    labels = summary['labels']
    gc_old = summary['global']['old']
    gc_new = summary['global']['new']
    def pct(c):
        tot = int(np.sum(c))
        return [f'{(v/tot*100 if tot else 0):.1f}%' for v in c]
    lines.append('## Global coverage (percent by band)')
    lines.append(f"Old: {dict(zip(labels, pct(gc_old)))}")
    lines.append(f"New: {dict(zip(labels, pct(gc_new)))}")
    lines.append('')
    lines.append('## Per-question thin-band flags (new method)')
    for q in sorted(summary['per_question'], key=lambda d: d['qidx']):
        newc = q['new']['counts']
        thin_low = newc[0] < 10  # 0.0–0.3 band
        thin_low2 = newc[1] < 10 # 0.3–0.5
        lines.append(f"- Q{q['qidx']}: {q['qtext']}")
        lines.append(f"  - new n={q['new']['n']}, range={q['new']['min']:.3f}–{q['new']['max']:.3f}")
        lines.append(f"  - bands: {dict(zip(labels, newc))}")
        if thin_low or thin_low2:
            reasons = []
            if thin_low: reasons.append('low 0.0–0.3 thin')
            if thin_low2: reasons.append('low-mid 0.3–0.5 thin')
            lines.append(f"  - flag: {', '.join(reasons)}")
        lines.append('')
    lines.append('## Recommendations')
    lines.append('- Increase returns for 0.1–0.3 and 0.3–0.5; keep refill-on until targets met (e.g., 30/30).')
    lines.append('- Add de-dup by SBERT (e.g., cosine > 0.92 considered duplicate) to avoid cluster saturation.')
    lines.append('- Provide explicit low-sim examples in prompt; strengthen constraints (replace entities, time, place, domain).')
    lines.append('- For stubborn questions, run a second-stage generator seeded by the least similar candidates to explore farther.')
    lines.append('')

    out_md = 'candidate_diversity_analysis.md'
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return out_md


def send_email(subject: str, body: str, attachments: List[str]) -> None:
    with open('email_config.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    msg = MIMEMultipart()
    msg['From'] = cfg['from_email']
    msg['To'] = cfg['to_email']
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    for path in attachments:
        if not os.path.exists(path):
            continue
        with open(path, 'rb') as fp:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(fp.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(path)}')
            msg.attach(part)
    server = smtplib.SMTP(cfg['smtp_server'], cfg['smtp_port'])
    server.starttls()
    server.login(cfg['from_email'], cfg['password'])
    server.sendmail(cfg['from_email'], cfg['to_email'], msg.as_string())
    server.quit()


def main():
    data = load_results()
    summary = analyze(data)
    plot = plot_global(summary)
    report = write_report(summary, plot)
    send_email(
        subject='Candidate Diversity Analysis and Recommendations',
        body='Attached are the latest candidate diversity analysis and global coverage plot.',
        attachments=[report, plot]
    )
    print('Report emailed and saved:', report, plot)


if __name__ == '__main__':
    main()



