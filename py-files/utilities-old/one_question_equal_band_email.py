#!/usr/bin/env python3
import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

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


def send_email(subject: str, body: str, attachments: list):
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
    import smtplib
    server = smtplib.SMTP(cfg['smtp_server'], cfg['smtp_port'])
    server.starttls()
    server.login(cfg['from_email'], cfg['password'])
    server.sendmail(cfg['from_email'], cfg['to_email'], msg.as_string())
    server.quit()


def main():
    load_dotenv()
    api = os.getenv('NEBIUS')
    if not api:
        raise RuntimeError('NEBIUS API key not found in env')

    client = OpenAI(api_key=api, base_url='https://api.studio.nebius.ai/v1/')
    model = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    question = 'What is the capital of France?'
    sbert = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate with equal-band target (e.g., 30 per band)
    candidates = utils.generate_sentence_replacements_with_nebius_diverse(
        client, model, question,
        num_return_sequences=20, num_api_calls=10,
        enforce_similarity_filter=True, filter_margin=0.05,
        low_band_quota_boost=True, refill_underfilled_bands=True,
        max_refill_retries=3, equal_band_target=30,
        verbose=True  # Show detailed candidate output
    )

    sims = np.array([sbert_sim(sbert, question, c) for c in candidates], dtype=float)
    pool_stats = f"n={len(candidates)}, min={sims.min():.3f}, max={sims.max():.3f}, mean={sims.mean():.3f}"

    # Verification and balancing
    bands = [(0.0,0.1),(0.1,0.2),(0.2,0.3),(0.3,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
    def assign_band(val):
        for i,(lo,hi) in enumerate(bands):
            if lo-0.05 <= val <= hi+0.05:
                return i
        return None
    idx_by_band = {i: [] for i in range(len(bands))}
    for i, s in enumerate(sims):
        b = assign_band(s)
        if b is not None:
            idx_by_band[b].append(i)
    raw_counts = [len(idx_by_band[i]) for i in range(len(bands))]

    # If not roughly equal, balance by subsampling to the minimum available
    target = min(raw_counts) if min(raw_counts) > 0 else 0
    balanced_idx = []
    if target > 0:
        rng = np.random.default_rng(42)
        for i in range(len(bands)):
            sel = rng.choice(idx_by_band[i], size=target, replace=False)
            balanced_idx.extend(sel.tolist())
        balanced_idx = sorted(set(balanced_idx))
    sims_bal = sims[balanced_idx] if balanced_idx else sims
    candidates_bal = [candidates[i] for i in balanced_idx] if balanced_idx else candidates
    bal_counts = None
    if balanced_idx:
        tmp = {i:0 for i in range(len(bands))}
        for s in sims_bal:
            tmp[assign_band(s)] += 1
        bal_counts = [tmp[i] for i in range(len(bands))]

    # Plot distribution
    os.makedirs('plots', exist_ok=True)
    dist_png = 'plots/one_question_equal_band_distribution.png'
    plt.figure(figsize=(7,4))
    plt.hist(sims, bins=25, color='purple', alpha=0.85)
    plt.xlabel('Similarity')
    plt.ylabel('Count')
    plt.title('Candidate similarity distribution (equal-band refill)')
    plt.xlim(0,1)
    # Set y-axis to show full range of data
    plt.ylim(0, max(50, int(max(np.histogram(sims, bins=25)[0]) * 1.1)))
    plt.tight_layout()
    plt.savefig(dist_png, dpi=200, bbox_inches='tight')
    plt.close()

    # Plot balanced pool if created
    dist_bal_png = None
    if balanced_idx:
        dist_bal_png = 'plots/one_question_equal_band_distribution_balanced.png'
        plt.figure(figsize=(7,4))
        plt.hist(sims_bal, bins=25, color='teal', alpha=0.85)
        plt.xlabel('Similarity')
        plt.ylabel('Count')
        plt.title('Candidate similarity distribution (balanced bands)')
        plt.xlim(0,1)
        # Set y-axis to show full range of data
        plt.ylim(0, max(10, int(max(np.histogram(sims_bal, bins=25)[0]) * 1.1)))
        plt.tight_layout()
        plt.savefig(dist_bal_png, dpi=200, bbox_inches='tight')
        plt.close()

    # Band bar plot for verification
    band_bar_png = 'plots/one_question_equal_band_bar.png'
    import matplotlib.ticker as mtick
    fig, ax = plt.subplots(figsize=(12,3.6))
    labels = ['0.0–0.1','0.1–0.2','0.2–0.3','0.3–0.4','0.4–0.5','0.5–0.6','0.6–0.7','0.7–0.8','0.8–0.9','0.9–1.0']
    x = np.arange(len(labels))
    raw = raw_counts
    bal = bal_counts if bal_counts else raw
    ax.bar(x-0.18, raw, width=0.36, label='raw', color='gray', alpha=0.7)
    ax.bar(x+0.18, bal, width=0.36, label='final', color='teal', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Count')
    ax.set_title('Band counts (verification)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(band_bar_png, dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Optional: equalize by fine-grained histogram bins (visual uniformity)
    bins = np.linspace(0.0, 1.0, 26)
    hist, _ = np.histogram(sims_bal if balanced_idx else sims, bins=bins)
    target_bin = hist.min() if hist.min() > 0 else 0
    fine_bal_idx = []
    if target_bin > 0:
        source = sims_bal if balanced_idx else sims
        idx_all = np.arange(len(source))
        for b in range(len(bins)-1):
            mask = (source >= bins[b]) & (source < bins[b+1]) if b < len(bins)-2 else (source >= bins[b]) & (source <= bins[b+1])
            idx_bin = idx_all[mask]
            if len(idx_bin) >= target_bin:
                sel = np.random.default_rng(123).choice(idx_bin, size=target_bin, replace=False)
                fine_bal_idx.extend(sel.tolist())
        fine_bal_idx = sorted(set(fine_bal_idx))
    fine_bal_png = None
    if fine_bal_idx:
        sims_fine = (sims_bal if balanced_idx else sims)[fine_bal_idx]
        fine_bal_png = 'plots/one_question_equal_bin_distribution_balanced.png'
        plt.figure(figsize=(7,4))
        plt.hist(sims_fine, bins=25, color='orange', alpha=0.85)
        plt.xlabel('Similarity')
        plt.ylabel('Count')
        plt.title('Candidate similarity distribution (balanced per 25 bins)')
        plt.xlim(0,1)
        # Set y-axis to show full range of data
        plt.ylim(0, max(10, int(max(np.histogram(sims_fine, bins=25)[0]) * 1.1)))
        plt.tight_layout()
        plt.savefig(fine_bal_png, dpi=200, bbox_inches='tight')
        plt.close()

    # Epsilon sweep
    EPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    K = 30
    lines = [f'Question: {question}', f'Pool: {pool_stats}']
    lines.append(f'Raw band counts (0.0–0.1,0.1–0.2,0.2–0.3,0.3–0.4,0.4–0.5,0.5–0.6,0.6–0.7,0.7–0.8,0.8–0.9,0.9–1.0): {raw_counts}')
    if bal_counts:
        lines.append(f'Balanced band counts: {bal_counts} (target per band={target})')
    lines.append('')
    lines.append('ε  mean  sem  expected')
    for eps in EPS:
        cur = sims_bal if balanced_idx else sims
        m, sem = sample_mean(cur, eps, K)
        exp = softmax_expectation(cur, eps)
        lines.append(f"{eps:.1f}  {m:.3f}  {sem:.3f}  {exp:.3f}")
    report_txt = '\n'.join(lines)

    # Save text report
    out_txt = 'one_question_equal_band_results.txt'
    with open(out_txt, 'w') as f:
        f.write(report_txt + '\n')

    # Email
    atts = [out_txt, dist_png, band_bar_png]
    if dist_bal_png:
        atts.append(dist_bal_png)
    if fine_bal_png:
        atts.append(fine_bal_png)
    send_email(
        subject='One-question equal-band refill results',
        body=report_txt,
        attachments=atts
    )

    print('Saved and emailed:', out_txt, dist_png)


if __name__ == '__main__':
    main()
