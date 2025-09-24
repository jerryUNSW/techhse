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
    sbert = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 10 test questions
    questions = [
        'What is the capital of France?',
        'Who wrote Romeo and Juliet?',
        'What is the largest planet in our solar system?',
        'What is the chemical symbol for gold?',
        'In which year did World War II end?',
        'What is the speed of light in vacuum?',
        'Who painted the Mona Lisa?',
        'What is the smallest country in the world?',
        'What is the currency of Japan?',
        'Who discovered penicillin?'
    ]
    
    # Epsilon values and sampling parameters
    EPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    K = 30  # Samples per epsilon
    
    # Results storage
    all_results = []
    per_question_plots = []
    
    print(f"Running 10-question epsilon test with K={K} samples per epsilon...")
    print(f"Epsilon values: {EPS}")
    print()
    
    for q_idx, question in enumerate(questions):
        print(f"Processing question {q_idx + 1}/10: {question}")
        
        # Generate balanced candidates for this question
        candidates = utils.generate_sentence_replacements_with_nebius_diverse(
            client, model, question,
            num_return_sequences=20, num_api_calls=10,
            enforce_similarity_filter=True, filter_margin=0.05,
            low_band_quota_boost=True, refill_underfilled_bands=True,
            max_refill_retries=3, equal_band_target=30,
            verbose=False
        )
        
        # Calculate similarities
        sims = np.array([sbert_sim(sbert, question, c) for c in candidates], dtype=float)
        
        # Create balanced subset
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
        target = min(raw_counts) if min(raw_counts) > 0 else 0
        
        balanced_idx = []
        if target > 0:
            rng = np.random.default_rng(42)
            for i in range(len(bands)):
                if len(idx_by_band[i]) >= target:
                    sel = rng.choice(idx_by_band[i], size=target, replace=False)
                    balanced_idx.extend(sel.tolist())
            balanced_idx = sorted(set(balanced_idx))
        
        balanced_sims = sims[balanced_idx]
        
        # Use all candidates if balanced subset is too small
        if len(balanced_sims) < 5:
            print(f"  Warning: Only {len(balanced_sims)} balanced candidates, using all {len(sims)} candidates instead...")
            balanced_sims = sims
        
        # Epsilon sweep for this question
        means = []
        sems = []
        expected = []
        
        for eps in EPS:
            m, sem = sample_mean(balanced_sims, eps, K)
            exp = softmax_expectation(balanced_sims, eps)
            means.append(m)
            sems.append(sem)
            expected.append(exp)
        
        # Store results
        question_result = {
            'question': question,
            'total_candidates': len(candidates),
            'balanced_candidates': len(balanced_sims),
            'similarity_range': [float(sims.min()), float(sims.max())],
            'mean_similarity': float(sims.mean()),
            'band_counts': raw_counts,
            'balanced_target': target,
            'epsilon_results': []
        }
        
        for i, eps in enumerate(EPS):
            question_result['epsilon_results'].append({
                'epsilon': eps,
                'observed_mean': means[i],
                'observed_sem': sems[i],
                'expected_mean': expected[i]
            })
        
        all_results.append(question_result)
        
        # Create per-question plot
        os.makedirs('plots/per_question', exist_ok=True)
        plot_file = f'plots/per_question/question_{q_idx:02d}_epsilon_trend.png'
        
        plt.figure(figsize=(10, 6))
        plt.errorbar(EPS, means, yerr=sems, 
                    marker='o', markersize=8, linewidth=2, 
                    label='Observed (K=30)', color='blue', capsize=5)
        plt.plot(EPS, expected, 
                marker='s', markersize=8, linewidth=2, 
                label='Expected (theoretical)', color='red', linestyle='--')
        
        # Add trend line
        z = np.polyfit(EPS, means, 1)
        p = np.poly1d(z)
        plt.plot(EPS, p(EPS), "g--", alpha=0.7, linewidth=1, label=f'Trend (slope={z[0]:.3f})')
        
        plt.xlabel('Epsilon (ε)', fontsize=12)
        plt.ylabel('Average Selected Similarity', fontsize=12)
        plt.title(f'Question {q_idx+1}: {question[:50]}{"..." if len(question) > 50 else ""}\nEpsilon vs Average Selected Similarity', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.xlim(0.3, 3.2)
        plt.ylim(min(min(means), min(expected)) - 0.05, 
                 max(max(means), max(expected)) + 0.05)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=200, bbox_inches='tight')
        plt.close()
        
        per_question_plots.append(plot_file)
        
        print(f"  Candidates: {len(candidates)} total, {len(balanced_sims)} balanced")
        print(f"  Similarity range: {sims.min():.3f} to {sims.max():.3f}")
        print(f"  Trend slope: {z[0]:.3f}")
        print()
    
    # Create summary plot
    print("Creating summary plot...")
    summary_plot = 'plots/ten_question_summary.png'
    
    plt.figure(figsize=(12, 8))
    
    # Plot all questions
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for q_idx, result in enumerate(all_results):
        means = [r['observed_mean'] for r in result['epsilon_results']]
        plt.plot(EPS, means, marker='o', linewidth=2, markersize=6,
                label=f"Q{q_idx+1}", color=colors[q_idx], alpha=0.8)
    
    plt.xlabel('Epsilon (ε)', fontsize=12)
    plt.ylabel('Average Selected Similarity', fontsize=12)
    plt.title('10-Question Epsilon Test: All Questions Overlay\n(K=30 samples per epsilon)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.3, 3.2)
    plt.tight_layout()
    plt.savefig(summary_plot, dpi=200, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics
    print("Calculating summary statistics...")
    summary_stats = {
        'total_questions': len(questions),
        'epsilon_values': EPS,
        'samples_per_epsilon': K,
        'per_question_results': []
    }
    
    for q_idx, result in enumerate(all_results):
        # Calculate trend slope
        means = [r['observed_mean'] for r in result['epsilon_results']]
        z = np.polyfit(EPS, means, 1)
        slope = z[0]
        
        # Check monotonicity
        is_monotonic = all(means[i] <= means[i+1] for i in range(len(means)-1))
        
        summary_stats['per_question_results'].append({
            'question_idx': q_idx + 1,
            'question': result['question'],
            'total_candidates': result['total_candidates'],
            'balanced_candidates': result['balanced_candidates'],
            'similarity_range': result['similarity_range'],
            'mean_similarity': result['mean_similarity'],
            'trend_slope': slope,
            'is_monotonic': is_monotonic,
            'epsilon_results': result['epsilon_results']
        })
    
    # Save results to JSON
    results_file = 'ten_question_epsilon_results.json'
    with open(results_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Create text report
    report_lines = [
        "10-Question Epsilon Test Results",
        "=" * 50,
        f"Total questions: {len(questions)}",
        f"Epsilon values: {EPS}",
        f"Samples per epsilon: {K}",
        f"Date: {np.datetime64('now')}",
        "",
        "PER-QUESTION SUMMARY:",
        ""
    ]
    
    for result in summary_stats['per_question_results']:
        report_lines.extend([
            f"Question {result['question_idx']}: {result['question']}",
            f"  Candidates: {result['total_candidates']} total, {result['balanced_candidates']} balanced",
            f"  Similarity range: {result['similarity_range'][0]:.3f} to {result['similarity_range'][1]:.3f}",
            f"  Mean similarity: {result['mean_similarity']:.3f}",
            f"  Trend slope: {result['trend_slope']:.3f}",
            f"  Monotonic trend: {'Yes' if result['is_monotonic'] else 'No'}",
            ""
        ])
    
    # Add epsilon summary
    report_lines.extend([
        "EPSILON SUMMARY:",
        ""
    ])
    
    for eps in EPS:
        eps_means = []
        for result in summary_stats['per_question_results']:
            eps_result = next(r for r in result['epsilon_results'] if r['epsilon'] == eps)
            eps_means.append(eps_result['observed_mean'])
        
        overall_mean = np.mean(eps_means)
        overall_std = np.std(eps_means)
        report_lines.append(f"ε = {eps:.1f}: Mean = {overall_mean:.3f} ± {overall_std:.3f} (across {len(eps_means)} questions)")
    
    report_text = '\n'.join(report_lines)
    
    # Save text report
    report_file = 'ten_question_epsilon_report.txt'
    with open(report_file, 'w') as f:
        f.write(report_text)
    
    print("Test completed!")
    print(f"Results saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    print(f"Summary plot: {summary_plot}")
    print(f"Per-question plots: {len(per_question_plots)} plots in plots/per_question/")
    
    # Email results
    print("Sending email...")
    attachments = [results_file, report_file, summary_plot] + per_question_plots
    
    send_email(
        subject='10-Question Epsilon Test Results',
        body=report_text,
        attachments=attachments
    )
    
    print("Email sent successfully!")

if __name__ == '__main__':
    main()
