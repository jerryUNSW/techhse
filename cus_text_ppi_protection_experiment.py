#!/usr/bin/env python3
"""
CusText PII Protection Experiment
Tests how well CusText/CusText+ (token-level DP with counter-fitted vectors) protects PII.

Usage:
  conda run -n priv-env python cus_text_ppi_protection_experiment.py --start 0 --rows 10 --eps 1.0 --top_k 20 --save_stop_words True
"""

import os
import re
import json
import time
import math
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords

VECTORS_PATH = "/home/yizhang/tech4HSE/external/CusText/CusText/embeddings/ct_vectors.txt"
DATASET_CSV = "/home/yizhang/tech4HSE/pii_external_dataset.csv"


def detect_pii_patterns(text: str) -> Dict[str, List[str]]:
    patterns = {
        'emails': re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text),
        'phones': re.findall(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\(?\d{2}\)?[-.\s]?\d{4}[-.\s]?\d{4}", text),
        'addresses': re.findall(r"\b\d+\s+[A-Za-z\s]+(?:Street|Avenue|Road|Drive|Lane|Boulevard|Way|Place|Court|Terrace)\b", text, re.IGNORECASE),
        'names': re.findall(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text)
    }
    return patterns


def _normalize_for_leak(text: str, keep_at: bool = False) -> str:
    t = text.lower()
    if keep_at:
        t = re.sub(r"[!\"#\$%&'\(\)\*\+,\-\./:;<=>\?\[\\\]\^_`\{\|\}~]", "", t)
    else:
        t = re.sub(r"[!\"#\$%&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}~]", "", t)
    t = re.sub(r"\s+", "", t)
    return t


def load_counter_fitting_vectors(path: str) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Counter-Fitting vectors not found at {path}")
    words: List[str] = []
    vecs: List[List[float]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.rstrip().split()
            if len(parts) < 301:
                continue
            word = parts[0]
            try:
                vals = list(map(float, parts[1:]))
            except Exception:
                continue
            words.append(word)
            vecs.append(vals)
    mat = np.asarray(vecs, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    word2idx = {w: i for i, w in enumerate(words)}
    return mat, words, word2idx


def exponential_mechanism_sample(scores: np.ndarray, epsilon: float) -> int:
    # scores assumed higher=better; apply EM over normalized scores
    if scores.size == 0:
        return 0
    s_min = float(np.min(scores))
    s_max = float(np.max(scores))
    if s_max == s_min:
        probs = np.ones_like(scores) / scores.size
    else:
        norm_scores = (scores - s_min) / (s_max - s_min)
        weights = np.exp((epsilon / 2.0) * norm_scores)
        probs = weights / np.sum(weights)
    return int(np.random.choice(len(scores), p=probs))


def sanitize_with_custext(text: str, epsilon: float, top_k: int, save_stop_words: bool,
                           emb_matrix: np.ndarray, idx2word: List[str], word2idx: Dict[str, int],
                           stop_set: set) -> str:
    tokens = text.split()
    out: List[str] = []
    for tok in tokens:
        base = tok
        key = tok.lower()
        # Stopword preservation for CusText+
        if save_stop_words and key in stop_set:
            out.append(base)
            continue
        # Keep numbers and OOV unchanged
        if key not in word2idx:
            out.append(base)
            continue
        i = word2idx[key]
        v = emb_matrix[i]
        sims = emb_matrix @ v
        # Get top_k neighbors by cosine (including self)
        nn_idx = np.argpartition(-sims, range(min(top_k, sims.size)))[:top_k]
        nn_sims = sims[nn_idx]
        sel = exponential_mechanism_sample(nn_sims, epsilon)
        new_word = idx2word[int(nn_idx[sel])]
        # Preserve capitalization pattern (simple heuristic)
        if tok.istitle():
            new_word = new_word.capitalize()
        elif tok.isupper():
            new_word = new_word.upper()
        out.append(new_word)
    return " ".join(out)


def run_custext_ppi_protection(start_idx: int, num_rows: int, epsilon: float, top_k: int, save_stop_words: bool):
    nltk.download('stopwords', quiet=True)
    stop_set = set(stopwords.words('english'))

    if not os.path.exists(DATASET_CSV):
        raise FileNotFoundError(f"Dataset not found at {DATASET_CSV}. Run download_pii_dataset.py first.")
    df = pd.read_csv(DATASET_CSV)

    end_idx = min(len(df), start_idx + num_rows)
    if end_idx <= start_idx:
        raise ValueError("Empty slice; adjust --start/--rows")
    df = df.iloc[start_idx:end_idx].copy()

    emb_matrix, idx2word, word2idx = load_counter_fitting_vectors(VECTORS_PATH)

    results = {
        'epsilon': epsilon,
        'top_k': top_k,
        'save_stop_words': save_stop_words,
        'overall': [],
        'emails': [],
        'phones': [],
        'addresses': [],
        'names': [],
        'samples': []
    }

    for ridx, row in df.iterrows():
        original_text = row.get('document', '')
        if not isinstance(original_text, str):
            original_text = str(original_text)
        orig_pii = detect_pii_patterns(original_text)

        sanitized = sanitize_with_custext(original_text, epsilon, top_k, save_stop_words,
                                          emb_matrix, idx2word, word2idx, stop_set)
        san_pii = detect_pii_patterns(sanitized)

        st_email_norm = _normalize_for_leak(sanitized, keep_at=True)
        st_norm = _normalize_for_leak(sanitized, keep_at=False)

        def leaked_email(spans):
            for s in spans:
                s_norm = _normalize_for_leak(str(s or ''), keep_at=True)
                if s_norm and s_norm in st_email_norm:
                    return True
            return False

        def leaked_generic(spans):
            for s in spans:
                s_norm = _normalize_for_leak(str(s or ''), keep_at=False)
                if s_norm and s_norm in st_norm:
                    return True
            return False

        email_protected = 0 if leaked_email(orig_pii['emails']) else 1
        phone_protected = 0 if leaked_generic(orig_pii['phones']) else 1
        addr_protected  = 0 if leaked_generic(orig_pii['addresses']) else 1
        name_protected  = 0 if leaked_generic(orig_pii['names']) else 1

        present_and_scores = []
        if len(orig_pii['emails']) > 0: present_and_scores.append(email_protected)
        if len(orig_pii['phones']) > 0: present_and_scores.append(phone_protected)
        if len(orig_pii['addresses']) > 0: present_and_scores.append(addr_protected)
        if len(orig_pii['names']) > 0: present_and_scores.append(name_protected)
        overall = float(np.mean(present_and_scores)) if present_and_scores else 1.0

        results['emails'].append(email_protected)
        results['phones'].append(phone_protected)
        results['addresses'].append(addr_protected)
        results['names'].append(name_protected)
        results['overall'].append(overall)
        results['samples'].append({
            'row': int(ridx),
            'original': original_text,
            'sanitized': sanitized
        })

    summary = {
        'epsilon': epsilon,
        'top_k': top_k,
        'save_stop_words': save_stop_words,
        'emails': float(np.mean(results['emails'])) if results['emails'] else 0.0,
        'phones': float(np.mean(results['phones'])) if results['phones'] else 0.0,
        'addresses': float(np.mean(results['addresses'])) if results['addresses'] else 0.0,
        'names': float(np.mean(results['names'])) if results['names'] else 0.0,
        'overall': float(np.mean(results['overall'])) if results['overall'] else 0.0,
    }

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = f"/home/yizhang/tech4HSE/results/custext_ppi_protection_{ts}.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'summary': summary, 'results': results}, f, indent=2)
    print(f"Saved results to {out_path}")

    print("Summary:")
    print(json.dumps(summary, indent=2))
    return out_path


def main():
    parser = argparse.ArgumentParser(description='CusText PII Protection Experiment')
    parser.add_argument('--start', type=int, default=0, help='Start index in dataset')
    parser.add_argument('--rows', type=int, default=10, help='Number of rows to test')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon')
    parser.add_argument('--top_k', type=int, default=20, help='Top-K neighbors')
    parser.add_argument('--save_stop_words', type=str, default='True', help='CusText+ stopword preservation (True/False)')
    args = parser.parse_args()
    save_sw = str(args.save_stop_words).lower() in ('1','true','yes','y')
    run_custext_ppi_protection(args.start, args.rows, args.eps, args.top_k, save_sw)


if __name__ == '__main__':
    main()


