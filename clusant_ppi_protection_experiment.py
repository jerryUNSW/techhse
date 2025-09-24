#!/usr/bin/env python3
"""
CluSanT PII/PPI Protection Experiment (10-row demo)

Mimics the existing PII protection experiment pipeline but applies the CluSanT
sanitization mechanism to short PII-containing snippets from the external dataset.

Notes:
- Uses all-MiniLM-L6-v2 embeddings and the type-aware, no-identity CluSanT we configured.
- For simplicity, we perform replacements for any tokens present in the CluSanT embeddings
  that appear in the text (case-insensitive, word boundaries). This focuses on LOC/ORG-like tokens
  which CluSanT currently supports out-of-the-box.
"""

import os
import re
import json
import ast
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer


def _normalize_for_leak(text: str, keep_at: bool = False) -> str:
    t = text.lower()
    if keep_at:
        t = re.sub(r"[!\"#\$%&'\(\)\*\+,\-\./:;<=>\?\[\\\]\^_`\{\|\}~]", "", t)
    else:
        t = re.sub(r"[!\"#\$%&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}~]", "", t)
    t = re.sub(r"\s+", "", t)
    return t


def extract_pii_from_labels(tokens: List[str], trailing_whitespace: List[bool], labels: List[str]) -> Dict[str, List[str]]:
    spans: List[Tuple[int, int, str]] = []
    current_type = None
    start_idx = None

    def span_text(si: int, ei: int) -> str:
        parts: List[str] = []
        for i in range(si, ei + 1):
            parts.append(tokens[i])
            if i < len(trailing_whitespace) and trailing_whitespace[i]:
                parts.append(' ')
        return ''.join(parts)

    for i, lab in enumerate(labels):
        if lab.startswith('B-'):
            if current_type is not None and start_idx is not None:
                spans.append((start_idx, i - 1, current_type))
            current_type = lab[2:]
            start_idx = i
        elif lab.startswith('I-'):
            continue
        else:
            if current_type is not None and start_idx is not None:
                spans.append((start_idx, i - 1, current_type))
            current_type = None
            start_idx = None

    if current_type is not None and start_idx is not None:
        spans.append((start_idx, len(labels) - 1, current_type))

    out: Dict[str, List[str]] = {'emails': [], 'phones': [], 'addresses': [], 'names': []}
    for si, ei, t in spans:
        text_span = span_text(si, ei)
        if 'EMAIL' in t:
            out['emails'].append(text_span)
        elif 'PHONE' in t:
            out['phones'].append(text_span)
        elif 'ADDRESS' in t:
            out['addresses'].append(text_span)
        elif 'NAME' in t:
            out['names'].append(text_span)
    return out


def reconstruct_text_from_tokens(tokens: List[str], trailing_whitespace: List[bool]) -> str:
    parts: List[str] = []
    for i, tok in enumerate(tokens):
        parts.append(tok)
        if i < len(trailing_whitespace) and trailing_whitespace[i]:
            parts.append(' ')
    return ''.join(parts)


def extract_pii_sentences(full_text: str, pii_spans: Dict[str, List[str]]) -> str:
    spans = set(sum(pii_spans.values(), []))
    if not spans:
        return full_text
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    kept = []
    for s in sentences:
        if any(span and (span in s) for span in spans):
            kept.append(s)
    return ' '.join(kept) if kept else full_text


def run_clusant_ppi_experiment(start_idx: int = 0, num_rows: int = 10, epsilons: List[float] = [1.0, 1.5, 2.0, 2.5, 3.0]):
    # Load dataset
    df = pd.read_csv('/home/yizhang/tech4HSE/pii_external_dataset.csv')
    end_idx = min(start_idx + num_rows, len(df))
    sample_df = df.iloc[start_idx:end_idx].copy()

    # Prepare CluSanT
    import sys
    sys.path.append('/home/yizhang/tech4HSE/CluSanT/src')
    from embedding_handler import EmbeddingHandler
    from clusant import CluSanT

    # Ensure we run inside CluSanT root so its relative paths (clusters/, inter/, intra/) resolve
    original_cwd = os.getcwd()
    clusant_root = '/home/yizhang/tech4HSE/CluSanT'
    os.chdir(clusant_root)

    emb_dir = os.path.join(clusant_root, 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    emb_path = os.path.join(emb_dir, 'all-MiniLM-L6-v2.txt')
    handler = EmbeddingHandler(model_name='all-MiniLM-L6-v2')
    if not os.path.exists(emb_path):
        handler.generate_and_save_embeddings([
            '/home/yizhang/tech4HSE/CluSanT/clusters/gpt-4/LOC.json',
            '/home/yizhang/tech4HSE/CluSanT/clusters/gpt-4/ORG.json',
        ], emb_dir)
    embeddings = handler.load_embeddings(emb_path)

    results = {'CluSanT': {}}

    for eps in epsilons:
        clus = CluSanT(
            embedding_file='all-MiniLM-L6-v2',
            embeddings=embeddings,
            epsilon=eps,
            num_clusters=336,
            mechanism='clusant',
            metric_to_create_cluster='euclidean',
            distance_metric_for_cluster='euclidean',
            distance_metric_for_words='euclidean',
            dp_type='metric',
            K=16,
        )

        protection_rates = { 'emails': [], 'phones': [], 'addresses': [], 'names': [], 'overall': [] }
        samples: List[Dict[str, str]] = []

        for idx, row in sample_df.iterrows():
            try:
                original_text_full = row['document']
                labels = ast.literal_eval(row['labels'])
                tokens = ast.literal_eval(row['tokens']) if isinstance(row['tokens'], str) else row['tokens']
                trailing_ws = ast.literal_eval(row['trailing_whitespace']) if isinstance(row['trailing_whitespace'], str) else row['trailing_whitespace']
                reconstructed = reconstruct_text_from_tokens(tokens, trailing_ws) if isinstance(tokens, list) and isinstance(trailing_ws, list) else original_text_full
                pii_spans = extract_pii_from_labels(tokens, trailing_ws, labels) if isinstance(labels, list) else {'emails': [], 'phones': [], 'addresses': [], 'names': []}
                original_text = extract_pii_sentences(reconstructed, pii_spans)

                # CluSanT sanitization: replace any tokens that match embeddings words (longest-first to prefer multi-word)
                sanitized_text = original_text
                # Build a set of candidate targets present in text
                # Consider up to 3-word phrases from embeddings for coverage
                # Start with multi-word first
                targets_present = []
                for w in embeddings.keys():
                    if ' ' in w:
                        if re.search(rf"\b{re.escape(w)}\b", sanitized_text, flags=re.IGNORECASE):
                            targets_present.append(w)
                # Then single words
                for w in embeddings.keys():
                    if ' ' not in w:
                        if re.search(rf"\b{re.escape(w)}\b", sanitized_text, flags=re.IGNORECASE):
                            targets_present.append(w)

                # Deduplicate, process longer first
                targets_present = sorted(set(targets_present), key=lambda x: (-len(x), x))

                for t in targets_present:
                    new = clus.replace_word(t)
                    if not new:
                        continue
                    pattern = re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE)
                    if pattern.search(sanitized_text):
                        sanitized_text = pattern.sub(new, sanitized_text)

                # Evaluate protection same as main experiment (binary presence of original PII substrings)
                st_email_norm = _normalize_for_leak(sanitized_text, keep_at=True)
                st_norm = _normalize_for_leak(sanitized_text, keep_at=False)

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

                email_protected = 0 if leaked_email(pii_spans['emails']) else 1
                phone_protected = 0 if leaked_generic(pii_spans['phones']) else 1
                addr_protected  = 0 if leaked_generic(pii_spans['addresses']) else 1
                name_protected  = 0 if leaked_generic(pii_spans['names']) else 1

                present_and_scores = []
                if len(pii_spans['emails']) > 0:
                    present_and_scores.append(email_protected)
                if len(pii_spans['phones']) > 0:
                    present_and_scores.append(phone_protected)
                if len(pii_spans['addresses']) > 0:
                    present_and_scores.append(addr_protected)
                if len(pii_spans['names']) > 0:
                    present_and_scores.append(name_protected)
                overall_protected = float(np.mean(present_and_scores)) if present_and_scores else 1.0

                for k, v in [('emails', email_protected), ('phones', phone_protected), ('addresses', addr_protected), ('names', name_protected)]:
                    protection_rates[k].append(v)
                protection_rates['overall'].append(overall_protected)
                samples.append({'row': int(idx), 'original': original_text, 'sanitized': sanitized_text})

            except Exception as e:
                # Skip problematic rows
                continue

        avg_rates = {k: (np.mean(v) if v else 0.0) for k, v in protection_rates.items()}
        avg_rates['samples'] = samples
        results['CluSanT'][eps] = avg_rates

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = f"/home/yizhang/tech4HSE/results/clusant_ppi_protection_{ts}.json"
    os.makedirs('/home/yizhang/tech4HSE/results', exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Restore original working directory
    os.chdir(original_cwd)

    print(f"Saved CluSanT PPI protection results to: {out_path}")
    return out_path


if __name__ == '__main__':
    run_clusant_ppi_experiment(start_idx=0, num_rows=10, epsilons=[1.0, 1.5, 2.0, 2.5, 3.0])


