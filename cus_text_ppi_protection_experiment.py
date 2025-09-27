#!/usr/bin/env python3
"""
CusText+ PPI Protection Experiment - Fixed Data Analysis Script
"""

import pandas as pd
import ast
import re
import numpy as np
import json
import time
from typing import Dict, List
import argparse

def _normalize_for_leak(text: str, keep_at: bool = False) -> str:
    """Normalize text for leak checking: case-insensitive, strip punctuation; keep '@' if requested.
    Also remove whitespace. Used to compare original PII substrings vs sanitized text.
    """
    t = text.lower()
    # Remove punctuation; optionally keep '@'
    if keep_at:
        # Remove all punctuation except '@'
        t = re.sub(r"[!\"#\$%&'\(\)\*\+,\-\./:;<=>\?\[\\\]\^_`\{\|\}~]", "", t)
    else:
        t = re.sub(r"[!\"#\$%&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}~]", "", t)
    # Remove all whitespace
    t = re.sub(r"\s+", "", t)
    return t

def reconstruct_text_from_tokens(tokens: List[str], trailing_whitespace: List[bool]) -> str:
    """Reconstruct text from tokens and trailing whitespace information."""
    parts = []
    for i, tok in enumerate(tokens):
        parts.append(tok)
        if i < len(trailing_whitespace) and trailing_whitespace[i]:
            parts.append(' ')
    return ''.join(parts)

def extract_pii_from_labels(tokens: List[str], trailing_whitespace: List[bool], labels: List[str]) -> Dict[str, List[str]]:
    """Extract exact PII spans from BIO labels using tokens + trailing whitespace."""
    spans = []
    current_type = None
    start_idx = None

    def span_text(si: int, ei: int) -> str:
        parts = []
        for i in range(si, ei + 1):
            if i < len(tokens):
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

    out = {'emails': [], 'phones': [], 'addresses': [], 'names': []}
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

def extract_pii_sentences(full_text: str, pii_spans: Dict[str, List[str]]) -> str:
    """Return only sentences from full_text that contain any PII span."""
    spans = set(sum(pii_spans.values(), []))
    if not spans:
        return full_text
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    kept = []
    for s in sentences:
        if any(span and (span in s) for span in spans):
            kept.append(s)
    return ' '.join(kept) if kept else full_text

def load_counter_fitting_vectors(vectors_path: str):
    """Load Counter-Fitting word vectors."""
    print(f"Loading vectors from {vectors_path}")
    emb_matrix = []
    idx2word = []
    word2idx = {}
    
    with open(vectors_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            word = parts[0]
            vector = [float(x) for x in parts[1:]]
            
            emb_matrix.append(vector)
            idx2word.append(word)
            word2idx[word] = i
    
    return np.array(emb_matrix), idx2word, word2idx

def exponential_mechanism_sample(scores: np.ndarray, epsilon: float) -> int:
    """Sample from exponential mechanism."""
    # scores assumed higher=better; apply EM over normalized scores
    if len(scores) == 1:
        return 0
    
    # Normalize scores to [0, 1]
    min_score = np.min(scores)
    max_score = np.max(scores)
    if max_score == min_score:
        return np.random.choice(len(scores))
    
    normalized = (scores - min_score) / (max_score - min_score)
    # Higher scores get higher probabilities
    probs = np.exp(epsilon * normalized)
    probs = probs / np.sum(probs)
    
    return int(np.random.choice(len(scores), p=probs))

def sanitize_with_custext(text: str, epsilon: float, top_k: int, save_stop_words: bool,
                           emb_matrix: np.ndarray, idx2word: List[str], word2idx: Dict[str, int],
                           stop_set: set) -> str:
    """Apply CusText+ sanitization with differential privacy."""
    tokens = text.split()
    out = []
    
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

def calculate_protection_rates(original_pii: Dict[str, List[str]], sanitized_text: str) -> Dict[str, int]:
    """Calculate protection rates using BIO labels (ground truth)."""
    # Normalize sanitized text for comparison
    st_email_norm = _normalize_for_leak(sanitized_text, keep_at=True)
    st_norm = _normalize_for_leak(sanitized_text, keep_at=False)

    def leaked_email(spans):
        """Check if any email span is leaked."""
        for s in spans:
            s_norm = _normalize_for_leak(str(s or ''), keep_at=True)
            if s_norm and s_norm in st_email_norm:
                return True
        return False

    def leaked_generic(spans):
        """Check if any generic PII span is leaked."""
        for s in spans:
            s_norm = _normalize_for_leak(str(s or ''), keep_at=False)
            if s_norm and s_norm in st_norm:
                return True
        return False

    # Calculate protection for each PII type (1 = protected, 0 = leaked)
    email_protected = 0 if leaked_email(original_pii['emails']) else 1
    phone_protected = 0 if leaked_generic(original_pii['phones']) else 1
    addr_protected = 0 if leaked_generic(original_pii['addresses']) else 1
    name_protected = 0 if leaked_generic(original_pii['names']) else 1

    return {
        'emails': email_protected,
        'phones': phone_protected,
        'addresses': addr_protected,
        'names': name_protected
    }

def run_custext_ppi_protection(start_idx: int, num_rows: int, epsilon: float, top_k: int, save_stop_words: bool):
    """Run CusText+ PPI protection experiment with correct data analysis."""
    print(f"Running CusText+ PPI protection experiment:")
    print(f"  Start index: {start_idx}")
    print(f"  Number of rows: {num_rows}")
    print(f"  Epsilon: {epsilon}")
    print(f"  Top-k: {top_k}")
    print(f"  Save stop words: {save_stop_words}")
    
    # Load data
    df = pd.read_csv('/home/yizhang/tech4HSE/pii_external_dataset.csv')
    sample_df = df.iloc[start_idx:start_idx + num_rows].copy()
    
    # Initialize CusText+ components
    try:
        from nltk.corpus import stopwords
        import nltk
        nltk.download('stopwords', quiet=True)
        VECTORS_PATH = "/home/yizhang/tech4HSE/external/CusText/CusText/embeddings/ct_vectors.txt"
        emb_matrix, idx2word, word2idx = load_counter_fitting_vectors(VECTORS_PATH)
        stop_set = set(stopwords.words('english'))
        print(f"✓ Loaded CusText+ vectors: {len(idx2word)} words")
    except Exception as e:
        print(f"✗ Failed to load CusText+ vectors: {e}")
        return

    # Initialize results
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

    print(f"\nProcessing {len(sample_df)} rows...")
    
    for ridx, row in sample_df.iterrows():
        print(f"\n--- Row {ridx} ---")
        
        # Extract data from row
        original_text = row.get('document', '')
        tokens = row.get('tokens')
        labels = row.get('labels')
        trailing_ws = row.get('trailing_whitespace')
        
        try:
            # Parse string representations of lists
            if isinstance(tokens, str):
                tokens = ast.literal_eval(tokens)
            if isinstance(labels, str):
                labels = ast.literal_eval(labels)
            if isinstance(trailing_ws, str):
                trailing_ws = ast.literal_eval(trailing_ws)
        except Exception as e:
            print(f"  Error parsing row data: {e}")
            continue

        # Reconstruct text and extract PII-containing sentences
        if isinstance(tokens, list) and isinstance(labels, list) and isinstance(trailing_ws, list) and len(tokens) == len(labels) == len(trailing_ws):
            reconstructed = reconstruct_text_from_tokens(tokens, trailing_ws)
            pii_preview = extract_pii_from_labels(tokens, trailing_ws, labels)
            original_text = extract_pii_sentences(reconstructed, pii_preview)
        
        print(f"  Original text: {original_text[:100]}...")
        
        # Extract original PII using BIO labels (ground truth)
        original_pii = extract_pii_from_labels(tokens, trailing_ws, labels)
        print(f"  Original PII: emails={len(original_pii['emails'])}, phones={len(original_pii['phones'])}, addresses={len(original_pii['addresses'])}, names={len(original_pii['names'])}")
        
        # Apply CusText+ sanitization
        sanitized = sanitize_with_custext(original_text, epsilon, top_k, save_stop_words,
                                          emb_matrix, idx2word, word2idx, stop_set)
        print(f"  Sanitized text: {sanitized[:100]}...")
        
        # Calculate protection rates using correct logic
        protection_rates = calculate_protection_rates(original_pii, sanitized)
        print(f"  Protection rates: {protection_rates}")
        
        # DEBUG: Check name protection in detail
        if original_pii['names']:
            print(f"  DEBUG - Name protection details:")
            for name in original_pii['names']:
                name_norm = _normalize_for_leak(name, keep_at=False)
                sani_norm = _normalize_for_leak(sanitized, keep_at=False)
                leaked = name_norm in sani_norm
                print(f"    Name: '{name}'")
                print(f"    Normalized: '{name_norm}'")
                print(f"    Sanitized normalized: '{sani_norm[:50]}...'")
                print(f"    Leaked: {leaked}")
                print(f"    Protection: {0 if leaked else 1}")
        
        # Calculate overall protection
        present_scores = []
        for pii_type in ['emails', 'phones', 'addresses', 'names']:
            if len(original_pii[pii_type]) > 0:
                present_scores.append(protection_rates[pii_type])
        
        overall = float(np.mean(present_scores)) if present_scores else 1.0
        print(f"  Overall protection: {overall}")
        
        # Store results
        results['emails'].append(protection_rates['emails'])
        results['phones'].append(protection_rates['phones'])
        results['addresses'].append(protection_rates['addresses'])
        results['names'].append(protection_rates['names'])
        results['overall'].append(overall)
        results['samples'].append({
            'row': int(ridx),
            'original': original_text,
            'sanitized': sanitized
        })

    # Calculate summary statistics
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

    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    out_path = f"/home/yizhang/tech4HSE/results/custext_ppi_protection_{timestamp}.json"
    
    output_data = {
        'summary': summary,
        'results': results
    }
    
    with open(out_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {out_path}")
    print(f"Summary:")
    print(json.dumps(summary, indent=2))

def main():
    parser = argparse.ArgumentParser(description='CusText+ PPI Protection Experiment')
    parser.add_argument('--start', type=int, default=0, help='Start index')
    parser.add_argument('--rows', type=int, default=100, help='Number of rows to process')
    parser.add_argument('--eps', type=float, default=1.0, help='Epsilon value')
    parser.add_argument('--top_k', type=int, default=20, help='Top-k candidates')
    parser.add_argument('--save_stop_words', type=bool, default=True, help='Save stop words')
    
    args = parser.parse_args()
    
    run_custext_ppi_protection(
        start_idx=args.start,
        num_rows=args.rows,
        epsilon=args.eps,
        top_k=args.top_k,
        save_stop_words=args.save_stop_words
    )

if __name__ == '__main__':
    main()
