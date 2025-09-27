#!/usr/bin/env python3
"""
PII Protection Experiment
Tests how well PhraseDP, InferDPT, SANTEXT+, and CusText+ protect PII information
"""

import pandas as pd
import numpy as np
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import ast
from typing import Dict, List, Tuple
import warnings
import argparse
warnings.filterwarnings('ignore')

# Import your existing mechanisms
from utils import phrase_DP_perturbation_old
from santext_integration import create_santext_mechanism
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
from openai import OpenAI
import time
import sys
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

# Ensure unbuffered/line-buffered stdout so progress prints appear immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

def detect_pii_patterns(text: str) -> Dict[str, List[str]]:
    """Detect PII patterns in text using regex"""
    patterns = {
        'emails': re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text),
        'phones': re.findall(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\(?\d{2}\)?[-.\s]?\d{4}[-.\s]?\d{4}', text),
        'addresses': re.findall(r'\d+\s+[A-Za-z\s]+(?:Street|Avenue|Road|Drive|Lane|Boulevard|Way|Place|Court|Terrace)', text, re.IGNORECASE),
        'names': re.findall(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b', text)  # Simple name pattern
    }
    return patterns

def _normalize_for_leak(text: str, keep_at: bool = False) -> str:
    """Normalize text for leak checking: case-insensitive, strip punctuation; keep '@' if requested.
    Also remove whitespace. Used to compare original PII substrings vs sanitized text.
    """
    t = text.lower()
    # Remove punctuation; optionally keep '@'
    if keep_at:
        # Remove all punctuation except '@'
        t = re.sub(r"[\p{P}&&[^@]]", "", t) if False else re.sub(r"[!\"#\$%&'\(\)\*\+,\-\./:;<=>\?\[\\\]\^_`\{\|\}~]", "", t)
        # Re-add '@' occurrences by replacing removal rule above except '@' was preserved by regex; included fallback for environments lacking \p{P}
    else:
        t = re.sub(r"[!\"#\$%&'\(\)\*\+,\-\./:;<=>\?@\[\\\]\^_`\{\|\}~]", "", t)
    # Remove all whitespace
    t = re.sub(r"\s+", "", t)
    return t

def reconstruct_text_from_tokens(tokens: List[str], trailing_whitespace: List[bool]) -> str:
    parts: List[str] = []
    for i, tok in enumerate(tokens):
        parts.append(tok)
        if i < len(trailing_whitespace) and trailing_whitespace[i]:
            parts.append(' ')
    return ''.join(parts)

def extract_pii_from_labels(tokens: List[str], trailing_whitespace: List[bool], labels: List[str]) -> Dict[str, List[str]]:
    """Extract exact PII spans from BIO labels using tokens + trailing whitespace."""
    spans: List[Tuple[int, int, str]] = []
    current_type = None
    start_idx = None

    def span_text(si: int, ei: int) -> str:
        # si..ei inclusive indices
        parts: List[str] = []
        for i in range(si, ei + 1):
            parts.append(tokens[i])
            if i < len(trailing_whitespace) and trailing_whitespace[i]:
                parts.append(' ')
        return ''.join(parts)

    for i, lab in enumerate(labels):
        if lab.startswith('B-'):
            # close previous
            if current_type is not None and start_idx is not None:
                spans.append((start_idx, i - 1, current_type))
            current_type = lab[2:]
            start_idx = i
        elif lab.startswith('I-'):
            # continue
            continue
        else:
            # Outside
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

def render_highlighted_from_labels(tokens: List[str], trailing_whitespace: List[bool], labels: List[str]) -> str:
    """Render text with PII tokens highlighted using BIO labels and ANSI colors."""
    RED = "\033[91m"; RESET = "\033[0m"; UNDER = "\033[4m"; ENDUNDER = "\033[24m"
    parts: List[str] = []
    for i, tok in enumerate(tokens):
        lab = labels[i] if i < len(labels) else 'O'
        highlighted = f"{RED}{UNDER}{tok}{ENDUNDER}{RESET}" if lab.startswith(('B-','I-')) else tok
        parts.append(highlighted)
        if i < len(trailing_whitespace) and trailing_whitespace[i]:
            parts.append(' ')
    return ''.join(parts)

def highlight_pii_regex(text: str) -> str:
    """Highlight regex-detected PII in a text with ANSI colors (best-effort)."""
    RED = "\033[91m"; RESET = "\033[0m"; UNDER = "\033[4m"; ENDUNDER = "\033[24m"
    patterns = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\(?\d{2}\)?[-.\s]?\d{4}[-.\s]?\d{4}",
        r"\b\d+\s+[A-Za-z\s]+(?:Street|Avenue|Road|Drive|Lane|Boulevard|Way|Place|Court|Terrace)\b",
    ]
    out = text
    for pat in patterns:
        out = re.sub(pat, lambda m: f"{RED}{UNDER}{m.group(0)}{ENDUNDER}{RESET}", out)
    return out

def extract_pii_sentences(full_text: str, pii_spans: Dict[str, List[str]]) -> str:
    """Return only sentences from full_text that contain any PII span.
    Fallback to full_text if no spans or no match.
    """
    import re
    spans = set(sum(pii_spans.values(), []))
    if not spans:
        return full_text
    # Simple sentence split on punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', full_text)
    kept = []
    for s in sentences:
        if any(span and (span in s) for span in spans):
            kept.append(s)
    return ' '.join(kept) if kept else full_text

def calculate_pii_protection_rate(original_pii: Dict, sanitized_pii: Dict) -> Dict[str, float]:
    """Calculate protection rate for each PII type"""
    protection_rates = {}
    
    for pii_type in ['emails', 'phones', 'addresses', 'names']:
        original_count = len(original_pii.get(pii_type, []))
        sanitized_count = len(sanitized_pii.get(pii_type, []))
        
        if original_count == 0:
            protection_rates[pii_type] = 1.0  # No PII to protect
        else:
            protection_rates[pii_type] = 1.0 - (sanitized_count / original_count)
    
    return protection_rates

def load_existing_results(results_file: str = None) -> Dict:
    """Load existing results from file for resumption"""
    if not results_file or not os.path.exists(results_file):
        return {}
    try:
        with open(results_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load existing results from {results_file}: {e}")
        return {}

def save_incremental_results(results: Dict, results_file: str):
    """Save results incrementally after each mechanism/epsilon completion"""
    try:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved incrementally to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save incremental results: {e}")

class EmbeddingCache:
    """Cache for embeddings to avoid reloading"""
    def __init__(self):
        self.sbert_model = None
        self.ct_embeddings = None
        self.clusant_embeddings = None
        self.inferdpt_embeddings = None
        self.phrase_cache = {}

    def get_sbert_model(self):
        if self.sbert_model is None:
            print("Loading SBERT model...")
            self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.sbert_model

    def get_custext_embeddings(self):
        if self.ct_embeddings is None:
            try:
                print("Loading CusText+ embeddings...")
                from cus_text_ppi_protection_experiment import load_counter_fitting_vectors
                from nltk.corpus import stopwords
                import nltk
                nltk.download('stopwords', quiet=True)
                VECTORS_PATH = "/home/yizhang/tech4HSE/external/CusText/CusText/embeddings/ct_vectors.txt"
                ct_emb_matrix, ct_idx2word, ct_word2idx = load_counter_fitting_vectors(VECTORS_PATH)
                ct_stop_set = set(stopwords.words('english'))
                self.ct_embeddings = (ct_emb_matrix, ct_idx2word, ct_word2idx, ct_stop_set)
            except Exception as e:
                print(f"Warning: CusText+ embedding init failed: {e}")
                self.ct_embeddings = (None, None, None, set())
        return self.ct_embeddings

    def get_inferdpt_embeddings(self):
        if self.inferdpt_embeddings is None:
            try:
                print("Loading InferDPT embeddings...")
                from inferdpt import initialize_embeddings
                # Initialize once and cache the result
                token_to_vector_dict, sorted_distance_data, delta_f_new = initialize_embeddings(epsilon=1.0)  # epsilon doesn't affect loading
                self.inferdpt_embeddings = (token_to_vector_dict, sorted_distance_data, delta_f_new)
            except Exception as e:
                print(f"Warning: InferDPT embedding init failed: {e}")
                self.inferdpt_embeddings = (None, None, None)
        return self.inferdpt_embeddings

    def get_clusant_embeddings(self):
        if self.clusant_embeddings is None:
            try:
                print("Loading CluSanT embeddings...")
                clusant_root = '/home/yizhang/tech4HSE/CluSanT'
                import sys
                sys.path.append('/home/yizhang/tech4HSE/CluSanT/src')
                from embedding_handler import EmbeddingHandler  # type: ignore
                clus_handler = EmbeddingHandler(model_name='all-MiniLM-L6-v2')
                emb_dir = f"{clusant_root}/embeddings"
                os.makedirs(emb_dir, exist_ok=True)
                emb_path = f"{emb_dir}/all-MiniLM-L6-v2.txt"
                if not os.path.exists(emb_path):
                    clus_handler.generate_and_save_embeddings([
                        f"{clusant_root}/clusters/gpt-4/LOC.json",
                        f"{clusant_root}/clusters/gpt-4/ORG.json",
                    ], emb_dir)
                clus_embeddings = clus_handler.load_embeddings(emb_path)
                self.clusant_embeddings = (clus_embeddings, clusant_root)
            except Exception as e:
                print(f"Warning: CluSanT embedding init failed: {e}")
                self.clusant_embeddings = (None, None)
        return self.clusant_embeddings

def run_pii_protection_experiment(start_idx: int = 0, num_rows: int = 10, resume_file: str = None):
    """Run the PII protection experiment"""
    print("Loading PII dataset...")
    
    # Load the dataset
    try:
        df = pd.read_csv('/home/yizhang/tech4HSE/pii_external_dataset.csv')
        print(f"Loaded {len(df)} rows from PII dataset")
    except FileNotFoundError:
        print("PII dataset not found. Please run download_pii_dataset.py first.")
        return
    
    # Select rows based on start index and number of rows
    end_idx = start_idx + num_rows
    if end_idx > len(df):
        print(f"Warning: Requested end index {end_idx} exceeds dataset size {len(df)}. Using available rows.")
        end_idx = len(df)
        num_rows = end_idx - start_idx
    
    sample_df = df.iloc[start_idx:end_idx].copy()
    total_rows = len(sample_df)
    print(f"Using rows {start_idx} to {end_idx-1} ({total_rows} rows) for experiment")
    
    # Initialize embedding cache
    embedding_cache = EmbeddingCache()

    # Setup results file for incremental saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = resume_file or f'/home/yizhang/tech4HSE/pii_protection_results_{timestamp}.json'

    # Load existing results if resuming
    results = load_existing_results(results_file)
    if not results:
        results = {
            'PhraseDP': {},
            'InferDPT': {},
            'SANTEXT+': {},
            'CusText+': {},
            'CluSanT': {}
        }
    else:
        print(f"Resuming experiment from: {results_file}")
    
    # Run epsilon sweep
    epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    
    # Models will be loaded on-demand via embedding cache
    
    print("Starting PII protection experiment...")

    # Initialize Nebius client for PhraseDP candidate generation
    load_dotenv()
    nebius_api_key = os.getenv("NEBIUS_API") or os.getenv("NEBIUS") or os.getenv("NEBIUS_KEY")
    if not nebius_api_key:
        print("Warning: NEBIUS API key not found in environment; PhraseDP candidate generation will fail.")
    nebius_client = OpenAI(base_url="https://api.studio.nebius.ai/v1/", api_key=nebius_api_key) if nebius_api_key else None
    # Use the project-standard model default
    nebius_model_name = os.getenv("NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # Cache for SANTEXT+: load model and build vocabulary once
    santext_global = None
    santext_vocab_built = False

    # State for CluSanT
    clus_saved_cwd = None

    for mechanism_name in ['PhraseDP', 'InferDPT', 'SANTEXT+','CusText+','CluSanT']:
        print(f"\n=== Mechanism: {mechanism_name} ===")
        results[mechanism_name] = {}
        
        for epsilon in epsilon_values:
            # Skip if this mechanism/epsilon combination already completed
            if mechanism_name in results and epsilon in results[mechanism_name]:
                print(f"\n-- Epsilon: {epsilon} (already completed, skipping)")
                continue

            print(f"\n-- Epsilon: {epsilon}")
            start_eps = time.time()

            # Initialize mechanism
            try:
                if mechanism_name == 'SANTEXT+':
                    if santext_global is None:
                        print("  [Init] Creating SANTEXT+ mechanism (global)...")
                        init_t0 = time.time()
                        santext_global = create_santext_mechanism(epsilon=epsilon, p=0.1)
                        print(f"  [Init] SANTEXT+ ready in {time.time()-init_t0:.2f}s")
                    # Load global vocabulary once
                    if not santext_vocab_built:
                        print("  [Init] Loading SANTEXT+ global vocabulary...")
                        vocab_t0 = time.time()
                        ok = santext_global.load_vocabulary_from_files(
                            vocab_path='/home/yizhang/tech4HSE/global_vocab/vocab.json',
                            embeddings_path='/home/yizhang/tech4HSE/global_vocab/embeddings.npy')
                        print(f"  [Init] Global vocab load {'ok' if ok else 'failed'} (t={time.time()-vocab_t0:.2f}s)")
                        if not ok:
                            # Try GloVe path first, fallback to BERT
                            print("  [Init] Trying SANTEXT+ global GloVe embeddings...")
                            glove_ok = santext_global.load_global_glove_embeddings('/home/yizhang/tech4HSE/SanText/data/glove.840B.300d.txt')
                            if not glove_ok:
                                print("  [Init] Trying SANTEXT+ global BERT embeddings...")
                                bert_ok = santext_global.load_global_bert_embeddings('bert-base-uncased')
                                santext_vocab_built = bert_ok
                            else:
                                santext_vocab_built = True
                        else:
                            santext_vocab_built = True
                elif mechanism_name == 'CluSanT':
                    # Get CluSanT embeddings from cache
                    clus_embeddings, clusant_root = embedding_cache.get_clusant_embeddings()
                    if clus_embeddings is not None and clus_saved_cwd is None:
                        # Switch to CluSanT root so its relative paths resolve
                        clus_saved_cwd = os.getcwd()
                        os.chdir(clusant_root)
                        print("  [Init] CluSanT ready.")
            except Exception as e:
                print(f"    Error initializing {mechanism_name}: {e}")
                continue
            
            protection_rates = {
                'emails': [],
                'phones': [],
                'addresses': [],
                'names': [],
                'overall': []
            }
            # Collect example texts per row
            samples: List[Dict[str, str]] = []
            
            for idx, row in sample_df.iterrows():
                row_t0 = time.time()
                print(f"  [Row {idx}] Starting...")
                try:
                    original_text = row['document']
                    labels = ast.literal_eval(row['labels'])
                    # Use dataset tokens + trailing whitespace
                    tokens = ast.literal_eval(row['tokens']) if isinstance(row['tokens'], str) else row['tokens']
                    trailing_ws = ast.literal_eval(row['trailing_whitespace']) if isinstance(row['trailing_whitespace'], str) else row['trailing_whitespace']
                    if isinstance(tokens, list) and isinstance(trailing_ws, list) and len(tokens) == len(trailing_ws) and len(tokens) == len(labels):
                        highlighted_orig = render_highlighted_from_labels(tokens, trailing_ws, labels)
                        reconstructed = reconstruct_text_from_tokens(tokens, trailing_ws)
                        # Extract only PII-containing sentences
                        pii_preview = extract_pii_from_labels(tokens, trailing_ws, labels)
                        original_text = extract_pii_sentences(reconstructed, pii_preview)
                    else:
                        highlighted_orig = original_text
                    print(f"    [Row {idx}] Original (PII highlighted):\n        {highlighted_orig}")
                    
                    # Extract original PII
                    pii_t0 = time.time()
                    original_pii = extract_pii_from_labels(tokens, trailing_ws, labels)
                    print(f"    [Row {idx}] Original PII counts: emails={len(original_pii['emails'])}, phones={len(original_pii['phones'])}, addresses={len(original_pii['addresses'])}, names={len(original_pii['names'])} (t={time.time()-pii_t0:.2f}s)")
                    
                    # Apply privacy mechanism
                    if mechanism_name == 'PhraseDP':
                        # Use old efficient PhraseDP (1 API call instead of 10)
                        sbert_model = embedding_cache.get_sbert_model()
                        if nebius_client is not None:
                            phrase_t0 = time.time()
                            print(f"    [Row {idx}] Running old efficient PhraseDP (1 API call)...")
                            sanitized_text = phrase_DP_perturbation_old(
                                nebius_client=nebius_client,
                                nebius_model_name=nebius_model_name,
                                input_sentence=original_text,
                                epsilon=epsilon,
                                sbert_model=sbert_model
                            )
                            print(f"    [Row {idx}] PhraseDP completed (t={time.time()-phrase_t0:.2f}s)")
                        else:
                            print(f"    [Row {idx}] No Nebius client; keeping original text.")
                            sanitized_text = original_text
                    
                    elif mechanism_name == 'InferDPT':
                        # Use real InferDPT perturbation with cached embeddings
                        print(f"    [Row {idx}] Running InferDPT perturbation...")
                        try:
                            from inferdpt import perturb_sentence
                            token_to_vector_dict, sorted_distance_data, delta_f_new = embedding_cache.get_inferdpt_embeddings()
                            if token_to_vector_dict is not None:
                                sanitized_text = perturb_sentence(
                                    original_text,
                                    epsilon,
                                    token_to_vector_dict=token_to_vector_dict,
                                    sorted_distance_data=sorted_distance_data,
                                    delta_f_new=delta_f_new
                                )
                            else:
                                sanitized_text = original_text
                        except Exception as e:
                            print(f"    [Row {idx}] InferDPT error: {e}")
                            sanitized_text = original_text
                    
                    elif mechanism_name == 'SANTEXT+':
                        print(f"    [Row {idx}] SANTEXT+ sanitizing...")
                        st_t0 = time.time()
                        # Reuse global mechanism; if epsilon were to affect probs, we could update internal parameter here if supported
                        sanitized_text = santext_global.sanitize_text(original_text)
                        print(f"    [Row {idx}] SANTEXT+ done (t={time.time()-st_t0:.2f}s)")
                    elif mechanism_name == 'CusText+':
                        ct_emb_matrix, ct_idx2word, ct_word2idx, ct_stop_set = embedding_cache.get_custext_embeddings()
                        if ct_emb_matrix is None:
                            sanitized_text = original_text
                        else:
                            from cus_text_ppi_protection_experiment import sanitize_with_custext
                            # Use stopword preservation
                            sanitized_text = sanitize_with_custext(
                                original_text,
                                epsilon=epsilon,
                                top_k=20,
                                save_stop_words=True,
                                emb_matrix=ct_emb_matrix,
                                idx2word=ct_idx2word,
                                word2idx=ct_word2idx,
                                stop_set=ct_stop_set
                            )
                    elif mechanism_name == 'CluSanT':
                        # Build a CluSanT instance for this epsilon
                        clus_embeddings, _ = embedding_cache.get_clusant_embeddings()
                        if clus_embeddings is None:
                            sanitized_text = original_text
                        else:
                            try:
                                from clusant import CluSanT  # type: ignore
                                clus = CluSanT(
                                    embedding_file='all-MiniLM-L6-v2',
                                    embeddings=clus_embeddings,
                                    epsilon=epsilon,
                                    num_clusters=336,
                                    mechanism='clusant',
                                    metric_to_create_cluster='euclidean',
                                    distance_metric_for_cluster='euclidean',
                                    distance_metric_for_words='euclidean',
                                    dp_type='metric',
                                    K=16,
                                )
                                sanitized_text = original_text
                                # Find targets present in text (multi-word first)
                                targets_present = []
                                for w in clus_embeddings.keys():
                                    if ' ' in w and re.search(rf"\\b{re.escape(w)}\\b", sanitized_text, flags=re.IGNORECASE):
                                        targets_present.append(w)
                                for w in clus_embeddings.keys():
                                    if ' ' not in w and re.search(rf"\\b{re.escape(w)}\\b", sanitized_text, flags=re.IGNORECASE):
                                        targets_present.append(w)
                                targets_present = sorted(set(targets_present), key=lambda x: (-len(x), x))
                                for t in targets_present:
                                    new = clus.replace_word(t)
                                    if not new:
                                        continue
                                    pattern = re.compile(rf"\\b{re.escape(t)}\\b", flags=re.IGNORECASE)
                                    if pattern.search(sanitized_text):
                                        sanitized_text = pattern.sub(new, sanitized_text)
                            except Exception as e:
                                print(f"    [Row {idx}] CluSanT error: {e}")
                                sanitized_text = original_text
                    
                    # Print raw perturbed text for this mechanism/row in blue
                    try:
                        BLUE = "\033[94m"; RESET = "\033[0m"
                        print(f"[{mechanism_name}][eps={epsilon}][row={int(idx)}] {BLUE}{sanitized_text}{RESET}")
                    except Exception:
                        pass

                    # Detect PII in sanitized text (for reference only)
                    det_t0 = time.time()
                    sanitized_pii = detect_pii_patterns(sanitized_text)
                    print(f"    [Row {idx}] Sanitized PII (regex) counts: emails={len(sanitized_pii['emails'])}, phones={len(sanitized_pii['phones'])}, addresses={len(sanitized_pii['addresses'])}, names={len(sanitized_pii['names'])} (t={time.time()-det_t0:.2f}s)")
                    
                    # Breach rule: check original PII substrings presence in sanitized text
                    # Normalize: case-insensitive, strip punctuation; keep '@' for emails
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

                    email_protected = 0 if leaked_email(original_pii['emails']) else 1
                    phone_protected = 0 if leaked_generic(original_pii['phones']) else 1
                    addr_protected  = 0 if leaked_generic(original_pii['addresses']) else 1
                    name_protected  = 0 if leaked_generic(original_pii['names']) else 1

                    # Overall: average across PII types that are present in the original text
                    present_and_scores = []
                    if len(original_pii['emails']) > 0:
                        present_and_scores.append(email_protected)
                    if len(original_pii['phones']) > 0:
                        present_and_scores.append(phone_protected)
                    if len(original_pii['addresses']) > 0:
                        present_and_scores.append(addr_protected)
                    if len(original_pii['names']) > 0:
                        present_and_scores.append(name_protected)
                    if present_and_scores:
                        overall_protected = float(np.mean(present_and_scores))
                    else:
                        overall_protected = 1.0

                    rates = {
                        'emails': email_protected,
                        'phones': phone_protected,
                        'addresses': addr_protected,
                        'names': name_protected
                    }
                    print(f"    [Row {idx}] Protection (binary): {rates}, overall={overall_protected} (row t={time.time()-row_t0:.2f}s)")
                    # Row-completion progress line
                    try:
                        print(f"  [{mechanism_name}][eps={epsilon}] Completed row {int(idx)+1}/{total_rows} (total elapsed for this eps {time.time()-start_eps:.1f}s)")
                    except Exception:
                        pass
                    
                    for pii_type in rates:
                        protection_rates[pii_type].append(rates[pii_type])
                    
                    # Overall protection per row (binary)
                    protection_rates['overall'].append(overall_protected)
                    # Store sample texts
                    samples.append({
                        'row': int(idx),
                        'original': original_text,
                        'sanitized': sanitized_text
                    })
                    
                except Exception as e:
                    print(f"    Error processing row {idx}: {e}")
                    continue
            
            # Calculate average protection rates
            avg_rates = {}
            for pii_type in protection_rates:
                if protection_rates[pii_type]:
                    avg_rates[pii_type] = np.mean(protection_rates[pii_type])
                else:
                    avg_rates[pii_type] = 0.0
            
            results[mechanism_name][epsilon] = avg_rates
            # Attach samples under this epsilon entry
            results[mechanism_name][epsilon]['samples'] = samples
            print(f"  [Epsilon {epsilon}] Average protection rates: {avg_rates} (t={time.time()-start_eps:.2f}s)")

            # Save results incrementally after each epsilon completion
            save_incremental_results(results, results_file)
    # Restore CWD if CluSanT changed it
    try:
        if clus_saved_cwd is not None:
            os.chdir(clus_saved_cwd)
    except Exception:
        pass
    
    # Final save
    save_incremental_results(results, results_file)
    print(f"\nFinal results saved to: {results_file}")
    
    # Create plots
    plot_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_pii_protection_plots(results, plot_timestamp)

    return results, results_file, f'/home/yizhang/tech4HSE/pii_protection_plots_{plot_timestamp}.png'

def send_experiment_email(subject_suffix: str, body: str, attachments: List[str]):
    """Send email using config at /data1/yizhangh/email_config.json, fallback to project email_config.json."""
    config_paths = ['/data1/yizhangh/email_config.json', '/home/yizhang/tech4HSE/email_config.json']
    config = None
    for p in config_paths:
        try:
            with open(p, 'r') as f:
                config = json.load(f)
                break
        except Exception:
            continue
    if not config:
        print("Email config not found; skipping email.")
        return False
    try:
        msg = MIMEMultipart()
        host = socket.gethostname()
        msg['From'] = config['from_email']
        msg['To'] = config['to_email']
        msg['Subject'] = f"PII Protection Results {subject_suffix} | Host: {host}"
        msg.attach(MIMEText(body, 'plain'))
        for path in attachments:
            if not path or not os.path.exists(path):
                continue
            part = MIMEBase('application', 'octet-stream')
            with open(path, 'rb') as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
            msg.attach(part)
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['from_email'], config['password'])
        server.sendmail(config['from_email'], config['to_email'], msg.as_string())
        server.quit()
        print(f"Email sent to {config['to_email']}")
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def create_pii_protection_plots(results: Dict, timestamp: str):
    """Create visualization plots for PII protection results"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PII Protection Analysis Across Privacy Mechanisms', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
    mechanisms = list(results.keys())
    # Infer epsilon values from any mechanism's keys and sort
    eps_set = set()
    for mech in mechanisms:
        eps_set.update([e for e in results[mech].keys() if isinstance(e, float)])
    epsilon_values = sorted(eps_set)
    pii_types = ['emails', 'phones', 'addresses', 'names']
    
    # Plot 1: Overall Protection Rate
    ax1 = axes[0, 0]
    for mechanism in mechanisms:
        if mechanism in results:
            overall_rates = [results[mechanism].get(eps, {}).get('overall', 0) for eps in epsilon_values]
            ax1.plot(epsilon_values, overall_rates, marker='o', linewidth=2, label=mechanism)
    
    ax1.set_xlabel('Epsilon Value')
    ax1.set_ylabel('Overall PII Protection Rate')
    ax1.set_title('Overall PII Protection Rate vs Epsilon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # Plot 2: PII Type-specific Protection Rates (Epsilon = 2.0)
    ax2 = axes[0, 1]
    eps_2_data = {}
    for mechanism in mechanisms:
        if mechanism in results and 2.0 in results[mechanism]:
            eps_2_data[mechanism] = [results[mechanism][2.0].get(pii_type, 0) for pii_type in pii_types]
    
    x = np.arange(len(pii_types))
    width = 0.25
    
    for i, mechanism in enumerate(mechanisms):
        if mechanism in eps_2_data:
            ax2.bar(x + i*width, eps_2_data[mechanism], width, label=mechanism)
    
    ax2.set_xlabel('PII Type')
    ax2.set_ylabel('Protection Rate')
    ax2.set_title('PII Type-specific Protection (Epsilon = 2.0)')
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(pii_types)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Heatmap of Protection Rates
    ax3 = axes[1, 0]
    heatmap_data = []
    for mechanism in mechanisms:
        if mechanism in results:
            row = [results[mechanism].get(eps, {}).get('overall', 0) for eps in epsilon_values]
            heatmap_data.append(row)
    
    if heatmap_data:
        im = ax3.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax3.set_xticks(range(len(epsilon_values)))
        ax3.set_xticklabels(epsilon_values)
        ax3.set_yticks(range(len(mechanisms)))
        ax3.set_yticklabels(mechanisms)
        ax3.set_xlabel('Epsilon Value')
        ax3.set_ylabel('Privacy Mechanism')
        ax3.set_title('PII Protection Rate Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Protection Rate')
        
        # Add text annotations
        for i in range(len(mechanisms)):
            for j in range(len(epsilon_values)):
                text = ax3.text(j, i, f'{heatmap_data[i][j]:.2f}', 
                               ha="center", va="center", color="black", fontweight='bold')
    
    # Plot 4: Mechanism Comparison (Average across all epsilons)
    ax4 = axes[1, 1]
    avg_protection = {}
    for mechanism in mechanisms:
        if mechanism in results:
            all_rates = []
            for eps in epsilon_values:
                if eps in results[mechanism]:
                    all_rates.append(results[mechanism][eps].get('overall', 0))
            avg_protection[mechanism] = np.mean(all_rates) if all_rates else 0
    
    mechanisms_list = list(avg_protection.keys())
    avg_rates_list = list(avg_protection.values())
    
    bars = ax4.bar(mechanisms_list, avg_rates_list, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_ylabel('Average Protection Rate')
    ax4.set_title('Average PII Protection Across All Epsilon Values')
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, avg_rates_list):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = f'/home/yizhang/tech4HSE/pii_protection_plots_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    
    plt.show()

def print_summary_report(results: Dict):
    """Print a summary report of the results"""
    print("\n" + "="*60)
    print("PII PROTECTION EXPERIMENT SUMMARY")
    print("="*60)
    
    eps_set = set()
    for mech in results:
        eps_set.update([e for e in results[mech].keys() if isinstance(e, float)])
    epsilon_values = sorted(eps_set)
    
    for mechanism in results:
        print(f"\n{mechanism}:")
        print("-" * 40)
        
        for epsilon in epsilon_values:
            if epsilon in results[mechanism]:
                rates = results[mechanism][epsilon]
                print(f"  Epsilon {epsilon}:")
                print(f"    Overall Protection: {rates.get('overall', 0):.3f}")
                print(f"    Email Protection:   {rates.get('emails', 0):.3f}")
                print(f"    Phone Protection:   {rates.get('phones', 0):.3f}")
                print(f"    Address Protection: {rates.get('addresses', 0):.3f}")
                print(f"    Name Protection:    {rates.get('names', 0):.3f}")
    
    # Find best performing mechanism
    best_mechanism = None
    best_rate = 0
    
    for mechanism in results:
        avg_rate = 0
        count = 0
        for epsilon in epsilon_values:
            if epsilon in results[mechanism]:
                avg_rate += results[mechanism][epsilon].get('overall', 0)
                count += 1
        
        if count > 0:
            avg_rate /= count
            if avg_rate > best_rate:
                best_rate = avg_rate
                best_mechanism = mechanism
    
    print(f"\nBEST PERFORMING MECHANISM: {best_mechanism}")
    print(f"Average Protection Rate: {best_rate:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PII Protection Experiment')
    parser.add_argument('--start', type=int, default=0, 
                       help='Start index for dataset rows (default: 0)')
    parser.add_argument('--rows', type=int, default=10,
                       help='Number of rows to process (default: 10)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from existing results file')
    
    args = parser.parse_args()
    
    print("Starting PII Protection Experiment...")
    print(f"This will test PhraseDP, InferDPT, SANTEXT+, and CusText+ on {args.rows} PII samples")
    print(f"Starting from row {args.start}")
    print("Epsilon values: 1.0, 1.5, 2.0, 2.5, 3.0")
    print("="*60)
    
    out = run_pii_protection_experiment(start_idx=args.start, num_rows=args.rows, resume_file=args.resume)
    
    if out:
        results, results_path, plot_path = out
        print_summary_report(results)
        print("\nExperiment completed successfully!")
        # Email
        summary_lines = []
        for mech, eps_map in results.items():
            for eps, vals in eps_map.items():
                if isinstance(eps, str):
                    continue
                summary_lines.append(
                    f"{mech} | eps={eps}: overall={vals.get('overall', 0):.3f}, email={vals.get('emails',0):.3f}, phone={vals.get('phones',0):.3f}, addr={vals.get('addresses',0):.3f}, name={vals.get('names',0):.3f}"
                )
        body = "PII Protection results (10 rows, eps=1,2)\n\n" + "\n".join(summary_lines) + f"\n\nResults: {results_path}\nPlot: {plot_path}"
        send_experiment_email(subject_suffix="(10 rows, eps 1 & 2)", body=body, attachments=[results_path, plot_path])
    else:
        print("Experiment failed. Please check the error messages above.")
