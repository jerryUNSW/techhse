#!/usr/bin/env python3
"""
PII Protection Experiment - Remaining Mechanisms
Tests InferDPT and SANTEXT+ on the same 182 rows that were processed with PhraseDP
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
from utils import generate_sentence_replacements_with_nebius_diverse
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

def run_pii_protection_experiment_remaining(start_idx: int = 0, num_rows: int = 182):
    """Run the PII protection experiment for remaining mechanisms (InferDPT and SANTEXT+)"""
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
    
    # Initialize results storage - only for remaining mechanisms
    results = {
        'InferDPT': {},
        'SANTEXT+': {}
    }
    
    # Run epsilon 1.0 and 2.0
    epsilon_values = [1.0, 2.0]
    
    # Initialize models
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Starting PII protection experiment for remaining mechanisms...")

    # Cache for SANTEXT+: load model and build vocabulary once
    santext_global = None
    santext_vocab_built = False

    # Only test InferDPT and SANTEXT+
    for mechanism_name in ['InferDPT', 'SANTEXT+']:
        print(f"\n=== Mechanism: {mechanism_name} ===")
        results[mechanism_name] = {}
        
        for epsilon in epsilon_values:
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
                    if mechanism_name == 'InferDPT':
                        # Use real InferDPT perturbation with global embeddings
                        print(f"    [Row {idx}] Running InferDPT perturbation...")
                        try:
                            from inferdpt import perturb_sentence
                            sanitized_text = perturb_sentence(original_text, epsilon)
                        except Exception as e:
                            print(f"    [Row {idx}] InferDPT error: {e}")
                            sanitized_text = original_text
                    
                    elif mechanism_name == 'SANTEXT+':
                        print(f"    [Row {idx}] SANTEXT+ sanitizing...")
                        st_t0 = time.time()
                        # Reuse global mechanism; if epsilon were to affect probs, we could update internal parameter here if supported
                        sanitized_text = santext_global.sanitize_text(original_text)
                        print(f"    [Row {idx}] SANTEXT+ done (t={time.time()-st_t0:.2f}s)")
                    
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
    
    # Load existing PhraseDP results from the previous experiment
    print("\nLoading existing PhraseDP results...")
    try:
        # Read the existing results from the log file
        phrase_dp_results = load_phrase_dp_results_from_log('/home/yizhang/tech4HSE/ppi-protection-182.txt')
        results['PhraseDP'] = phrase_dp_results
        print("Successfully loaded PhraseDP results from log file")
    except Exception as e:
        print(f"Warning: Could not load PhraseDP results: {e}")
        print("Proceeding with only InferDPT and SANTEXT+ results")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'/home/yizhang/tech4HSE/pii_protection_results_complete_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create plots
    create_pii_protection_plots(results, timestamp)
    
    return results, results_file, f'/home/yizhang/tech4HSE/pii_protection_plots_complete_{timestamp}.png'

def load_phrase_dp_results_from_log(log_file: str) -> Dict:
    """Load PhraseDP results from the log file"""
    results = {1.0: {'emails': [], 'phones': [], 'addresses': [], 'names': [], 'overall': []}}
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            if 'Protection (binary):' in line and 'overall=' in line:
                # Extract protection data
                # Format: [Row X] Protection (binary): {'emails': 1, 'phones': 1, 'addresses': 1, 'names': 1}, overall=1.0
                try:
                    # Extract the dictionary part
                    start = line.find('{')
                    end = line.find('}', start) + 1
                    if start != -1 and end != 0:
                        dict_str = line[start:end]
                        protection_dict = ast.literal_eval(dict_str)
                        
                        # Extract overall value
                        overall_start = line.find('overall=') + 8
                        overall_end = line.find(' ', overall_start)
                        if overall_end == -1:
                            overall_end = line.find('(', overall_start)
                        overall = float(line[overall_start:overall_end])
                        
                        # Add to results
                        results[1.0]['emails'].append(protection_dict['emails'])
                        results[1.0]['phones'].append(protection_dict['phones'])
                        results[1.0]['addresses'].append(protection_dict['addresses'])
                        results[1.0]['names'].append(protection_dict['names'])
                        results[1.0]['overall'].append(overall)
                except Exception as e:
                    continue
        
        # Calculate averages
        for pii_type in ['emails', 'phones', 'addresses', 'names', 'overall']:
            if results[1.0][pii_type]:
                results[1.0][pii_type] = np.mean(results[1.0][pii_type])
            else:
                results[1.0][pii_type] = 0.0
        
        print(f"Loaded {len([x for x in results[1.0]['overall'] if isinstance(x, (int, float))])} PhraseDP results")
        
    except Exception as e:
        print(f"Error loading PhraseDP results: {e}")
        return {1.0: {'emails': 0.0, 'phones': 0.0, 'addresses': 0.0, 'names': 0.0, 'overall': 0.0}}
    
    return results

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
    epsilon_values = [1.0, 2.0]
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
    plot_file = f'/home/yizhang/tech4HSE/pii_protection_plots_complete_{timestamp}.png'
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Plots saved to: {plot_file}")
    
    plt.show()

def print_summary_report(results: Dict):
    """Print a summary report of the results"""
    print("\n" + "="*60)
    print("PII PROTECTION EXPERIMENT SUMMARY")
    print("="*60)
    
    epsilon_values = [1.0, 2.0]
    
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
    parser = argparse.ArgumentParser(description='PII Protection Experiment - Remaining Mechanisms')
    parser.add_argument('--start', type=int, default=0, 
                       help='Start index for dataset rows (default: 0)')
    parser.add_argument('--rows', type=int, default=182, 
                       help='Number of rows to process (default: 182)')
    
    args = parser.parse_args()
    
    print("Starting PII Protection Experiment for Remaining Mechanisms...")
    print(f"This will test InferDPT and SANTEXT+ on {args.rows} PII samples")
    print(f"Starting from row {args.start}")
    print("Epsilon values: 1.0, 2.0")
    print("PhraseDP results will be loaded from existing log file")
    print("="*60)
    
    out = run_pii_protection_experiment_remaining(start_idx=args.start, num_rows=args.rows)
    
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
        body = f"PII Protection results ({args.rows} rows, eps=1,2) - Complete comparison\n\n" + "\n".join(summary_lines) + f"\n\nResults: {results_path}\nPlot: {plot_path}"
        send_experiment_email(subject_suffix=f"({args.rows} rows, eps 1 & 2, complete)", body=body, attachments=[results_path, plot_path])
    else:
        print("Experiment failed. Please check the error messages above.")
