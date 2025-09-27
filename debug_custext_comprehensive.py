#!/usr/bin/env python3
"""
Debug script to find the bug in CusText+ protection calculation in comprehensive script
"""

import pandas as pd
import ast
import re
import numpy as np

def _normalize_for_leak(text: str, keep_at: bool = False) -> str:
    t = text.lower()
    if keep_at:
        t = re.sub(r'[!\"#\$%&\'\(\)\*\+,\-\./:;<=>\?\[\\\\\]\^_`\{\|\}~]', '', t)
    else:
        t = re.sub(r'[!\"#\$%&\'\(\)\*\+,\-\./:;<=>\?@\[\\\\\]\^_`\{\|\}~]', '', t)
    t = re.sub(r'\s+', '', t)
    return t

def extract_pii_from_labels(tokens, trailing_ws, labels):
    def span_text(si, ei):
        parts = []
        for i in range(si, ei + 1):
            if i < len(tokens):
                parts.append(tokens[i])
                if i < len(trailing_ws) and trailing_ws[i]:
                    parts.append(' ')
        return ''.join(parts)

    spans = []
    current_type = None
    start_idx = None

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

def reconstruct_text_from_tokens(tokens, trailing_whitespace):
    parts = []
    for i, tok in enumerate(tokens):
        parts.append(tok)
        if i < len(trailing_whitespace) and trailing_whitespace[i]:
            parts.append(' ')
    return ''.join(parts)

def extract_pii_sentences(full_text: str, pii_spans):
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

def test_custext_logic():
    """Test the exact logic used in comprehensive script"""
    print('=== Testing CusText+ Logic from Comprehensive Script ===')
    
    # Load data
    df = pd.read_csv('/home/yizhang/tech4HSE/pii_external_dataset.csv')
    
    # Test with first 2 rows (same as comprehensive script)
    for idx in range(2):
        row = df.iloc[idx]
        print(f'\n--- Row {idx} ---')
        
        # Replicate the exact logic from comprehensive script
        original_text = row['document']
        labels = ast.literal_eval(row['labels'])
        tokens = ast.literal_eval(row['tokens']) if isinstance(row['tokens'], str) else row['tokens']
        trailing_ws = ast.literal_eval(row['trailing_whitespace']) if isinstance(row['trailing_whitespace'], str) else row['trailing_whitespace']
        
        if isinstance(tokens, list) and isinstance(trailing_ws, list) and len(tokens) == len(trailing_ws) and len(tokens) == len(labels):
            reconstructed = reconstruct_text_from_tokens(tokens, trailing_ws)
            # Extract only PII-containing sentences
            pii_preview = extract_pii_from_labels(tokens, trailing_ws, labels)
            original_text = extract_pii_sentences(reconstructed, pii_preview)
        
        print(f'Original text: {original_text[:100]}...')
        
        # Extract original PII (same as comprehensive script)
        original_pii = extract_pii_from_labels(tokens, trailing_ws, labels)
        print(f'Original PII extracted: {original_pii}')
        
        # Simulate CusText+ sanitization (using actual results from comprehensive script)
        if idx == 0:
            if 'My name is Aaliyah Popova' in original_text:
                sanitized_text = 'My appointment is Britney Popova, and I am a watchmaker with 13 age of experience.'
            else:
                sanitized_text = 'My baptism is Maximilian Becker, and I am a constructor with three ageing of experience.'
        else:
            if 'My name is Konstantin Becker' in original_text:
                sanitized_text = 'My baptism is Maximilian Becker, and I am a constructor with three ageing of experience.'
            else:
                sanitized_text = 'My appointment is Britney Popova, and I am a watchmaker with 13 age of experience.'
        
        print(f'Sanitized text: {sanitized_text[:100]}...')
        
        # Apply protection detection logic (exact same as comprehensive script)
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
        addr_protected = 0 if leaked_generic(original_pii['addresses']) else 1
        name_protected = 0 if leaked_generic(original_pii['names']) else 1
        
        rates = {
            'emails': email_protected,
            'phones': phone_protected,
            'addresses': addr_protected,
            'names': name_protected
        }
        
        print(f'Protection rates: {rates}')
        
        # Overall calculation (same as comprehensive script)
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
        
        print(f'Overall protection: {overall_protected}')
        
        # DEBUG: Check name protection in detail
        if original_pii['names']:
            print(f'DEBUG - Name protection details:')
            for name in original_pii['names']:
                name_norm = _normalize_for_leak(name, keep_at=False)
                sani_norm = _normalize_for_leak(sanitized_text, keep_at=False)
                leaked = name_norm in sani_norm
                print(f'  Name: "{name}"')
                print(f'  Normalized: "{name_norm}"')
                print(f'  Sanitized normalized: "{sani_norm[:50]}..."')
                print(f'  Leaked: {leaked}')
                print(f'  Protection: {0 if leaked else 1}')

if __name__ == '__main__':
    test_custext_logic()
