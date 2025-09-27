#!/usr/bin/env python3
"""
Recompute PPI Protection Rates - Simple Direct Approach

For each data sample (first 100) in pii_external_dataset.csv:
1. Extract labeled PII substrings from BIO labels
2. Check if those exact substrings occur in the perturbed text
3. If present = leak (0), if absent = protected (1)
4. Compute protection rates by epsilon/mechanism/PII type
"""

import json
import pandas as pd
import ast
import numpy as np
from collections import defaultdict

def extract_pii_substrings_from_bio(tokens, trailing_ws, labels):
    """
    Extract PII substrings from BIO labels exactly as they appear in text.
    Returns dict with lists of PII substrings for each type.
    """
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

    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if current_type is not None and start_idx is not None:
                spans.append((start_idx, i - 1, current_type))
            current_type = label[2:]
            start_idx = i
        elif label.startswith('I-'):
            continue
        else:
            if current_type is not None and start_idx is not None:
                spans.append((start_idx, i - 1, current_type))
            current_type = None
            start_idx = None

    if current_type is not None and start_idx is not None:
        spans.append((start_idx, len(labels) - 1, current_type))

    pii_substrings = {'emails': [], 'phones': [], 'addresses': [], 'names': []}

    for si, ei, label_type in spans:
        text_span = span_text(si, ei).strip()  # Remove trailing whitespace

        if 'EMAIL' in label_type:
            pii_substrings['emails'].append(text_span)
        elif 'PHONE' in label_type:
            pii_substrings['phones'].append(text_span)
        elif 'ADDRESS' in label_type:
            pii_substrings['addresses'].append(text_span)
        elif 'NAME' in label_type:
            pii_substrings['names'].append(text_span)

    return pii_substrings

def check_pii_protection(original_pii_substrings, perturbed_text):
    """
    Check if PII substrings are protected (absent) in perturbed text.
    Returns protection status for each PII type (1=protected, 0=leaked).
    """
    protection_status = {}

    for pii_type, substrings in original_pii_substrings.items():
        if not substrings:
            # No PII of this type = 100% protection
            protection_status[pii_type] = 1
        else:
            protected_count = 0
            total_count = len(substrings)

            for pii_substring in substrings:
                # Simple substring check - if PII still appears, it's leaked
                if pii_substring in perturbed_text:
                    # Leaked
                    pass
                else:
                    # Protected
                    protected_count += 1

            # Return 1 if all PII of this type is protected, 0 otherwise
            protection_status[pii_type] = 1 if protected_count == total_count else 0

    return protection_status

def load_data():
    """Load comprehensive results and PII dataset."""
    print("Loading data...")

    # Load comprehensive results file with sample data
    results_file = "comprehensive_ppi_protection_results_original.json"

    with open(results_file, 'r') as f:
        comprehensive_results = json.load(f)

    print(f"✓ Using results file: {results_file}")

    # Load PII dataset
    df = pd.read_csv('/home/yizhang/tech4HSE/pii_external_dataset.csv')

    print(f"✓ Loaded comprehensive results: {len(comprehensive_results)} mechanisms")
    print(f"✓ Loaded PII dataset: {len(df)} rows")

    return comprehensive_results, df

def recompute_protection_rates():
    """Main function to recompute all protection rates."""
    print("=== Recomputing PPI Protection Rates ===")
    print()

    # Load data
    comprehensive_results, df = load_data()

    # Initialize results storage
    new_results = {}
    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    pii_types = ['emails', 'phones', 'addresses', 'names']

    print("Processing mechanisms...")

    for mechanism in mechanisms:
        if mechanism not in comprehensive_results:
            print(f"⚠️  Mechanism {mechanism} not found in comprehensive results")
            continue

        print(f"\n🔍 Processing {mechanism}...")
        new_results[mechanism] = {}

        for eps in epsilon_values:
            eps_str = str(eps)
            if eps_str not in comprehensive_results[mechanism]:
                print(f"  ⚠️  Epsilon {eps} not found for {mechanism}")
                continue

            print(f"  Processing epsilon {eps}...")

            # Get samples for this mechanism and epsilon
            samples = comprehensive_results[mechanism][eps_str].get('samples', [])
            if not samples:
                print(f"    ⚠️  No samples found for {mechanism} epsilon {eps}")
                continue

            # Initialize protection counters
            protection_counts = {pii_type: [] for pii_type in pii_types}

            # Process each sample (should be first 100 from dataset)
            for sample in samples:
                row_idx = sample['row']
                original_text = sample['original']
                perturbed_text = sample['sanitized']

                # Extract BIO data for this row
                try:
                    tokens = ast.literal_eval(df.iloc[row_idx]['tokens'])
                    labels = ast.literal_eval(df.iloc[row_idx]['labels'])
                    trailing_ws = ast.literal_eval(df.iloc[row_idx]['trailing_whitespace'])
                except Exception as e:
                    print(f"    ⚠️  Error processing row {row_idx}: {e}")
                    continue

                # Extract PII substrings from BIO labels
                original_pii = extract_pii_substrings_from_bio(tokens, trailing_ws, labels)

                # Check protection for each PII type
                protection_status = check_pii_protection(original_pii, perturbed_text)

                # Record protection status
                for pii_type in pii_types:
                    protection_counts[pii_type].append(protection_status[pii_type])

            # Compute protection rates for this epsilon
            epsilon_results = {}
            overall_scores = []

            for pii_type in pii_types:
                if protection_counts[pii_type]:
                    protection_rate = np.mean(protection_counts[pii_type])
                    epsilon_results[pii_type] = protection_rate
                    overall_scores.append(protection_rate)
                else:
                    epsilon_results[pii_type] = 0.0

            # Compute overall protection rate
            epsilon_results['overall'] = np.mean(overall_scores) if overall_scores else 0.0

            # Store results
            new_results[mechanism][eps_str] = epsilon_results

            # Print summary
            print(f"    ✓ Processed {len(samples)} samples")
            print(f"      Overall: {epsilon_results['overall']:.3f}")
            print(f"      Names: {epsilon_results['names']:.3f}, Emails: {epsilon_results['emails']:.3f}")
            print(f"      Phones: {epsilon_results['phones']:.3f}, Addresses: {epsilon_results['addresses']:.3f}")

    return new_results

def save_results(results):
    """Save recomputed results to file."""
    output_file = "recomputed_ppi_protection_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_file}")
    return output_file

def print_summary(results):
    """Print summary of results."""
    print("\n" + "="*60)
    print("RECOMPUTED PROTECTION RATES SUMMARY")
    print("="*60)

    mechanisms = ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']
    epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]

    for mechanism in mechanisms:
        if mechanism not in results:
            continue

        print(f"\n{mechanism}:")
        print("  Eps  | Overall | Names  | Emails | Phones | Addrs")
        print("  -----|---------|--------|--------|--------|-------")

        for eps in epsilon_values:
            eps_str = str(eps)
            if eps_str in results[mechanism]:
                data = results[mechanism][eps_str]
                print(f"  {eps:3.1f}  |  {data['overall']:5.3f}  | {data['names']:5.3f}  | {data['emails']:5.3f}  | {data['phones']:5.3f}  | {data['addresses']:5.3f}")

def main():
    """Main execution function."""
    # Recompute protection rates
    results = recompute_protection_rates()

    # Save results
    output_file = save_results(results)

    # Print summary
    print_summary(results)

    print(f"\n🎉 Protection rates recomputed successfully!")
    print(f"📁 Results saved to: {output_file}")
    print("\nUse this file for generating accurate plots.")

if __name__ == "__main__":
    main()