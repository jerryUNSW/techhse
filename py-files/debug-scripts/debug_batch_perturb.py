#!/usr/bin/env python3
"""
Debug script to test batch_perturb_options_with_phrasedp function directly
"""

import os
import sys
import yaml
from dotenv import load_dotenv

# Import necessary components
import utils
from sanitization_methods import phrasedp_sanitize_text

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load environment variables from .env
load_dotenv()

# ANSI color codes for better console output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"

def batch_perturb_options_with_phrasedp(options, epsilon, nebius_client, nebius_model_name):
    print(f"{CYAN}Batch Perturbation Starting:{RESET}")
    print(f"Options Count: {len(options)}")
    print(f"Epsilon: {epsilon}")
    """Batch perturb all options together using PhraseDP for efficiency."""
    # Combine all options into a single text for batch processing
    print(f"{CYAN}Original Options:{RESET}")
    for key, value in options.items():
        print(f"  {key}) {value}")
    combined_text = ""
    for key, value in options.items():
        combined_text += f"Option {key}: {value}\n"
    print(f"{CYAN}Combined Text:{RESET}\n{combined_text}")
    # Apply PhraseDP to the combined text
    print(f"{YELLOW}Applying PhraseDP to combined text...{RESET}")
    print(f"{YELLOW}Combined Text:{RESET}\n{combined_text}")
    print(f"{YELLOW}Applying PhraseDP...{RESET}")
    perturbed_combined = phrasedp_sanitize_text(
        combined_text.strip(),
        epsilon=epsilon,
        nebius_client=nebius_client,
        nebius_model_name=nebius_model_name,
    )
    print(f"{CYAN}Perturbed Combined Text:{RESET}\n{perturbed_combined}")
    # Parse back to individual options
    perturbed_options = {}
    lines = perturbed_combined.split('\n')
    print(f"{YELLOW}Parsing Perturbed Options:{RESET}")
    for line in lines:
        line = line.strip()
        print(f"  Processing line: '{line}'")
        if line.startswith('Option ') and ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                option_key = parts[0].replace('Option ', '').strip()
                option_value = parts[1].strip()
                print(f"    Extracted: key='{option_key}', value='{option_value}'")
                if option_key in ['A', 'B', 'C', 'D']:
                    perturbed_options[option_key] = option_value
                    print(f"    {GREEN}Added to perturbed options{RESET}")
    # Fallback: if parsing fails, use original keys with split content
    if len(perturbed_options) != len(options):
        print(f"{YELLOW}Warning: Batch parsing partially failed. Parsed {len(perturbed_options)} out of {len(options)} options. Using fallback approach.{RESET}")
        option_keys = list(options.keys())
        lines = [l.strip() for l in perturbed_combined.split('\n') if l.strip()]
        for i, key in enumerate(option_keys):
            print(f"  Fallback processing for key: {key}")
            if i < len(lines):
                # Remove "Option X:" prefix if it exists
                line = lines[i]
                print(f"    Fallback line: '{line}'")
                if line.startswith(f'Option {key}:'):
                    line = line[len(f'Option {key}:'):].strip()
                perturbed_options[key] = line
                print(f"    {GREEN}Fallback: set {key} to '{line}'{RESET}")
            else:
                perturbed_options[key] = f"Perturbed option {key}"
                print(f"    {YELLOW}Fallback: no line for {key}, using default{RESET}")
    print(f"{CYAN}Final Perturbed Options:{RESET}")
    for key, value in perturbed_options.items():
        print(f"  {key}) {value}")
    return perturbed_options

def main():
    print(f"{BLUE}--- Testing Batch Perturbation Debug ---{RESET}")

    # Sample options to test
    test_options = {
        'A': 'Streptococcus viridans',
        'B': 'Enterococcus faecalis',
        'C': 'Staphylococcus epidermidis',
        'D': 'Bacillus cereus'
    }

    # Initialize Nebius client
    try:
        nebius_client = utils.get_nebius_client()
        nebius_model_name = config.get('local_model', 'microsoft/phi-4')
        epsilon = config.get('epsilon', 1.0)

        print(f"Using Nebius model: {nebius_model_name}")
        print(f"Using epsilon: {epsilon}")

        # Test the batch perturbation function
        result = batch_perturb_options_with_phrasedp(
            test_options,
            epsilon,
            nebius_client,
            nebius_model_name
        )

        print(f"\n{GREEN}Test completed successfully!{RESET}")

    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()