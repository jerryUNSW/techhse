#!/usr/bin/env python3
"""Test just the scenario 3.1.2.new function directly"""

import os
import yaml
import sys
from dotenv import load_dotenv

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
load_dotenv()

# Import all necessary components
import utils
from sanitization_methods import phrasedp_sanitize_text
from sentence_transformers import SentenceTransformer

# Define colors
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RED = "\033[91m"
RESET = "\033[0m"

def batch_perturb_options_with_phrasedp(options, epsilon, nebius_client, nebius_model_name):
    print(f"{CYAN}Batch Perturbation Starting:{RESET}")
    print(f"Options Count: {len(options)}")
    print(f"Epsilon: {epsilon}")
    """Perturb each option individually using PhraseDP while maintaining structure."""
    print(f"{CYAN}Original Options:{RESET}")
    for key, value in options.items():
        print(f"  {key}) {value}")

    # Perturb each option individually to preserve structure
    perturbed_options = {}
    for key, value in options.items():
        print(f"{YELLOW}Applying PhraseDP to option {key}: {value}{RESET}")
        perturbed_value = phrasedp_sanitize_text(
            value,
            epsilon=epsilon,
            nebius_client=nebius_client,
            nebius_model_name=nebius_model_name,
        )
        perturbed_options[key] = perturbed_value
        print(f"  {GREEN}{key}) {perturbed_value}{RESET}")

    print(f"{CYAN}Final Perturbed Options:{RESET}")
    for key, value in perturbed_options.items():
        print(f"  {key}) {value}")

    return perturbed_options

def main():
    print(f"{BLUE}Testing Scenario 3.1.2.new Batch Perturbation{RESET}")

    # Sample data like MedQA would have
    question = "A 65-year-old male is treated for anal carcinoma with therapy including external beam radiation. What organism is most likely responsible if he develops infectious diarrhea?"

    options = {
        'A': 'Streptococcus viridans',
        'B': 'Enterococcus faecalis',
        'C': 'Staphylococcus epidermidis',
        'D': 'Bacillus cereus'
    }

    try:
        # Initialize clients
        nebius_client = utils.get_nebius_client()
        nebius_model_name = config.get('local_model', 'microsoft/phi-4')
        epsilon = config.get('epsilon', 1.0)

        print(f"Question: {question}")
        print(f"Options: {options}")

        # Test the batch perturbation
        perturbed_options = batch_perturb_options_with_phrasedp(
            options,
            epsilon,
            nebius_client,
            nebius_model_name
        )

        print(f"\n{GREEN}SUCCESS!{RESET}")
        print(f"All {len(perturbed_options)} options were perturbed successfully!")

        # Show comparison
        print(f"\n{YELLOW}COMPARISON:{RESET}")
        for key in options.keys():
            print(f"{key}) Original: {options[key]}")
            print(f"{key}) Perturbed: {perturbed_options[key]}")
            print()

    except Exception as e:
        print(f"{RED}Error: {e}{RESET}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()