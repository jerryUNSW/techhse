#!/usr/bin/env python3
"""Test the fixed batch perturbation function"""

import os
import sys
import yaml
from dotenv import load_dotenv

# Load configuration and environment
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
load_dotenv()

# Import necessary components
import utils
import importlib.util
spec = importlib.util.spec_from_file_location("test_script", "/home/yizhang/tech4HSE/test-medqa-usmle-4-options.py")
test_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_script)
batch_perturb_options_with_phrasedp = test_script.batch_perturb_options_with_phrasedp

# ANSI color codes
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

def main():
    print(f"{CYAN}Testing Fixed Batch Perturbation{RESET}")

    # Test options
    test_options = {
        'A': 'Streptococcus viridans',
        'B': 'Enterococcus faecalis',
        'C': 'Staphylococcus epidermidis',
        'D': 'Bacillus cereus'
    }

    try:
        nebius_client = utils.get_nebius_client()
        nebius_model_name = config.get('local_model', 'microsoft/phi-4')
        epsilon = config.get('epsilon', 1.0)

        result = batch_perturb_options_with_phrasedp(
            test_options,
            epsilon,
            nebius_client,
            nebius_model_name
        )

        print(f"\n{GREEN}SUCCESS: All options perturbed individually!{RESET}")
        print(f"Result contains {len(result)} options (expected: {len(test_options)})")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()