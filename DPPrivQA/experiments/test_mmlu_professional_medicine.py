#!/usr/bin/env python3
"""
MMLU Professional Medicine Experiment Script
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.test_mmlu_professional_law import main as mmlu_main

if __name__ == "__main__":
    # This script uses the same structure as professional_law
    # Just update the dataset name
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-questions", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--epsilon-values", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--use-phrasedp-plus", action="store_true")
    parser.add_argument("--skip-epsilon-independent", action="store_true")
    parser.add_argument("--skip-epsilon-dependent", action="store_true")
    args = parser.parse_args()
    
    # Import and modify the main function
    from dpprivqa.datasets import MMLUDataset
    from dpprivqa.utils.logging import setup_logging
    
    dataset_name = "mmlu_professional_medicine"
    logger = setup_logging(dataset_name)
    logger.info("Starting MMLU Professional Medicine experiment")
    
    dataset = MMLUDataset(subset="professional_medicine")
    # Rest of the logic is the same as professional_law
    mmlu_main()


