#!/usr/bin/env python3
"""
Setup database script - Initialize SQLite database with all tables.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpprivqa.database.writer import ExperimentDBWriter
from dpprivqa.utils.config import load_config


def main():
    """Create database and all tables."""
    config = load_config()
    db_path = config.get("database", {}).get("path", "exp-results/results.db")
    
    print(f"Creating database at: {db_path}")
    
    # Create database writer (this will create all tables)
    db_writer = ExperimentDBWriter(db_path)
    
    print("Database created successfully!")
    print(f"Location: {db_path}")
    print("\nTables created:")
    print("  - experiments")
    
    datasets = [
        'medqa', 'medmcqa', 'hse_bench',
        'mmlu_professional_law', 'mmlu_professional_medicine',
        'mmlu_clinical_knowledge', 'mmlu_college_medicine'
    ]
    
    for dataset in datasets:
        print(f"  - {dataset}_epsilon_independent_results")
        print(f"  - {dataset}_epsilon_dependent_results")
    
    db_writer.close()
    print("\nSetup complete!")


if __name__ == "__main__":
    main()


