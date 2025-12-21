"""
Database schema definitions for experiment results storage.
"""

import sqlite3
from typing import List


# List of all datasets
DATASETS = [
    'medqa',
    'medmcqa',
    'hse_bench',
    'mmlu_professional_law',
    'mmlu_professional_medicine',
    'mmlu_clinical_knowledge',
    'mmlu_college_medicine',
]


def create_experiments_table(conn: sqlite3.Connection):
    """Create the experiments metadata table."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset_name TEXT NOT NULL,
            experiment_type TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            description TEXT,
            total_questions INTEGER,
            mechanisms TEXT,
            epsilon_values TEXT,
            local_model TEXT,
            remote_cot_model TEXT,
            remote_qa_model TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(dataset_name, experiment_type, timestamp)
        )
    """)


def create_epsilon_independent_table(conn: sqlite3.Connection, dataset_name: str):
    """
    Create epsilon-independent results table for a dataset.
    
    Args:
        conn: Database connection
        dataset_name: Name of the dataset
    """
    table_name = f"{dataset_name}_epsilon_independent_results"
    
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            question_idx INTEGER NOT NULL,
            scenario TEXT NOT NULL,
            original_question TEXT NOT NULL,
            options TEXT NOT NULL,
            cot_text TEXT,
            generated_answer TEXT NOT NULL,
            ground_truth_answer TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL,
            processing_time REAL,
            local_model TEXT,
            remote_model TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id),
            UNIQUE(experiment_id, question_idx, scenario)
        )
    """)
    
    # Create indexes
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{dataset_name}_ei_experiment_scenario 
        ON {table_name}(experiment_id, scenario)
    """)
    
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{dataset_name}_ei_question 
        ON {table_name}(question_idx)
    """)


def create_epsilon_dependent_table(conn: sqlite3.Connection, dataset_name: str):
    """
    Create epsilon-dependent results table for a dataset.
    
    Args:
        conn: Database connection
        dataset_name: Name of the dataset
    """
    table_name = f"{dataset_name}_epsilon_dependent_results"
    
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            question_idx INTEGER NOT NULL,
            mechanism TEXT NOT NULL,
            epsilon REAL NOT NULL,
            original_question TEXT NOT NULL,
            sanitized_question TEXT NOT NULL,
            options TEXT NOT NULL,
            induced_cot TEXT NOT NULL,
            generated_answer TEXT NOT NULL,
            ground_truth_answer TEXT NOT NULL,
            is_correct BOOLEAN NOT NULL,
            processing_time REAL,
            sanitization_time REAL,
            cot_generation_time REAL,
            local_model TEXT NOT NULL,
            remote_model TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (experiment_id) REFERENCES experiments (id),
            UNIQUE(experiment_id, question_idx, mechanism, epsilon)
        )
    """)
    
    # Create indexes
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{dataset_name}_ed_experiment_mechanism_epsilon 
        ON {table_name}(experiment_id, mechanism, epsilon)
    """)
    
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{dataset_name}_ed_question 
        ON {table_name}(question_idx)
    """)
    
    conn.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{dataset_name}_ed_epsilon 
        ON {table_name}(epsilon)
    """)


def create_all_tables(conn: sqlite3.Connection):
    """
    Create all database tables.
    
    Args:
        conn: Database connection
    """
    # Create experiments table
    create_experiments_table(conn)
    
    # Create tables for each dataset
    for dataset_name in DATASETS:
        create_epsilon_independent_table(conn, dataset_name)
        create_epsilon_dependent_table(conn, dataset_name)
    
    conn.commit()


