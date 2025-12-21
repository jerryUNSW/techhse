"""
Tests for database schema.
"""

import sqlite3
from dpprivqa.database.writer import ExperimentDBWriter


def test_schema_creation(temp_db):
    """Test that all tables are created correctly."""
    writer = ExperimentDBWriter(temp_db)
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    # Check experiments table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'")
    assert cursor.fetchone() is not None
    
    # Check dataset tables exist
    datasets = ['medqa', 'medmcqa', 'hse_bench', 'mmlu_professional_law']
    for dataset in datasets:
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' "
            f"AND name='{dataset}_epsilon_independent_results'"
        )
        assert cursor.fetchone() is not None
        
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' "
            f"AND name='{dataset}_epsilon_dependent_results'"
        )
        assert cursor.fetchone() is not None
    
    conn.close()


def test_experiments_table_structure(temp_db):
    """Test experiments table structure."""
    writer = ExperimentDBWriter(temp_db)
    conn = sqlite3.connect(temp_db)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(experiments)")
    columns = [row[1] for row in cursor.fetchall()]
    
    expected_columns = ['id', 'dataset_name', 'experiment_type', 'timestamp']
    for col in expected_columns:
        assert col in columns
    
    conn.close()


