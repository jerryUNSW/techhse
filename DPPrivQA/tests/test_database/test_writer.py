"""
Tests for database writer.
"""

from dpprivqa.database.writer import ExperimentDBWriter
import sqlite3


def test_create_experiment(db_writer):
    """Test creating experiment record."""
    exp_id = db_writer.create_experiment(
        dataset_name="medqa",
        experiment_type="epsilon_independent",
        total_questions=10
    )
    
    assert exp_id is not None
    assert isinstance(exp_id, int)


def test_write_epsilon_independent_result(db_writer, sample_question):
    """Test writing epsilon-independent result."""
    exp_id = db_writer.create_experiment("medqa", "epsilon_independent", 1)
    
    result = {
        "question": sample_question["question"],
        "options": sample_question["options"],
        "answer": "B",
        "ground_truth": "B",
        "is_correct": True,
        "processing_time": 1.5
    }
    
    db_writer.write_epsilon_independent_result("medqa", exp_id, 0, "local", result)
    
    # Verify data was written
    conn = sqlite3.connect(db_writer.db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM medqa_epsilon_independent_results WHERE experiment_id=?",
        (exp_id,)
    )
    assert cursor.fetchone() is not None
    conn.close()


def test_write_epsilon_dependent_result(db_writer, sample_question):
    """Test writing epsilon-dependent result."""
    exp_id = db_writer.create_experiment(
        "medqa",
        "epsilon_dependent",
        1,
        mechanisms=['phrasedp'],
        epsilon_values=[2.0]
    )
    
    result = {
        "question": sample_question["question"],
        "sanitized_question": "sanitized question",
        "options": sample_question["options"],
        "cot_text": "CoT reasoning",
        "answer": "B",
        "ground_truth": "B",
        "is_correct": True,
        "processing_time": 2.0,
        "sanitization_time": 0.5,
        "cot_generation_time": 1.0,
        "local_model": "test-local",
        "remote_model": "test-remote"
    }
    
    db_writer.write_epsilon_dependent_result("medqa", exp_id, 0, "phrasedp", 2.0, result)
    
    # Verify data was written
    conn = sqlite3.connect(db_writer.db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM medqa_epsilon_dependent_results WHERE experiment_id=?",
        (exp_id,)
    )
    assert cursor.fetchone() is not None
    conn.close()


