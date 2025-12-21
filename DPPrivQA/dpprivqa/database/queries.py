"""
Query utilities for retrieving experiment results.
"""

import sqlite3
from typing import List, Tuple, Optional


def get_experiment_summary(conn: sqlite3.Connection, experiment_id: int) -> Optional[Tuple]:
    """
    Get summary information for an experiment.
    
    Args:
        conn: Database connection
        experiment_id: Experiment ID
    
    Returns:
        Tuple with experiment information
    """
    cursor = conn.execute("""
        SELECT id, dataset_name, experiment_type, timestamp, description,
               total_questions, mechanisms, epsilon_values,
               local_model, remote_cot_model, remote_qa_model, created_at
        FROM experiments
        WHERE id = ?
    """, (experiment_id,))
    
    return cursor.fetchone()


def get_accuracy_summary(
    conn: sqlite3.Connection,
    dataset_name: str,
    experiment_id: Optional[int] = None,
    experiment_type: Optional[str] = None
) -> List[Tuple]:
    """
    Get accuracy summary for epsilon-independent or epsilon-dependent results.
    
    Args:
        conn: Database connection
        dataset_name: Name of the dataset
        experiment_id: Optional experiment ID to filter by
        experiment_type: 'epsilon_independent' or 'epsilon_dependent'
    
    Returns:
        List of tuples with (scenario/mechanism, epsilon, total, correct, accuracy)
    """
    if experiment_type == 'epsilon_independent':
        table_name = f"{dataset_name}_epsilon_independent_results"
        query = f"""
            SELECT scenario, 
                   COUNT(*) as total,
                   SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
                   ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END), 3) as accuracy
            FROM {table_name}
            WHERE experiment_id = COALESCE(?, experiment_id)
            GROUP BY scenario
            ORDER BY scenario
        """
    elif experiment_type == 'epsilon_dependent':
        table_name = f"{dataset_name}_epsilon_dependent_results"
        query = f"""
            SELECT mechanism, epsilon,
                   COUNT(*) as total,
                   SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct,
                   ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END), 3) as accuracy
            FROM {table_name}
            WHERE experiment_id = COALESCE(?, experiment_id)
            GROUP BY mechanism, epsilon
            ORDER BY mechanism, epsilon
        """
    else:
        raise ValueError("experiment_type must be 'epsilon_independent' or 'epsilon_dependent'")
    
    cursor = conn.execute(query, (experiment_id,))
    return cursor.fetchall()


