"""
Query utilities for retrieving experiment results.
"""

import sqlite3
from typing import List, Tuple, Optional, Set, Dict


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


def get_missing_epsilon_independent_results(
    conn: sqlite3.Connection,
    dataset_name: str,
    experiment_id: int,
    expected_question_indices: Set[int],
    expected_scenarios: List[str] = ['local', 'local_cot', 'remote']
) -> List[Tuple[int, str]]:
    """
    Find missing epsilon-independent results.
    
    Args:
        conn: Database connection
        dataset_name: Name of the dataset
        experiment_id: Experiment ID
        expected_question_indices: Set of expected question indices
        expected_scenarios: List of expected scenarios
    
    Returns:
        List of (question_idx, scenario) tuples that are missing
    """
    table_name = f"{dataset_name}_epsilon_independent_results"
    
    # Get existing results
    cursor = conn.execute(f"""
        SELECT question_idx, scenario
        FROM {table_name}
        WHERE experiment_id = ?
    """, (experiment_id,))
    
    existing = set(cursor.fetchall())
    
    # Find missing combinations
    missing = []
    for q_idx in expected_question_indices:
        for scenario in expected_scenarios:
            if (q_idx, scenario) not in existing:
                missing.append((q_idx, scenario))
    
    return missing


def get_missing_epsilon_dependent_results(
    conn: sqlite3.Connection,
    dataset_name: str,
    experiment_id: int,
    expected_question_indices: Set[int],
    expected_mechanisms: List[str] = ['phrasedp', 'phrasedp_plus'],
    expected_epsilons: List[float] = [1.0, 2.0, 3.0]
) -> List[Tuple[int, str, float]]:
    """
    Find missing epsilon-dependent results.
    
    Args:
        conn: Database connection
        dataset_name: Name of the dataset
        experiment_id: Experiment ID
        expected_question_indices: Set of expected question indices
        expected_mechanisms: List of expected mechanisms
        expected_epsilons: List of expected epsilon values
    
    Returns:
        List of (question_idx, mechanism, epsilon) tuples that are missing
    """
    table_name = f"{dataset_name}_epsilon_dependent_results"
    
    # Get existing results
    cursor = conn.execute(f"""
        SELECT question_idx, mechanism, epsilon
        FROM {table_name}
        WHERE experiment_id = ?
    """, (experiment_id,))
    
    existing = set(cursor.fetchall())
    
    # Find missing combinations
    missing = []
    for q_idx in expected_question_indices:
        for mechanism in expected_mechanisms:
            for epsilon in expected_epsilons:
                if (q_idx, mechanism, epsilon) not in existing:
                    missing.append((q_idx, mechanism, epsilon))
    
    return missing


def get_missing_results_summary(
    conn: sqlite3.Connection,
    dataset_name: str,
    experiment_id_ei: Optional[int] = None,
    experiment_id_ed: Optional[int] = None,
    total_questions: int = 500,
    expected_scenarios: List[str] = ['local', 'local_cot', 'remote'],
    expected_mechanisms: List[str] = ['phrasedp', 'phrasedp_plus'],
    expected_epsilons: List[float] = [1.0, 2.0, 3.0]
) -> Dict[str, any]:
    """
    Get summary of missing results for a dataset.
    
    Args:
        conn: Database connection
        dataset_name: Name of the dataset
        experiment_id_ei: Epsilon-independent experiment ID (uses latest if None)
        experiment_id_ed: Epsilon-dependent experiment ID (uses latest if None)
        total_questions: Total number of questions expected
        expected_scenarios: Expected scenarios for epsilon-independent
        expected_mechanisms: Expected mechanisms for epsilon-dependent
        expected_epsilons: Expected epsilon values
    
    Returns:
        Dictionary with missing results summary
    """
    # Get latest experiment IDs if not provided
    if experiment_id_ei is None:
        cursor = conn.execute("""
            SELECT id FROM experiments
            WHERE dataset_name = ? AND experiment_type = 'epsilon_independent'
            ORDER BY id DESC LIMIT 1
        """, (dataset_name,))
        row = cursor.fetchone()
        experiment_id_ei = row[0] if row else None
    
    if experiment_id_ed is None:
        cursor = conn.execute("""
            SELECT id FROM experiments
            WHERE dataset_name = ? AND experiment_type = 'epsilon_dependent'
            ORDER BY id DESC LIMIT 1
        """, (dataset_name,))
        row = cursor.fetchone()
        experiment_id_ed = row[0] if row else None
    
    expected_indices = set(range(total_questions))
    
    missing_ei = []
    missing_ed = []
    
    if experiment_id_ei:
        missing_ei = get_missing_epsilon_independent_results(
            conn, dataset_name, experiment_id_ei, expected_indices, expected_scenarios
        )
    
    if experiment_id_ed:
        missing_ed = get_missing_epsilon_dependent_results(
            conn, dataset_name, experiment_id_ed, expected_indices, expected_mechanisms, expected_epsilons
        )
    
    # Group missing by scenario/mechanism
    missing_by_scenario = {}
    for q_idx, scenario in missing_ei:
        if scenario not in missing_by_scenario:
            missing_by_scenario[scenario] = []
        missing_by_scenario[scenario].append(q_idx)
    
    missing_by_mechanism_epsilon = {}
    for q_idx, mechanism, epsilon in missing_ed:
        key = f"{mechanism}_eps{epsilon:.1f}"
        if key not in missing_by_mechanism_epsilon:
            missing_by_mechanism_epsilon[key] = []
        missing_by_mechanism_epsilon[key].append(q_idx)
    
    return {
        'epsilon_independent': {
            'total_missing': len(missing_ei),
            'missing_by_scenario': {k: sorted(v) for k, v in missing_by_scenario.items()},
            'missing_question_indices': sorted(set(q_idx for q_idx, _ in missing_ei))
        },
        'epsilon_dependent': {
            'total_missing': len(missing_ed),
            'missing_by_mechanism_epsilon': {k: sorted(v) for k, v in missing_by_mechanism_epsilon.items()},
            'missing_question_indices': sorted(set(q_idx for q_idx, _, _ in missing_ed))
        },
        'experiment_id_ei': experiment_id_ei,
        'experiment_id_ed': experiment_id_ed
    }


