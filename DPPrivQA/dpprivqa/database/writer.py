"""
Database writer for storing experiment results.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from dpprivqa.database.schema import create_all_tables, DATASETS


class ExperimentDBWriter:
    """Write experiment results to SQLite database."""
    
    def __init__(self, db_path: str = "exp-results/results.db"):
        """
        Initialize database writer.
        
        Args:
            db_path: Path to SQLite database file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        
        # Create all tables if they don't exist
        create_all_tables(self.conn)
    
    def create_experiment(
        self,
        dataset_name: str,
        experiment_type: str,
        total_questions: int,
        mechanisms: Optional[List[str]] = None,
        epsilon_values: Optional[List[float]] = None,
        local_model: Optional[str] = None,
        remote_cot_model: Optional[str] = None,
        remote_qa_model: Optional[str] = None,
        description: Optional[str] = None
    ) -> int:
        """
        Create a new experiment record.
        
        Args:
            dataset_name: Name of the dataset
            experiment_type: 'epsilon_independent' or 'epsilon_dependent'
            total_questions: Total number of questions in experiment
            mechanisms: List of mechanism names (for epsilon_dependent)
            epsilon_values: List of epsilon values (for epsilon_dependent)
            local_model: Local model name
            remote_cot_model: Remote CoT model name
            remote_qa_model: Remote QA model name
            description: Optional description
        
        Returns:
            Experiment ID
        """
        timestamp = datetime.now().isoformat()
        
        try:
            cursor = self.conn.execute("""
                INSERT INTO experiments (
                    dataset_name, experiment_type, timestamp, description,
                    total_questions, mechanisms, epsilon_values,
                    local_model, remote_cot_model, remote_qa_model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_name,
                experiment_type,
                timestamp,
                description,
                total_questions,
                json.dumps(mechanisms) if mechanisms else None,
                json.dumps(epsilon_values) if epsilon_values else None,
                local_model,
                remote_cot_model,
                remote_qa_model
            ))
            
            experiment_id = cursor.lastrowid
            self.conn.commit()
            
            # Verify the experiment was actually created
            if experiment_id == 0:
                raise ValueError("Failed to create experiment: lastrowid is 0")
            
            # Double-check by querying the database
            verify_cursor = self.conn.execute(
                "SELECT id, dataset_name FROM experiments WHERE id = ?",
                (experiment_id,)
            )
            verify_row = verify_cursor.fetchone()
            if not verify_row:
                raise ValueError(f"Experiment ID {experiment_id} was not found in database after creation")
            if verify_row[1] != dataset_name:
                raise ValueError(f"Experiment ID {experiment_id} is for wrong dataset: expected {dataset_name}, got {verify_row[1]}")
            
            return experiment_id
            
        except sqlite3.IntegrityError as e:
            # UNIQUE constraint violation - this shouldn't happen with timestamps
            raise ValueError(f"Failed to create experiment due to UNIQUE constraint: {e}. This may indicate a timestamp collision.")
        except Exception as e:
            self.conn.rollback()
            raise
    
    def write_epsilon_independent_result(
        self,
        dataset_name: str,
        experiment_id: int,
        question_idx: int,
        scenario: str,
        result: Dict[str, Any]
    ):
        """
        Write epsilon-independent result.
        
        Args:
            dataset_name: Name of the dataset
            experiment_id: Experiment ID
            question_idx: Question index
            scenario: Scenario name ('local', 'local_cot', 'remote')
            result: Result dictionary with keys:
                - question: Original question
                - options: Options dict (will be JSON-encoded)
                - answer: Generated answer
                - ground_truth: Ground truth answer
                - is_correct: Boolean
                - processing_time: Optional float
                - cot_text: Optional CoT text
                - local_model: Optional local model name
                - remote_model: Optional remote model name
        """
        table_name = f"{dataset_name}_epsilon_independent_results"
        
        self.conn.execute(f"""
            INSERT OR REPLACE INTO {table_name} (
                experiment_id, question_idx, scenario,
                original_question, options, cot_text,
                generated_answer, ground_truth_answer, is_correct,
                processing_time, local_model, remote_model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            question_idx,
            scenario,
            result.get('question', ''),
            json.dumps(result.get('options', {})),
            result.get('cot_text'),
            result.get('answer', ''),
            result.get('ground_truth', ''),
            result.get('is_correct', False),
            result.get('processing_time'),
            result.get('local_model'),
            result.get('remote_model')
        ))
        
        self.conn.commit()
    
    def write_epsilon_dependent_result(
        self,
        dataset_name: str,
        experiment_id: int,
        question_idx: int,
        mechanism: str,
        epsilon: float,
        result: Dict[str, Any]
    ):
        """
        Write epsilon-dependent result.
        
        Args:
            dataset_name: Name of the dataset
            experiment_id: Experiment ID
            question_idx: Question index
            mechanism: Mechanism name ('phrasedp', 'phrasedp_plus', etc.)
            epsilon: Epsilon value
            result: Result dictionary with keys:
                - question: Original question
                - sanitized_question: Sanitized question
                - options: Options dict (will be JSON-encoded)
                - cot_text: Induced CoT text
                - answer: Generated answer
                - ground_truth: Ground truth answer
                - is_correct: Boolean
                - processing_time: Optional float
                - sanitization_time: Optional float
                - cot_generation_time: Optional float
                - local_model: Local model name
                - remote_model: Remote model name
        """
        table_name = f"{dataset_name}_epsilon_dependent_results"
        
        self.conn.execute(f"""
            INSERT OR REPLACE INTO {table_name} (
                experiment_id, question_idx, mechanism, epsilon,
                original_question, sanitized_question, options, induced_cot,
                generated_answer, ground_truth_answer, is_correct,
                processing_time, sanitization_time, cot_generation_time,
                local_model, remote_model
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_id,
            question_idx,
            mechanism,
            epsilon,
            result.get('question', ''),
            result.get('sanitized_question', ''),
            json.dumps(result.get('options', {})),
            result.get('cot_text', ''),
            result.get('answer', ''),
            result.get('ground_truth', ''),
            result.get('is_correct', False),
            result.get('processing_time'),
            result.get('sanitization_time'),
            result.get('cot_generation_time'),
            result.get('local_model', ''),
            result.get('remote_model', '')
        ))
        
        self.conn.commit()
    
    def write_epsilon_independent_results_batch(
        self,
        dataset_name: str,
        experiment_id: int,
        results: List[Tuple[int, str, Dict[str, Any]]]
    ):
        """
        Write multiple epsilon-independent results in a batch.
        
        Args:
            dataset_name: Name of the dataset
            experiment_id: Experiment ID
            results: List of (question_idx, scenario, result_dict) tuples
        """
        table_name = f"{dataset_name}_epsilon_independent_results"
        
        for question_idx, scenario, result in results:
            self.conn.execute(f"""
                INSERT OR REPLACE INTO {table_name} (
                    experiment_id, question_idx, scenario,
                    original_question, options, cot_text,
                    generated_answer, ground_truth_answer, is_correct,
                    processing_time, local_model, remote_model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                question_idx,
                scenario,
                result.get('question', ''),
                json.dumps(result.get('options', {})),
                result.get('cot_text'),
                result.get('answer', ''),
                result.get('ground_truth', ''),
                result.get('is_correct', False),
                result.get('processing_time'),
                result.get('local_model'),
                result.get('remote_model')
            ))
        
        self.conn.commit()
    
    def write_epsilon_dependent_results_batch(
        self,
        dataset_name: str,
        experiment_id: int,
        results: List[Tuple[int, str, float, Dict[str, Any]]]
    ):
        """
        Write multiple epsilon-dependent results in a batch.
        
        Args:
            dataset_name: Name of the dataset
            experiment_id: Experiment ID
            results: List of (question_idx, mechanism, epsilon, result_dict) tuples
        """
        table_name = f"{dataset_name}_epsilon_dependent_results"
        
        for question_idx, mechanism, epsilon, result in results:
            self.conn.execute(f"""
                INSERT OR REPLACE INTO {table_name} (
                    experiment_id, question_idx, mechanism, epsilon,
                    original_question, sanitized_question, options, induced_cot,
                    generated_answer, ground_truth_answer, is_correct,
                    processing_time, sanitization_time, cot_generation_time,
                    local_model, remote_model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_id,
                question_idx,
                mechanism,
                epsilon,
                result.get('question', ''),
                result.get('sanitized_question', ''),
                json.dumps(result.get('options', {})),
                result.get('cot_text', ''),
                result.get('answer', ''),
                result.get('ground_truth', ''),
                result.get('is_correct', False),
                result.get('processing_time'),
                result.get('sanitization_time'),
                result.get('cot_generation_time'),
                result.get('local_model', ''),
                result.get('remote_model', '')
            ))
        
        self.conn.commit()
    
    def close(self):
        """Close database connection."""
        self.conn.close()


