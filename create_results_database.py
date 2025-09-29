#!/usr/bin/env python3
"""
Create SQLite Database for Tech4HSE Experiment Results

This script creates a SQLite database to store:
1. MedQA UME test results (500 questions × mechanisms × epsilon values)
2. PII protection results (mechanisms × epsilon values × protection rates)

Author: Tech4HSE Team
Date: 2025-01-29
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any

class ResultsDatabase:
    """SQLite database for storing experiment results."""
    
    def __init__(self, db_path: str = "tech4hse_results.db"):
        """Initialize database connection and create tables."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.create_tables()
        
    def create_tables(self):
        """Create all necessary tables."""
        
        # Table 1: Experiments metadata
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_type TEXT NOT NULL,  -- 'medqa_ume' or 'pii_protection'
                timestamp TEXT NOT NULL,
                description TEXT,
                total_questions INTEGER,
                mechanisms TEXT,  -- JSON array of mechanism names
                epsilon_values TEXT,  -- JSON array of epsilon values
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Table 2: MedQA UME Results
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS medqa_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                question_id INTEGER NOT NULL,
                mechanism TEXT NOT NULL,
                epsilon REAL NOT NULL,
                is_correct BOOLEAN NOT NULL,
                local_answer TEXT,
                correct_answer TEXT,
                processing_time REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id),
                UNIQUE(question_id, mechanism, epsilon)
            )
        """)
        
        # Table 3: PII Protection Results (aggregated)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pii_protection_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER,
                mechanism TEXT NOT NULL,
                epsilon REAL NOT NULL,
                overall_protection REAL NOT NULL,
                email_protection REAL NOT NULL,
                phone_protection REAL NOT NULL,
                address_protection REAL NOT NULL,
                name_protection REAL NOT NULL,
                num_samples INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id),
                UNIQUE(mechanism, epsilon)
            )
        """)
        
        # Table 4: PII Protection Samples (individual examples)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pii_protection_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                protection_result_id INTEGER,
                row_index INTEGER NOT NULL,
                original_text TEXT NOT NULL,
                sanitized_text TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (protection_result_id) REFERENCES pii_protection_results (id)
            )
        """)
        
        # Create indexes for better performance
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_medqa_question_mechanism ON medqa_results(question_id, mechanism)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_medqa_epsilon ON medqa_results(epsilon)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_pii_mechanism_epsilon ON pii_protection_results(mechanism, epsilon)")
        
        self.conn.commit()
        
    def add_medqa_experiment(self, 
                           timestamp: str,
                           total_questions: int = 500,
                           mechanisms: List[str] = None,
                           epsilon_values: List[float] = None) -> int:
        """Add a new MedQA UME experiment record."""
        if mechanisms is None:
            mechanisms = ["PhraseDP", "InferDPT", "SANTEXT+", "CusText+", "CluSanT"]
        if epsilon_values is None:
            epsilon_values = [2.0, 3.0]
            
        cursor = self.conn.execute("""
            INSERT INTO experiments (experiment_type, timestamp, description, total_questions, mechanisms, epsilon_values)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'medqa_ume',
            timestamp,
            f'MedQA UME test with {total_questions} questions, epsilon {epsilon_values}',
            total_questions,
            json.dumps(mechanisms),
            json.dumps(epsilon_values)
        ))
        
        experiment_id = cursor.lastrowid
        self.conn.commit()
        return experiment_id
        
    def add_medqa_result(self, 
                        experiment_id: int,
                        question_id: int,
                        mechanism: str,
                        epsilon: float,
                        is_correct: bool,
                        local_answer: str = None,
                        correct_answer: str = None,
                        processing_time: float = None):
        """Add a single MedQA result."""
        self.conn.execute("""
            INSERT OR REPLACE INTO medqa_results 
            (experiment_id, question_id, mechanism, epsilon, is_correct, local_answer, correct_answer, processing_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (experiment_id, question_id, mechanism, epsilon, is_correct, local_answer, correct_answer, processing_time))
        self.conn.commit()
        
    def add_pii_protection_experiment(self, 
                                    timestamp: str,
                                    mechanisms: List[str] = None,
                                    epsilon_values: List[float] = None) -> int:
        """Add a new PII protection experiment record."""
        if mechanisms is None:
            mechanisms = ["PhraseDP", "InferDPT", "SANTEXT+", "CusText+", "CluSanT"]
        if epsilon_values is None:
            epsilon_values = [1.0, 1.5, 2.0, 2.5, 3.0]
            
        cursor = self.conn.execute("""
            INSERT INTO experiments (experiment_type, timestamp, description, mechanisms, epsilon_values)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'pii_protection',
            timestamp,
            f'PII protection test with {len(mechanisms)} mechanisms, epsilon {epsilon_values}',
            json.dumps(mechanisms),
            json.dumps(epsilon_values)
        ))
        
        experiment_id = cursor.lastrowid
        self.conn.commit()
        return experiment_id
        
    def add_pii_protection_result(self, 
                                experiment_id: int,
                                mechanism: str,
                                epsilon: float,
                                overall_protection: float,
                                email_protection: float,
                                phone_protection: float,
                                address_protection: float,
                                name_protection: float,
                                num_samples: int) -> int:
        """Add PII protection aggregated results."""
        cursor = self.conn.execute("""
            INSERT OR REPLACE INTO pii_protection_results 
            (experiment_id, mechanism, epsilon, overall_protection, email_protection, 
             phone_protection, address_protection, name_protection, num_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (experiment_id, mechanism, epsilon, overall_protection, email_protection, 
              phone_protection, address_protection, name_protection, num_samples))
        
        result_id = cursor.lastrowid
        self.conn.commit()
        return result_id
        
    def add_pii_protection_samples(self, 
                                 protection_result_id: int,
                                 samples: List[Dict[str, Any]]):
        """Add individual PII protection samples."""
        for sample in samples:
            self.conn.execute("""
                INSERT INTO pii_protection_samples 
                (protection_result_id, row_index, original_text, sanitized_text)
                VALUES (?, ?, ?, ?)
            """, (protection_result_id, sample['row'], sample['original'], sample['sanitized']))
        self.conn.commit()
        
    def load_pii_protection_results_from_json(self, json_file: str):
        """Load PII protection results from JSON file."""
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        # Create experiment record
        timestamp = "2025-09-27T22:08:05"  # From filename
        experiment_id = self.add_pii_protection_experiment(timestamp)
        
        print(f"Loading PII protection results from {json_file}")
        print(f"Created experiment record with ID: {experiment_id}")
        
        # Load results for each mechanism and epsilon
        for mechanism, mechanism_data in data.items():
            if isinstance(mechanism_data, dict):
                for epsilon_str, epsilon_data in mechanism_data.items():
                    try:
                        epsilon = float(epsilon_str)
                        
                        # Skip samples for now, we'll add them separately
                        if 'samples' in epsilon_data:
                            samples = epsilon_data['samples']
                            del epsilon_data['samples']
                        else:
                            samples = []
                            
                        # Add protection results
                        protection_result_id = self.add_pii_protection_result(
                            experiment_id=experiment_id,
                            mechanism=mechanism,
                            epsilon=epsilon,
                            overall_protection=epsilon_data.get('overall', 0.0),
                            email_protection=epsilon_data.get('emails', 0.0),
                            phone_protection=epsilon_data.get('phones', 0.0),
                            address_protection=epsilon_data.get('addresses', 0.0),
                            name_protection=epsilon_data.get('names', 0.0),
                            num_samples=len(samples)
                        )
                        
                        # Add samples
                        if samples:
                            self.add_pii_protection_samples(protection_result_id, samples)
                            
                        print(f"  Loaded {mechanism} epsilon {epsilon}: {len(samples)} samples")
                        
                    except (ValueError, TypeError) as e:
                        print(f"  Skipping {mechanism}/{epsilon_str}: {e}")
                        
        print(f"✅ Successfully loaded PII protection results")
        
    def get_medqa_accuracy_summary(self, experiment_id: int = None):
        """Get accuracy summary for MedQA results."""
        query = """
            SELECT mechanism, epsilon, 
                   COUNT(*) as total_questions,
                   SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_answers,
                   ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END), 3) as accuracy
            FROM medqa_results
            WHERE experiment_id = COALESCE(?, experiment_id)
            GROUP BY mechanism, epsilon
            ORDER BY mechanism, epsilon
        """
        
        cursor = self.conn.execute(query, (experiment_id,))
        return cursor.fetchall()
        
    def get_pii_protection_summary(self, experiment_id: int = None):
        """Get protection rate summary for PII results."""
        query = """
            SELECT mechanism, epsilon, overall_protection, email_protection, 
                   phone_protection, address_protection, name_protection, num_samples
            FROM pii_protection_results
            WHERE experiment_id = COALESCE(?, experiment_id)
            ORDER BY mechanism, epsilon
        """
        
        cursor = self.conn.execute(query, (experiment_id,))
        return cursor.fetchall()
        
    def get_experiment_info(self):
        """Get information about all experiments."""
        query = """
            SELECT id, experiment_type, timestamp, description, total_questions, 
                   mechanisms, epsilon_values, created_at
            FROM experiments
            ORDER BY created_at DESC
        """
        
        cursor = self.conn.execute(query)
        return cursor.fetchall()
        
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Main function to create database and load data."""
    print("🔧 Creating Tech4HSE Results Database...")
    
    # Create database
    db = ResultsDatabase("tech4hse_results.db")
    
    # Load PII protection results if file exists
    pii_file = "pii_protection_results_20250927_220805.json"
    if os.path.exists(pii_file):
        print(f"\n📊 Loading PII protection results from {pii_file}...")
        db.load_pii_protection_results_from_json(pii_file)
    else:
        print(f"\n⚠️  PII protection results file not found: {pii_file}")
    
    # Create a sample MedQA experiment record (for future use)
    print(f"\n📝 Creating sample MedQA experiment record...")
    timestamp = datetime.now().isoformat()
    medqa_experiment_id = db.add_medqa_experiment(
        timestamp=timestamp,
        total_questions=500,
        mechanisms=["PhraseDP", "InferDPT", "SANTEXT+", "CusText+", "CluSanT"],
        epsilon_values=[2.0, 3.0]
    )
    print(f"Created MedQA experiment record with ID: {medqa_experiment_id}")
    
    # Display summary
    print(f"\n📈 Database Summary:")
    experiments = db.get_experiment_info()
    for exp in experiments:
        exp_id, exp_type, timestamp, description, total_questions, mechanisms, epsilon_values, created_at = exp
        print(f"  Experiment {exp_id}: {exp_type}")
        print(f"    Description: {description}")
        print(f"    Created: {created_at}")
        
        if exp_type == 'pii_protection':
            results = db.get_pii_protection_summary(exp_id)
            print(f"    Results: {len(results)} mechanism/epsilon combinations")
        elif exp_type == 'medqa_ume':
            results = db.get_medqa_accuracy_summary(exp_id)
            print(f"    Results: {len(results)} mechanism/epsilon combinations")
    
    # Close database
    db.close()
    
    print(f"\n✅ Database created successfully: tech4hse_results.db")
    print(f"📋 Ready to store:")
    print(f"   - MedQA UME results (500 questions × mechanisms × epsilon 2&3)")
    print(f"   - PII protection results (mechanisms × epsilon × protection rates)")
    print(f"   - Individual PII protection samples")

if __name__ == "__main__":
    main()