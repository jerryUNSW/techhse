#!/usr/bin/env python3
"""
Load MedQA Next 500 Results into Database
=========================================

This script loads the results from medqa_usmle_next_500_progress_20251001_092543.json
into the tech4hse_results.db database.

Author: Tech4HSE Team
Date: 2025-01-30
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any

class MedQAResultsLoader:
    """Load MedQA results into the database."""
    
    def __init__(self, db_path: str = "tech4hse_results.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
    
    def add_experiment(self, timestamp: str, epsilon: float = 3.0) -> int:
        """Add a new MedQA experiment record."""
        cursor = self.conn.execute("""
            INSERT INTO experiments (experiment_type, timestamp, description, total_questions, mechanisms, epsilon_values)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'medqa_ume',
            timestamp,
            f'MedQA-USMLE Next 500 Questions Test (Epsilon {epsilon})',
            500,
            json.dumps(["Local Model", "Local + CoT", "PhraseDP + CoT"]),
            json.dumps([epsilon])
        ))
        
        experiment_id = cursor.lastrowid
        self.conn.commit()
        return experiment_id
    
    def load_results_from_json(self, json_file: str, epsilon: float = 3.0):
        """Load results from the JSON progress file."""
        print(f"📊 Loading results from {json_file}...")
        
        # Load JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        if not data:
            print("❌ No data found in JSON file")
            return
        
        # Create experiment record
        timestamp = datetime.now().isoformat()
        experiment_id = self.add_experiment(timestamp, epsilon)
        print(f"✅ Created experiment record with ID: {experiment_id}")
        
        # Process each question
        total_questions = len(data)
        print(f"📝 Processing {total_questions} questions...")
        
        for i, item in enumerate(data):
            question_id = item['question_id']
            correct_answer = item['correct_answer']
            
            # Get results for this question
            results = item['results']
            
            # Add results for each mechanism
            mechanisms = [
                ("Local Model", results['local_alone_correct']),
                ("Local + CoT", results['non_private_cot_correct']),
                ("PhraseDP + CoT", results['old_phrase_dp_local_cot_correct'])
            ]
            
            # For the first question, we need to determine which mechanisms got it right
            if i == 0:
                # For first question, check if any mechanism got it right
                for mechanism_name, correct_count in mechanisms:
                    is_correct = correct_count > 0
                    self.add_result(
                        experiment_id, question_id, mechanism_name, epsilon,
                        is_correct, correct_answer
                    )
            else:
                # For subsequent questions, check if the count increased
                prev_results = data[i-1]['results']
                prev_mechanisms = [
                    ("Local Model", prev_results['local_alone_correct']),
                    ("Local + CoT", prev_results['non_private_cot_correct']),
                    ("PhraseDP + CoT", prev_results['old_phrase_dp_local_cot_correct'])
                ]
                
                for j, (mechanism_name, correct_count) in enumerate(mechanisms):
                    prev_correct = prev_mechanisms[j][1]
                    is_correct = correct_count > prev_correct
                    self.add_result(
                        experiment_id, question_id, mechanism_name, epsilon,
                        is_correct, correct_answer
                    )
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{total_questions} questions...")
        
        print(f"✅ Loaded {total_questions} questions into database")
        
        # Verify the data
        self.verify_data(experiment_id)
    
    def add_result(self, experiment_id: int, question_id: int, mechanism: str, 
                   epsilon: float, is_correct: bool, correct_answer: str):
        """Add a single MedQA result."""
        self.conn.execute("""
            INSERT OR REPLACE INTO medqa_results 
            (experiment_id, question_id, mechanism, epsilon, is_correct, correct_answer)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (experiment_id, question_id, mechanism, epsilon, is_correct, correct_answer))
        self.conn.commit()
    
    def verify_data(self, experiment_id: int):
        """Verify the loaded data."""
        cursor = self.conn.cursor()
        
        # Get total results for this experiment
        cursor.execute("""
            SELECT COUNT(*) FROM medqa_results WHERE experiment_id = ?
        """, (experiment_id,))
        total_results = cursor.fetchone()[0]
        
        # Get accuracy by mechanism
        cursor.execute("""
            SELECT mechanism, 
                   COUNT(*) as total,
                   SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct,
                   ROUND(AVG(CASE WHEN is_correct = 1 THEN 100.0 ELSE 0.0 END), 2) as accuracy
            FROM medqa_results 
            WHERE experiment_id = ?
            GROUP BY mechanism
            ORDER BY accuracy DESC
        """, (experiment_id,))
        
        results = cursor.fetchall()
        
        print(f"\n📈 VERIFICATION RESULTS:")
        print(f"Total records: {total_results}")
        print(f"Accuracy by mechanism:")
        for mechanism, total, correct, accuracy in results:
            print(f"  {mechanism}: {correct}/{total} = {accuracy}%")
    
    def get_experiment_summary(self):
        """Get summary of all experiments."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT e.id, e.experiment_type, e.timestamp, e.description, e.epsilon_values,
                   COUNT(mr.id) as total_results,
                   COUNT(DISTINCT mr.question_id) as total_questions
            FROM experiments e
            LEFT JOIN medqa_results mr ON e.id = mr.experiment_id
            WHERE e.experiment_type = 'medqa_ume'
            GROUP BY e.id
            ORDER BY e.timestamp DESC
        """)
        
        experiments = cursor.fetchall()
        
        print(f"\n📊 MEDQA EXPERIMENTS SUMMARY:")
        print("=" * 60)
        for exp in experiments:
            exp_id, exp_type, timestamp, description, epsilon_values, total_results, total_questions = exp
            print(f"Experiment {exp_id}: {description}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Epsilon: {epsilon_values}")
            print(f"  Questions: {total_questions}")
            print(f"  Results: {total_results}")
            print()
    
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Main function to load MedQA results."""
    import sys
    
    print("🔧 Loading MedQA Next 500 Results into Database")
    print("=" * 60)
    
    # Get command line arguments
    if len(sys.argv) < 2:
        print("Usage: python load_medqa_next_500_results.py <json_file> [epsilon]")
        print("Example: python load_medqa_next_500_results.py medqa_usmle_next_500_progress_20251001_133747.json 1.0")
        return
    
    json_file = sys.argv[1]
    epsilon = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    
    # Initialize loader
    loader = MedQAResultsLoader()
    
    # Load results from JSON file
    if os.path.exists(json_file):
        loader.load_results_from_json(json_file, epsilon=epsilon)
    else:
        print(f"❌ JSON file not found: {json_file}")
        return
    
    # Show experiment summary
    loader.get_experiment_summary()
    
    # Close database
    loader.close()
    
    print("✅ Results loaded successfully into tech4hse_results.db")

if __name__ == "__main__":
    main()
