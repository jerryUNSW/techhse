#!/usr/bin/env python3
"""
Query Tech4HSE Results Database

This script provides functions to query and display results from the SQLite database.
"""

import sqlite3
import json
from typing import List, Tuple

class ResultsQuery:
    """Query interface for the results database."""
    
    def __init__(self, db_path: str = "tech4hse_results.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        
    def get_experiments(self) -> List[Tuple]:
        """Get all experiments."""
        cursor = self.conn.execute("""
            SELECT id, experiment_type, timestamp, description, total_questions, 
                   mechanisms, epsilon_values, created_at
            FROM experiments
            ORDER BY created_at DESC
        """)
        return cursor.fetchall()
        
    def get_pii_protection_results(self, experiment_id: int = None) -> List[Tuple]:
        """Get PII protection results."""
        query = """
            SELECT mechanism, epsilon, overall_protection, email_protection, 
                   phone_protection, address_protection, name_protection, num_samples
            FROM pii_protection_results
            WHERE experiment_id = COALESCE(?, experiment_id)
            ORDER BY mechanism, epsilon
        """
        cursor = self.conn.execute(query, (experiment_id,))
        return cursor.fetchall()
        
    def get_medqa_results(self, experiment_id: int = None) -> List[Tuple]:
        """Get MedQA results."""
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
        
    def get_pii_samples(self, mechanism: str = None, epsilon: float = None) -> List[Tuple]:
        """Get PII protection samples."""
        query = """
            SELECT p.mechanism, p.epsilon, s.row_index, s.original_text, s.sanitized_text
            FROM pii_protection_samples s
            JOIN pii_protection_results p ON s.protection_result_id = p.id
            WHERE (p.mechanism = ? OR ? IS NULL)
              AND (p.epsilon = ? OR ? IS NULL)
            ORDER BY p.mechanism, p.epsilon, s.row_index
            LIMIT 10
        """
        cursor = self.conn.execute(query, (mechanism, mechanism, epsilon, epsilon))
        return cursor.fetchall()
        
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Main function to display database contents."""
    print("🔍 Querying Tech4HSE Results Database...")
    
    db = ResultsQuery()
    
    # Get experiments
    experiments = db.get_experiments()
    print(f"\n📋 Experiments ({len(experiments)}):")
    for exp in experiments:
        print(f"  ID {exp['id']}: {exp['experiment_type']} - {exp['description']}")
        print(f"    Created: {exp['created_at']}")
        if exp['mechanisms']:
            mechanisms = json.loads(exp['mechanisms'])
            epsilon_values = json.loads(exp['epsilon_values'])
            print(f"    Mechanisms: {mechanisms}")
            print(f"    Epsilon values: {epsilon_values}")
    
    # Get PII protection results
    pii_results = db.get_pii_protection_results()
    if pii_results:
        print(f"\n🛡️ PII Protection Results ({len(pii_results)} combinations):")
        print("  Mechanism | Epsilon | Overall | Email | Phone | Address | Name | Samples")
        print("  ----------|---------|---------|-------|-------|---------|------|--------")
        for result in pii_results:
            print(f"  {result['mechanism']:10} | {result['epsilon']:7.1f} | {result['overall_protection']:7.3f} | "
                  f"{result['email_protection']:5.3f} | {result['phone_protection']:5.3f} | "
                  f"{result['address_protection']:7.3f} | {result['name_protection']:4.3f} | "
                  f"{result['num_samples']:7}")
    
    # Get MedQA results
    medqa_results = db.get_medqa_results()
    if medqa_results:
        print(f"\n🏥 MedQA Results ({len(medqa_results)} combinations):")
        print("  Mechanism | Epsilon | Total | Correct | Accuracy")
        print("  ----------|---------|-------|---------|---------")
        for result in medqa_results:
            print(f"  {result['mechanism']:10} | {result['epsilon']:7.1f} | "
                  f"{result['total_questions']:5} | {result['correct_answers']:7} | "
                  f"{result['accuracy']:8.3f}")
    else:
        print(f"\n🏥 MedQA Results: No data yet (ready for 500 questions × mechanisms × epsilon 2&3)")
    
    # Show sample PII protection examples
    samples = db.get_pii_samples()
    if samples:
        print(f"\n📝 Sample PII Protection Examples (first 3):")
        for i, sample in enumerate(samples[:3]):
            print(f"\n  Example {i+1} ({sample['mechanism']}, ε={sample['epsilon']}):")
            print(f"    Original:  {sample['original_text'][:100]}...")
            print(f"    Sanitized: {sample['sanitized_text'][:100]}...")
    
    db.close()
    
    print(f"\n✅ Database query complete!")
    print(f"💡 Ready to store MedQA UME test results for epsilon 2.0 and 3.0")

if __name__ == "__main__":
    main()
