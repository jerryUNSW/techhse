#!/usr/bin/env python3
"""
Interactive Database Browser for Tech4HSE Results

Simple command-line interface to browse and query the SQLite database.
"""

import sqlite3
import json
from typing import List, Tuple

class DatabaseBrowser:
    """Interactive database browser."""
    
    def __init__(self, db_path: str = "tech4hse_results.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
    def run_query(self, query: str, params: tuple = ()) -> List[Tuple]:
        """Run a SQL query and return results."""
        cursor = self.conn.execute(query, params)
        return cursor.fetchall()
        
    def show_experiments(self):
        """Show all experiments."""
        print("\n📋 Experiments:")
        results = self.run_query("SELECT * FROM experiments ORDER BY created_at DESC")
        for row in results:
            print(f"  ID {row['id']}: {row['experiment_type']}")
            print(f"    Description: {row['description']}")
            print(f"    Created: {row['created_at']}")
            if row['mechanisms']:
                mechanisms = json.loads(row['mechanisms'])
                epsilon_values = json.loads(row['epsilon_values'])
                print(f"    Mechanisms: {mechanisms}")
                print(f"    Epsilon values: {epsilon_values}")
            print()
            
    def show_pii_results(self, epsilon: float = None):
        """Show PII protection results."""
        print(f"\n🛡️ PII Protection Results" + (f" (ε={epsilon})" if epsilon else ""))
        
        if epsilon:
            query = "SELECT * FROM pii_protection_results WHERE epsilon = ? ORDER BY overall_protection DESC"
            results = self.run_query(query, (epsilon,))
        else:
            query = "SELECT * FROM pii_protection_results ORDER BY mechanism, epsilon"
            results = self.run_query(query)
            
        print("  Mechanism | Epsilon | Overall | Email | Phone | Address | Name | Samples")
        print("  ----------|---------|---------|-------|-------|---------|------|--------")
        for row in results:
            print(f"  {row['mechanism']:10} | {row['epsilon']:7.1f} | {row['overall_protection']:7.3f} | "
                  f"{row['email_protection']:5.3f} | {row['phone_protection']:5.3f} | "
                  f"{row['address_protection']:7.3f} | {row['name_protection']:4.3f} | "
                  f"{row['num_samples']:7}")
                  
    def show_medqa_results(self):
        """Show MedQA results."""
        print(f"\n🏥 MedQA Results:")
        results = self.run_query("""
            SELECT mechanism, epsilon, 
                   COUNT(*) as total_questions,
                   SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_answers,
                   ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END), 3) as accuracy
            FROM medqa_results
            GROUP BY mechanism, epsilon
            ORDER BY mechanism, epsilon
        """)
        
        if results:
            print("  Mechanism | Epsilon | Total | Correct | Accuracy")
            print("  ----------|---------|-------|---------|---------")
            for row in results:
                print(f"  {row['mechanism']:10} | {row['epsilon']:7.1f} | "
                      f"{row['total_questions']:5} | {row['correct_answers']:7} | "
                      f"{row['accuracy']:8.3f}")
        else:
            print("  No MedQA results found (ready for 500 questions × mechanisms × epsilon 2&3)")
            
    def show_samples(self, mechanism: str = None, epsilon: float = None, limit: int = 3):
        """Show PII protection samples."""
        print(f"\n📝 PII Protection Samples" + 
              (f" ({mechanism}, ε={epsilon})" if mechanism or epsilon else ""))
        
        query = """
            SELECT p.mechanism, p.epsilon, s.row_index, s.original_text, s.sanitized_text
            FROM pii_protection_samples s
            JOIN pii_protection_results p ON s.protection_result_id = p.id
            WHERE (p.mechanism = ? OR ? IS NULL)
              AND (p.epsilon = ? OR ? IS NULL)
            ORDER BY p.mechanism, p.epsilon, s.row_index
            LIMIT ?
        """
        results = self.run_query(query, (mechanism, mechanism, epsilon, epsilon, limit))
        
        for i, row in enumerate(results):
            print(f"\n  Example {i+1} ({row['mechanism']}, ε={row['epsilon']}):")
            print(f"    Original:  {row['original_text'][:100]}...")
            print(f"    Sanitized: {row['sanitized_text'][:100]}...")
            
    def add_medqa_result(self, experiment_id: int, question_id: int, mechanism: str, 
                        epsilon: float, is_correct: bool, local_answer: str = None, 
                        correct_answer: str = None):
        """Add a MedQA result."""
        self.conn.execute("""
            INSERT OR REPLACE INTO medqa_results 
            (experiment_id, question_id, mechanism, epsilon, is_correct, local_answer, correct_answer)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (experiment_id, question_id, mechanism, epsilon, is_correct, local_answer, correct_answer))
        self.conn.commit()
        print(f"✅ Added MedQA result: Q{question_id} {mechanism} ε={epsilon} {'✓' if is_correct else '✗'}")
        
    def run_custom_query(self, query: str):
        """Run a custom SQL query."""
        try:
            results = self.run_query(query)
            if results:
                # Print column names
                print(f"\nColumns: {', '.join(results[0].keys())}")
                print("-" * 50)
                
                # Print results (limit to 10 rows)
                for i, row in enumerate(results[:10]):
                    print(f"Row {i+1}: {dict(row)}")
                    
                if len(results) > 10:
                    print(f"... and {len(results) - 10} more rows")
            else:
                print("No results found.")
        except Exception as e:
            print(f"❌ Query error: {e}")
            
    def interactive_mode(self):
        """Run interactive mode."""
        print("🔍 Tech4HSE Database Browser")
        print("Available commands:")
        print("  1 - Show experiments")
        print("  2 - Show PII results")
        print("  3 - Show PII results for epsilon 2.0")
        print("  4 - Show MedQA results")
        print("  5 - Show samples")
        print("  6 - Add MedQA result")
        print("  7 - Custom query")
        print("  q - Quit")
        
        while True:
            try:
                cmd = input("\n> ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == '1':
                    self.show_experiments()
                elif cmd == '2':
                    self.show_pii_results()
                elif cmd == '3':
                    self.show_pii_results(epsilon=2.0)
                elif cmd == '4':
                    self.show_medqa_results()
                elif cmd == '5':
                    mechanism = input("Mechanism (or Enter for all): ").strip() or None
                    epsilon = input("Epsilon (or Enter for all): ").strip()
                    epsilon = float(epsilon) if epsilon else None
                    self.show_samples(mechanism, epsilon)
                elif cmd == '6':
                    print("Add MedQA result:")
                    experiment_id = int(input("Experiment ID (2): ") or "2")
                    question_id = int(input("Question ID: "))
                    mechanism = input("Mechanism: ")
                    epsilon = float(input("Epsilon: "))
                    is_correct = input("Correct (y/n): ").lower() == 'y'
                    local_answer = input("Local answer (optional): ") or None
                    correct_answer = input("Correct answer (optional): ") or None
                    self.add_medqa_result(experiment_id, question_id, mechanism, epsilon, is_correct, local_answer, correct_answer)
                elif cmd == '7':
                    query = input("SQL query: ")
                    self.run_custom_query(query)
                else:
                    print("Unknown command. Use 1-7 or 'q' to quit.")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Main function."""
    db = DatabaseBrowser()
    
    try:
        db.interactive_mode()
    finally:
        db.close()

if __name__ == "__main__":
    main()
