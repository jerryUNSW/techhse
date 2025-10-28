#!/usr/bin/env python3
"""
Script to populate HSE-bench local model results into the database
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime

def populate_hse_bench_results():
    """Populate the database with recent HSE-bench local model results."""
    
    # Connect to database
    conn = sqlite3.connect('tech4hse_results.db')
    cursor = conn.cursor()
    
    # Define the recent results files and their categories
    results_files = [
        {
            'file': 'experiment_results/QA-results/hse-bench/hse_bench_local_only_results_meta-llama_Meta-Llama-3.1-8B-Instruct_-1q_20250930_201711.json',
            'category': 'regulation',
            'experiment_id': 'hse_bench_20250930_201711'
        },
        {
            'file': 'experiment_results/QA-results/hse-bench/hse_bench_local_only_results_meta-llama_Meta-Llama-3.1-8B-Instruct_-1q_20250930_201418.json',
            'category': 'safety_exam',
            'experiment_id': 'hse_bench_20250930_201418'
        },
        {
            'file': 'experiment_results/QA-results/hse-bench/hse_bench_local_only_results_meta-llama_Meta-Llama-3.1-8B-Instruct_-1q_20250930_194756.json',
            'category': 'court_case',
            'experiment_id': 'hse_bench_20250930_194756'
        }
    ]
    
    for result_info in results_files:
        file_path = Path(result_info['file'])
        category = result_info['category']
        experiment_id = result_info['experiment_id']
        
        if not file_path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
            
        try:
            # Load the JSON results
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract basic results
            total_questions = data['results']['total_questions']
            correct_answers = data['results']['local_alone_correct']
            overall_accuracy = (correct_answers / total_questions * 100) if total_questions > 0 else 0
            
            # Extract task type results
            task_results = data.get('task_type_results', {})
            
            # Insert into database
            insert_sql = '''
            INSERT OR REPLACE INTO hse_bench_local_results (
                experiment_id, category, model_name, total_questions, correct_answers, overall_accuracy,
                rule_recall_correct, rule_recall_total, rule_recall_accuracy,
                rule_application_correct, rule_application_total, rule_application_accuracy,
                issue_spotting_correct, issue_spotting_total, issue_spotting_accuracy,
                rule_conclusion_correct, rule_conclusion_total, rule_conclusion_accuracy,
                start_time, end_time, results_file_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            '''
            
            cursor.execute(insert_sql, (
                experiment_id,
                category,
                data.get('model_name', 'meta-llama/Meta-Llama-3.1-8B-Instruct'),
                total_questions,
                correct_answers,
                overall_accuracy,
                task_results.get('rule_recall', {}).get('correct', 0),
                task_results.get('rule_recall', {}).get('total', 0),
                (task_results.get('rule_recall', {}).get('correct', 0) / task_results.get('rule_recall', {}).get('total', 1) * 100) if task_results.get('rule_recall', {}).get('total', 0) > 0 else 0,
                task_results.get('rule_application', {}).get('correct', 0),
                task_results.get('rule_application', {}).get('total', 0),
                (task_results.get('rule_application', {}).get('correct', 0) / task_results.get('rule_application', {}).get('total', 1) * 100) if task_results.get('rule_application', {}).get('total', 0) > 0 else 0,
                task_results.get('issue_spotting', {}).get('correct', 0),
                task_results.get('issue_spotting', {}).get('total', 0),
                (task_results.get('issue_spotting', {}).get('correct', 0) / task_results.get('issue_spotting', {}).get('total', 1) * 100) if task_results.get('issue_spotting', {}).get('total', 0) > 0 else 0,
                task_results.get('rule_conclusion', {}).get('correct', 0),
                task_results.get('rule_conclusion', {}).get('total', 0),
                (task_results.get('rule_conclusion', {}).get('correct', 0) / task_results.get('rule_conclusion', {}).get('total', 1) * 100) if task_results.get('rule_conclusion', {}).get('total', 0) > 0 else 0,
                data.get('start_time', ''),
                data.get('end_time', ''),
                str(file_path)
            ))
            
            print(f"✅ Inserted {category} results: {correct_answers}/{total_questions} = {overall_accuracy:.1f}%")
            
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
    
    # Commit changes
    conn.commit()
    
    # Display summary
    print("\n📊 Database Summary:")
    cursor.execute('SELECT category, total_questions, correct_answers, overall_accuracy FROM hse_bench_local_results ORDER BY overall_accuracy DESC')
    results = cursor.fetchall()
    
    for row in results:
        category, total, correct, accuracy = row
        print(f"   {category.upper()}: {correct}/{total} = {accuracy:.1f}%")
    
    conn.close()
    print("\n✅ HSE-bench results populated successfully!")

if __name__ == "__main__":
    populate_hse_bench_results()

