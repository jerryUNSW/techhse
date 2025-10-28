#!/usr/bin/env python3
"""
Load MedQA Detailed Results into Database
========================================

This script parses the detailed MedQA experiment results from text files
and loads them into the database with proper question IDs (not -1).

Author: Tech4HSE Team
Date: 2025-01-30
"""

import sqlite3
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple
import datetime

def parse_medqa_results_file(file_path: str) -> List[Dict]:
    """Parse a MedQA results text file and extract individual question results."""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by question blocks
    question_blocks = re.split(r'--- Question (\d+)/500 \(Dataset idx: (\d+)\) ---', content)
    
    results = []
    epsilon = None
    
    # Extract epsilon from filename
    if 'epsilon1' in file_path:
        epsilon = 1.0
    elif 'epsilon-2' in file_path or 'epsilon2' in file_path:
        epsilon = 2.0
    elif 'epsilon-3' in file_path or 'epsilon3' in file_path:
        epsilon = 3.0
    
    for i in range(1, len(question_blocks), 3):
        if i + 2 < len(question_blocks):
            question_num = int(question_blocks[i])
            dataset_idx = int(question_blocks[i + 1])
            question_content = question_blocks[i + 2]
            
            # Parse question details
            question_data = parse_question_block(question_content, question_num, dataset_idx, epsilon)
            if question_data:
                results.append(question_data)
    
    return results

def parse_question_block(content: str, question_num: int, dataset_idx: int, epsilon: float) -> Dict:
    """Parse a single question block and extract all mechanism results."""
    
    # Extract question text
    question_match = re.search(r'Question: (.+?)(?=Options:)', content, re.DOTALL)
    if not question_match:
        return None
    
    question_text = question_match.group(1).strip()
    
    # Extract options
    options_match = re.search(r'Options:\s*(.+?)(?=Correct Answer:)', content, re.DOTALL)
    if not options_match:
        return None
    
    options_text = options_match.group(1).strip()
    
    # Extract correct answer
    correct_match = re.search(r'Correct Answer: ([ABCD])', content)
    if not correct_match:
        return None
    
    correct_answer = correct_match.group(1)
    
    # Parse mechanism results
    mechanism_results = {}
    
    # Scenario 1: Purely Local Model (Baseline)
    local_match = re.search(r'--- Scenario 1: Purely Local Model \(Baseline\) ---.*?Local Answer: ([ABCD]).*?Result: (Correct|Incorrect)', content, re.DOTALL)
    if local_match:
        mechanism_results['Purely Local Model (Baseline)'] = {
            'answer': local_match.group(1),
            'is_correct': local_match.group(2) == 'Correct'
        }
    
    # Scenario 2: Non-Private Local Model + Remote CoT
    cot_match = re.search(r'--- Scenario 2: Non-Private Local Model \+ Remote CoT ---.*?Local Answer \(Non-Private CoT-Aided\): ([ABCD]).*?Result: (Correct|Incorrect)', content, re.DOTALL)
    if cot_match:
        mechanism_results['Non-Private Local Model + Remote CoT'] = {
            'answer': cot_match.group(1),
            'is_correct': cot_match.group(2) == 'Correct'
        }
    
    # Scenario 3: Private Local Model + CoT (Phrase DP)
    phrasedp_match = re.search(r'--- Scenario 3: Private Local Model \+ CoT \(Phrase DP \(Old\)\) ---.*?Local Answer \(Private CoT-Aided\): ([ABCD]).*?Result: (Correct|Incorrect)', content, re.DOTALL)
    if phrasedp_match:
        mechanism_results['Private Local Model + CoT (Old Phrase DP)'] = {
            'answer': phrasedp_match.group(1),
            'is_correct': phrasedp_match.group(2) == 'Correct'
        }
    
    # Scenario 3: Private Local Model + CoT (InferDPT with Batch Options)
    inferdpt_match = re.search(r'--- Scenario 3: Private Local Model \+ CoT \(InferDPT with Batch Options\) ---.*?Local Answer \(Fully Private CoT-Aided\): ([ABCD]).*?Result: (Correct|Incorrect)', content, re.DOTALL)
    if inferdpt_match:
        mechanism_results['Private Local Model + CoT (InferDPT)'] = {
            'answer': inferdpt_match.group(1),
            'is_correct': inferdpt_match.group(2) == 'Correct'
        }
    
    # Scenario 3: Private Local Model + CoT (SANTEXT+ with Batch Options)
    santext_match = re.search(r'--- Scenario 3: Private Local Model \+ CoT \(SANTEXT\+ with Batch Options\) ---.*?Local Answer \(Fully Private CoT-Aided\): ([ABCD]).*?Result: (Correct|Incorrect)', content, re.DOTALL)
    if santext_match:
        mechanism_results['Private Local Model + CoT (SANTEXT+)'] = {
            'answer': santext_match.group(1),
            'is_correct': santext_match.group(2) == 'Correct'
        }
    
    # Scenario 4: Purely Remote Model
    remote_match = re.search(r'--- Scenario 4: Purely Remote Model ---.*?Purely Remote Answer: ([ABCD]).*?Result: (Correct|Incorrect)', content, re.DOTALL)
    if remote_match:
        mechanism_results['Purely Remote Model'] = {
            'answer': remote_match.group(1),
            'is_correct': remote_match.group(2) == 'Correct'
        }
    
    return {
        'question_id': question_num,
        'dataset_idx': dataset_idx,
        'epsilon': epsilon,
        'question_text': question_text,
        'options_text': options_text,
        'correct_answer': correct_answer,
        'mechanism_results': mechanism_results
    }

def create_detailed_medqa_table():
    """Create a new table for detailed MedQA results."""
    conn = sqlite3.connect('tech4hse_results.db')
    cursor = conn.cursor()
    
    # Create detailed results table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS medqa_detailed_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id INTEGER NOT NULL,
            dataset_idx INTEGER NOT NULL,
            epsilon REAL NOT NULL,
            mechanism TEXT NOT NULL,
            question_text TEXT,
            options_text TEXT,
            correct_answer TEXT,
            predicted_answer TEXT,
            is_correct BOOLEAN,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(question_id, epsilon, mechanism)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Created medqa_detailed_results table")

def load_detailed_results_to_db(results: List[Dict]):
    """Load detailed results into the database."""
    conn = sqlite3.connect('tech4hse_results.db')
    cursor = conn.cursor()
    
    # Clear existing detailed results for this epsilon
    if results:
        epsilon = results[0]['epsilon']
        cursor.execute('DELETE FROM medqa_detailed_results WHERE epsilon = ?', (epsilon,))
        print(f"🗑️  Cleared existing detailed results for epsilon {epsilon}")
    
    inserted_count = 0
    
    for result in results:
        question_id = result['question_id']
        dataset_idx = result['dataset_idx']
        epsilon = result['epsilon']
        question_text = result['question_text']
        options_text = result['options_text']
        correct_answer = result['correct_answer']
        
        for mechanism, mechanism_result in result['mechanism_results'].items():
            predicted_answer = mechanism_result['answer']
            is_correct = mechanism_result['is_correct']
            
            try:
                cursor.execute('''
                    INSERT INTO medqa_detailed_results 
                    (question_id, dataset_idx, epsilon, mechanism, question_text, options_text, 
                     correct_answer, predicted_answer, is_correct)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (question_id, dataset_idx, epsilon, mechanism, question_text, options_text,
                      correct_answer, predicted_answer, is_correct))
                inserted_count += 1
            except sqlite3.IntegrityError:
                # Skip duplicates
                continue
    
    conn.commit()
    conn.close()
    print(f"✅ Inserted {inserted_count} detailed results into database")

def main():
    """Main function to load MedQA detailed results."""
    print("🔍 Loading MedQA Detailed Results into Database")
    print("=" * 50)
    
    # Create the detailed results table
    create_detailed_medqa_table()
    
    # Define the result files
    result_files = [
        'experiment_results/QA-results/MedQA-UME-results/MedQA-UME_epsilon1_comprehensive_mechanisms.txt',
        'experiment_results/QA-results/MedQA-UME-results/test-500-new-epsilon-2.txt',
        'experiment_results/QA-results/MedQA-UME-results/test-500-new-epsilon-3.txt'
    ]
    
    total_loaded = 0
    
    for file_path in result_files:
        if not Path(file_path).exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        print(f"\n📁 Processing: {file_path}")
        
        # Parse the file
        results = parse_medqa_results_file(file_path)
        print(f"📊 Parsed {len(results)} questions")
        
        if results:
            # Load into database
            load_detailed_results_to_db(results)
            total_loaded += len(results)
    
    print(f"\n✅ Total questions loaded: {total_loaded}")
    
    # Verify the data
    conn = sqlite3.connect('tech4hse_results.db')
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM medqa_detailed_results')
    total_records = cursor.fetchone()[0]
    
    cursor.execute('SELECT DISTINCT epsilon FROM medqa_detailed_results ORDER BY epsilon')
    epsilons = [row[0] for row in cursor.fetchall()]
    
    cursor.execute('SELECT DISTINCT mechanism FROM medqa_detailed_results')
    mechanisms = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    print(f"📊 Database verification:")
    print(f"  Total records: {total_records}")
    print(f"  Epsilon values: {epsilons}")
    print(f"  Mechanisms: {len(mechanisms)}")
    for mechanism in mechanisms:
        print(f"    - {mechanism}")

if __name__ == "__main__":
    main()
