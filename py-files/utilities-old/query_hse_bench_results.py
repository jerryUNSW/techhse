#!/usr/bin/env python3
"""
Script to query and analyze HSE-bench local model results from the database
"""

import sqlite3
import pandas as pd

def query_hse_bench_results():
    """Query and display HSE-bench results from the database."""
    
    # Connect to database
    conn = sqlite3.connect('tech4hse_results.db')
    
    # Query all results
    query = '''
    SELECT 
        category,
        total_questions,
        correct_answers,
        overall_accuracy,
        rule_recall_accuracy,
        rule_application_accuracy,
        issue_spotting_accuracy,
        rule_conclusion_accuracy,
        start_time,
        end_time
    FROM hse_bench_local_results
    ORDER BY overall_accuracy DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    
    print("📊 HSE-Bench Local Model Results Summary")
    print("=" * 60)
    
    for _, row in df.iterrows():
        category = row['category'].upper()
        total = row['total_questions']
        correct = row['correct_answers']
        overall_acc = row['overall_accuracy']
        
        print(f"\n🏛️ {category} ({total} questions)")
        print(f"   Overall Accuracy: {correct}/{total} = {overall_acc:.1f}%")
        print(f"   Task Type Breakdown:")
        print(f"     Rule Recall: {row['rule_recall_accuracy']:.1f}%")
        print(f"     Rule Application: {row['rule_application_accuracy']:.1f}%")
        print(f"     Issue Spotting: {row['issue_spotting_accuracy']:.1f}%")
        print(f"     Rule Conclusion: {row['rule_conclusion_accuracy']:.1f}%")
        print(f"   Experiment Time: {row['start_time']} to {row['end_time']}")
    
    # Summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"   Total Categories Tested: {len(df)}")
    print(f"   Average Accuracy: {df['overall_accuracy'].mean():.1f}%")
    print(f"   Best Category: {df.loc[df['overall_accuracy'].idxmax(), 'category'].upper()} ({df['overall_accuracy'].max():.1f}%)")
    print(f"   Total Questions: {df['total_questions'].sum()}")
    print(f"   Total Correct: {df['correct_answers'].sum()}")
    
    # Task type analysis
    print(f"\n🎯 Task Type Performance:")
    task_types = ['rule_recall', 'rule_application', 'issue_spotting', 'rule_conclusion']
    for task_type in task_types:
        avg_acc = df[f'{task_type}_accuracy'].mean()
        best_acc = df[f'{task_type}_accuracy'].max()
        worst_acc = df[f'{task_type}_accuracy'].min()
        print(f"   {task_type.replace('_', ' ').title()}: {avg_acc:.1f}% (Best: {best_acc:.1f}%, Worst: {worst_acc:.1f}%)")
    
    conn.close()

def get_task_type_breakdown():
    """Get detailed task type breakdown."""
    
    conn = sqlite3.connect('tech4hse_results.db')
    
    query = '''
    SELECT 
        category,
        rule_recall_correct || '/' || rule_recall_total as rule_recall,
        rule_application_correct || '/' || rule_application_total as rule_application,
        issue_spotting_correct || '/' || issue_spotting_total as issue_spotting,
        rule_conclusion_correct || '/' || rule_conclusion_total as rule_conclusion
    FROM hse_bench_local_results
    ORDER BY overall_accuracy DESC
    '''
    
    df = pd.read_sql_query(query, conn)
    
    print("\n📋 Detailed Task Type Breakdown:")
    print("=" * 80)
    print(f"{'Category':<15} {'Rule Recall':<15} {'Rule Application':<18} {'Issue Spotting':<16} {'Rule Conclusion':<16}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        category = row['category'].upper()
        print(f"{category:<15} {row['rule_recall']:<15} {row['rule_application']:<18} {row['issue_spotting']:<16} {row['rule_conclusion']:<16}")
    
    conn.close()

if __name__ == "__main__":
    query_hse_bench_results()
    get_task_type_breakdown()

