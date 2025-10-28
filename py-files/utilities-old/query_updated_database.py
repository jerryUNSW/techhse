#!/usr/bin/env python3
"""
Query Updated Database with All PII Protection Results

This script provides comprehensive queries to view all PII protection results
in the database, including the newly updated CluSanT data.
"""

import sqlite3
import pandas as pd
from tabulate import tabulate

def connect_to_database():
    """Connect to the SQLite database."""
    db_path = '/home/yizhang/tech4HSE/tech4hse_results.db'
    return sqlite3.connect(db_path)

def query_all_protection_results():
    """Query all PII protection results."""
    conn = connect_to_database()
    
    query = """
    SELECT 
        mechanism,
        epsilon,
        overall_protection,
        email_protection,
        phone_protection,
        address_protection,
        name_protection,
        num_samples,
        created_at
    FROM pii_protection_results 
    ORDER BY mechanism, epsilon
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def query_mechanism_summary():
    """Query summary statistics by mechanism."""
    conn = connect_to_database()
    
    query = """
    SELECT 
        mechanism,
        COUNT(*) as num_epsilon_values,
        AVG(overall_protection) as avg_overall_protection,
        AVG(email_protection) as avg_email_protection,
        AVG(phone_protection) as avg_phone_protection,
        AVG(address_protection) as avg_address_protection,
        AVG(name_protection) as avg_name_protection,
        SUM(num_samples) as total_samples
    FROM pii_protection_results 
    GROUP BY mechanism
    ORDER BY avg_overall_protection DESC
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def query_epsilon_comparison():
    """Query protection rates by epsilon value."""
    conn = connect_to_database()
    
    query = """
    SELECT 
        epsilon,
        AVG(overall_protection) as avg_overall_protection,
        AVG(email_protection) as avg_email_protection,
        AVG(phone_protection) as avg_phone_protection,
        AVG(address_protection) as avg_address_protection,
        AVG(name_protection) as avg_name_protection,
        COUNT(DISTINCT mechanism) as num_mechanisms
    FROM pii_protection_results 
    GROUP BY epsilon
    ORDER BY epsilon
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def query_sample_counts():
    """Query sample counts by mechanism and epsilon."""
    conn = connect_to_database()
    
    query = """
    SELECT 
        p.mechanism,
        p.epsilon,
        p.num_samples,
        COUNT(s.id) as actual_samples
    FROM pii_protection_results p
    LEFT JOIN pii_protection_samples s ON p.id = s.protection_result_id
    GROUP BY p.mechanism, p.epsilon
    ORDER BY p.mechanism, p.epsilon
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df

def format_percentage(value):
    """Format decimal as percentage."""
    return f"{value:.1%}"

def main():
    """Main function to display all database queries."""
    print("="*80)
    print("COMPREHENSIVE PII PROTECTION DATABASE QUERY RESULTS")
    print("="*80)
    print()
    
    # Query 1: All protection results
    print("1. ALL PII PROTECTION RESULTS")
    print("-" * 50)
    df_all = query_all_protection_results()
    
    # Format the dataframe for display
    df_display = df_all.copy()
    for col in ['overall_protection', 'email_protection', 'phone_protection', 
                'address_protection', 'name_protection']:
        df_display[col] = df_display[col].apply(format_percentage)
    
    print(tabulate(df_display, headers='keys', tablefmt='grid', showindex=False))
    print()
    
    # Query 2: Mechanism summary
    print("2. MECHANISM SUMMARY (Average Performance)")
    print("-" * 50)
    df_summary = query_mechanism_summary()
    
    # Format percentages
    for col in ['avg_overall_protection', 'avg_email_protection', 'avg_phone_protection', 
                'avg_address_protection', 'avg_name_protection']:
        df_summary[col] = df_summary[col].apply(format_percentage)
    
    print(tabulate(df_summary, headers='keys', tablefmt='grid', showindex=False))
    print()
    
    # Query 3: Epsilon comparison
    print("3. EPSILON VALUE COMPARISON (Average Across All Mechanisms)")
    print("-" * 50)
    df_epsilon = query_epsilon_comparison()
    
    # Format percentages
    for col in ['avg_overall_protection', 'avg_email_protection', 'avg_phone_protection', 
                'avg_address_protection', 'avg_name_protection']:
        df_epsilon[col] = df_epsilon[col].apply(format_percentage)
    
    print(tabulate(df_epsilon, headers='keys', tablefmt='grid', showindex=False))
    print()
    
    # Query 4: Sample counts
    print("4. SAMPLE COUNTS BY MECHANISM AND EPSILON")
    print("-" * 50)
    df_samples = query_sample_counts()
    print(tabulate(df_samples, headers='keys', tablefmt='grid', showindex=False))
    print()
    
    # Summary insights
    print("5. KEY INSIGHTS")
    print("-" * 50)
    
    # Best performing mechanism
    best_mechanism = df_summary.loc[df_summary['avg_overall_protection'].idxmax(), 'mechanism']
    best_rate = df_summary['avg_overall_protection'].max()
    print(f"🏆 Best Overall Mechanism: {best_mechanism} ({best_rate:.1%} average protection)")
    
    # Most consistent mechanism
    consistency_scores = []
    for mechanism in df_all['mechanism'].unique():
        mechanism_data = df_all[df_all['mechanism'] == mechanism]['overall_protection']
        consistency = 1 - mechanism_data.std()  # Lower std = higher consistency
        consistency_scores.append((mechanism, consistency))
    
    most_consistent = max(consistency_scores, key=lambda x: x[1])
    print(f"📊 Most Consistent Mechanism: {most_consistent[0]} (std: {most_consistent[1]:.4f})")
    
    # Total experiments
    total_experiments = len(df_all)
    total_mechanisms = df_all['mechanism'].nunique()
    total_samples = df_all['num_samples'].sum()
    print(f"📈 Total Experiments: {total_experiments}")
    print(f"🔬 Total Mechanisms: {total_mechanisms}")
    print(f"📝 Total Samples: {total_samples}")
    
    print()
    print("✅ Database query completed successfully!")

if __name__ == "__main__":
    main()
