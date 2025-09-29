#!/usr/bin/env python3
"""
Simple Database Query without external dependencies

This script provides basic queries to view all PII protection results
in the database, including the newly updated CluSanT data.
"""

import sqlite3

def connect_to_database():
    """Connect to the SQLite database."""
    db_path = '/home/yizhang/tech4HSE/tech4hse_results.db'
    return sqlite3.connect(db_path)

def query_all_protection_results():
    """Query all PII protection results."""
    conn = connect_to_database()
    cursor = conn.cursor()
    
    query = """
    SELECT 
        mechanism,
        epsilon,
        overall_protection,
        email_protection,
        phone_protection,
        address_protection,
        name_protection,
        num_samples
    FROM pii_protection_results 
    ORDER BY mechanism, epsilon
    """
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    return results

def query_mechanism_summary():
    """Query summary statistics by mechanism."""
    conn = connect_to_database()
    cursor = conn.cursor()
    
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
    
    cursor.execute(query)
    results = cursor.fetchall()
    conn.close()
    
    return results

def format_percentage(value):
    """Format decimal as percentage."""
    return f"{value:.1%}"

def print_table(headers, rows, title=""):
    """Print a simple table."""
    if title:
        print(f"\n{title}")
        print("-" * len(title))
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_row = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(row_str)

def main():
    """Main function to display all database queries."""
    print("="*80)
    print("COMPREHENSIVE PII PROTECTION DATABASE QUERY RESULTS")
    print("="*80)
    
    # Query 1: All protection results
    results = query_all_protection_results()
    headers = ['Mechanism', 'Epsilon', 'Overall', 'Email', 'Phone', 'Address', 'Name', 'Samples']
    
    # Format results for display
    formatted_results = []
    for row in results:
        formatted_row = [
            row[0],  # mechanism
            f"{row[1]:.1f}",  # epsilon
            format_percentage(row[2]),  # overall
            format_percentage(row[3]),  # email
            format_percentage(row[4]),  # phone
            format_percentage(row[5]),  # address
            format_percentage(row[6]),  # name
            str(row[7])  # samples
        ]
        formatted_results.append(formatted_row)
    
    print_table(headers, formatted_results, "1. ALL PII PROTECTION RESULTS")
    
    # Query 2: Mechanism summary
    summary_results = query_mechanism_summary()
    summary_headers = ['Mechanism', 'Epsilon Count', 'Avg Overall', 'Avg Email', 'Avg Phone', 'Avg Address', 'Avg Name', 'Total Samples']
    
    formatted_summary = []
    for row in summary_results:
        formatted_row = [
            row[0],  # mechanism
            str(row[1]),  # num_epsilon_values
            format_percentage(row[2]),  # avg_overall
            format_percentage(row[3]),  # avg_email
            format_percentage(row[4]),  # avg_phone
            format_percentage(row[5]),  # avg_address
            format_percentage(row[6]),  # avg_name
            str(row[7])  # total_samples
        ]
        formatted_summary.append(formatted_row)
    
    print_table(summary_headers, formatted_summary, "2. MECHANISM SUMMARY (Average Performance)")
    
    # Key insights
    print("\n3. KEY INSIGHTS")
    print("-" * 20)
    
    if summary_results:
        # Best performing mechanism
        best_mechanism = summary_results[0]  # Already sorted by avg_overall_protection DESC
        print(f"🏆 Best Overall Mechanism: {best_mechanism[0]} ({format_percentage(best_mechanism[2])} average protection)")
        
        # Most consistent mechanism (lowest standard deviation)
        consistency_scores = []
        mechanisms = set(row[0] for row in results)
        
        for mechanism in mechanisms:
            mechanism_data = [row[2] for row in results if row[0] == mechanism]  # overall_protection values
            if len(mechanism_data) > 1:
                mean_val = sum(mechanism_data) / len(mechanism_data)
                variance = sum((x - mean_val) ** 2 for x in mechanism_data) / len(mechanism_data)
                std_dev = variance ** 0.5
                consistency_scores.append((mechanism, std_dev))
        
        if consistency_scores:
            most_consistent = min(consistency_scores, key=lambda x: x[1])
            print(f"📊 Most Consistent Mechanism: {most_consistent[0]} (std: {most_consistent[1]:.4f})")
        
        # Total statistics
        total_experiments = len(results)
        total_mechanisms = len(mechanisms)
        total_samples = sum(row[7] for row in results)
        print(f"📈 Total Experiments: {total_experiments}")
        print(f"🔬 Total Mechanisms: {total_mechanisms}")
        print(f"📝 Total Samples: {total_samples}")
    
    print("\n✅ Database query completed successfully!")

if __name__ == "__main__":
    main()
