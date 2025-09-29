#!/usr/bin/env python3
"""
Update SQLite Database with Latest CluSanT Results

This script loads the latest CluSanT PII protection results and updates the SQLite database.
"""

import json
import sqlite3
import numpy as np
from datetime import datetime

def load_clusant_results():
    """Load the latest CluSanT results from JSON file."""
    filename = '/home/yizhang/tech4HSE/pii_protection_results_20250929_071204.json'
    
    print(f"Loading CluSanT results from: {filename}")
    with open(filename, 'r') as f:
        data = json.load(f)
    
    return data

def update_database(db_path, clusant_results):
    """Update the database with new CluSanT results."""
    
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # First, check if we need to create a new experiment entry
        experiment_type = 'pii_protection'
        timestamp = datetime.now().isoformat()
        description = 'CluSanT PII Protection Experiment (Fixed Regex Bug)'
        
        # Check if this experiment already exists
        cursor.execute("""
            SELECT id FROM experiments 
            WHERE experiment_type = ? AND description = ?
            ORDER BY created_at DESC LIMIT 1
        """, (experiment_type, description))
        
        experiment_row = cursor.fetchone()
        if experiment_row:
            experiment_id = experiment_row[0]
            print(f"Using existing experiment ID: {experiment_id}")
        else:
            # Create new experiment entry
            cursor.execute("""
                INSERT INTO experiments 
                (experiment_type, timestamp, description, total_questions, mechanisms, epsilon_values)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                experiment_type,
                timestamp,
                description,
                10,  # Number of samples in the experiment
                json.dumps(['CluSanT']),
                json.dumps([1.0, 1.5, 2.0, 2.5, 3.0])
            ))
            experiment_id = cursor.lastrowid
            print(f"Created new experiment with ID: {experiment_id}")
        
        # Process CluSanT results
        if 'CluSanT' in clusant_results:
            clusant_data = clusant_results['CluSanT']
            
            for epsilon_str, epsilon_data in clusant_data.items():
                epsilon = float(epsilon_str)
                
                print(f"Processing CluSanT results for epsilon {epsilon}...")
                
                # Extract protection rates
                overall_protection = float(epsilon_data.get('overall', 0.0))
                email_protection = float(epsilon_data.get('emails', 0.0))
                phone_protection = float(epsilon_data.get('phones', 0.0))
                address_protection = float(epsilon_data.get('addresses', 0.0))
                name_protection = float(epsilon_data.get('names', 0.0))
                
                # Count samples
                num_samples = len(epsilon_data.get('samples', []))
                
                # Insert or update PII protection results
                cursor.execute("""
                    INSERT OR REPLACE INTO pii_protection_results 
                    (experiment_id, mechanism, epsilon, overall_protection, email_protection, 
                     phone_protection, address_protection, name_protection, num_samples)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    'CluSanT',
                    epsilon,
                    overall_protection,
                    email_protection,
                    phone_protection,
                    address_protection,
                    name_protection,
                    num_samples
                ))
                
                protection_result_id = cursor.lastrowid
                print(f"  ✅ Updated protection results (ID: {protection_result_id})")
                
                # Insert sample data
                if 'samples' in epsilon_data:
                    # First, delete existing samples for this protection result
                    cursor.execute("""
                        DELETE FROM pii_protection_samples 
                        WHERE protection_result_id = ?
                    """, (protection_result_id,))
                    
                    # Insert new samples
                    for i, sample in enumerate(epsilon_data['samples']):
                        cursor.execute("""
                            INSERT INTO pii_protection_samples 
                            (protection_result_id, row_index, original_text, sanitized_text)
                            VALUES (?, ?, ?, ?)
                        """, (
                            protection_result_id,
                            sample.get('row', i),
                            sample.get('original', ''),
                            sample.get('sanitized', '')
                        ))
                    
                    print(f"  ✅ Inserted {len(epsilon_data['samples'])} samples")
        
        # Commit all changes
        conn.commit()
        print("✅ Database updated successfully!")
        
        # Print summary
        print("\n" + "="*60)
        print("DATABASE UPDATE SUMMARY")
        print("="*60)
        
        cursor.execute("""
            SELECT mechanism, epsilon, overall_protection, email_protection, 
                   phone_protection, address_protection, name_protection, num_samples
            FROM pii_protection_results 
            WHERE mechanism = 'CluSanT'
            ORDER BY epsilon
        """)
        
        results = cursor.fetchall()
        for row in results:
            mechanism, epsilon, overall, email, phone, address, name, samples = row
            print(f"\n{mechanism} (ε={epsilon}):")
            print(f"  Overall Protection: {overall:.3f}")
            print(f"  Email Protection:   {email:.3f}")
            print(f"  Phone Protection:   {phone:.3f}")
            print(f"  Address Protection: {address:.3f}")
            print(f"  Name Protection:    {name:.3f}")
            print(f"  Number of Samples:  {samples}")
        
    except Exception as e:
        print(f"❌ Error updating database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def main():
    """Main function to update the database."""
    print("=== Updating Database with Latest CluSanT Results ===")
    print()
    
    # Load CluSanT results
    clusant_results = load_clusant_results()
    print("✅ CluSanT results loaded successfully")
    print()
    
    # Update database
    db_path = '/home/yizhang/tech4HSE/tech4hse_results.db'
    update_database(db_path, clusant_results)
    
    print()
    print("🎉 Database update completed successfully!")

if __name__ == "__main__":
    main()
