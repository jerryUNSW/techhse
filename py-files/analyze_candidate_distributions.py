#!/usr/bin/env python3
"""
Analyze candidate similarity distributions to test the hypothesis that
candidates aren't diverse enough for proper epsilon sensitivity.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_candidate_distributions():
    """Analyze candidate similarity distributions to understand epsilon sensitivity issues."""
    print("🔍 CANDIDATE DISTRIBUTION ANALYSIS")
    print("=" * 80)
    
    # Load results
    with open('scaled_epsilon_comparison_results_20250920_170437.json', 'r') as f:
        results = json.load(f)
    
    epsilon_values = results['epsilon_values']
    questions = results['questions']
    
    print(f"Hypothesis: Candidates aren't diverse enough for proper epsilon sensitivity")
    print(f"Specifically: Lowest similarity candidates are still too similar for epsilon=0.5")
    print()
    
    # Analyze candidate distributions
    all_old_candidates = []
    all_new_candidates = []
    epsilon_analysis = {eps: {'old': [], 'new': []} for eps in epsilon_values}
    
    for question_data in questions:
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                epsilon = eps_test['epsilon']
                
                # Collect all candidate similarities
                old_candidates = eps_test['old_method']['candidate_similarities']
                new_candidates = eps_test['new_method']['candidate_similarities']
                
                all_old_candidates.extend(old_candidates)
                all_new_candidates.extend(new_candidates)
                
                epsilon_analysis[epsilon]['old'].extend(old_candidates)
                epsilon_analysis[epsilon]['new'].extend(new_candidates)
    
    # Overall distribution analysis
    print("📊 OVERALL CANDIDATE DISTRIBUTION")
    print("-" * 50)
    
    old_min, old_max = min(all_old_candidates), max(all_old_candidates)
    new_min, new_max = min(all_new_candidates), max(all_new_candidates)
    old_mean, new_mean = np.mean(all_old_candidates), np.mean(all_new_candidates)
    old_std, new_std = np.std(all_old_candidates), np.std(all_new_candidates)
    
    print(f"Old Method Candidates:")
    print(f"  Range: {old_min:.3f} to {old_max:.3f} (span: {old_max - old_min:.3f})")
    print(f"  Mean: {old_mean:.3f} ± {old_std:.3f}")
    print(f"  Percentiles: 5%={np.percentile(all_old_candidates, 5):.3f}, 25%={np.percentile(all_old_candidates, 25):.3f}, 75%={np.percentile(all_old_candidates, 75):.3f}, 95%={np.percentile(all_old_candidates, 95):.3f}")
    print()
    
    print(f"New Method Candidates:")
    print(f"  Range: {new_min:.3f} to {new_max:.3f} (span: {new_max - new_min:.3f})")
    print(f"  Mean: {new_mean:.3f} ± {new_std:.3f}")
    print(f"  Percentiles: 5%={np.percentile(all_new_candidates, 5):.3f}, 25%={np.percentile(all_new_candidates, 25):.3f}, 75%={np.percentile(all_new_candidates, 75):.3f}, 95%={np.percentile(all_new_candidates, 95):.3f}")
    print()
    
    # Test the hypothesis: Are candidates diverse enough for epsilon=0.5?
    print("🎯 HYPOTHESIS TESTING: EPSILON=0.5 SENSITIVITY")
    print("-" * 50)
    
    # For epsilon=0.5, we expect to select candidates with lower similarity
    # If the lowest candidates are still too similar, epsilon=0.5 won't work properly
    
    old_lowest_5_percent = np.percentile(all_old_candidates, 5)
    new_lowest_5_percent = np.percentile(all_new_candidates, 5)
    old_lowest_10_percent = np.percentile(all_old_candidates, 10)
    new_lowest_10_percent = np.percentile(all_new_candidates, 10)
    
    print(f"Lowest 5% of candidates:")
    print(f"  Old Method: {old_lowest_5_percent:.3f}")
    print(f"  New Method: {new_lowest_5_percent:.3f}")
    print()
    
    print(f"Lowest 10% of candidates:")
    print(f"  Old Method: {old_lowest_10_percent:.3f}")
    print(f"  New Method: {new_lowest_10_percent:.3f}")
    print()
    
    # Expected similarity ranges for different epsilon values
    print("📈 EXPECTED SIMILARITY RANGES FOR EPSILON VALUES")
    print("-" * 50)
    print("For proper epsilon sensitivity, we need:")
    print("  Epsilon=0.5: Should select candidates with similarity ~0.2-0.4 (low privacy, high utility)")
    print("  Epsilon=1.0: Should select candidates with similarity ~0.4-0.6 (medium)")
    print("  Epsilon=1.5: Should select candidates with similarity ~0.6-0.8 (high privacy, low utility)")
    print("  Epsilon=2.0: Should select candidates with similarity ~0.8-0.9 (very high privacy)")
    print()
    
    # Check if we have candidates in the expected ranges
    print("🔍 AVAILABILITY OF CANDIDATES IN EXPECTED RANGES")
    print("-" * 50)
    
    ranges = [
        (0.0, 0.3, "Very Low (ε=0.5)"),
        (0.3, 0.5, "Low (ε=0.5-1.0)"),
        (0.5, 0.7, "Medium (ε=1.0-1.5)"),
        (0.7, 0.9, "High (ε=1.5-2.0)"),
        (0.9, 1.0, "Very High (ε=2.0+)")
    ]
    
    for min_sim, max_sim, description in ranges:
        old_count = sum(1 for x in all_old_candidates if min_sim <= x < max_sim)
        new_count = sum(1 for x in all_new_candidates if min_sim <= x < max_sim)
        old_pct = (old_count / len(all_old_candidates)) * 100
        new_pct = (new_count / len(all_new_candidates)) * 100
        
        print(f"{description}:")
        print(f"  Old: {old_count:4d} candidates ({old_pct:5.1f}%)")
        print(f"  New: {new_count:4d} candidates ({new_pct:5.1f}%)")
        print()
    
    # Analyze what epsilon=0.5 actually selects
    print("🎯 WHAT EPSILON=0.5 ACTUALLY SELECTS")
    print("-" * 50)
    
    epsilon_05_old = epsilon_analysis[0.5]['old']
    epsilon_05_new = epsilon_analysis[0.5]['new']
    
    if epsilon_05_old and epsilon_05_new:
        old_selected_mean = np.mean(epsilon_05_old)
        new_selected_mean = np.mean(epsilon_05_new)
        old_selected_min = min(epsilon_05_old)
        new_selected_min = min(epsilon_05_new)
        
        print(f"Epsilon=0.5 Selected Similarities:")
        print(f"  Old Method: Mean={old_selected_mean:.3f}, Min={old_selected_min:.3f}")
        print(f"  New Method: Mean={new_selected_mean:.3f}, Min={new_selected_min:.3f}")
        print()
        
        print(f"Expected for ε=0.5: ~0.2-0.4 (low privacy, high utility)")
        print(f"Actual for ε=0.5: {old_selected_mean:.3f} (old), {new_selected_mean:.3f} (new)")
        print()
        
        if old_selected_mean > 0.4 or new_selected_mean > 0.4:
            print("🚨 HYPOTHESIS CONFIRMED: Epsilon=0.5 is selecting candidates that are too similar!")
            print("   This explains why epsilon sensitivity is poor.")
        else:
            print("✅ HYPOTHESIS REJECTED: Epsilon=0.5 is selecting appropriately low similarity candidates.")
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 50)
    
    if old_lowest_5_percent > 0.3 or new_lowest_5_percent > 0.3:
        print("🔧 CANDIDATE GENERATION NEEDS IMPROVEMENT:")
        print("   1. Generate more diverse candidates with lower similarity scores")
        print("   2. Target similarity ranges: 0.1-0.9 instead of current 0.3-0.9")
        print("   3. Use more aggressive anonymization for low-similarity candidates")
        print("   4. Implement explicit similarity targeting in prompts")
    else:
        print("✅ CANDIDATE DIVERSITY IS ADEQUATE")
        print("   The problem may be in the exponential mechanism implementation itself.")
    
    print()
    print("🔧 EXPONENTIAL MECHANISM CALIBRATION:")
    print("   1. Check if the exponential mechanism is properly using epsilon")
    print("   2. Verify probability calculations match differential privacy theory")
    print("   3. Consider temperature scaling or different utility functions")
    print("   4. Test with synthetic data where epsilon sensitivity is known")

if __name__ == "__main__":
    try:
        analyze_candidate_distributions()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
