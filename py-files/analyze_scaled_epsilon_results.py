#!/usr/bin/env python3
"""
Comprehensive analysis of scaled epsilon comparison results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def analyze_scaled_epsilon_results():
    """Analyze the scaled epsilon comparison results comprehensively."""
    print("🔬 SCALED EPSILON COMPARISON ANALYSIS")
    print("=" * 80)
    
    # Load results
    with open('scaled_epsilon_comparison_results_20250920_170437.json', 'r') as f:
        results = json.load(f)
    
    epsilon_values = results['epsilon_values']
    questions = results['questions']
    
    print(f"📊 Dataset Overview:")
    print(f"  - Questions tested: {len(questions)}")
    print(f"  - Epsilon values: {epsilon_values}")
    print(f"  - Old method: {results['test_parameters']['old_method']['total_candidates']} candidates")
    print(f"  - New method: {results['test_parameters']['new_method']['total_candidates']} candidates")
    print(f"  - Total tests: {len(questions) * len(epsilon_values) * 2}")
    print()
    
    # Extract data for analysis
    old_similarities = []
    new_similarities = []
    old_candidate_ranges = []
    new_candidate_ranges = []
    epsilon_data = {eps: {'old': [], 'new': []} for eps in epsilon_values}
    
    for question_data in questions:
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                epsilon = eps_test['epsilon']
                
                # Selected similarities
                old_sim = eps_test['old_method']['similarity_to_original']
                new_sim = eps_test['new_method']['similarity_to_original']
                old_similarities.append(old_sim)
                new_similarities.append(new_sim)
                epsilon_data[epsilon]['old'].append(old_sim)
                epsilon_data[epsilon]['new'].append(new_sim)
                
                # Candidate diversity (range)
                old_candidates = eps_test['old_method']['candidate_similarities']
                new_candidates = eps_test['new_method']['candidate_similarities']
                old_range = max(old_candidates) - min(old_candidates) if old_candidates else 0
                new_range = max(new_candidates) - min(new_candidates) if new_candidates else 0
                old_candidate_ranges.append(old_range)
                new_candidate_ranges.append(new_range)
    
    # 1. OVERALL PERFORMANCE COMPARISON
    print("📈 OVERALL PERFORMANCE COMPARISON")
    print("-" * 50)
    
    old_mean_sim = np.mean(old_similarities)
    new_mean_sim = np.mean(new_similarities)
    old_std_sim = np.std(old_similarities)
    new_std_sim = np.std(new_similarities)
    
    old_mean_range = np.mean(old_candidate_ranges)
    new_mean_range = np.mean(new_candidate_ranges)
    old_std_range = np.std(old_candidate_ranges)
    new_std_range = np.std(new_candidate_ranges)
    
    print(f"Selected Similarity:")
    print(f"  Old Method: {old_mean_sim:.3f} ± {old_std_sim:.3f}")
    print(f"  New Method: {new_mean_sim:.3f} ± {new_std_sim:.3f}")
    print(f"  Difference: {new_mean_sim - old_mean_sim:+.3f}")
    print()
    
    print(f"Candidate Diversity (Range):")
    print(f"  Old Method: {old_mean_range:.3f} ± {old_std_range:.3f}")
    print(f"  New Method: {new_mean_range:.3f} ± {new_std_range:.3f}")
    print(f"  Improvement: {((new_mean_range - old_mean_range) / old_mean_range * 100):+.1f}%")
    print()
    
    # 2. EPSILON SENSITIVITY ANALYSIS
    print("🎯 EPSILON SENSITIVITY ANALYSIS")
    print("-" * 50)
    
    for epsilon in epsilon_values:
        old_eps_sims = epsilon_data[epsilon]['old']
        new_eps_sims = epsilon_data[epsilon]['new']
        
        old_mean = np.mean(old_eps_sims)
        new_mean = np.mean(new_eps_sims)
        old_std = np.std(old_eps_sims)
        new_std = np.std(new_eps_sims)
        
        print(f"Epsilon {epsilon}:")
        print(f"  Old: {old_mean:.3f} ± {old_std:.3f} (n={len(old_eps_sims)})")
        print(f"  New: {new_mean:.3f} ± {new_std:.3f} (n={len(new_eps_sims)})")
        print(f"  Difference: {new_mean - old_mean:+.3f}")
        print()
    
    # 3. EPSILON CORRELATION ANALYSIS
    print("📊 EPSILON CORRELATION ANALYSIS")
    print("-" * 50)
    
    # Calculate correlation between epsilon and selected similarity
    eps_array = []
    old_sim_array = []
    new_sim_array = []
    
    for question_data in questions:
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                eps_array.append(eps_test['epsilon'])
                old_sim_array.append(eps_test['old_method']['similarity_to_original'])
                new_sim_array.append(eps_test['new_method']['similarity_to_original'])
    
    old_corr = np.corrcoef(eps_array, old_sim_array)[0, 1]
    new_corr = np.corrcoef(eps_array, new_sim_array)[0, 1]
    
    print(f"Epsilon-Similarity Correlation:")
    print(f"  Old Method: {old_corr:.3f}")
    print(f"  New Method: {new_corr:.3f}")
    print(f"  Expected: Positive correlation (higher epsilon → higher similarity)")
    print()
    
    # 4. QUESTION-BY-QUESTION ANALYSIS
    print("📝 QUESTION-BY-QUESTION ANALYSIS")
    print("-" * 50)
    
    question_improvements = []
    for i, question_data in enumerate(questions):
        question_text = question_data['question_text']
        old_sims = []
        new_sims = []
        old_ranges = []
        new_ranges = []
        
        for eps_test in question_data['epsilon_tests']:
            if 'error' not in eps_test:
                old_sims.append(eps_test['old_method']['similarity_to_original'])
                new_sims.append(eps_test['new_method']['similarity_to_original'])
                
                old_candidates = eps_test['old_method']['candidate_similarities']
                new_candidates = eps_test['new_method']['candidate_similarities']
                old_ranges.append(max(old_candidates) - min(old_candidates) if old_candidates else 0)
                new_ranges.append(max(new_candidates) - min(new_candidates) if new_candidates else 0)
        
        old_mean_sim = np.mean(old_sims)
        new_mean_sim = np.mean(new_sims)
        old_mean_range = np.mean(old_ranges)
        new_mean_range = np.mean(new_ranges)
        
        range_improvement = ((new_mean_range - old_mean_range) / old_mean_range * 100) if old_mean_range > 0 else 0
        question_improvements.append(range_improvement)
        
        print(f"Q{i+1:2d}: {question_text[:50]}...")
        print(f"     Similarity: {old_mean_sim:.3f} → {new_mean_sim:.3f} ({new_mean_sim - old_mean_sim:+.3f})")
        print(f"     Diversity:  {old_mean_range:.3f} → {new_mean_range:.3f} ({range_improvement:+.1f}%)")
        print()
    
    # 5. STATISTICAL SIGNIFICANCE
    print("📊 STATISTICAL SIGNIFICANCE")
    print("-" * 50)
    
    from scipy import stats
    
    # T-test for similarity differences
    t_stat_sim, p_val_sim = stats.ttest_rel(new_similarities, old_similarities)
    print(f"Similarity T-test:")
    print(f"  T-statistic: {t_stat_sim:.3f}")
    print(f"  P-value: {p_val_sim:.6f}")
    print(f"  Significant: {'Yes' if p_val_sim < 0.05 else 'No'} (α=0.05)")
    print()
    
    # T-test for diversity differences
    t_stat_range, p_val_range = stats.ttest_rel(new_candidate_ranges, old_candidate_ranges)
    print(f"Diversity T-test:")
    print(f"  T-statistic: {t_stat_range:.3f}")
    print(f"  P-value: {p_val_range:.6f}")
    print(f"  Significant: {'Yes' if p_val_range < 0.05 else 'No'} (α=0.05)")
    print()
    
    # 6. SUMMARY AND RECOMMENDATIONS
    print("🎯 SUMMARY AND RECOMMENDATIONS")
    print("-" * 50)
    
    print("Key Findings:")
    print(f"1. Candidate Diversity: New method shows {((new_mean_range - old_mean_range) / old_mean_range * 100):+.1f}% improvement")
    print(f"2. Selected Similarity: {'Higher' if new_mean_sim > old_mean_sim else 'Lower'} by {abs(new_mean_sim - old_mean_sim):.3f}")
    print(f"3. Epsilon Sensitivity: Old={old_corr:.3f}, New={new_corr:.3f} (both {'good' if abs(old_corr) > 0.3 and abs(new_corr) > 0.3 else 'poor'})")
    print(f"4. Statistical Significance: Diversity improvement is {'significant' if p_val_range < 0.05 else 'not significant'}")
    print()
    
    print("Recommendations:")
    if new_mean_range > old_mean_range:
        print("✅ Use the new diverse candidate generation method")
    else:
        print("❌ Consider further improvements to candidate diversity")
    
    if abs(old_corr) < 0.3 or abs(new_corr) < 0.3:
        print("⚠️  Epsilon sensitivity is poor - consider calibrating the exponential mechanism")
    else:
        print("✅ Epsilon sensitivity is adequate")
    
    if p_val_range < 0.05:
        print("✅ Diversity improvement is statistically significant")
    else:
        print("⚠️  Diversity improvement may not be statistically significant")
    
    print()
    print("Next Steps:")
    print("1. Use new method for production if diversity improvement is meaningful")
    print("2. Investigate epsilon calibration if sensitivity is poor")
    print("3. Test with more complex questions to validate improvements")
    print("4. Consider different similarity metrics or candidate generation strategies")

if __name__ == "__main__":
    try:
        analyze_scaled_epsilon_results()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
