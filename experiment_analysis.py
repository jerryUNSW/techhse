#!/usr/bin/env python3
"""
Comprehensive analysis of MedQA experiment results.
"""

import re
import json
from datetime import datetime

def analyze_experiment_results():
    """Analyze the current experiment results."""
    
    # Current results from the monitoring script
    results = {
        'Baseline (No Privacy)': {'correct': 299, 'total': 499, 'percentage': 59.92},
        'Non-Private CoT': {'correct': 415, 'total': 499, 'percentage': 83.17},
        'Old PhraseDP': {'correct': 372, 'total': 499, 'percentage': 74.55},
        'Old PhraseDP + Batch Options': {'correct': 294, 'total': 499, 'percentage': 58.92},
        'InferDPT + Batch Options': {'correct': 294, 'total': 499, 'percentage': 58.92},
        'SANTEXT+ + Batch Options': {'correct': 283, 'total': 498, 'percentage': 56.83},
        'Remote Model': {'correct': 447, 'total': 498, 'percentage': 89.76}
    }
    
    print("=" * 80)
    print("MEDQA EXPERIMENT ANALYSIS")
    print("=" * 80)
    print(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Questions Completed: 499/500 (99.8%)")
    print()
    
    # Sort methods by performance
    sorted_methods = sorted(results.items(), key=lambda x: x[1]['percentage'], reverse=True)
    
    print("PERFORMANCE RANKING:")
    print("-" * 50)
    for i, (method, stats) in enumerate(sorted_methods, 1):
        print(f"{i}. {method:<30} {stats['correct']:3d}/{stats['total']:3d} = {stats['percentage']:5.1f}%")
    
    print()
    
    # Performance Analysis
    print("PERFORMANCE ANALYSIS:")
    print("-" * 50)
    
    # Best and worst performers
    best_method = sorted_methods[0]
    worst_method = sorted_methods[-1]
    
    print(f"🏆 Best Performer: {best_method[0]} ({best_method[1]['percentage']:.1f}%)")
    print(f"📉 Worst Performer: {worst_method[0]} ({worst_method[1]['percentage']:.1f}%)")
    print(f"📊 Performance Gap: {best_method[1]['percentage'] - worst_method[1]['percentage']:.1f} percentage points")
    print()
    
    # Privacy vs Non-Privacy Analysis
    print("PRIVACY vs NON-PRIVACY ANALYSIS:")
    print("-" * 50)
    
    non_private_methods = ['Non-Private CoT', 'Remote Model']
    private_methods = [method for method in results.keys() if method not in non_private_methods]
    
    non_private_avg = sum(results[method]['percentage'] for method in non_private_methods) / len(non_private_methods)
    private_avg = sum(results[method]['percentage'] for method in private_methods) / len(private_methods)
    
    print(f"Non-Private Methods Average: {non_private_avg:.1f}%")
    print(f"Private Methods Average: {private_avg:.1f}%")
    print(f"Privacy Cost: {non_private_avg - private_avg:.1f} percentage points")
    print()
    
    # CoT Analysis
    print("CHAIN-OF-THOUGHT (CoT) ANALYSIS:")
    print("-" * 50)
    
    baseline = results['Baseline (No Privacy)']['percentage']
    non_private_cot = results['Non-Private CoT']['percentage']
    old_phrasedp = results['Old PhraseDP']['percentage']
    
    print(f"Baseline (No CoT): {baseline:.1f}%")
    print(f"Non-Private CoT: {non_private_cot:.1f}%")
    print(f"Private CoT (Old PhraseDP): {old_phrasedp:.1f}%")
    print()
    print(f"CoT Benefit (Non-Private): +{non_private_cot - baseline:.1f} percentage points")
    print(f"CoT Benefit (Private): +{old_phrasedp - baseline:.1f} percentage points")
    print(f"Privacy Cost of CoT: {non_private_cot - old_phrasedp:.1f} percentage points")
    print()
    
    # Batch Options Analysis
    print("BATCH OPTIONS ANALYSIS:")
    print("-" * 50)
    
    old_phrasedp_single = results['Old PhraseDP']['percentage']
    old_phrasedp_batch = results['Old PhraseDP + Batch Options']['percentage']
    inferdpt_batch = results['InferDPT + Batch Options']['percentage']
    santext_batch = results['SANTEXT+ + Batch Options']['percentage']
    
    print(f"Old PhraseDP (Single): {old_phrasedp_single:.1f}%")
    print(f"Old PhraseDP (Batch): {old_phrasedp_batch:.1f}%")
    print(f"InferDPT (Batch): {inferdpt_batch:.1f}%")
    print(f"SANTEXT+ (Batch): {santext_batch:.1f}%")
    print()
    print(f"Batch Options Cost (Old PhraseDP): {old_phrasedp_single - old_phrasedp_batch:.1f} percentage points")
    print(f"Best Batch Method: {'InferDPT' if inferdpt_batch > santext_batch else 'SANTEXT+'} ({max(inferdpt_batch, santext_batch):.1f}%)")
    print()
    
    # Privacy Mechanism Comparison
    print("PRIVACY MECHANISM COMPARISON:")
    print("-" * 50)
    
    privacy_mechanisms = {
        'Old PhraseDP': old_phrasedp_single,
        'Old PhraseDP + Batch': old_phrasedp_batch,
        'InferDPT + Batch': inferdpt_batch,
        'SANTEXT+ + Batch': santext_batch
    }
    
    sorted_privacy = sorted(privacy_mechanisms.items(), key=lambda x: x[1], reverse=True)
    
    for i, (mechanism, accuracy) in enumerate(sorted_privacy, 1):
        print(f"{i}. {mechanism:<25} {accuracy:5.1f}%")
    
    print()
    
    # Key Insights
    print("KEY INSIGHTS:")
    print("-" * 50)
    
    insights = []
    
    # Performance insights
    if best_method[1]['percentage'] > 85:
        insights.append("✅ Remote model achieves excellent performance (>85%)")
    
    if non_private_cot > baseline + 20:
        insights.append("✅ Non-private CoT provides significant benefit (+20% over baseline)")
    
    if old_phrasedp > baseline + 10:
        insights.append("✅ Private CoT still provides meaningful benefit (+10% over baseline)")
    
    if old_phrasedp_batch < old_phrasedp_single - 10:
        insights.append("⚠️  Batch options significantly hurt performance (-10% vs single)")
    
    if abs(inferdpt_batch - santext_batch) < 5:
        insights.append("📊 InferDPT and SANTEXT+ show similar performance")
    
    if private_avg < non_private_avg - 20:
        insights.append("🔒 Privacy comes at significant cost (-20% vs non-private)")
    
    for insight in insights:
        print(insight)
    
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    print("-" * 50)
    
    print("1. 🎯 For maximum accuracy: Use Remote Model (89.8%)")
    print("2. 🔒 For privacy with good performance: Use Old PhraseDP (74.5%)")
    print("3. ⚖️  For balanced privacy/performance: Use Non-Private CoT (83.2%)")
    print("4. 🚫 Avoid batch options for Old PhraseDP (significant performance drop)")
    print("5. 🔄 Consider single-option perturbation over batch processing")
    
    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze_experiment_results()

