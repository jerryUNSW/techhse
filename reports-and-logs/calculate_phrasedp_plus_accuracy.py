#!/usr/bin/env python3
"""
Calculate Full PhraseDP+ Accuracy
=================================

Calculate the projected accuracy of PhraseDP+ by combining:
1. Questions that original PhraseDP got correct (stays correct)
2. Questions that medical mode improved (wrong -> correct)

Author: Tech4HSE Team
Date: 2025-01-30
"""

import sqlite3

def calculate_phrasedp_plus_accuracy():
    """Calculate PhraseDP+ accuracy for each epsilon."""
    
    conn = sqlite3.connect('tech4hse_results.db')
    cursor = conn.cursor()
    
    print("="*80)
    print("PhraseDP+ PROJECTED ACCURACY CALCULATION")
    print("="*80)
    print()
    
    results = {}
    
    for epsilon in [1.0, 2.0, 3.0]:
        print(f"\n{'='*80}")
        print(f"EPSILON = {epsilon}")
        print(f"{'='*80}")
        
        # 1. Get total questions
        cursor.execute("""
            SELECT COUNT(DISTINCT question_id)
            FROM medqa_detailed_results 
            WHERE epsilon = ? AND mechanism = 'Private Local Model + CoT (Old Phrase DP)'
        """, (epsilon,))
        
        total_questions = cursor.fetchone()[0]
        print(f"Total Questions: {total_questions}")
        
        # 2. Get questions PhraseDP got correct
        cursor.execute("""
            SELECT COUNT(*), question_id
            FROM medqa_detailed_results 
            WHERE epsilon = ? 
            AND mechanism = 'Private Local Model + CoT (Old Phrase DP)' 
            AND is_correct = 1
        """, (epsilon,))
        
        phrasedp_correct_count = cursor.fetchone()[0]
        print(f"\n1. Original PhraseDP Correct: {phrasedp_correct_count}/{total_questions}")
        
        # 3. Get questions that medical mode improved (was wrong, now correct)
        cursor.execute("""
            SELECT COUNT(*)
            FROM medical_improvement_results 
            WHERE epsilon = ? AND improvement = 1
        """, (epsilon,))
        
        improvements = cursor.fetchone()[0]
        print(f"2. Medical Mode Improvements: {improvements}")
        
        # 4. Calculate PhraseDP+ total correct
        phrasedp_plus_correct = phrasedp_correct_count + improvements
        phrasedp_plus_accuracy = (phrasedp_plus_correct / total_questions * 100)
        
        print(f"\n{'─'*80}")
        print(f"PhraseDP+ Projected Performance:")
        print(f"{'─'*80}")
        print(f"Total Correct: {phrasedp_plus_correct}/{total_questions}")
        print(f"Accuracy: {phrasedp_plus_accuracy:.2f}%")
        
        # 5. Get original PhraseDP accuracy for comparison
        phrasedp_accuracy = (phrasedp_correct_count / total_questions * 100)
        improvement_points = phrasedp_plus_accuracy - phrasedp_accuracy
        
        print(f"\nComparison:")
        print(f"  Original PhraseDP: {phrasedp_accuracy:.2f}%")
        print(f"  PhraseDP+:         {phrasedp_plus_accuracy:.2f}%")
        print(f"  Improvement:       +{improvement_points:.2f} percentage points")
        
        # 6. Compare with other mechanisms
        print(f"\n{'─'*80}")
        print(f"Comparison with Other Mechanisms:")
        print(f"{'─'*80}")
        
        # Get Local + CoT accuracy
        cursor.execute("""
            SELECT COUNT(*)
            FROM medqa_detailed_results 
            WHERE epsilon = ? 
            AND mechanism = 'Non-Private Local Model + Remote CoT' 
            AND is_correct = 1
        """, (epsilon,))
        
        local_cot_correct = cursor.fetchone()[0]
        local_cot_accuracy = (local_cot_correct / total_questions * 100)
        
        # Get Remote accuracy
        cursor.execute("""
            SELECT COUNT(*)
            FROM medqa_detailed_results 
            WHERE epsilon = ? 
            AND mechanism = 'Purely Remote Model' 
            AND is_correct = 1
        """, (epsilon,))
        
        remote_correct = cursor.fetchone()[0]
        remote_accuracy = (remote_correct / total_questions * 100)
        
        # Get InferDPT accuracy
        cursor.execute("""
            SELECT COUNT(*)
            FROM medqa_detailed_results 
            WHERE epsilon = ? 
            AND mechanism = 'Private Local Model + CoT (InferDPT)' 
            AND is_correct = 1
        """, (epsilon,))
        
        inferdpt_correct = cursor.fetchone()[0]
        inferdpt_accuracy = (inferdpt_correct / total_questions * 100)
        
        # Get SANTEXT+ accuracy
        cursor.execute("""
            SELECT COUNT(*)
            FROM medqa_detailed_results 
            WHERE epsilon = ? 
            AND mechanism = 'Private Local Model + CoT (SANTEXT+)' 
            AND is_correct = 1
        """, (epsilon,))
        
        santext_correct = cursor.fetchone()[0]
        santext_accuracy = (santext_correct / total_questions * 100)
        
        print(f"Local + CoT:       {local_cot_accuracy:.2f}% ({local_cot_correct}/{total_questions})")
        print(f"PhraseDP:          {phrasedp_accuracy:.2f}% ({phrasedp_correct_count}/{total_questions})")
        print(f"PhraseDP+:         {phrasedp_plus_accuracy:.2f}% ({phrasedp_plus_correct}/{total_questions}) ⭐")
        print(f"InferDPT:          {inferdpt_accuracy:.2f}% ({inferdpt_correct}/{total_questions})")
        print(f"SANTEXT+:          {santext_accuracy:.2f}% ({santext_correct}/{total_questions})")
        print(f"Remote:            {remote_accuracy:.2f}% ({remote_correct}/{total_questions})")
        
        # Store results
        results[epsilon] = {
            'total_questions': total_questions,
            'phrasedp_correct': phrasedp_correct_count,
            'improvements': improvements,
            'phrasedp_plus_correct': phrasedp_plus_correct,
            'phrasedp_accuracy': phrasedp_accuracy,
            'phrasedp_plus_accuracy': phrasedp_plus_accuracy,
            'improvement_points': improvement_points,
            'local_cot_accuracy': local_cot_accuracy,
            'inferdpt_accuracy': inferdpt_accuracy,
            'santext_accuracy': santext_accuracy,
            'remote_accuracy': remote_accuracy
        }
    
    conn.close()
    
    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Epsilon':<10} {'PhraseDP':<12} {'PhraseDP+':<12} {'Improvement':<15} {'Local+CoT':<12}")
    print(f"{'─'*80}")
    
    for epsilon in [1.0, 2.0, 3.0]:
        r = results[epsilon]
        print(f"{epsilon:<10.1f} {r['phrasedp_accuracy']:>10.2f}% {r['phrasedp_plus_accuracy']:>10.2f}% "
              f"{r['improvement_points']:>13.2f}pp {r['local_cot_accuracy']:>10.2f}%")
    
    print(f"\n{'='*80}")
    print("KEY FINDINGS")
    print(f"{'='*80}")
    
    for epsilon in [1.0, 2.0, 3.0]:
        r = results[epsilon]
        gap_to_local = r['local_cot_accuracy'] - r['phrasedp_plus_accuracy']
        
        if gap_to_local < 0:
            print(f"✅ Epsilon {epsilon}: PhraseDP+ EXCEEDS Local+CoT by {abs(gap_to_local):.2f}pp!")
        elif gap_to_local < 2:
            print(f"✅ Epsilon {epsilon}: PhraseDP+ nearly matches Local+CoT (gap: {gap_to_local:.2f}pp)")
        else:
            print(f"📊 Epsilon {epsilon}: PhraseDP+ trails Local+CoT by {gap_to_local:.2f}pp")
    
    return results

if __name__ == "__main__":
    results = calculate_phrasedp_plus_accuracy()
    print("\n✅ PhraseDP+ accuracy calculation complete!")

