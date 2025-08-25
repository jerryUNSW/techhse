import re
import numpy as np

def parse_epsilon_results(filename):
    """Parse epsilon experiment results and extract the progression."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by questions
    questions = content.split('ORIGINAL QUESTION:')[1:]
    
    results = {}
    
    for question_block in questions:
        lines = question_block.strip().split('\n')
        
        # Extract original question
        original_question = lines[0].strip()
        
        # Find epsilon results section
        epsilon_results = []
        in_epsilon_section = False
        
        for line in lines:
            if 'Epsilon  Similarity   Selected Replacement' in line:
                in_epsilon_section = True
                continue
            elif in_epsilon_section and line.startswith('ALL CANDIDATES'):
                break
            elif in_epsilon_section and line.strip() and not line.startswith('-'):
                # Parse epsilon line
                parts = line.split('     ')
                if len(parts) >= 3:
                    epsilon = parts[0].strip()
                    similarity = parts[1].strip()
                    replacement = parts[2].strip()
                    epsilon_results.append({
                        'epsilon': float(epsilon),
                        'similarity': float(similarity),
                        'replacement': replacement
                    })
        
        results[original_question] = epsilon_results
    
    return results

def analyze_smoothness(epsilon_data):
    """Analyze how smooth the epsilon progression is."""
    similarities = [data['similarity'] for data in epsilon_data]
    epsilons = [data['epsilon'] for data in epsilon_data]
    
    # Calculate differences between consecutive similarities
    differences = [abs(similarities[i] - similarities[i-1]) for i in range(1, len(similarities))]
    
    # Calculate statistics
    mean_diff = np.mean(differences)
    max_diff = np.max(differences)
    std_diff = np.std(differences)
    
    # Count significant jumps (>0.1 change)
    significant_jumps = sum(1 for diff in differences if diff > 0.1)
    
    return {
        'mean_difference': mean_diff,
        'max_difference': max_diff,
        'std_difference': std_diff,
        'significant_jumps': significant_jumps,
        'total_transitions': len(differences),
        'jump_percentage': significant_jumps / len(differences) * 100 if differences else 0
    }

def compare_experiments():
    """Compare the two experiments."""
    print("=" * 100)
    print("COMPARISON: 25 CANDIDATES vs 100 CANDIDATES")
    print("=" * 100)
    
    # Load both experiments
    results_25 = parse_epsilon_results('epsilon_experiment_results.txt')
    results_100 = parse_epsilon_results('epsilon_experiment_results_100candidates.txt')
    
    print(f"\n{'Question':<50} {'25 Candidates':<30} {'100 Candidates':<30}")
    print("-" * 110)
    
    for i, (question, data_25) in enumerate(results_25.items(), 1):
        if question in results_100:
            data_100 = results_100[question]
            
            # Analyze smoothness
            smoothness_25 = analyze_smoothness(data_25)
            smoothness_100 = analyze_smoothness(data_100)
            
            print(f"Q{i}: {question[:47]}...")
            print(f"{'':<50} Mean diff: {smoothness_25['mean_difference']:.3f}     Mean diff: {smoothness_100['mean_difference']:.3f}")
            print(f"{'':<50} Max diff:  {smoothness_25['max_difference']:.3f}     Max diff:  {smoothness_100['max_difference']:.3f}")
            print(f"{'':<50} Jumps:     {smoothness_25['significant_jumps']}/{smoothness_25['total_transitions']}     Jumps:     {smoothness_100['significant_jumps']}/{smoothness_100['total_transitions']}")
            print(f"{'':<50} Jump %:    {smoothness_25['jump_percentage']:.1f}%        Jump %:    {smoothness_100['jump_percentage']:.1f}%")
            print()
    
    # Overall comparison
    print("\n" + "=" * 100)
    print("OVERALL COMPARISON")
    print("=" * 100)
    
    all_smoothness_25 = []
    all_smoothness_100 = []
    
    for question in results_25:
        if question in results_100:
            all_smoothness_25.append(analyze_smoothness(results_25[question]))
            all_smoothness_100.append(analyze_smoothness(results_100[question]))
    
    avg_25 = {
        'mean_diff': np.mean([s['mean_difference'] for s in all_smoothness_25]),
        'max_diff': np.mean([s['max_difference'] for s in all_smoothness_25]),
        'jump_percentage': np.mean([s['jump_percentage'] for s in all_smoothness_25])
    }
    
    avg_100 = {
        'mean_diff': np.mean([s['mean_difference'] for s in all_smoothness_100]),
        'max_diff': np.mean([s['max_difference'] for s in all_smoothness_100]),
        'jump_percentage': np.mean([s['jump_percentage'] for s in all_smoothness_100])
    }
    
    print(f"25 Candidates - Average mean difference: {avg_25['mean_diff']:.3f}")
    print(f"100 Candidates - Average mean difference: {avg_100['mean_diff']:.3f}")
    print(f"Improvement: {((avg_25['mean_diff'] - avg_100['mean_diff']) / avg_25['mean_diff'] * 100):.1f}%")
    
    print(f"\n25 Candidates - Average max difference: {avg_25['max_diff']:.3f}")
    print(f"100 Candidates - Average max difference: {avg_100['max_diff']:.3f}")
    print(f"Improvement: {((avg_25['max_diff'] - avg_100['max_diff']) / avg_25['max_diff'] * 100):.1f}%")
    
    print(f"\n25 Candidates - Average jump percentage: {avg_25['jump_percentage']:.1f}%")
    print(f"100 Candidates - Average jump percentage: {avg_100['jump_percentage']:.1f}%")
    print(f"Improvement: {((avg_25['jump_percentage'] - avg_100['jump_percentage']) / avg_25['jump_percentage'] * 100):.1f}%")

def show_detailed_progression():
    """Show detailed progression for one question."""
    results_100 = parse_epsilon_results('epsilon_experiment_results_100candidates.txt')
    
    # Show first question in detail
    for question, data in results_100.items():
        print(f"\nDETAILED PROGRESSION: {question}")
        print("-" * 100)
        print(f"{'Epsilon':<8} {'Similarity':<12} {'Change':<12} {'Privacy Level'}")
        print("-" * 100)
        
        prev_similarity = None
        for item in data:
            epsilon = item['epsilon']
            similarity = item['similarity']
            
            if prev_similarity is not None:
                change = similarity - prev_similarity
                change_str = f"{change:+.3f}"
            else:
                change_str = "N/A"
            
            if similarity < 0.4:
                privacy_level = "HIGH"
            elif similarity < 0.7:
                privacy_level = "MODERATE"
            else:
                privacy_level = "LOW"
            
            print(f"{epsilon:<8.2f} {similarity:<12.4f} {change_str:<12} {privacy_level}")
            prev_similarity = similarity
        
        break  # Only show first question

if __name__ == "__main__":
    compare_experiments()
    show_detailed_progression()
