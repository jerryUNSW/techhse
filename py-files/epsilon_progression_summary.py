import re

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
                        'epsilon': epsilon,
                        'similarity': similarity,
                        'replacement': replacement
                    })
        
        results[original_question] = epsilon_results
    
    return results

def display_epsilon_progression(results):
    """Display the epsilon progression for each question."""
    print("=" * 100)
    print("EPSILON PROGRESSION ANALYSIS")
    print("=" * 100)
    
    for i, (original_question, epsilon_data) in enumerate(results.items(), 1):
        print(f"\nQUESTION {i}: {original_question}")
        print("-" * 100)
        print(f"{'Epsilon':<8} {'Similarity':<12} {'Privacy Level':<15} {'Selected Replacement'}")
        print("-" * 100)
        
        for data in epsilon_data:
            epsilon = data['epsilon']
            similarity = float(data['similarity'])
            replacement = data['replacement']
            
            # Determine privacy level
            if similarity < 0.4:
                privacy_level = "HIGH PRIVACY"
            elif similarity < 0.7:
                privacy_level = "MODERATE"
            else:
                privacy_level = "LOW PRIVACY"
            
            print(f"{epsilon:<8} {similarity:<12.4f} {privacy_level:<15} {replacement}")
        
        print()

def display_privacy_transitions(results):
    """Show where privacy transitions occur."""
    print("\n" + "=" * 100)
    print("PRIVACY TRANSITION ANALYSIS")
    print("=" * 100)
    
    for i, (original_question, epsilon_data) in enumerate(results.items(), 1):
        print(f"\nQUESTION {i}: {original_question}")
        print("-" * 100)
        
        prev_similarity = None
        transitions = []
        
        for data in epsilon_data:
            epsilon = data['epsilon']
            similarity = float(data['similarity'])
            
            if prev_similarity is not None:
                diff = similarity - prev_similarity
                if abs(diff) > 0.1:  # Significant change
                    direction = "↑" if diff > 0 else "↓"
                    transitions.append(f"Epsilon {epsilon}: {direction} {abs(diff):.3f} change")
            
            prev_similarity = similarity
        
        if transitions:
            print("Significant privacy transitions:")
            for transition in transitions:
                print(f"  {transition}")
        else:
            print("No significant privacy transitions detected")
        
        print()

if __name__ == "__main__":
    # Parse the results
    results = parse_epsilon_results('epsilon_experiment_results.txt')
    
    # Display progression
    display_epsilon_progression(results)
    
    # Display transitions
    display_privacy_transitions(results)
