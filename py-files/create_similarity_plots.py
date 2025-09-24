import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

def parse_similarity_data(filename):
    """Parse similarity data from the testing-phraseDP.txt file."""
    with open(filename, 'r') as f:
        content = f.read()
    
    # Split by questions
    questions = content.split('QUESTION ')[1:]  # Skip the header
    
    similarity_data = {}
    
    for question_block in questions:
        # Extract question number and original question
        lines = question_block.split('\n')
        question_num = lines[0].split('/')[0]
        original_question = ""
        
        # Find the original question
        for line in lines:
            if line.startswith('Original Question:'):
                original_question = line.replace('Original Question:', '').strip()
                break
        
        # Extract similarities
        similarities = []
        for line in lines:
            if 'Similarity:' in line:
                similarity_match = re.search(r'Similarity: ([\d.]+)', line)
                if similarity_match:
                    similarities.append(float(similarity_match.group(1)))
        
        if similarities:
            similarity_data[question_num] = {
                'original_question': original_question,
                'similarities': similarities
            }
    
    return similarity_data

def create_distribution_plots(similarity_data):
    """Create distribution plots for each question."""
    num_questions = len(similarity_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    # Colors for the plots
    colors = plt.cm.Set3(np.linspace(0, 1, num_questions))
    
    for i, (question_num, data) in enumerate(similarity_data.items()):
        similarities = data['similarities']
        original_question = data['original_question']
        
        # Create histogram
        ax = axes[i]
        ax.hist(similarities, bins=15, alpha=0.7, color=colors[i], edgecolor='black')
        
        # Add statistics
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        
        # Add vertical lines for mean and range
        ax.axvline(mean_sim, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_sim:.3f}')
        ax.axvline(min_sim, color='green', linestyle=':', linewidth=1, label=f'Min: {min_sim:.3f}')
        ax.axvline(max_sim, color='blue', linestyle=':', linewidth=1, label=f'Max: {max_sim:.3f}')
        
        # Customize plot
        ax.set_title(f'Q{question_num}: Similarity Distribution\n({len(similarities)} candidates)', fontsize=10)
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Mean: {mean_sim:.3f}\nStd: {std_sim:.3f}\nRange: {max_sim-min_sim:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('similarity_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_summary_plot(similarity_data):
    """Create a summary plot showing all questions together."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Box plot
    labels = []
    data_for_box = []
    
    for question_num, data in similarity_data.items():
        labels.append(f'Q{question_num}')
        data_for_box.append(data['similarities'])
    
    ax1.boxplot(data_for_box, labels=labels)
    ax1.set_title('Similarity Score Distribution by Question')
    ax1.set_ylabel('Similarity Score')
    ax1.set_xlabel('Question Number')
    ax1.grid(True, alpha=0.3)
    
    # Violin plot
    ax2.violinplot(data_for_box, positions=range(1, len(labels) + 1))
    ax2.set_title('Similarity Score Density by Question')
    ax2.set_ylabel('Similarity Score')
    ax2.set_xlabel('Question Number')
    ax2.set_xticks(range(1, len(labels) + 1))
    ax2.set_xticklabels(labels)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('similarity_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def print_statistics(similarity_data):
    """Print summary statistics for each question."""
    print("=" * 80)
    print("SIMILARITY SCORE STATISTICS BY QUESTION")
    print("=" * 80)
    
    for question_num, data in similarity_data.items():
        similarities = data['similarities']
        original_question = data['original_question']
        
        print(f"\nQuestion {question_num}:")
        print(f"Original: {original_question}")
        print(f"Number of candidates: {len(similarities)}")
        print(f"Mean similarity: {np.mean(similarities):.3f}")
        print(f"Std deviation: {np.std(similarities):.3f}")
        print(f"Min similarity: {np.min(similarities):.3f}")
        print(f"Max similarity: {np.max(similarities):.3f}")
        print(f"Range: {np.max(similarities) - np.min(similarities):.3f}")
        print("-" * 60)

if __name__ == "__main__":
    # Parse the data
    similarity_data = parse_similarity_data('testing-phraseDP.txt')
    
    # Print statistics
    print_statistics(similarity_data)
    
    # Create plots
    print("\nCreating distribution plots...")
    create_distribution_plots(similarity_data)
    
    print("\nCreating summary plots...")
    create_summary_plot(similarity_data)
    
    print("\nPlots saved as 'similarity_distributions.png' and 'similarity_summary.png'")
