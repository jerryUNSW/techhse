import matplotlib.pyplot as plt
import numpy as np
import json
import re
from collections import defaultdict

def extract_similarity_data_from_txt():
    """
    Extract similarity data from the text results file
    """
    with open('phrase_dp_comparison_results.txt', 'r') as f:
        content = f.read()
    
    # Split by questions
    questions = content.split('QUESTION ')[1:]  # Skip the header
    
    question_data = []
    
    for i, question_block in enumerate(questions):
        lines = question_block.split('\n')
        
        # Extract question text
        question_text = lines[2].replace('Original Question: ', '')
        
        # Find OLD and NEW sections
        old_similarities = []
        new_similarities = []
        
        in_old_section = False
        in_new_section = False
        
        for line in lines:
            if 'OLD PHRASE DP IMPLEMENTATION:' in line:
                in_old_section = True
                in_new_section = False
                continue
            elif 'NEW DIVERSE PHRASE DP IMPLEMENTATION:' in line:
                in_old_section = False
                in_new_section = True
                continue
            elif 'Similarity range:' in line and in_old_section:
                # Extract range info
                range_match = re.search(r'(\d+\.\d+) - (\d+\.\d+)', line)
                if range_match:
                    old_min, old_max = float(range_match.group(1)), float(range_match.group(2))
            elif 'Similarity range:' in line and in_new_section:
                # Extract range info
                range_match = re.search(r'(\d+\.\d+) - (\d+\.\d+)', line)
                if range_match:
                    new_min, new_max = float(range_match.group(1)), float(range_match.group(2))
            elif 'Similarity mean:' in line and in_old_section:
                old_mean = float(re.search(r'(\d+\.\d+)', line).group(1))
            elif 'Similarity mean:' in line and in_new_section:
                new_mean = float(re.search(r'(\d+\.\d+)', line).group(1))
            elif 'Similarity std:' in line and in_old_section:
                old_std = float(re.search(r'(\d+\.\d+)', line).group(1))
            elif 'Similarity std:' in line and in_new_section:
                new_std = float(re.search(r'(\d+\.\d+)', line).group(1))
            elif line.strip().startswith('[') and ']' in line and in_old_section:
                # Extract similarity value
                sim_match = re.search(r'\[(\d+\.\d+)\]', line)
                if sim_match:
                    old_similarities.append(float(sim_match.group(1)))
            elif line.strip().startswith('[') and ']' in line and in_new_section:
                # Extract similarity value
                sim_match = re.search(r'\[(\d+\.\d+)\]', line)
                if sim_match:
                    new_similarities.append(float(sim_match.group(1)))
        
        if old_similarities and new_similarities:
            question_data.append({
                'question_num': i + 1,
                'question_text': question_text,
                'old_similarities': old_similarities,
                'new_similarities': new_similarities,
                'old_range': (old_min, old_max),
                'new_range': (new_min, new_max),
                'old_mean': old_mean,
                'new_mean': new_mean,
                'old_std': old_std,
                'new_std': new_std
            })
    
    return question_data

def create_individual_plots(question_data):
    """
    Create individual distribution plots for each question
    """
    # Set up the plotting style
    plt.style.use('default')
    
    for data in question_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f'Question {data["question_num"]}: {data["question_text"][:60]}...', 
                     fontsize=14, fontweight='bold')
        
        # Plot 1: Histogram comparison
        ax1.hist(data['old_similarities'], bins=20, alpha=0.7, label='Old Implementation', 
                color='red', density=True, edgecolor='black', linewidth=0.5)
        ax1.hist(data['new_similarities'], bins=20, alpha=0.7, label='New Implementation', 
                color='blue', density=True, edgecolor='black', linewidth=0.5)
        ax1.set_xlabel('Similarity Score')
        ax1.set_ylabel('Density')
        ax1.set_title('Similarity Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        old_range = data['old_range'][1] - data['old_range'][0]
        new_range = data['new_range'][1] - data['new_range'][0]
        range_improvement = ((new_range - old_range) / old_range) * 100
        
        stats_text = f"""Old: Range={old_range:.3f}, Mean={data['old_mean']:.3f}, Std={data['old_std']:.3f}
New: Range={new_range:.3f}, Mean={data['new_mean']:.3f}, Std={data['new_std']:.3f}
Range Improvement: {range_improvement:+.1f}%"""
        
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Plot 2: Box plot comparison
        box_data = [data['old_similarities'], data['new_similarities']]
        box_plot = ax2.boxplot(box_data, labels=['Old', 'New'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('red')
        box_plot['boxes'][1].set_facecolor('blue')
        ax2.set_ylabel('Similarity Score')
        ax2.set_title('Similarity Score Box Plot')
        ax2.grid(True, alpha=0.3)
        
        # Add range comparison
        ax2.text(0.02, 0.98, f'Range Comparison:\nOld: {old_range:.3f}\nNew: {new_range:.3f}\nImprovement: {range_improvement:+.1f}%', 
                transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save individual plot
        filename = f'question_{data["question_num"]:02d}_distribution_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def create_summary_plot(question_data):
    """
    Create a summary plot showing range improvements for all questions
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Extract data for summary
    question_nums = []
    old_ranges = []
    new_ranges = []
    range_improvements = []
    old_means = []
    new_means = []
    
    for data in question_data:
        question_nums.append(data['question_num'])
        old_range = data['old_range'][1] - data['old_range'][0]
        new_range = data['new_range'][1] - data['new_range'][0]
        old_ranges.append(old_range)
        new_ranges.append(new_range)
        range_improvements.append(((new_range - old_range) / old_range) * 100)
        old_means.append(data['old_mean'])
        new_means.append(data['new_mean'])
    
    # Plot 1: Range comparison
    x_pos = np.arange(len(question_nums))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, old_ranges, width, label='Old Implementation', 
                   color='red', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, new_ranges, width, label='New Implementation', 
                   color='blue', alpha=0.7)
    
    ax1.set_xlabel('Question Number')
    ax1.set_ylabel('Similarity Range')
    ax1.set_title('Similarity Range Comparison Across All Questions')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'Q{i}' for i in question_nums])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add improvement percentages on bars
    for i, (old, new, improvement) in enumerate(zip(old_ranges, new_ranges, range_improvements)):
        ax1.text(i, max(old, new) + 0.01, f'{improvement:+.1f}%', 
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 2: Mean comparison
    bars3 = ax1.bar(x_pos - width/2, old_means, width, label='Old Mean', 
                   color='red', alpha=0.7)
    bars4 = ax1.bar(x_pos + width/2, new_means, width, label='New Mean', 
                   color='blue', alpha=0.7)
    
    ax2.set_xlabel('Question Number')
    ax2.set_ylabel('Mean Similarity')
    ax2.set_title('Mean Similarity Comparison Across All Questions')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Q{i}' for i in question_nums])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('summary_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: summary_distribution_comparison.png")
    plt.close()

if __name__ == "__main__":
    print("Extracting similarity data from results file...")
    question_data = extract_similarity_data_from_txt()
    
    print(f"Found data for {len(question_data)} questions")
    
    print("Creating individual distribution plots...")
    create_individual_plots(question_data)
    
    print("Creating summary plot...")
    create_summary_plot(question_data)
    
    print("\nAll plots created successfully!")
    print("Individual question plots: question_XX_distribution_comparison.png")
    print("Summary plot: summary_distribution_comparison.png")

