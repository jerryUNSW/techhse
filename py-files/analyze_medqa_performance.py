#!/usr/bin/env python3
"""
MedQA Performance Analysis Script
================================

This script analyzes the performance of different methods from the MedQA experiment
results stored in xxx-500-qa file.

Author: Tech4HSE Team
Date: 2025-08-26
"""

import re
import json
from collections import defaultdict
from datetime import datetime

# Try to import matplotlib, but don't fail if it's not available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Chart generation will be skipped.")

class MedQAPerformanceAnalyzer:
    """Analyzer for MedQA experiment performance results."""
    
    def __init__(self, results_file="xxx-500-qa"):
        self.results_file = results_file
        self.results = {
            'total_questions': 0,
            'scenarios': {
                'local_alone': {'correct': 0, 'incorrect': 0},
                'non_private_cot': {'correct': 0, 'incorrect': 0},
                'phrase_dp_cot': {'correct': 0, 'incorrect': 0},
                'inferdpt_cot': {'correct': 0, 'incorrect': 0},
                'purely_remote': {'correct': 0, 'incorrect': 0}
            },
            'question_details': []
        }
    
    def parse_results_file(self):
        """Parse the results file and extract performance metrics."""
        print(f"Parsing results file: {self.results_file}")
        
        current_question = None
        current_scenario = None
        
        with open(self.results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Extract question number
                question_match = re.search(r'Question (\d+)/500', line)
                if question_match:
                    current_question = int(question_match.group(1))
                    self.results['total_questions'] = max(self.results['total_questions'], current_question)
                    print(f"Processing Question {current_question}/500...")
                    continue
                
                # Extract scenario information
                if 'Scenario 1:' in line:
                    current_scenario = 'local_alone'
                elif 'Scenario 2:' in line:
                    current_scenario = 'non_private_cot'
                elif 'Scenario 3.1:' in line:
                    current_scenario = 'phrase_dp_cot'
                elif 'Scenario 3.2:' in line:
                    current_scenario = 'inferdpt_cot'
                elif 'Scenario 4:' in line:
                    current_scenario = 'purely_remote'
                
                # Extract result (Correct/Incorrect)
                if 'Result:' in line and current_scenario:
                    if 'Correct' in line:
                        self.results['scenarios'][current_scenario]['correct'] += 1
                    elif 'Incorrect' in line:
                        self.results['scenarios'][current_scenario]['incorrect'] += 1
    
    def calculate_accuracy(self, correct, total):
        """Calculate accuracy percentage."""
        if total == 0:
            return 0.0
        return (correct / total) * 100
    
    def compute_performance_metrics(self):
        """Compute comprehensive performance metrics."""
        metrics = {}
        
        for scenario, counts in self.results['scenarios'].items():
            total = counts['correct'] + counts['incorrect']
            accuracy = self.calculate_accuracy(counts['correct'], total)
            
            metrics[scenario] = {
                'correct': counts['correct'],
                'incorrect': counts['incorrect'],
                'total': total,
                'accuracy': accuracy
            }
        
        return metrics
    
    def print_performance_summary(self, metrics):
        """Print a comprehensive performance summary."""
        print("\n" + "="*80)
        print("MEDQA EXPERIMENT PERFORMANCE ANALYSIS")
        print("="*80)
        print(f"Total Questions Processed: {self.results['total_questions']}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Scenario names mapping
        scenario_names = {
            'local_alone': '1. Purely Local Model (Baseline)',
            'non_private_cot': '2. Non-Private Local + Remote CoT',
            'phrase_dp_cot': '3.1. Private Local + CoT (Phrase DP)',
            'inferdpt_cot': '3.2. Private Local + CoT (InferDPT)',
            'purely_remote': '4. Purely Remote Model'
        }
        
        # Print individual scenario results
        print("INDIVIDUAL SCENARIO PERFORMANCE:")
        print("-" * 60)
        for scenario, name in scenario_names.items():
            if scenario in metrics:
                m = metrics[scenario]
                print(f"{name:<45} {m['correct']:>3}/{m['total']:<3} = {m['accuracy']:>6.2f}%")
        
        print()
        
        # Performance comparisons
        print("PERFORMANCE COMPARISONS:")
        print("-" * 60)
        
        if 'local_alone' in metrics and 'non_private_cot' in metrics:
            local_acc = metrics['local_alone']['accuracy']
            non_private_acc = metrics['non_private_cot']['accuracy']
            gain = non_private_acc - local_acc
            print(f"CoT-Aiding Gain (Non-Private vs Local Alone): {gain:+.2f}%")
        
        if 'non_private_cot' in metrics and 'phrase_dp_cot' in metrics:
            non_private_acc = metrics['non_private_cot']['accuracy']
            phrase_dp_acc = metrics['phrase_dp_cot']['accuracy']
            privacy_cost = phrase_dp_acc - non_private_acc
            print(f"Privacy Cost (Phrase DP vs Non-Private CoT): {privacy_cost:+.2f}%")
        
        if 'non_private_cot' in metrics and 'inferdpt_cot' in metrics:
            non_private_acc = metrics['non_private_cot']['accuracy']
            inferdpt_acc = metrics['inferdpt_cot']['accuracy']
            privacy_cost = inferdpt_acc - non_private_acc
            print(f"Privacy Cost (InferDPT vs Non-Private CoT): {privacy_cost:+.2f}%")
        
        if 'phrase_dp_cot' in metrics and 'inferdpt_cot' in metrics:
            phrase_dp_acc = metrics['phrase_dp_cot']['accuracy']
            inferdpt_acc = metrics['inferdpt_cot']['accuracy']
            difference = phrase_dp_acc - inferdpt_acc
            print(f"Phrase DP vs InferDPT Performance: {difference:+.2f}%")
        
        if 'local_alone' in metrics and 'purely_remote' in metrics:
            local_acc = metrics['local_alone']['accuracy']
            remote_acc = metrics['purely_remote']['accuracy']
            gap = remote_acc - local_acc
            print(f"Remote vs Local Performance Gap: {gap:+.2f}%")
        
        print()
        
        # Key insights
        print("KEY INSIGHTS:")
        print("-" * 60)
        
        # Find best performing method
        best_scenario = max(metrics.items(), key=lambda x: x[1]['accuracy'])
        print(f"Best Performing Method: {scenario_names.get(best_scenario[0], best_scenario[0])} ({best_scenario[1]['accuracy']:.2f}%)")
        
        # Find worst performing method
        worst_scenario = min(metrics.items(), key=lambda x: x[1]['accuracy'])
        print(f"Worst Performing Method: {scenario_names.get(worst_scenario[0], worst_scenario[0])} ({worst_scenario[1]['accuracy']:.2f}%)")
        
        # Privacy-utility trade-off analysis
        if 'non_private_cot' in metrics and 'phrase_dp_cot' in metrics and 'inferdpt_cot' in metrics:
            non_private_acc = metrics['non_private_cot']['accuracy']
            phrase_dp_acc = metrics['phrase_dp_cot']['accuracy']
            inferdpt_acc = metrics['inferdpt_cot']['accuracy']
            
            phrase_dp_cost = non_private_acc - phrase_dp_acc
            inferdpt_cost = non_private_acc - inferdpt_acc
            
            print(f"Privacy-Utility Trade-off:")
            print(f"  - Phrase DP cost: {phrase_dp_cost:.2f}% accuracy loss")
            print(f"  - InferDPT cost: {inferdpt_cost:.2f}% accuracy loss")
            
            if phrase_dp_cost < inferdpt_cost:
                print(f"  - Phrase DP provides better privacy-utility balance")
            else:
                print(f"  - InferDPT provides better privacy-utility balance")
    
    def create_performance_chart(self, metrics, output_file="medqa_performance_chart.png"):
        """Create a performance comparison chart."""
        scenario_names = {
            'local_alone': 'Purely Local',
            'non_private_cot': 'Non-Private CoT',
            'phrase_dp_cot': 'Phrase DP CoT',
            'inferdpt_cot': 'InferDPT CoT',
            'purely_remote': 'Purely Remote'
        }
        
        scenarios = []
        accuracies = []
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (scenario, name) in enumerate(scenario_names.items()):
            if scenario in metrics:
                scenarios.append(name)
                accuracies.append(metrics[scenario]['accuracy'])
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(scenarios, accuracies, color=colors[:len(scenarios)])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.title('MedQA Experiment Performance Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Method', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.ylim(0, max(accuracies) * 1.1)
        plt.grid(axis='y', alpha=0.3)
        
        # Add total questions processed
        plt.figtext(0.02, 0.02, f'Total Questions: {self.results["total_questions"]}/500', 
                   fontsize=10, style='italic')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nPerformance chart saved as: {output_file}")
    
    def save_detailed_results(self, metrics, output_file="medqa_detailed_results.json"):
        """Save detailed results to JSON file."""
        output_data = {
            'analysis_date': datetime.now().isoformat(),
            'total_questions_processed': self.results['total_questions'],
            'total_questions_target': 500,
            'completion_percentage': (self.results['total_questions'] / 500) * 100,
            'performance_metrics': metrics,
            'scenario_names': {
                'local_alone': '1. Purely Local Model (Baseline)',
                'non_private_cot': '2. Non-Private Local + Remote CoT',
                'phrase_dp_cot': '3.1. Private Local + CoT (Phrase DP)',
                'inferdpt_cot': '3.2. Private Local + CoT (InferDPT)',
                'purely_remote': '4. Purely Remote Model'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Detailed results saved as: {output_file}")
    
    def run_analysis(self):
        """Run the complete performance analysis."""
        print("Starting MedQA Performance Analysis...")
        
        # Parse the results file
        self.parse_results_file()
        
        # Compute performance metrics
        metrics = self.compute_performance_metrics()
        
        # Print summary
        self.print_performance_summary(metrics)
        
        # Create visualization
        if MATPLOTLIB_AVAILABLE:
            self.create_performance_chart(metrics)
        else:
            print("Matplotlib not available. Skipping chart generation.")
        
        # Save detailed results
        self.save_detailed_results(metrics)
        
        return metrics

def main():
    """Main function to run the analysis."""
    analyzer = MedQAPerformanceAnalyzer()
    metrics = analyzer.run_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
