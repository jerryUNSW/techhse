#!/usr/bin/env python3
"""
Show Medical QA Examples
========================

A simple script to display medical question-answer examples from the EMRQA-MSQUAD dataset
without running any privacy-preserving methods.

Author: Tech4HSE Team
Date: 2025-08-26
"""

from datasets import load_dataset

def show_medical_qa_examples(num_examples=10):
    """
    Load and display medical question-answer examples from the EMRQA-MSQUAD dataset.
    """
    print("="*80)
    print("MEDICAL QUESTION-ANSWER EXAMPLES FROM EMRQA-MSQUAD DATASET")
    print("="*80)
    
    try:
        # Load the dataset
        print("Loading Eladio/emrqa-msquad dataset...")
        dataset = load_dataset("Eladio/emrqa-msquad")
        
        # Use validation split for cleaner examples
        split_data = dataset['validation']
        
        print(f"✓ Successfully loaded dataset")
        print(f"Using validation split: {len(split_data)} examples")
        print(f"Showing first {num_examples} examples:")
        print("="*80)
        
        for i in range(min(num_examples, len(split_data))):
            example = split_data[i]
            
            print(f"\n{'='*60}")
            print(f"EXAMPLE {i+1}/{num_examples}")
            print(f"{'='*60}")
            
            # Display full context
            context = example['context']
            print(f"CONTEXT:")
            print(f"{context}")
            print()
            
            # Display question
            print(f"QUESTION:")
            print(f"{example['question']}")
            print()
            
            # Display answer
            answers = example['answers']
            if 'text' in answers and answers['text']:
                print(f"ANSWER:")
                for j, answer_text in enumerate(answers['text']):
                    print(f"  {answer_text}")
            else:
                print(f"ANSWER: No answer provided")
            
            print("-" * 60)
        
        # Show dataset statistics
        print(f"\n{'='*80}")
        print("DATASET STATISTICS")
        print(f"{'='*80}")
        print(f"Train split: {len(dataset['train'])} examples")
        print(f"Validation split: {len(dataset['validation'])} examples")
        print(f"Total examples: {len(dataset['train']) + len(dataset['validation'])}")
        
        # Show field information
        if len(split_data) > 0:
            print(f"\nAvailable fields: {list(split_data[0].keys())}")
        
        print("="*80)
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None

def show_sample_questions_only(num_examples=5):
    """
    Show just the questions and answers without context for a quick overview.
    """
    print("="*80)
    print("QUICK OVERVIEW: QUESTIONS & ANSWERS ONLY")
    print("="*80)
    
    try:
        dataset = load_dataset("Eladio/emrqa-msquad")
        split_data = dataset['validation']
        
        for i in range(min(num_examples, len(split_data))):
            example = split_data[i]
            
            print(f"\n{i+1}. QUESTION: {example['question']}")
            
            answers = example['answers']
            if 'text' in answers and answers['text']:
                print(f"   ANSWER: {', '.join(answers['text'])}")
            else:
                print(f"   ANSWER: No answer provided")
        
        print("="*80)
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    # Show detailed examples with context
    show_medical_qa_examples(3)
    
    print("\n" + "="*80)
    
    # Show quick overview without context
    show_sample_questions_only(5)
