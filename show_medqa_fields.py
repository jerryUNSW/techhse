#!/usr/bin/env python3
"""
Show MedQA-USMLE-4-options Dataset Fields
=========================================

A script to display all fields and structure of the MedQA-USMLE-4-options dataset.

Author: Tech4HSE Team
Date: 2025-08-26
"""

from datasets import load_dataset

def show_medqa_dataset_fields():
    """
    Load and display all fields of the MedQA-USMLE-4-options dataset.
    """
    print("="*80)
    print("MEDQA-USMLE-4-OPTIONS DATASET FIELDS ANALYSIS")
    print("="*80)
    
    try:
        # Load the dataset
        print("Loading MedQA-USMLE-4-options dataset...")
        dataset = load_dataset('GBaker/MedQA-USMLE-4-options')
        
        print(f"✓ Successfully loaded dataset")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Show dataset info for each split
        for split_name, split_data in dataset.items():
            print(f"\n{split_name.upper()} split:")
            print(f"  Total samples: {len(split_data)}")
            
            if len(split_data) > 0:
                print(f"  Available fields: {list(split_data[0].keys())}")
        
        # Show first 5 examples with all fields
        first_split = list(dataset.keys())[0]
        split_data = dataset[first_split]
        
        print(f"\n{'='*80}")
        print(f"FIRST 5 EXAMPLES FROM {first_split.upper()} SPLIT")
        print(f"{'='*80}")
        
        for i in range(min(5, len(split_data))):
            example = split_data[i]
            print(f"\n--- Example {i+1} ---")
            
            # Print all fields
            for key, value in example.items():
                if isinstance(value, str):
                    if len(value) > 200:
                        print(f"{key}: {value[:200]}...")
                    else:
                        print(f"{key}: {value}")
                elif isinstance(value, dict):
                    print(f"{key}: {value}")
                else:
                    print(f"{key}: {value}")
            
            print("-" * 50)
        
        # Show field statistics
        print(f"\n{'='*80}")
        print("FIELD ANALYSIS")
        print(f"{'='*80}")
        
        if len(split_data) > 0:
            first_example = split_data[0]
            print(f"Total fields: {len(first_example.keys())}")
            print(f"Field names: {list(first_example.keys())}")
            
            print(f"\nDetailed field information:")
            for field_name, field_value in first_example.items():
                field_type = type(field_value).__name__
                if isinstance(field_value, str):
                    field_length = len(field_value)
                    print(f"  {field_name}: {field_type} (length: {field_length})")
                elif isinstance(field_value, dict):
                    dict_keys = list(field_value.keys())
                    print(f"  {field_name}: {field_type} (keys: {dict_keys})")
                else:
                    print(f"  {field_name}: {field_type}")
        
        return dataset
        
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None

def show_sample_questions_only(num_examples=3):
    """
    Show just the questions and answers without other fields for a quick overview.
    """
    print("="*80)
    print("SAMPLE MEDQA QUESTIONS & ANSWERS")
    print("="*80)
    
    try:
        dataset = load_dataset('GBaker/MedQA-USMLE-4-options')
        first_split = list(dataset.keys())[0]
        split_data = dataset[first_split]
        
        for i in range(min(num_examples, len(split_data))):
            example = split_data[i]
            
            print(f"\n{i+1}. QUESTION: {example.get('question', 'No question')}")
            
            if 'options' in example:
                options = example['options']
                print(f"   OPTIONS:")
                for key, value in options.items():
                    print(f"     {key}) {value}")
            
            if 'answer_idx' in example:
                print(f"   CORRECT ANSWER: {example['answer_idx']}")
        
        print("="*80)
        
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    # Show detailed field analysis
    dataset = show_medqa_dataset_fields()
    
    print("\n" + "="*80)
    
    # Show quick overview
    show_sample_questions_only(3)
