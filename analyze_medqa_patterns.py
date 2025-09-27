#!/usr/bin/env python3
"""
Analyze MedQA USMLE Dataset Patterns
====================================

This script analyzes the MedQA USMLE dataset to understand:
1. Question types and patterns
2. Whether there's a progression from knowledge-based to patient-based questions
3. Content analysis across different indices
4. Medical domain distribution

Author: Tech4HSE Team
Date: 2025-01-27
"""

import re
from datasets import load_dataset
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_question_patterns(dataset, start_idx=0, end_idx=100):
    """
    Analyze question patterns in the MedQA dataset.
    """
    print("="*80)
    print("MEDQA USMLE DATASET PATTERN ANALYSIS")
    print("="*80)
    
    # Extract questions from the specified range
    questions = []
    for i in range(start_idx, min(end_idx, len(dataset))):
        item = dataset[i]
        questions.append({
            'index': i,
            'question': item['question'],
            'options': item['options'],
            'answer': item['answer_idx']
        })
    
    print(f"Analyzing {len(questions)} questions (indices {start_idx}-{end_idx-1})")
    
    # Analyze question types
    question_types = analyze_question_types(questions)
    
    # Analyze medical domains
    medical_domains = analyze_medical_domains(questions)
    
    # Analyze question complexity
    complexity_analysis = analyze_question_complexity(questions)
    
    # Check for progression patterns
    progression_analysis = analyze_progression_patterns(questions)
    
    return {
        'question_types': question_types,
        'medical_domains': medical_domains,
        'complexity': complexity_analysis,
        'progression': progression_analysis
    }

def analyze_question_types(questions):
    """
    Categorize questions by type (knowledge-based vs patient-based).
    """
    print("\n" + "="*60)
    print("QUESTION TYPE ANALYSIS")
    print("="*60)
    
    knowledge_based = []
    patient_based = []
    other = []
    
    # Keywords to identify question types
    patient_keywords = [
        'patient', 'presents', 'complains', 'symptoms', 'history', 'examination',
        'diagnosis', 'treatment', 'management', 'presents with', 'complains of',
        'physical exam', 'laboratory', 'imaging', 'vital signs', 'age', 'year-old',
        'male', 'female', 'adult', 'child', 'infant', 'elderly'
    ]
    
    knowledge_keywords = [
        'which of the following', 'what is', 'what are', 'define', 'explain',
        'mechanism', 'function', 'structure', 'process', 'pathway', 'system',
        'characteristics', 'properties', 'types', 'classification', 'theory'
    ]
    
    for q in questions:
        question_text = q['question'].lower()
        
        # Check for patient-based indicators
        patient_score = sum(1 for keyword in patient_keywords if keyword in question_text)
        
        # Check for knowledge-based indicators
        knowledge_score = sum(1 for keyword in knowledge_keywords if keyword in question_text)
        
        if patient_score > knowledge_score and patient_score > 0:
            patient_based.append(q)
        elif knowledge_score > patient_score and knowledge_score > 0:
            knowledge_based.append(q)
        else:
            other.append(q)
    
    print(f"Knowledge-based questions: {len(knowledge_based)} ({len(knowledge_based)/len(questions)*100:.1f}%)")
    print(f"Patient-based questions: {len(patient_based)} ({len(patient_based)/len(questions)*100:.1f}%)")
    print(f"Other/Unclear: {len(other)} ({len(other)/len(questions)*100:.1f}%)")
    
    # Show examples
    print(f"\n--- KNOWLEDGE-BASED EXAMPLES ---")
    for i, q in enumerate(knowledge_based[:3]):
        print(f"{i+1}. Index {q['index']}: {q['question'][:100]}...")
    
    print(f"\n--- PATIENT-BASED EXAMPLES ---")
    for i, q in enumerate(patient_based[:3]):
        print(f"{i+1}. Index {q['index']}: {q['question'][:100]}...")
    
    return {
        'knowledge_based': knowledge_based,
        'patient_based': patient_based,
        'other': other
    }

def analyze_medical_domains(questions):
    """
    Analyze medical domains covered in questions.
    """
    print("\n" + "="*60)
    print("MEDICAL DOMAIN ANALYSIS")
    print("="*60)
    
    domain_keywords = {
        'Cardiology': ['heart', 'cardiac', 'myocardial', 'coronary', 'chest pain', 'angina', 'arrhythmia'],
        'Neurology': ['brain', 'neurological', 'seizure', 'stroke', 'headache', 'consciousness', 'neurological'],
        'Gastroenterology': ['stomach', 'gastrointestinal', 'liver', 'pancreas', 'digestive', 'abdomen'],
        'Pulmonology': ['lung', 'respiratory', 'breathing', 'pneumonia', 'asthma', 'COPD'],
        'Endocrinology': ['diabetes', 'hormone', 'thyroid', 'insulin', 'glucose', 'endocrine'],
        'Oncology': ['cancer', 'tumor', 'malignancy', 'chemotherapy', 'radiation', 'metastasis'],
        'Infectious Disease': ['infection', 'bacterial', 'viral', 'fever', 'antibiotic', 'pathogen'],
        'Pharmacology': ['drug', 'medication', 'pharmacology', 'dose', 'side effect', 'interaction'],
        'Pathology': ['disease', 'pathology', 'biopsy', 'histology', 'morphology'],
        'Physiology': ['function', 'mechanism', 'process', 'regulation', 'homeostasis']
    }
    
    domain_counts = Counter()
    
    for q in questions:
        question_text = q['question'].lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_text for keyword in keywords):
                domain_counts[domain] += 1
    
    print("Medical domain distribution:")
    for domain, count in domain_counts.most_common():
        print(f"  {domain}: {count} ({count/len(questions)*100:.1f}%)")
    
    return domain_counts

def analyze_question_complexity(questions):
    """
    Analyze question complexity based on length and medical terminology.
    """
    print("\n" + "="*60)
    print("QUESTION COMPLEXITY ANALYSIS")
    print("="*60)
    
    complexities = []
    
    for q in questions:
        question_text = q['question']
        
        # Complexity metrics
        word_count = len(question_text.split())
        char_count = len(question_text)
        
        # Medical terminology count (rough estimate)
        medical_terms = len(re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', question_text))
        
        # Question complexity score
        complexity_score = word_count * 0.3 + medical_terms * 2 + (char_count / 100)
        
        complexities.append({
            'index': q['index'],
            'word_count': word_count,
            'medical_terms': medical_terms,
            'complexity_score': complexity_score,
            'question': question_text
        })
    
    # Sort by complexity
    complexities.sort(key=lambda x: x['complexity_score'])
    
    print(f"Average word count: {np.mean([c['word_count'] for c in complexities]):.1f}")
    print(f"Average medical terms: {np.mean([c['medical_terms'] for c in complexities]):.1f}")
    print(f"Average complexity score: {np.mean([c['complexity_score'] for c in complexities]):.1f}")
    
    print(f"\n--- SIMPLEST QUESTIONS ---")
    for i, c in enumerate(complexities[:3]):
        print(f"{i+1}. Index {c['index']}: {c['question'][:80]}... (Score: {c['complexity_score']:.1f})")
    
    print(f"\n--- MOST COMPLEX QUESTIONS ---")
    for i, c in enumerate(complexities[-3:]):
        print(f"{i+1}. Index {c['index']}: {c['question'][:80]}... (Score: {c['complexity_score']:.1f})")
    
    return complexities

def analyze_progression_patterns(questions):
    """
    Analyze whether there's a progression pattern in question types.
    """
    print("\n" + "="*60)
    print("PROGRESSION PATTERN ANALYSIS")
    print("="*60)
    
    # Divide questions into chunks
    chunk_size = len(questions) // 5  # 5 chunks
    chunks = [questions[i:i+chunk_size] for i in range(0, len(questions), chunk_size)]
    
    print(f"Analyzing progression across {len(chunks)} chunks of ~{chunk_size} questions each")
    
    chunk_analysis = []
    
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
            
        # Analyze this chunk
        patient_count = 0
        knowledge_count = 0
        
        for q in chunk:
            question_text = q['question'].lower()
            
            # Simple classification
            if any(word in question_text for word in ['patient', 'presents', 'complains', 'symptoms']):
                patient_count += 1
            elif any(word in question_text for word in ['which of the following', 'what is', 'define']):
                knowledge_count += 1
        
        total = len(chunk)
        patient_pct = (patient_count / total) * 100 if total > 0 else 0
        knowledge_pct = (knowledge_count / total) * 100 if total > 0 else 0
        
        chunk_analysis.append({
            'chunk': i,
            'start_idx': chunk[0]['index'],
            'end_idx': chunk[-1]['index'],
            'patient_pct': patient_pct,
            'knowledge_pct': knowledge_pct,
            'total': total
        })
        
        print(f"Chunk {i+1} (indices {chunk[0]['index']}-{chunk[-1]['index']}): "
              f"Patient-based: {patient_pct:.1f}%, Knowledge-based: {knowledge_pct:.1f}%")
    
    # Check for trends
    patient_trend = [c['patient_pct'] for c in chunk_analysis]
    knowledge_trend = [c['knowledge_pct'] for c in chunk_analysis]
    
    print(f"\n--- TREND ANALYSIS ---")
    print(f"Patient-based trend: {patient_trend}")
    print(f"Knowledge-based trend: {knowledge_trend}")
    
    # Simple trend detection
    if len(patient_trend) >= 2:
        patient_increasing = patient_trend[-1] > patient_trend[0]
        knowledge_increasing = knowledge_trend[-1] > knowledge_trend[0]
        
        print(f"Patient-based questions {'increasing' if patient_increasing else 'decreasing'} over time")
        print(f"Knowledge-based questions {'increasing' if knowledge_increasing else 'decreasing'} over time")
    
    return chunk_analysis

def show_sample_questions_by_index(dataset, indices):
    """
    Show sample questions at specific indices to understand patterns.
    """
    print("\n" + "="*80)
    print("SAMPLE QUESTIONS BY INDEX")
    print("="*80)
    
    for idx in indices:
        if idx < len(dataset):
            item = dataset[idx]
            print(f"\n--- QUESTION {idx} ---")
            print(f"Question: {item['question']}")
            print(f"Options:")
            for key, value in item['options'].items():
                print(f"  {key}) {value}")
            print(f"Answer: {item['answer_idx']}")
            print("-" * 60)

def main():
    """
    Main analysis function.
    """
    print("Loading MedQA USMLE dataset...")
    try:
        dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')
        print(f"✅ Loaded dataset with {len(dataset)} questions")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Analyze different ranges
    ranges_to_analyze = [
        (0, 50, "First 50 questions"),
        (50, 100, "Questions 50-100"),
        (100, 150, "Questions 100-150"),
        (200, 250, "Questions 200-250"),
        (400, 450, "Questions 400-450")
    ]
    
    for start, end, description in ranges_to_analyze:
        print(f"\n{'='*80}")
        print(f"ANALYZING {description.upper()}")
        print(f"{'='*80}")
        
        analysis = analyze_question_patterns(dataset, start, end)
        
        # Show sample questions from this range
        sample_indices = [start, start + (end-start)//2, end-1]
        show_sample_questions_by_index(dataset, sample_indices)
    
    # Overall dataset analysis
    print(f"\n{'='*80}")
    print("OVERALL DATASET ANALYSIS")
    print(f"{'='*80}")
    
    # Analyze first 200 questions for overall patterns
    overall_analysis = analyze_question_patterns(dataset, 0, 200)
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("✅ MedQA USMLE dataset analysis completed")
    print("✅ Question type patterns identified")
    print("✅ Medical domain distribution analyzed")
    print("✅ Complexity patterns examined")
    print("✅ Progression patterns investigated")

if __name__ == "__main__":
    main()
