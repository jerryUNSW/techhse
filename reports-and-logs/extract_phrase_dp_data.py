#!/usr/bin/env python3
"""
Extract phrase DP candidates and final replacements from testing-phraseDP.txt
"""

import re
import json
from typing import Dict, List, Tuple

def parse_phrase_dp_file(file_path: str) -> List[Dict]:
    """
    Parse the phrase DP testing file and extract candidates and final replacements for each question.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split content into question sections
    question_sections = content.split('QUESTION ')[1:]  # Skip the header
    
    results = []
    
    for section in question_sections:
        # Extract question number and original question
        lines = section.strip().split('\n')
        
        # Find the original question
        original_question = None
        for line in lines:
            if line.startswith('Original Question:'):
                original_question = line.replace('Original Question:', '').strip()
                break
        
        if not original_question:
            continue
        
        # Find the perturbed question (final DP replacement)
        perturbed_question = None
        for line in lines:
            if line.startswith('Perturbed:'):
                perturbed_question = line.replace('Perturbed:', '').strip()
                break
        
        # Extract candidates and their similarity scores
        candidates = []
        in_candidates_section = False
        
        for line in lines:
            if 'All Generated Candidates with Similarities:' in line:
                in_candidates_section = True
                continue
            elif in_candidates_section and line.startswith('---'):
                break
            elif in_candidates_section and line.strip():
                # Parse candidate lines like "  1. Does the nationality of two notable figures from the film industry overlap?"
                match = re.match(r'\s*(\d+)\.\s*(.+?)\s*Similarity:\s*([\d.]+)', line)
                if match:
                    candidate_num = int(match.group(1))
                    candidate_text = match.group(2).strip()
                    similarity = float(match.group(3))
                    candidates.append({
                        'number': candidate_num,
                        'text': candidate_text,
                        'similarity': similarity
                    })
        
        # Create result entry
        result = {
            'original_question': original_question,
            'final_dp_replacement': perturbed_question,
            'candidates': candidates,
            'num_candidates': len(candidates)
        }
        
        results.append(result)
    
    return results

def save_results_to_json(results: List[Dict], output_file: str):
    """Save results to JSON file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

def save_results_to_txt(results: List[Dict], output_file: str):
    """Save results to human-readable text file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("PHRASE DP CANDIDATES AND FINAL REPLACEMENTS EXTRACTION\n")
        f.write("=" * 60 + "\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"QUESTION {i}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Original Question: {result['original_question']}\n")
            f.write(f"Final DP Replacement: {result['final_dp_replacement']}\n")
            f.write(f"Number of Candidates: {result['num_candidates']}\n\n")
            
            f.write("Candidates with Similarity Scores:\n")
            for candidate in result['candidates']:
                f.write(f"  {candidate['number']:2d}. {candidate['text']}\n")
                f.write(f"      Similarity: {candidate['similarity']:.4f}\n")
            f.write("\n" + "=" * 60 + "\n\n")

def main():
    input_file = "txt-files/testing-phraseDP.txt"
    json_output = "phrase_dp_extracted_data.json"
    txt_output = "phrase_dp_extracted_data.txt"
    
    print(f"Parsing {input_file}...")
    results = parse_phrase_dp_file(input_file)
    
    print(f"Found {len(results)} questions with phrase DP data")
    
    # Save to JSON
    save_results_to_json(results, json_output)
    print(f"Saved structured data to {json_output}")
    
    # Save to human-readable text
    save_results_to_txt(results, txt_output)
    print(f"Saved human-readable format to {txt_output}")
    
    # Print summary
    print("\nSummary:")
    for i, result in enumerate(results, 1):
        print(f"Question {i}: {result['num_candidates']} candidates")
        if result['candidates']:
            avg_similarity = sum(c['similarity'] for c in result['candidates']) / len(result['candidates'])
            print(f"  Average similarity: {avg_similarity:.4f}")
            max_similarity = max(c['similarity'] for c in result['candidates'])
            min_similarity = min(c['similarity'] for c in result['candidates'])
            print(f"  Similarity range: {min_similarity:.4f} - {max_similarity:.4f}")

if __name__ == "__main__":
    main()
