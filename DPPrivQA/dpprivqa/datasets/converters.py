"""
Format conversion utilities for different dataset formats.
"""

from typing import Dict, Any, List


def convert_mmlu_to_standard(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MMLU format to standard format.
    
    MMLU format:
    {
        'question': str,
        'choices': List[str],  # 4 choices
        'answer': int  # 0-3
    }
    
    Standard format:
    {
        'question': str,
        'options': Dict[str, str],  # {'A': '...', 'B': '...', ...}
        'answer_idx': str  # 'A', 'B', 'C', or 'D'
    }
    """
    question = item["question"]
    choices = item["choices"]  # List of 4 strings
    answer = item["answer"]  # Integer 0-3
    
    # Convert choices list to options dict
    options = {chr(65 + i): choice for i, choice in enumerate(choices)}
    
    # Convert answer integer to letter
    answer_idx = chr(65 + answer)  # 0->'A', 1->'B', 2->'C', 3->'D'
    
    return {
        "question": question,
        "options": options,
        "answer_idx": answer_idx
    }


def convert_arc_to_standard(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert ARC format to standard format.
    
    ARC format:
    {
        'question': str,
        'choices': List[Dict],  # [{'label': 'A', 'text': '...'}, ...]
        'answerKey': str  # 'A', 'B', 'C', or 'D'
    }
    
    Standard format:
    {
        'question': str,
        'options': Dict[str, str],
        'answer_idx': str
    }
    """
    question = item["question"]
    choices = item["choices"]  # List of dicts with 'label' and 'text'
    answer_key = item["answerKey"]  # String like 'A', 'B', 'C', 'D'
    
    # Convert choices list to options dict
    options = {choice["label"]: choice["text"] for choice in choices}
    
    return {
        "question": question,
        "options": options,
        "answer_idx": answer_key
    }


def convert_medqa_to_standard(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert MedQA format to standard format.
    
    MedQA format varies, but typically:
    {
        'question': str,
        'options': Dict[str, str] or List[str],
        'answer': str or int
    }
    """
    question = item.get("question", "")
    
    # Handle different option formats
    if isinstance(item.get("options"), dict):
        options = item["options"]
    elif isinstance(item.get("options"), list):
        options = {chr(65 + i): opt for i, opt in enumerate(item["options"])}
    else:
        # Try to extract from other fields
        options = {}
        for key in ['A', 'B', 'C', 'D']:
            if key in item:
                options[key] = item[key]
    
    # Handle different answer formats
    answer = item.get("answer") or item.get("answer_idx") or item.get("answerKey")
    if isinstance(answer, int):
        answer_idx = chr(65 + answer)
    elif isinstance(answer, str) and len(answer) == 1:
        answer_idx = answer.upper()
    else:
        answer_idx = str(answer).upper()
    
    return {
        "question": question,
        "options": options,
        "answer_idx": answer_idx
    }


