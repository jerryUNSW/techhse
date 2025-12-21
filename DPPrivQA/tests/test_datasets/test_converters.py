"""
Tests for format converters.
"""

from dpprivqa.datasets.converters import (
    convert_mmlu_to_standard,
    convert_arc_to_standard,
    convert_medqa_to_standard
)


def test_convert_mmlu_to_standard():
    """Test MMLU format conversion."""
    item = {
        "question": "What is the capital of France?",
        "choices": ["London", "Paris", "Berlin", "Madrid"],
        "answer": 1
    }
    
    result = convert_mmlu_to_standard(item)
    
    assert result['question'] == "What is the capital of France?"
    assert result['options']['A'] == "London"
    assert result['options']['B'] == "Paris"
    assert result['answer_idx'] == 'B'


def test_convert_arc_to_standard():
    """Test ARC format conversion."""
    item = {
        "question": "What is 2+2?",
        "choices": [
            {"label": "A", "text": "3"},
            {"label": "B", "text": "4"},
            {"label": "C", "text": "5"}
        ],
        "answerKey": "B"
    }
    
    result = convert_arc_to_standard(item)
    
    assert result['question'] == "What is 2+2?"
    assert result['options']['A'] == "3"
    assert result['options']['B'] == "4"
    assert result['answer_idx'] == 'B'


def test_convert_medqa_to_standard():
    """Test MedQA format conversion."""
    # Test with dict options
    item1 = {
        "question": "What is hypertension?",
        "options": {"A": "High blood pressure", "B": "Low blood pressure"},
        "answer": "A"
    }
    
    result1 = convert_medqa_to_standard(item1)
    assert result1['answer_idx'] == 'A'
    
    # Test with list options
    item2 = {
        "question": "What is diabetes?",
        "options": ["Type 1", "Type 2"],
        "answer": 0
    }
    
    result2 = convert_medqa_to_standard(item2)
    assert result2['answer_idx'] == 'A'


