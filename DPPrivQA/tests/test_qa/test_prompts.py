"""
Tests for prompt utilities.
"""

from dpprivqa.qa.prompts import (
    format_question_with_options,
    extract_letter_from_answer,
    check_mcq_correctness
)


def test_format_question_with_options():
    """Test formatting question with options."""
    question = "What is 2+2?"
    options = {"A": "3", "B": "4", "C": "5"}
    
    formatted = format_question_with_options(question, options)
    
    assert question in formatted
    assert "A) 3" in formatted
    assert "B) 4" in formatted
    assert "Answer:" in formatted


def test_extract_letter_from_answer():
    """Test extracting letter from answer."""
    assert extract_letter_from_answer("A") == "A"
    assert extract_letter_from_answer("The answer is B") == "B"
    assert extract_letter_from_answer("C is correct") == "C"
    assert extract_letter_from_answer("invalid") == "Error"


def test_check_mcq_correctness():
    """Test MCQ correctness checking."""
    assert check_mcq_correctness("A", "A") == True
    assert check_mcq_correctness("B", "A") == False
    assert check_mcq_correctness("a", "A") == True  # Case insensitive
    assert check_mcq_correctness("A", "a") == True


