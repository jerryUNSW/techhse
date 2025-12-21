"""
Prompt loading and formatting utilities.
"""

import os
from typing import Dict, Any


def load_system_prompt(prompt_name: str = "system_prompt.txt") -> str:
    """
    Load system prompt from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file
    
    Returns:
        The system prompt content
    
    Raises:
        FileNotFoundError: If prompt file not found
    """
    prompt_path = os.path.join("prompts", prompt_name)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to resolve relative to repository root
        module_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(module_dir)))
        alt_path = os.path.join(repo_root, "prompts", prompt_name)
        try:
            with open(alt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"System prompt file not found: {prompt_path} or {alt_path}")


def load_user_prompt_template(template_name: str = "user_prompt_template.txt") -> str:
    """
    Load user prompt template from the prompts directory.
    
    Args:
        template_name: Name of the template file
    
    Returns:
        The user prompt template content
    
    Raises:
        FileNotFoundError: If template file not found
    """
    template_path = os.path.join("prompts", template_name)
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        # Fallback to resolve relative to repository root
        module_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(module_dir)))
        alt_path = os.path.join(repo_root, "prompts", template_name)
        try:
            with open(alt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"User prompt template file not found: {template_path} or {alt_path}")


def format_user_prompt(template: str, **kwargs) -> str:
    """
    Format a user prompt template with the provided arguments.
    
    Args:
        template: The prompt template
        **kwargs: Arguments to format into the template
    
    Returns:
        The formatted user prompt
    
    Raises:
        KeyError: If required argument is missing
    """
    try:
        return template.format(**kwargs)
    except KeyError as e:
        raise KeyError(f"Missing required argument in prompt template: {e}")


def format_question_with_options(question: str, options: Dict[str, str] = None) -> str:
    """
    Format question with options for LLM input.
    
    Args:
        question: Question text
        options: Dictionary of options {'A': '...', 'B': '...', ...}
    
    Returns:
        Formatted question string
    """
    formatted = f"{question}"
    if options:
        formatted += "\n\nOptions:\n"
        for key, value in options.items():
            formatted += f"{key}) {value}\n"
    formatted += "\n\nAnswer:"
    return formatted


def extract_letter_from_answer(answer: str) -> str:
    """
    Extract the letter (A, B, C, D) from the model's answer.
    
    Args:
        answer: Model's answer text
    
    Returns:
        Extracted letter ('A', 'B', 'C', or 'D'), or 'Error' if not found
    """
    answer = answer.strip().upper()
    
    # Look for single letters
    for letter in ['A', 'B', 'C', 'D']:
        if answer == letter or answer.startswith(letter) or f" {letter}" in answer:
            return letter
    
    # If no letter found, return Error
    return "Error"


def check_mcq_correctness(predicted: str, ground_truth: str) -> bool:
    """
    Check if predicted answer matches ground truth.
    
    Args:
        predicted: Predicted answer letter
        ground_truth: Ground truth answer letter
    
    Returns:
        True if correct, False otherwise
    """
    return predicted.upper() == ground_truth.upper()


