"""
QA scenario implementations for different testing approaches.
"""

import time
from typing import Dict, Any, Optional
from openai import OpenAI

from dpprivqa.qa.models import (
    create_completion_with_model_support,
    find_working_nebius_model
)
from dpprivqa.qa.prompts import (
    format_question_with_options,
    extract_letter_from_answer,
    check_mcq_correctness
)


def run_local_only(
    local_client: OpenAI,
    local_model_name: str,
    question: str,
    options: Dict[str, str],
    max_tokens: int = 256
) -> Dict[str, Any]:
    """
    Run local-only scenario (baseline).
    
    Args:
        local_client: Local LLM client
        local_model_name: Name of local model
        question: Question text
        options: Options dictionary
        max_tokens: Maximum tokens for response
    
    Returns:
        Result dictionary with answer, is_correct, processing_time, etc.
    """
    start_time = time.time()
    
    formatted_question = format_question_with_options(question, options)
    
    try:
        local_model = find_working_nebius_model(local_client, local_model_name)
        response = create_completion_with_model_support(
            local_client, local_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert. Answer the multiple choice question by providing only the letter (A, B, C, or D) of the correct option."
                },
                {"role": "user", "content": formatted_question}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        answer_text = response.choices[0].message.content.strip()
        predicted = extract_letter_from_answer(answer_text)
    except Exception as e:
        predicted = "Error"
        answer_text = f"Error: {e}"
    
    processing_time = time.time() - start_time
    
    return {
        "answer": predicted,
        "answer_text": answer_text,
        "question": question,
        "options": options,
        "processing_time": processing_time,
        "local_model": local_model_name,
        "scenario": "local"
    }


def run_local_with_cot(
    local_client: OpenAI,
    remote_client: OpenAI,
    local_model_name: str,
    remote_model_name: str,
    question: str,
    options: Dict[str, str],
    max_tokens: int = 256,
    cot_max_tokens: int = 512
) -> Dict[str, Any]:
    """
    Run local + CoT scenario (non-private).
    
    Args:
        local_client: Local LLM client
        remote_client: Remote LLM client
        local_model_name: Name of local model
        remote_model_name: Name of remote model
        question: Question text
        options: Options dictionary
        max_tokens: Maximum tokens for answer
        cot_max_tokens: Maximum tokens for CoT generation
    
    Returns:
        Result dictionary with answer, cot_text, is_correct, processing_time, etc.
    """
    start_time = time.time()
    
    # Step 1: Generate CoT from remote (using original question, no options for privacy)
    cot_start = time.time()
    prompt_lines = [
        "Here is the question:",
        question,
        "",
        "Please provide a clear, step-by-step chain-of-thought reasoning to solve this question. Do NOT provide the final answer; provide only the reasoning steps."
    ]
    cot_prompt = "\n".join(prompt_lines)
    
    try:
        cot_response = create_completion_with_model_support(
            remote_client, remote_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert reasoner. Provide a clear, step-by-step chain of thought to analyze the given question. Focus on domain-appropriate reasoning and knowledge."
                },
                {"role": "user", "content": cot_prompt}
            ],
            max_tokens=cot_max_tokens,
            temperature=0.0
        )
        cot_text = cot_response.choices[0].message.content.strip()
    except Exception as e:
        cot_text = f"Error: {e}"
    
    cot_time = time.time() - cot_start
    
    # Step 2: Get answer from local model with CoT
    formatted_question = format_question_with_options(question, options)
    full_prompt = f"{formatted_question}\n\nChain of Thought:\n{cot_text}\n\nBased on the chain of thought above, what is the correct answer? Provide only the letter (A, B, C, or D):"
    
    try:
        local_model = find_working_nebius_model(local_client, local_model_name)
        response = create_completion_with_model_support(
            local_client, local_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert. Use the provided chain of thought to answer the multiple choice question. Provide only the letter (A, B, C, or D) of the correct option."
                },
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        answer_text = response.choices[0].message.content.strip()
        predicted = extract_letter_from_answer(answer_text)
    except Exception as e:
        predicted = "Error"
        answer_text = f"Error: {e}"
    
    processing_time = time.time() - start_time
    
    return {
        "answer": predicted,
        "answer_text": answer_text,
        "cot_text": cot_text,
        "question": question,
        "options": options,
        "processing_time": processing_time,
        "cot_generation_time": cot_time,
        "local_model": local_model_name,
        "remote_model": remote_model_name,
        "scenario": "local_cot"
    }


def run_remote_only(
    remote_client: OpenAI,
    remote_model_name: str,
    question: str,
    options: Dict[str, str],
    max_tokens: int = 256  # Will be increased to 4096 for GPT-5 automatically
) -> Dict[str, Any]:
    """
    Run remote-only scenario (baseline).
    
    Args:
        remote_client: Remote LLM client
        remote_model_name: Name of remote model
        question: Question text
        options: Options dictionary
        max_tokens: Maximum tokens for response
    
    Returns:
        Result dictionary with answer, is_correct, processing_time, etc.
    """
    start_time = time.time()
    
    formatted_question = format_question_with_options(question, options)
    
    try:
        response = create_completion_with_model_support(
            remote_client, remote_model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert. Answer the multiple choice question by providing only the letter (A, B, C, or D) of the correct option."
                },
                {"role": "user", "content": formatted_question}
            ],
            max_tokens=max_tokens,
            temperature=0.0
        )
        content = response.choices[0].message.content
        answer_text = content.strip() if content else ""
        predicted = extract_letter_from_answer(answer_text)
    except Exception as e:
        predicted = "Error"
        answer_text = f"Error: {e}"
    
    processing_time = time.time() - start_time
    
    return {
        "answer": predicted,
        "answer_text": answer_text,
        "question": question,
        "options": options,
        "processing_time": processing_time,
        "remote_model": remote_model_name,
        "scenario": "remote"
    }


def run_local_with_selective_cot(
    local_client: OpenAI,
    remote_client: OpenAI,
    local_model_name: str,
    remote_model_name: str,
    question: str,
    options: Dict[str, str],
    max_tokens: int = 256,
    cot_max_tokens: int = 512,
    dataset_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run local + selective CoT scenario.
    Decides whether to use CoT based on question characteristics.
    
    Args:
        local_client: Local LLM client
        remote_client: Remote LLM client
        local_model_name: Name of local model
        remote_model_name: Name of remote model
        question: Question text
        options: Options dictionary
        max_tokens: Maximum tokens for answer
        cot_max_tokens: Maximum tokens for CoT generation
        dataset_name: Optional dataset name for dataset-specific rules (e.g., 'clinical_knowledge')
    
    Returns:
        Result dictionary with answer, is_correct, processing_time, etc.
        Includes cot_used (bool), cot_reason (str), and cot_text (str or None)
    """
    from dpprivqa.qa.selective_cot import should_use_cot, should_use_cot_clinical_knowledge
    
    # Use dataset-specific rules if available
    if dataset_name == 'clinical_knowledge' or dataset_name == 'mmlu_clinical_knowledge':
        use_cot, reason = should_use_cot_clinical_knowledge(question, options)
    else:
        # Use generic rules
        use_cot, reason = should_use_cot(question, options)
    
    if not use_cot:
        # Use local only
        result = run_local_only(
            local_client, local_model_name,
            question, options, max_tokens
        )
        result['scenario'] = 'local_selective_cot'
        result['cot_used'] = False
        result['cot_reason'] = reason
        result['cot_text'] = None
        result['cot_generation_time'] = 0.0
        result['remote_model'] = None
        return result
    
    # Use CoT
    result = run_local_with_cot(
        local_client, remote_client,
        local_model_name, remote_model_name,
        question, options, max_tokens, cot_max_tokens
    )
    result['scenario'] = 'local_selective_cot'
    result['cot_used'] = True
    result['cot_reason'] = reason
    return result

