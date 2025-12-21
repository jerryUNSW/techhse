"""
DPPrivQA pipeline: Local → PhraseDP sanitization → Remote CoT → Local answer.
"""

import time
from typing import Dict, Any, Optional, Callable
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from dpprivqa.qa.models import (
    create_completion_with_model_support,
    find_working_nebius_model
)
from dpprivqa.qa.prompts import (
    format_question_with_options,
    extract_letter_from_answer
)
from dpprivqa.mechanisms.phrasedp import phrasedp_sanitize
from dpprivqa.mechanisms.phrasedp_plus import phrasedp_plus_sanitize


def run_dpprivqa(
    local_client: OpenAI,
    remote_client: OpenAI,
    local_model_name: str,
    remote_model_name: str,
    question: str,
    options: Dict[str, str],
    mechanism: str = 'phrasedp',
    epsilon: float = 2.0,
    sbert_model: Optional[SentenceTransformer] = None,
    medical_mode: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
    metamap_phrases: Optional[list] = None,
    max_tokens: int = 256,
    cot_max_tokens: int = 512
) -> Dict[str, Any]:
    """
    Run DPPrivQA pipeline: sanitize question → generate CoT → generate answer.
    
    Args:
        local_client: Local LLM client
        remote_client: Remote LLM client
        local_model_name: Name of local model
        remote_model_name: Name of remote model
        question: Original question text
        options: Options dictionary
        mechanism: Mechanism name ('phrasedp' or 'phrasedp_plus')
        epsilon: Privacy parameter
        sbert_model: Sentence-BERT model (will be loaded if not provided)
        medical_mode: If True, preserve medical terminology
        metadata: Metadata for PhraseDP+ (optional)
        metamap_phrases: Medical phrases to preserve (optional)
        max_tokens: Maximum tokens for answer
        cot_max_tokens: Maximum tokens for CoT generation
    
    Returns:
        Result dictionary with sanitized_question, cot_text, answer, processing times, etc.
    """
    start_time = time.time()
    
    # Step 1: Sanitize question with PhraseDP
    sanitization_start = time.time()
    
    if mechanism == 'phrasedp_plus':
        sanitized_question = phrasedp_plus_sanitize(
            text=question,
            epsilon=epsilon,
            nebius_client=local_client,
            nebius_model_name=local_model_name,
            sbert_model=sbert_model,
            medical_mode=medical_mode,
            metadata=metadata,
            metamap_phrases=metamap_phrases
        )
    else:  # Default to phrasedp
        sanitized_question = phrasedp_sanitize(
            text=question,
            epsilon=epsilon,
            nebius_client=local_client,
            nebius_model_name=local_model_name,
            sbert_model=sbert_model,
            medical_mode=medical_mode,
            metamap_phrases=metamap_phrases
        )
    
    sanitization_time = time.time() - sanitization_start
    
    # Step 2: Generate CoT from remote using sanitized question (NO options for privacy)
    cot_start = time.time()
    prompt_lines = [
        "Here is the (possibly perturbed) question:",
        sanitized_question,
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
    
    # Step 3: Generate answer from local model using original question + CoT
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
        "sanitized_question": sanitized_question,
        "cot_text": cot_text,
        "question": question,
        "options": options,
        "processing_time": processing_time,
        "sanitization_time": sanitization_time,
        "cot_generation_time": cot_time,
        "local_model": local_model_name,
        "remote_model": remote_model_name,
        "mechanism": mechanism,
        "epsilon": epsilon,
        "scenario": "dpprivqa"
    }


