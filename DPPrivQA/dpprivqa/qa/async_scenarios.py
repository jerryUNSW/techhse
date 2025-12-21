"""
Async wrappers for QA scenario functions.

These wrappers allow concurrent execution of scenario functions using asyncio.
"""

import asyncio
from typing import Dict, Any, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from dpprivqa.qa.scenarios import (
    run_local_only,
    run_local_with_cot,
    run_remote_only
)
from dpprivqa.qa.dpprivqa import run_dpprivqa


async def async_run_local_only(
    local_client: OpenAI,
    local_model_name: str,
    question: str,
    options: Dict[str, str],
    max_tokens: int = 256
) -> Dict[str, Any]:
    """
    Async wrapper for run_local_only.
    
    Args:
        local_client: Local LLM client
        local_model_name: Name of local model
        question: Question text
        options: Options dictionary
        max_tokens: Maximum tokens for response
    
    Returns:
        Result dictionary with answer, is_correct, processing_time, etc.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        run_local_only,
        local_client,
        local_model_name,
        question,
        options,
        max_tokens
    )


async def async_run_local_with_cot(
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
    Async wrapper for run_local_with_cot.
    
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
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_local_with_cot(
            local_client,
            remote_client,
            local_model_name,
            remote_model_name,
            question,
            options,
            max_tokens,
            cot_max_tokens
        )
    )


async def async_run_remote_only(
    remote_client: OpenAI,
    remote_model_name: str,
    question: str,
    options: Dict[str, str],
    max_tokens: int = 256
) -> Dict[str, Any]:
    """
    Async wrapper for run_remote_only.
    
    Args:
        remote_client: Remote LLM client
        remote_model_name: Name of remote model
        question: Question text
        options: Options dictionary
        max_tokens: Maximum tokens for response
    
    Returns:
        Result dictionary with answer, is_correct, processing_time, etc.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        run_remote_only,
        remote_client,
        remote_model_name,
        question,
        options,
        max_tokens
    )


async def async_run_dpprivqa(
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
    Async wrapper for run_dpprivqa.
    
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
        medical_mode: Whether to use medical mode for PhraseDP+
        metadata: Optional metadata
        metamap_phrases: Optional MetaMap phrases for PhraseDP+
        max_tokens: Maximum tokens for answer
        cot_max_tokens: Maximum tokens for CoT generation
    
    Returns:
        Result dictionary with answer, sanitized_question, cot_text, etc.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: run_dpprivqa(
            local_client,
            remote_client,
            local_model_name,
            remote_model_name,
            question,
            options,
            mechanism,
            epsilon,
            sbert_model,
            medical_mode,
            metadata,
            metamap_phrases,
            max_tokens,
            cot_max_tokens
        )
    )


