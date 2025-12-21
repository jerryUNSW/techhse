"""
LLM client wrappers for local and remote models.
"""

import os
from typing import Optional, List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_nebius_client() -> OpenAI:
    """
    Get Nebius API client for local models.
    
    Returns:
        OpenAI-compatible Nebius client
    
    Raises:
        ValueError: If API key not found
    """
    api_key = os.getenv("NEBIUS_API") or os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("Nebius API key not found. Set NEBIUS_API or NEBIUS_API_KEY in .env")
    
    base_url = os.getenv("NEBIUS_BASE_URL") or "https://api.studio.nebius.ai/v1/"
    return OpenAI(base_url=base_url, api_key=api_key)


def get_remote_llm_client() -> OpenAI:
    """
    Get remote LLM client (OpenAI).
    
    Returns:
        OpenAI client
    
    Raises:
        ValueError: If API key not found
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)


def create_completion_with_model_support(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.0
):
    """
    Create completion with model-specific support (handles GPT-5, Anthropic, etc.).
    
    Args:
        client: OpenAI-compatible client
        model_name: Name of the model
        messages: List of message dicts
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Completion response
    """
    # Handle GPT-5 which uses max_completion_tokens instead of max_tokens
    if "gpt-5" in model_name.lower() or "gpt-5-chat-latest" in model_name.lower():
        # GPT-5 uses internal "reasoning tokens" that count against the limit
        # If all tokens are used for reasoning, there's nothing left for output content
        # Use 4096 to allow both reasoning (~2000 tokens) and output (~2000 tokens)
        if max_tokens < 4096:
            max_tokens = 4096
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens
            # GPT-5 doesn't support temperature parameter
        )
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )


def find_working_nebius_model(client: OpenAI, model_name: str) -> str:
    """
    Probe Nebius to find a working local model ID.
    
    Args:
        client: Nebius client
        model_name: Desired model name
    
    Returns:
        Working model name
    
    Raises:
        ValueError: If model doesn't work
    """
    try:
        resp = create_completion_with_model_support(
            client,
            model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0,
        )
        # If no exception, model works
        return model_name
    except Exception:
        raise ValueError(f"Local model {model_name} is not working")

