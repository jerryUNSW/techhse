"""
Mechanism registry for easy access to privacy-preserving mechanisms.
"""

from typing import Dict, Callable, Optional
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from dpprivqa.mechanisms.phrasedp import phrasedp_sanitize
from dpprivqa.mechanisms.phrasedp_plus import phrasedp_plus_sanitize


# Registry of mechanism functions
MECHANISM_REGISTRY: Dict[str, Callable] = {}


def register_mechanism(name: str, func: Callable):
    """Register a mechanism function."""
    MECHANISM_REGISTRY[name] = func


def get_mechanism(name: str) -> Optional[Callable]:
    """Get a mechanism function by name."""
    return MECHANISM_REGISTRY.get(name)


def create_phrasedp_function(nebius_client: OpenAI, nebius_model_name: str, medical_mode: bool = False):
    """Create a PhraseDP sanitization function with bound parameters."""
    def sanitize(text: str, epsilon: float, **kwargs):
        return phrasedp_sanitize(
            text=text,
            epsilon=epsilon,
            nebius_client=nebius_client,
            nebius_model_name=nebius_model_name,
            medical_mode=medical_mode,
            **kwargs
        )
    return sanitize


def create_phrasedp_plus_function(nebius_client: OpenAI, nebius_model_name: str, medical_mode: bool = False, metadata: Optional[Dict] = None):
    """Create a PhraseDP+ sanitization function with bound parameters."""
    def sanitize(text: str, epsilon: float, **kwargs):
        return phrasedp_plus_sanitize(
            text=text,
            epsilon=epsilon,
            nebius_client=nebius_client,
            nebius_model_name=nebius_model_name,
            medical_mode=medical_mode,
            metadata=metadata,
            **kwargs
        )
    return sanitize


# Register default mechanisms
# Note: These will be properly initialized when clients are available
register_mechanism('phrasedp', phrasedp_sanitize)
register_mechanism('phrasedp_plus', phrasedp_plus_sanitize)


