"""
PhraseDP+: PhraseDP with field-specific metadata context.

This module extends PhraseDP to incorporate background metadata
(e.g., field-specific context) for improved sanitization.
"""

from typing import Optional, Dict, Any, List
from sentence_transformers import SentenceTransformer
from openai import OpenAI

from dpprivqa.mechanisms.base import PrivacyMechanism
from dpprivqa.mechanisms.phrasedp import phrasedp_sanitize


def phrasedp_plus_sanitize(
    text: str,
    epsilon: float,
    nebius_client: OpenAI,
    nebius_model_name: str,
    sbert_model: Optional[SentenceTransformer] = None,
    metadata: Optional[Dict[str, Any]] = None,
    medical_mode: bool = False,
    metamap_phrases: Optional[List[str]] = None
) -> str:
    """
    Sanitize text using PhraseDP+ with field-specific metadata context.
    
    Args:
        text: Input text to sanitize
        epsilon: Privacy parameter
        nebius_client: Nebius API client for candidate generation
        nebius_model_name: Name of the Nebius model to use
        sbert_model: Sentence-BERT model (will be loaded if not provided)
        metadata: Dictionary with field-specific context (e.g., {"field": "medicine", "context": "clinical"})
        medical_mode: If True, preserve medical terminology
        metamap_phrases: List of medical phrases to preserve (for medical mode)
    
    Returns:
        Sanitized text using PhraseDP+
    """
    # PhraseDP+ enhances the prompt with metadata context
    # For now, we use PhraseDP with enhanced metadata in metamap_phrases
    # In a full implementation, this would modify the prompt generation
    
    # If metadata is provided, incorporate it into the sanitization
    enhanced_metamap_phrases = list(metamap_phrases) if metamap_phrases else []
    
    if metadata:
        # Extract field-specific context from metadata
        field = metadata.get('field', '')
        context = metadata.get('context', '')
        
        # Add field-specific guidance to metamap phrases
        if field:
            enhanced_metamap_phrases.append(f"field: {field}")
        if context:
            enhanced_metamap_phrases.append(f"context: {context}")
    
    # Use PhraseDP with enhanced context
    return phrasedp_sanitize(
        text=text,
        epsilon=epsilon,
        nebius_client=nebius_client,
        nebius_model_name=nebius_model_name,
        sbert_model=sbert_model,
        medical_mode=medical_mode,
        metamap_phrases=enhanced_metamap_phrases if enhanced_metamap_phrases else None
    )


class PhraseDPPlus(PrivacyMechanism):
    """PhraseDP+ mechanism implementation with metadata context."""
    
    def __init__(
        self,
        nebius_client: OpenAI,
        nebius_model_name: str,
        sbert_model: Optional[SentenceTransformer] = None,
        medical_mode: bool = False,
        default_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PhraseDP+ mechanism.
        
        Args:
            nebius_client: Nebius API client for candidate generation
            nebius_model_name: Name of the Nebius model to use
            sbert_model: Sentence-BERT model (will be loaded if not provided)
            medical_mode: If True, preserve medical terminology
            default_metadata: Default metadata to use if not provided in sanitize()
        """
        self.nebius_client = nebius_client
        self.nebius_model_name = nebius_model_name
        self.sbert_model = sbert_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.medical_mode = medical_mode
        self.default_metadata = default_metadata or {}
    
    def sanitize(
        self,
        text: str,
        epsilon: float,
        **kwargs
    ) -> str:
        """
        Sanitize text using PhraseDP+.
        
        Args:
            text: Input text to sanitize
            epsilon: Privacy parameter
            **kwargs: Additional parameters (metadata, metamap_phrases)
        
        Returns:
            Sanitized text
        """
        metadata = kwargs.get('metadata', self.default_metadata)
        metamap_phrases = kwargs.get('metamap_phrases', None)
        
        return phrasedp_plus_sanitize(
            text=text,
            epsilon=epsilon,
            nebius_client=self.nebius_client,
            nebius_model_name=self.nebius_model_name,
            sbert_model=self.sbert_model,
            metadata=metadata,
            medical_mode=self.medical_mode,
            metamap_phrases=metamap_phrases
        )
    
    def get_name(self) -> str:
        """Return the name of this mechanism."""
        return "phrasedp_plus"


