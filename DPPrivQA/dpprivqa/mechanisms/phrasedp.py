"""
PhraseDP: Phrase-level Differential Privacy for text sanitization.

This module implements the PhraseDP mechanism that uses phrase-level
differential privacy to sanitize text while preserving semantic meaning.
"""

import os
import numpy as np
from typing import Optional, List, Dict, Any
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from dpprivqa.mechanisms.base import PrivacyMechanism
from dpprivqa.qa.prompts import load_system_prompt, load_user_prompt_template


def get_embedding(model: SentenceTransformer, text: str) -> np.ndarray:
    """Get embedding for a given text."""
    return model.encode(text, convert_to_tensor=True)


def differentially_private_replacement(
    target_phrase: str,
    epsilon: float,
    candidate_phrases: List[str],
    candidate_embeddings: Dict[str, np.ndarray],
    sbert_model: SentenceTransformer
) -> str:
    """
    Apply exponential mechanism to select a differentially private replacement.
    
    Args:
        target_phrase: Original phrase to replace
        epsilon: Privacy parameter
        candidate_phrases: List of candidate replacement phrases
        candidate_embeddings: Dictionary mapping phrases to their embeddings
        sbert_model: Sentence-BERT model for embedding computation
    
    Returns:
        Selected replacement phrase
    """
    # Compute embedding for target phrase
    target_embedding = get_embedding(sbert_model, target_phrase).cpu().numpy()
    
    # Ensure target_embedding is 2D (1, n_features)
    if target_embedding.ndim == 1:
        target_embedding = target_embedding.reshape(1, -1)
    
    # Stack precomputed candidate embeddings
    candidate_embeddings_matrix = np.vstack([
        candidate_embeddings[phrase] for phrase in candidate_phrases
    ])
    
    # Compute cosine similarity
    similarities = cosine_similarity(target_embedding, candidate_embeddings_matrix)[0]
    
    # Convert similarity to distance
    distances = 1 - similarities
    
    # Apply the exponential mechanism
    p_unnorm = np.exp(-epsilon * distances)
    p_norm = p_unnorm / np.sum(p_unnorm)  # Normalize to make it a probability distribution
    
    # Sample a replacement
    return np.random.choice(candidate_phrases, p=p_norm)


def generate_sentence_replacements_with_nebius_diverse(
    nebius_client: OpenAI,
    nebius_model_name: str,
    input_sentence: str,
    num_return_sequences: int = 10,
    max_tokens: int = 150,
    num_api_calls: int = 10,
    enforce_similarity_filter: bool = True,
    filter_margin: float = 0.05,
    low_band_quota_boost: bool = True,
    refill_underfilled_bands: bool = True,
    max_refill_retries: int = 2,
    equal_band_target: Optional[int] = None,
    global_equalize_max_loops: int = 5,
    verbose: bool = False,
    mode: str = "normal",
    metamap_phrases: Optional[List[str]] = None
) -> List[str]:
    """
    Generate diverse candidate paraphrases with targeted similarity levels.
    
    This function generates candidates across different similarity bands (0.1-0.9)
    to provide better exponential mechanism effectiveness.
    
    Args:
        nebius_client: Nebius API client
        nebius_model_name: Name of the Nebius model to use
        input_sentence: Original sentence to paraphrase
        num_return_sequences: Number of candidates per API call
        max_tokens: Maximum tokens per candidate
        num_api_calls: Number of API calls to make (default: 10, but uses 5 bands)
        mode: Text processing mode - "normal" or "medqa-ume" (medical mode)
        metamap_phrases: List of medical phrases to preserve (for medical mode)
        **kwargs: Additional parameters for band management
    
    Returns:
        List of unique paraphrased sentences
    """
    # Load base prompts
    try:
        system_prompt = load_system_prompt()
        user_prompt_template = load_user_prompt_template()
    except FileNotFoundError:
        # Fallback prompts if files don't exist
        system_prompt = """
        You are a highly skilled AI assistant specializing in natural language processing.
        Your primary goal is to rephrase user questions in a generalized manner,
        preserving their core logical structure and intent, but explicitly removing
        or generalizing any sensitive personal identifying information (PII) or
        unique details that could lead to re-identification.
        """
        user_prompt_template = """
        Generalize and paraphrase the following sentence.
        Ensure the logical structure and core meaning are maintained, but remove or generalize
        any sensitive personal information.
        
        Original: {input_sentence}
        Paraphrase:
        """
    
    # Append medical mode instructions if needed
    if mode == "medqa-ume":
        system_prompt += """
        MEDICAL TEXT SPECIAL REQUIREMENTS:
        - PRESERVE ALL MEDICAL TERMINOLOGY: Keep every medical term, diagnosis, symptom, treatment
        - MAINTAIN DIAGNOSTIC ACCURACY: Ensure medical questions remain diagnostically meaningful
        - REMOVE ONLY PII: Eliminate only personally identifiable information (names, ages, locations, dates)
        - PRESERVE CLINICAL CONTEXT: Keep all medical reasoning, symptom relationships, and diagnostic pathways
        """
        
        if metamap_phrases:
            system_prompt += f"""
            CRITICAL MEDICAL CONCEPTS TO PRESERVE (from metamap analysis):
            {', '.join(metamap_phrases)}
            - DO NOT PERTURB any of the above medical concepts and phrases
            - MASK ONLY PII within these phrases
            """
    
    # Define similarity bands (5 bands for diverse generation)
    similarity_bands = [
        {'level': 'band_0.0-0.2', 'target': '0.0-0.2', 'description': 'Extreme abstraction'},
        {'level': 'band_0.2-0.4', 'target': '0.2-0.4', 'description': 'Heavy abstraction'},
        {'level': 'band_0.4-0.6', 'target': '0.4-0.6', 'description': 'Moderate abstraction'},
        {'level': 'band_0.6-0.8', 'target': '0.6-0.8', 'description': 'Light abstraction'},
        {'level': 'band_0.8-1.0', 'target': '0.8-1.0', 'description': 'Minimal abstraction'},
    ]
    
    all_candidates = []
    
    # Generate candidates for each similarity band
    for band in similarity_bands[:5]:  # Use first 5 bands
        user_prompt = f"""
        Generate paraphrases with similarity to the original between {band['target']}.
        {band['description']}: {band['description']}
        
        Original: {input_sentence}
        Paraphrase:
        """
        
        try:
            response = nebius_client.chat.completions.create(
                model=nebius_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.95,
                n=num_return_sequences
            )
            
            for choice in response.choices:
                candidate = choice.message.content.strip()
                if candidate and candidate.lower() != input_sentence.lower():
                    if len(candidate) > 10:
                        all_candidates.append(candidate)
        
        except Exception as e:
            if verbose:
                print(f"Error generating candidates for {band['level']}: {e}")
            continue
    
    # Remove duplicates while preserving order
    seen = set()
    unique_candidates = []
    for candidate in all_candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)
    
    return unique_candidates


def phrasedp_sanitize(
    text: str,
    epsilon: float,
    nebius_client: OpenAI,
    nebius_model_name: str,
    sbert_model: Optional[SentenceTransformer] = None,
    medical_mode: bool = False,
    metamap_phrases: Optional[List[str]] = None
) -> str:
    """
    Sanitize text using PhraseDP (phrase-level differential privacy).
    
    Args:
        text: Input text to sanitize
        epsilon: Privacy parameter (higher = less privacy, more utility)
        nebius_client: Nebius API client for candidate generation
        nebius_model_name: Name of the Nebius model to use
        sbert_model: Sentence-BERT model (will be loaded if not provided)
        medical_mode: If True, preserve medical terminology
        metamap_phrases: List of medical phrases to preserve (for medical mode)
    
    Returns:
        Sanitized text using PhraseDP
    """
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {epsilon}")
    
    if not text or not text.strip():
        return text
    
    # Load Sentence-BERT model if not provided
    if sbert_model is None:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Determine mode
    mode = "medqa-ume" if medical_mode else "normal"
    
    # Step 1: Generate diverse candidate sentence-level replacements
    candidate_sentences = generate_sentence_replacements_with_nebius_diverse(
        nebius_client=nebius_client,
        nebius_model_name=nebius_model_name,
        input_sentence=text,
        num_return_sequences=10,
        mode=mode,
        metamap_phrases=metamap_phrases
    )
    
    if not candidate_sentences:
        raise ValueError("No candidate sentences were generated. Check the Nebius API call.")
    
    # Step 2: Precompute embeddings for all candidates
    candidate_embeddings = {
        sent: get_embedding(sbert_model, sent).cpu().numpy()
        for sent in candidate_sentences
    }
    
    # Step 3: Select a replacement using exponential mechanism
    dp_replacement = differentially_private_replacement(
        target_phrase=text,
        epsilon=epsilon,
        candidate_phrases=candidate_sentences,
        candidate_embeddings=candidate_embeddings,
        sbert_model=sbert_model
    )
    
    return dp_replacement


class PhraseDP(PrivacyMechanism):
    """PhraseDP mechanism implementation."""
    
    def __init__(
        self,
        nebius_client: OpenAI,
        nebius_model_name: str,
        sbert_model: Optional[SentenceTransformer] = None,
        medical_mode: bool = False
    ):
        """
        Initialize PhraseDP mechanism.
        
        Args:
            nebius_client: Nebius API client for candidate generation
            nebius_model_name: Name of the Nebius model to use
            sbert_model: Sentence-BERT model (will be loaded if not provided)
            medical_mode: If True, preserve medical terminology
        """
        self.nebius_client = nebius_client
        self.nebius_model_name = nebius_model_name
        self.sbert_model = sbert_model or SentenceTransformer('all-MiniLM-L6-v2')
        self.medical_mode = medical_mode
    
    def sanitize(
        self,
        text: str,
        epsilon: float,
        **kwargs
    ) -> str:
        """
        Sanitize text using PhraseDP.
        
        Args:
            text: Input text to sanitize
            epsilon: Privacy parameter
            **kwargs: Additional parameters (metamap_phrases for medical mode)
        
        Returns:
            Sanitized text
        """
        metamap_phrases = kwargs.get('metamap_phrases', None)
        return phrasedp_sanitize(
            text=text,
            epsilon=epsilon,
            nebius_client=self.nebius_client,
            nebius_model_name=self.nebius_model_name,
            sbert_model=self.sbert_model,
            medical_mode=self.medical_mode,
            metamap_phrases=metamap_phrases
        )
    
    def get_name(self) -> str:
        """Return the name of this mechanism."""
        return "phrasedp"


