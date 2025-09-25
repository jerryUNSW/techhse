#!/usr/bin/env python3
"""
Unified Privacy-Preserving Text Sanitization Methods
====================================================

This module provides a unified interface for all privacy-preserving text sanitization methods:
- PhraseDP (phrase-level differential privacy)
- InferDPT (inference-based differential privacy for text)
- SANTEXT+ (sanitized text plus)
- CUSTEXT+ (customized text)
- CluSanT (clustering-based sanitization)

Each method is provided as a clean function that can be easily imported and used.
"""

import os
import numpy as np
import yaml
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Import the individual method implementations
from santext_integration import create_santext_mechanism
from inferdpt import perturb_sentence as _inferdpt_perturb_sentence
from cus_text_ppi_protection_experiment import sanitize_with_custext as _custext_sanitize
from cus_text_ppi_protection_experiment import load_counter_fitting_vectors as _load_custext_vectors
import cus_text_ppi_protection_experiment as _custext_mod
from clusant_ppi_protection_experiment import run_clusant_ppi_experiment
import sys
import re
import utils

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Load environment variables from .env
load_dotenv()

# Global instances for methods that need initialization
_santext_mechanism = None
_custext_components = None
_clusant_mechanism = None
_clusant_resources = None  # cache embeddings/paths for CluSanT
_sbert_model = None

def _get_sbert_model():
    """Get or create Sentence-BERT model."""
    global _sbert_model
    if _sbert_model is None:
        _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _sbert_model

def _get_santext_mechanism(epsilon: float = None, p: float = 0.3):
    """Get or create SANTEXT+ mechanism."""
    global _santext_mechanism
    if _santext_mechanism is None:
        if epsilon is None:
            epsilon = config.get('epsilon', 1.0)
        _santext_mechanism = create_santext_mechanism(epsilon=epsilon, p=p)
    else:
        # Update epsilon if different
        if epsilon is not None and epsilon != _santext_mechanism.epsilon:
            _santext_mechanism.epsilon = epsilon
    return _santext_mechanism

def _get_custext_components():
    """Get or create CUSTEXT+ components (embeddings, vocab, stopwords)."""
    global _custext_components
    if _custext_components is None:
        # Load counter-fitted vectors using the same logic as PPI experiment
        try:
            emb_matrix, idx2word, word2idx = _load_custext_vectors(_custext_mod.VECTORS_PATH)
        except Exception as e:
            raise RuntimeError(f"Failed to load CusText vectors: {e}")

        # Load stopwords
        try:
            import nltk
            from nltk.corpus import stopwords
            nltk.download('stopwords', quiet=True)
            stop_set = set(stopwords.words('english'))
        except Exception:
            stop_set = set()

        _custext_components = {
            'emb_matrix': emb_matrix,
            'idx2word': idx2word,
            'word2idx': word2idx,
            'stop_set': stop_set,
        }
    return _custext_components

def _get_clusant_resources():
    """Prepare and cache CluSanT resources (embeddings, paths, classes)."""
    global _clusant_resources
    if _clusant_resources is not None:
        return _clusant_resources

    clusant_root = '/home/yizhang/tech4HSE/CluSanT'
    src_path = os.path.join(clusant_root, 'src')
    if src_path not in sys.path:
        sys.path.append(src_path)
    from embedding_handler import EmbeddingHandler
    from clusant import CluSanT  # noqa: F401

    # Ensure embeddings exist
    emb_dir = os.path.join(clusant_root, 'embeddings')
    os.makedirs(emb_dir, exist_ok=True)
    emb_path = os.path.join(emb_dir, 'all-MiniLM-L6-v2.txt')
    handler = EmbeddingHandler(model_name='all-MiniLM-L6-v2')

    # Generate from cluster wordlists if missing
    if not os.path.exists(emb_path):
        wordlists = [
            os.path.join(clusant_root, 'clusters', 'gpt-4', 'LOC.json'),
            os.path.join(clusant_root, 'clusters', 'gpt-4', 'ORG.json'),
        ]
        handler.generate_and_save_embeddings(wordlists, emb_dir)
    embeddings = handler.load_embeddings(emb_path)

    _clusant_resources = {
        'root': clusant_root,
        'embeddings': embeddings,
        'EmbeddingHandler': EmbeddingHandler,
    }
    return _clusant_resources

# =============================================================================
# PUBLIC API FUNCTIONS
# =============================================================================

def phrasedp_sanitize_text(text: str, epsilon: float = None, nebius_client=None, nebius_model_name=None) -> str:
    """
    Sanitize text using PhraseDP (phrase-level differential privacy).
    
    Args:
        text: Input text to sanitize
        epsilon: Privacy parameter (default from config)
        nebius_client: Nebius API client (required)
        nebius_model_name: Nebius model name (required)
    
    Returns:
        Sanitized text using PhraseDP
    """
    if epsilon is None:
        epsilon = config.get('epsilon', 1.0)
    
    if nebius_client is None or nebius_model_name is None:
        raise ValueError("PhraseDP requires nebius_client and nebius_model_name parameters")
    
    sbert_model = _get_sbert_model()
    return utils.phrase_DP_perturbation_diverse(nebius_client, nebius_model_name, text, epsilon, sbert_model)

def inferdpt_sanitize_text(text: str, epsilon: float = None) -> str:
    """
    Sanitize text using InferDPT (inference-based differential privacy for text).
    
    Args:
        text: Input text to sanitize
        epsilon: Privacy parameter (default from config)
    
    Returns:
        Sanitized text using InferDPT
    """
    if epsilon is None:
        epsilon = config.get('epsilon', 1.0)
    
    return _inferdpt_perturb_sentence(text, epsilon)

def santext_sanitize_text(text: str, epsilon: float = None, p: float = 0.3) -> str:
    """
    Sanitize text using SANTEXT+ (sanitized text plus).
    
    Args:
        text: Input text to sanitize
        epsilon: Privacy parameter (default from config)
        p: Probability of sanitizing non-sensitive words
    
    Returns:
        Sanitized text using SANTEXT+
    """
    if epsilon is None:
        epsilon = config.get('epsilon', 1.0)
    
    mechanism = _get_santext_mechanism(epsilon, p)
    
    # Build vocabulary with the text if not already built
    if not mechanism.vocab_words:
        mechanism.build_vocabulary([text])
    
    return mechanism.sanitize_text(text)

def custext_sanitize_text(text: str, epsilon: float = None, top_k: int = 5) -> str:
    """
    Sanitize text using CUSTEXT+ (customized text).
    
    Args:
        text: Input text to sanitize
        epsilon: Privacy parameter (default from config)
        top_k: Number of top candidates to consider
    
    Returns:
        Sanitized text using CUSTEXT+
    """
    if epsilon is None:
        epsilon = config.get('epsilon', 1.0)
    
    components = _get_custext_components()
    
    return _custext_sanitize(
        text=text,
        epsilon=epsilon,
        top_k=top_k,
        save_stop_words=True,
        emb_matrix=components['emb_matrix'],
        idx2word=components['idx2word'],
        word2idx=components['word2idx'],
        stop_set=components['stop_set']
    )

def clusant_sanitize_text(text: str, epsilon: float = None) -> str:
    """
    Sanitize text using CluSanT (clustering-based sanitization), mirroring the PPI experiment flow.
    """
    if epsilon is None:
        epsilon = config.get('epsilon', 1.0)

    res = _get_clusant_resources()
    clusant_root = res['root']
    embeddings = res['embeddings']

    # Import here after sys.path updated
    from clusant import CluSanT

    # Ensure CluSanT's relative directories exist and run inside its root
    original_cwd = os.getcwd()
    try:
        os.makedirs(os.path.join(clusant_root, 'clusters'), exist_ok=True)
        os.makedirs(os.path.join(clusant_root, 'centroids'), exist_ok=True)
        os.makedirs(os.path.join(clusant_root, 'intra'), exist_ok=True)
        os.makedirs(os.path.join(clusant_root, 'inter'), exist_ok=True)
        os.chdir(clusant_root)

        # Instantiate CluSanT with same config as PPI script
        clus = CluSanT(
            embedding_file='all-MiniLM-L6-v2',
            embeddings=embeddings,
            epsilon=epsilon,
            num_clusters=336,
            mechanism='clusant',
            metric_to_create_cluster='euclidean',
            distance_metric_for_cluster='euclidean',
            distance_metric_for_words='euclidean',
            dp_type='metric',
            K=16,
        )

        sanitized_text = text

        # Detect present targets (multi-word first), then single word, using word boundaries
        targets_present = []
        for w in embeddings.keys():
            if ' ' in w and re.search(rf"\b{re.escape(w)}\b", sanitized_text, flags=re.IGNORECASE):
                targets_present.append(w)
        for w in embeddings.keys():
            if ' ' not in w and re.search(rf"\b{re.escape(w)}\b", sanitized_text, flags=re.IGNORECASE):
                targets_present.append(w)

        # Deduplicate, process longer first
        targets_present = sorted(set(targets_present), key=lambda x: (-len(x), x))

        for t in targets_present:
            new = clus.replace_word(t)
            if not new:
                continue
            pattern = re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE)
            if pattern.search(sanitized_text):
                sanitized_text = pattern.sub(new, sanitized_text)

        return sanitized_text
    finally:
        os.chdir(original_cwd)

# =============================================================================
# BATCH PROCESSING FUNCTIONS
# =============================================================================

def phrasedp_sanitize_batch(texts: List[str], epsilon: float = None, nebius_client=None, nebius_model_name=None) -> List[str]:
    """Sanitize a batch of texts using PhraseDP."""
    return [phrasedp_sanitize_text(text, epsilon, nebius_client, nebius_model_name) for text in texts]

def inferdpt_sanitize_batch(texts: List[str], epsilon: float = None) -> List[str]:
    """Sanitize a batch of texts using InferDPT."""
    return [inferdpt_sanitize_text(text, epsilon) for text in texts]

def santext_sanitize_batch(texts: List[str], epsilon: float = None, p: float = 0.3) -> List[str]:
    """Sanitize a batch of texts using SANTEXT+."""
    if epsilon is None:
        epsilon = config.get('epsilon', 1.0)
    
    mechanism = _get_santext_mechanism(epsilon, p)
    
    # Build vocabulary with all texts
    if not mechanism.vocab_words:
        mechanism.build_vocabulary(texts)
    
    return [mechanism.sanitize_text(text) for text in texts]

def custext_sanitize_batch(texts: List[str], epsilon: float = None, top_k: int = 5) -> List[str]:
    """Sanitize a batch of texts using CUSTEXT+."""
    return [custext_sanitize_text(text, epsilon, top_k) for text in texts]

def clusant_sanitize_batch(texts: List[str], epsilon: float = None) -> List[str]:
    """Sanitize a batch of texts using CluSanT."""
    return [clusant_sanitize_text(text, epsilon) for text in texts]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_available_methods() -> List[str]:
    """Get list of available sanitization methods."""
    return ['phrasedp', 'inferdpt', 'santext', 'custext', 'clusant']

def sanitize_text_with_method(text: str, method: str, **kwargs) -> str:
    """
    Sanitize text using the specified method.
    
    Args:
        text: Input text to sanitize
        method: Method name ('phrasedp', 'inferdpt', 'santext', 'custext', 'clusant')
        **kwargs: Additional parameters for the specific method
    
    Returns:
        Sanitized text
    """
    method_map = {
        'phrasedp': phrasedp_sanitize_text,
        'inferdpt': inferdpt_sanitize_text,
        'santext': santext_sanitize_text,
        'custext': custext_sanitize_text,
        'clusant': clusant_sanitize_text
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}. Available methods: {get_available_methods()}")
    
    return method_map[method](text, **kwargs)

def reset_global_instances():
    """Reset all global instances (useful for testing)."""
    global _santext_mechanism, _custext_components, _clusant_mechanism, _sbert_model
    _santext_mechanism = None
    _custext_components = None
    _clusant_mechanism = None
    _sbert_model = None

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example usage
    sample_text = "A 65-year-old male is treated for anal carcinoma with therapy including external beam radiation."
    
    print("Testing sanitization methods:")
    print(f"Original: {sample_text}")
    print("-" * 60)
    
    # # Test PhraseDP (requires Nebius client)
    # try:
    #     # Use unified Nebius getter
    #     nebius_client = utils.get_nebius_client()
    #     nebius_model_name = config.get('local_model', 'microsoft/phi-4')

    #     phrasedp_result = phrasedp_sanitize_text(sample_text, epsilon=1.0, nebius_client=nebius_client, nebius_model_name=nebius_model_name)
    #     print(f"PhraseDP: {phrasedp_result}")
    # except Exception as e:
    #     print(f"PhraseDP: [ERROR - {e}] (requires Nebius client)")
    
    # # Test InferDPT
    # try:
    #     inferdpt_result = inferdpt_sanitize_text(sample_text, epsilon=1.0)
    #     print(f"InferDPT: {inferdpt_result}")
    # except Exception as e:
    #     print(f"InferDPT: [ERROR - {e}]")
    
    # # Test SANTEXT+
    # try:
    #     santext_result = santext_sanitize_text(sample_text, epsilon=1.0)
    #     print(f"SANTEXT+: {santext_result}")
    # except Exception as e:
    #     print(f"SANTEXT+: [ERROR - {e}]")
    
    # Test CUSTEXT+
    try:
        custext_result = custext_sanitize_text(sample_text, epsilon=1.0)
        print(f"CUSTEXT+: {custext_result}")
    except Exception as e:
        print(f"CUSTEXT+: [ERROR - {e}]")
    
    # Test CluSanT
    try:
        clusant_result = clusant_sanitize_text(sample_text, epsilon=1.0)
        print(f"CluSanT: {clusant_result}")
    except Exception as e:
        print(f"CluSanT: [ERROR - {e}]")
    
    print("-" * 60)
    print("Available methods:", get_available_methods())
