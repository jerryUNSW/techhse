"""
InferDPT: Inference-based Differential Privacy for Text.

This module wraps the real InferDPT implementation from the root directory.
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path to import the real inferdpt
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Try importing directly from inferdpt first
    from inferdpt import perturb_sentence as _perturb_sentence, initialize_embeddings
    REAL_IMPL_AVAILABLE = True
    USE_DIRECT_IMPORT = True
except ImportError:
    try:
        # Fallback to sanitization_methods wrapper
        from sanitization_methods import inferdpt_sanitize_text as _inferdpt_sanitize_text
        REAL_IMPL_AVAILABLE = True
        USE_DIRECT_IMPORT = False
    except ImportError as e:
        REAL_IMPL_AVAILABLE = False
        IMPORT_ERROR = str(e)
        USE_DIRECT_IMPORT = False

from dpprivqa.mechanisms.base import PrivacyMechanism


def inferdpt_sanitize_text(text: str, epsilon: float = 1.0, sbert_model: Optional = None) -> str:
    """
    Sanitize text using InferDPT (inference-based differential privacy for text).
    
    This function uses the real InferDPT implementation from the root directory.
    
    Args:
        text: Input text to sanitize
        epsilon: Privacy parameter
        sbert_model: Not used (kept for compatibility with interface)
    
    Returns:
        Sanitized text using InferDPT
    """
    if not REAL_IMPL_AVAILABLE:
        raise ImportError(
            f"Real InferDPT implementation not available: {IMPORT_ERROR}\n"
            "Please ensure inferdpt.py and sanitization_methods.py are in the project root."
        )
    
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {epsilon}")
    
    if not text or not text.strip():
        return text
    
    # Use the real implementation
    if USE_DIRECT_IMPORT:
        # Initialize embeddings if needed (they're cached globally in the real implementation)
        token_to_vector_dict, sorted_distance_data, delta_f_new = initialize_embeddings(epsilon)
        return _perturb_sentence(text, epsilon, 
                                token_to_vector_dict=token_to_vector_dict,
                                sorted_distance_data=sorted_distance_data,
                                delta_f_new=delta_f_new)
    else:
        return _inferdpt_sanitize_text(text, epsilon=epsilon)


class InferDPT(PrivacyMechanism):
    """InferDPT mechanism implementation."""
    
    def sanitize(self, text: str, epsilon: float, **kwargs) -> str:
        """Sanitize text using InferDPT."""
        return inferdpt_sanitize_text(text, epsilon)
    
    def get_name(self) -> str:
        """Return the name of this mechanism."""
        return "inferdpt"
