"""
SANTEXT+: Sanitized Text Plus mechanism.

This module wraps the real SANTEXT+ implementation from the root directory.
"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path to import the real santext
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    # Try importing directly from santext_integration first
    from santext_integration import create_santext_mechanism
    REAL_IMPL_AVAILABLE = True
    USE_DIRECT_IMPORT = True
except ImportError:
    try:
        # Fallback to sanitization_methods wrapper
        from sanitization_methods import santext_sanitize_text as _santext_sanitize_text
        REAL_IMPL_AVAILABLE = True
        USE_DIRECT_IMPORT = False
    except ImportError as e:
        REAL_IMPL_AVAILABLE = False
        IMPORT_ERROR = str(e)
        USE_DIRECT_IMPORT = False

# Global mechanism instance for SANTEXT+
_santext_mechanism = None

from dpprivqa.mechanisms.base import PrivacyMechanism


def santext_sanitize_text(text: str, epsilon: float = 1.0, p: float = 0.3, sbert_model: Optional = None) -> str:
    """
    Sanitize text using SANTEXT+ (sanitized text plus).
    
    This function uses the real SANTEXT+ implementation from the root directory.
    
    Args:
        text: Input text to sanitize
        epsilon: Privacy parameter
        p: Probability of sanitizing non-sensitive words (default: 0.3)
        sbert_model: Not used (kept for compatibility with interface)
    
    Returns:
        Sanitized text using SANTEXT+
    """
    if not REAL_IMPL_AVAILABLE:
        raise ImportError(
            f"Real SANTEXT+ implementation not available: {IMPORT_ERROR}\n"
            "Please ensure santext_integration.py and sanitization_methods.py are in the project root."
        )
    
    if epsilon <= 0:
        raise ValueError(f"Epsilon must be positive, got {epsilon}")
    
    if not text or not text.strip():
        return text
    
    # Use the real implementation
    global _santext_mechanism
    if USE_DIRECT_IMPORT:
        # Create or reuse mechanism instance
        if _santext_mechanism is None or _santext_mechanism.epsilon != epsilon or _santext_mechanism.p != p:
            _santext_mechanism = create_santext_mechanism(epsilon=epsilon, p=p)
        # Build vocabulary if needed (it builds from the text)
        if not _santext_mechanism.vocab_words:
            _santext_mechanism.build_vocabulary([text])
        return _santext_mechanism.sanitize_text(text)
    else:
        return _santext_sanitize_text(text, epsilon=epsilon, p=p)


class SanText(PrivacyMechanism):
    """SANTEXT+ mechanism implementation."""
    
    def sanitize(self, text: str, epsilon: float, **kwargs) -> str:
        """Sanitize text using SANTEXT+."""
        p = kwargs.get('p', 0.3)
        return santext_sanitize_text(text, epsilon, p)
    
    def get_name(self) -> str:
        """Return the name of this mechanism."""
        return "santext"
