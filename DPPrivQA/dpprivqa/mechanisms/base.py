"""
Base interface for privacy-preserving text sanitization mechanisms.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class PrivacyMechanism(ABC):
    """Abstract base class for privacy-preserving text sanitization mechanisms."""
    
    @abstractmethod
    def sanitize(
        self,
        text: str,
        epsilon: float,
        **kwargs
    ) -> str:
        """
        Sanitize text with given epsilon parameter.
        
        Args:
            text: Input text to sanitize
            epsilon: Privacy parameter (higher = less privacy, more utility)
            **kwargs: Additional mechanism-specific parameters
        
        Returns:
            Sanitized text
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this mechanism."""
        pass


