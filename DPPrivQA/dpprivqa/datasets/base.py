"""
Base interface for dataset loaders.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Dataset(ABC):
    """Abstract base class for dataset loaders."""
    
    @abstractmethod
    def load(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load dataset and return standardized format.
        
        Args:
            split: Dataset split to load ('train', 'test', 'validation')
        
        Returns:
            List of questions in standardized format:
            {
                'question': str,
                'options': Dict[str, str],  # {'A': '...', 'B': '...', ...}
                'answer_idx': str  # 'A', 'B', 'C', or 'D'
            }
        """
        pass
    
    @abstractmethod
    def convert_to_standard(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a raw dataset item to standardized format.
        
        Args:
            item: Raw item from dataset
        
        Returns:
            Standardized format dict
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of this dataset."""
        pass


