"""
MedMCQA dataset loader.
"""

from typing import List, Dict, Any
from datasets import load_dataset

from dpprivqa.datasets.base import Dataset
from dpprivqa.datasets.converters import convert_medqa_to_standard


class MedMCQADataset(Dataset):
    """Loader for MedMCQA dataset."""
    
    def __init__(self):
        """Initialize MedMCQA dataset loader."""
        self.dataset_name = "medmcqa"
    
    def load(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load MedMCQA dataset and return standardized format.
        
        Args:
            split: Dataset split to load
        
        Returns:
            List of questions in standardized format
        """
        try:
            dataset = load_dataset(self.dataset_name, split=split)
        except Exception as e:
            raise ValueError(f"Failed to load MedMCQA dataset: {e}")
        
        converted = []
        for item in dataset:
            try:
                standardized = self.convert_to_standard(item)
                converted.append(standardized)
            except Exception as e:
                # Skip items that can't be converted
                continue
        
        return converted
    
    def convert_to_standard(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert MedMCQA item to standardized format.
        
        Args:
            item: Raw item from MedMCQA dataset
        
        Returns:
            Standardized format dict
        """
        return convert_medqa_to_standard(item)
    
    def get_name(self) -> str:
        """Return the name of this dataset."""
        return "medmcqa"


