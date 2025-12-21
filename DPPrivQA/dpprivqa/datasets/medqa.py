"""
MedQA-USMLE dataset loader.
"""

from typing import List, Dict, Any
from datasets import load_dataset

from dpprivqa.datasets.base import Dataset
from dpprivqa.datasets.converters import convert_medqa_to_standard


class MedQADataset(Dataset):
    """Loader for MedQA-USMLE dataset."""
    
    def __init__(self):
        """Initialize MedQA dataset loader."""
        self.dataset_name = "bigbio/med_qa"
        self.config_name = "med_qa_en_4options"
    
    def load(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load MedQA dataset and return standardized format.
        
        Args:
            split: Dataset split to load
        
        Returns:
            List of questions in standardized format
        """
        try:
            dataset = load_dataset(self.dataset_name, self.config_name, split=split)
        except Exception as e:
            # Fallback: try alternative dataset name
            try:
                dataset = load_dataset("medqa", split=split)
            except Exception:
                raise ValueError(f"Failed to load MedQA dataset: {e}")
        
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
        Convert MedQA item to standardized format.
        
        Args:
            item: Raw item from MedQA dataset
        
        Returns:
            Standardized format dict
        """
        return convert_medqa_to_standard(item)
    
    def get_name(self) -> str:
        """Return the name of this dataset."""
        return "medqa"


