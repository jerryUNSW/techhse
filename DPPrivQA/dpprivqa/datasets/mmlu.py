"""
MMLU dataset loader for all subsets.
"""

from typing import List, Dict, Any, Optional
from datasets import load_dataset

from dpprivqa.datasets.base import Dataset
from dpprivqa.datasets.converters import convert_mmlu_to_standard


class MMLUDataset(Dataset):
    """Loader for MMLU dataset subsets."""
    
    # Supported MMLU subsets
    SUBSETS = {
        'professional_law': 'MMLU Professional Law',
        'professional_medicine': 'MMLU Professional Medicine',
        'clinical_knowledge': 'MMLU Clinical Knowledge',
        'college_medicine': 'MMLU College Medicine',
    }
    
    def __init__(self, subset: str = "professional_law"):
        """
        Initialize MMLU dataset loader.
        
        Args:
            subset: MMLU subset name (professional_law, professional_medicine, etc.)
        """
        if subset not in self.SUBSETS:
            raise ValueError(f"Unknown MMLU subset: {subset}. Available: {list(self.SUBSETS.keys())}")
        
        self.dataset_name = "cais/mmlu"
        self.subset = subset
        self.display_name = self.SUBSETS[subset]
    
    def load(self, split: str = "test") -> List[Dict[str, Any]]:
        """
        Load MMLU dataset and return standardized format.
        
        Args:
            split: Dataset split to load
        
        Returns:
            List of questions in standardized format
        """
        try:
            # Try loading with trust_remote_code=False first
            dataset = load_dataset(self.dataset_name, self.subset, split=split, trust_remote_code=False)
        except Exception as e1:
            try:
                # Try alternative dataset source
                dataset = load_dataset("lighteval/mmlu", self.subset, split=split, trust_remote_code=False)
            except Exception as e2:
                try:
                    # Try without trust_remote_code parameter (for older versions)
                    dataset = load_dataset(self.dataset_name, self.subset, split=split)
                except Exception as e3:
                    raise ValueError(
                        f"Failed to load MMLU {self.subset} dataset. Tried:\n"
                        f"1. {self.dataset_name}/{self.subset}: {e1}\n"
                        f"2. lighteval/mmlu/{self.subset}: {e2}\n"
                        f"3. {self.dataset_name}/{self.subset} (no trust_remote_code): {e3}"
                    )
        
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
        Convert MMLU item to standardized format.
        
        Args:
            item: Raw item from MMLU dataset
        
        Returns:
            Standardized format dict
        """
        return convert_mmlu_to_standard(item)
    
    def get_name(self) -> str:
        """Return the name of this dataset."""
        return f"mmlu_{self.subset}"


