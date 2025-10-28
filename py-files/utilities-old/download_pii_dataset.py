#!/usr/bin/env python3
"""
Script to download and explore the PII External Dataset from Kaggle
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

def download_and_explore_pii_dataset():
    """Download the PII dataset and explore its contents"""
    
    print("Downloading PII External Dataset from Kaggle...")
    
    try:
        # Try different common file names for PII datasets
        possible_files = ["train.csv", "test.csv", "data.csv", "pii_data.csv", "dataset.csv", ""]
        
        df = None
        for file_name in possible_files:
            try:
                print(f"Trying to load file: '{file_name}'")
                df = kagglehub.load_dataset(
                    KaggleDatasetAdapter.PANDAS,
                    "alejopaullier/pii-external-dataset",
                    file_name,
                )
                print(f"Successfully loaded file: '{file_name}'")
                break
            except Exception as e:
                print(f"Failed to load '{file_name}': {e}")
                continue
        
        if df is None:
            raise Exception("Could not load any file from the dataset")
        
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 records:")
        print(df.head())
        
        print("\nDataset info:")
        print(df.info())
        
        print("\nBasic statistics:")
        print(df.describe())
        
        # Save the dataset to a local file
        output_file = "/home/yizhang/tech4HSE/pii_external_dataset.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDataset saved to: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
        # Try to get more information about the dataset structure
        try:
            print("Trying alternative approaches...")
            # Try to download the dataset as a zip file
            path = kagglehub.dataset_download("alejopaullier/pii-external-dataset")
            print(f"Dataset downloaded to: {path}")
            
            # List files in the downloaded directory
            import os
            files = os.listdir(path)
            print(f"Files in dataset: {files}")
            
            # Try to load the first CSV file we find
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                csv_path = os.path.join(path, csv_files[0])
                print(f"Loading CSV file: {csv_path}")
                df = pd.read_csv(csv_path)
                print("Dataset loaded successfully from downloaded file!")
                print(f"Dataset shape: {df.shape}")
                print(f"Columns: {list(df.columns)}")
                print("\nFirst 5 records:")
                print(df.head())
                
                # Save the dataset to a local file
                output_file = "/home/yizhang/tech4HSE/pii_external_dataset.csv"
                df.to_csv(output_file, index=False)
                print(f"\nDataset saved to: {output_file}")
                
                return df
            
        except Exception as e2:
            print(f"Could not download dataset: {e2}")
        
        return None

def create_dataset_explanation(df):
    """Create an explanation of the dataset"""
    
    explanation = f"""
# PII External Dataset Analysis

## Dataset Overview
- **Source**: Kaggle - alejopaullier/pii-external-dataset
- **Shape**: {df.shape[0]} rows, {df.shape[1]} columns
- **Columns**: {', '.join(df.columns)}

## Column Analysis
"""
    
    for col in df.columns:
        explanation += f"\n### {col}\n"
        explanation += f"- **Data Type**: {df[col].dtype}\n"
        explanation += f"- **Non-null values**: {df[col].count()}/{len(df)}\n"
        if df[col].dtype == 'object':
            explanation += f"- **Unique values**: {df[col].nunique()}\n"
            explanation += f"- **Sample values**: {list(df[col].dropna().unique()[:5])}\n"
        else:
            explanation += f"- **Min**: {df[col].min()}\n"
            explanation += f"- **Max**: {df[col].max()}\n"
            explanation += f"- **Mean**: {df[col].mean():.2f}\n"
    
    explanation += f"""
## Sample Data
```python
{df.head().to_string()}
```

## Dataset Purpose
This dataset appears to be related to Personally Identifiable Information (PII) detection or classification. 
It could be useful for:
- Training PII detection models
- Privacy-preserving text sanitization research
- Named Entity Recognition (NER) for sensitive information
- Privacy evaluation benchmarks

## Usage in Privacy Research
Given the context of this Tech4HSE project focusing on privacy-preserving text sanitization, 
this dataset could be valuable for:
1. **Evaluation**: Testing how well privacy mechanisms preserve or remove PII
2. **Training**: Creating models that can identify PII in text
3. **Benchmarking**: Comparing different privacy-preserving approaches
4. **Research**: Understanding PII patterns in real-world data
"""
    
    return explanation

if __name__ == "__main__":
    print("Starting PII dataset download and analysis...")
    
    # Download and explore the dataset
    df = download_and_explore_pii_dataset()
    
    if df is not None:
        # Create explanation
        explanation = create_dataset_explanation(df)
        
        # Save explanation to file
        explanation_file = "/home/yizhang/tech4HSE/pii_dataset_explanation.md"
        with open(explanation_file, 'w') as f:
            f.write(explanation)
        
        print(f"\nDataset explanation saved to: {explanation_file}")
        print("\n" + "="*50)
        print("DATASET EXPLANATION")
        print("="*50)
        print(explanation)
    else:
        print("Failed to download the dataset. Please check the Kaggle dataset URL and try again.")
