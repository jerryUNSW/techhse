#!/usr/bin/env python3
"""
Convert PDF to text format
"""

import PyPDF2
import sys
import os

def convert_pdf_to_txt(pdf_path, output_path):
    """Convert PDF file to text format."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            print(f"Converting PDF with {len(pdf_reader.pages)} pages...")
            
            for page_num, page in enumerate(pdf_reader.pages):
                print(f"Processing page {page_num + 1}...")
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
                text += "\n"
            
            # Write to output file
            with open(output_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
            
            print(f"Successfully converted PDF to text: {output_path}")
            print(f"Total characters: {len(text)}")
            
    except Exception as e:
        print(f"Error converting PDF: {e}")
        return False
    
    return True

if __name__ == "__main__":
    pdf_path = "__pycache__/relatedworks/Tree of Attacks- Jailbreaking Black-Box LLMs Automatically.pdf"
    output_path = "relatedwork/tree_of_attacks.txt"
    
    if os.path.exists(pdf_path):
        convert_pdf_to_txt(pdf_path, output_path)
    else:
        print(f"PDF file not found: {pdf_path}")

