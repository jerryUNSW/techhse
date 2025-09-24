#!/usr/bin/env python3
"""
Fetch BibTeX entries for papers in relatedworks folder using citation APIs
"""

import requests
import json
import re
import time
from pathlib import Path

def get_bibtex_from_doi(doi):
    """
    Fetch BibTeX entry from CrossRef API using DOI
    
    Args:
        doi: Digital Object Identifier
        
    Returns:
        BibTeX entry as string or None
    """
    try:
        url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching DOI {doi}: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching DOI {doi}: {e}")
        return None

def search_paper_crossref(title, author=None, year=None):
    """
    Search for a paper using CrossRef API
    
    Args:
        title: Paper title
        author: Author name (optional)
        year: Publication year (optional)
        
    Returns:
        List of matching papers with DOIs
    """
    try:
        # Clean title for search
        clean_title = re.sub(r'[^\w\s]', ' ', title).strip()
        
        params = {
            'query.title': clean_title,
            'rows': 5
        }
        
        if author:
            params['query.author'] = author
        if year:
            params['filter': f'from-pub-date:{year},until-pub-date:{year}']
        
        url = "https://api.crossref.org/works"
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('message', {}).get('items', [])
        else:
            print(f"Error searching CrossRef: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error searching CrossRef: {e}")
        return []

def extract_doi_from_pdf(pdf_path):
    """
    Try to extract DOI from PDF metadata or text
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        DOI string or None
    """
    try:
        # This is a simplified approach - in practice you might need PyPDF2 or similar
        # For now, we'll return None and rely on title search
        return None
    except Exception as e:
        print(f"Error extracting DOI from {pdf_path}: {e}")
        return None

def get_paper_info_from_filename(filename):
    """
    Extract paper information from filename
    
    Args:
        filename: PDF filename
        
    Returns:
        Dictionary with title and other info
    """
    # Remove .pdf extension
    title = filename.replace('.pdf', '')
    
    # Common patterns in academic paper filenames
    # Try to extract year if present
    year_match = re.search(r'(\d{4})', title)
    year = year_match.group(1) if year_match else None
    
    # Clean up title
    title = re.sub(r'\d{4}', '', title)  # Remove year
    title = re.sub(r'[-_]', ' ', title)  # Replace dashes/underscores with spaces
    title = title.strip()
    
    return {
        'title': title,
        'year': year,
        'filename': filename
    }

def fetch_bibtex_for_papers(relatedworks_dir):
    """
    Fetch BibTeX entries for all papers in relatedworks directory
    
    Args:
        relatedworks_dir: Path to relatedworks directory
        
    Returns:
        Dictionary mapping filenames to BibTeX entries
    """
    relatedworks_path = Path(relatedworks_dir)
    bibtex_entries = {}
    
    # Get all PDF files
    pdf_files = list(relatedworks_path.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files in {relatedworks_dir}")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        # Extract paper info from filename
        paper_info = get_paper_info_from_filename(pdf_file.name)
        title = paper_info['title']
        year = paper_info['year']
        
        print(f"  Title: {title}")
        if year:
            print(f"  Year: {year}")
        
        # Try to extract DOI from PDF first
        doi = extract_doi_from_pdf(pdf_file)
        
        bibtex_entry = None
        
        if doi:
            print(f"  Found DOI: {doi}")
            bibtex_entry = get_bibtex_from_doi(doi)
        else:
            # Search CrossRef for the paper
            print("  Searching CrossRef...")
            papers = search_paper_crossref(title, year=year)
            
            if papers:
                print(f"  Found {len(papers)} potential matches")
                
                # Try the first few matches
                for i, paper in enumerate(papers[:3]):
                    paper_title = paper.get('title', [''])[0]
                    paper_doi = paper.get('DOI')
                    
                    print(f"    Match {i+1}: {paper_title[:100]}...")
                    
                    if paper_doi:
                        print(f"    DOI: {paper_doi}")
                        bibtex_entry = get_bibtex_from_doi(paper_doi)
                        
                        if bibtex_entry:
                            print("    ✓ Successfully retrieved BibTeX")
                            break
                        else:
                            print("    ✗ Failed to retrieve BibTeX")
            else:
                print("  No matches found in CrossRef")
        
        if bibtex_entry:
            bibtex_entries[pdf_file.name] = bibtex_entry
            print("  ✓ BibTeX entry retrieved")
        else:
            print("  ✗ No BibTeX entry found")
        
        # Be nice to the API
        time.sleep(1)
    
    return bibtex_entries

def update_references_bib(bibtex_entries, references_bib_path):
    """
    Update references.bib file with new BibTeX entries
    
    Args:
        bibtex_entries: Dictionary of BibTeX entries
        references_bib_path: Path to references.bib file
    """
    # Read existing references.bib
    existing_entries = set()
    if Path(references_bib_path).exists():
        with open(references_bib_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Extract existing citation keys
            existing_keys = re.findall(r'@\w+\{([^,]+),', content)
            existing_entries = set(existing_keys)
    else:
        content = ""
    
    # Add new entries
    new_entries = []
    for filename, bibtex_entry in bibtex_entries.items():
        # Extract citation key from BibTeX entry
        key_match = re.search(r'@\w+\{([^,]+),', bibtex_entry)
        if key_match:
            citation_key = key_match.group(1)
            if citation_key not in existing_entries:
                new_entries.append(bibtex_entry)
                print(f"Adding new entry: {citation_key}")
            else:
                print(f"Entry already exists: {citation_key}")
        else:
            print(f"Could not extract citation key from {filename}")
    
    # Write updated references.bib
    if new_entries:
        with open(references_bib_path, 'a', encoding='utf-8') as f:
            if content and not content.endswith('\n'):
                f.write('\n')
            f.write('\n'.join(new_entries))
            f.write('\n')
        
        print(f"\nAdded {len(new_entries)} new entries to {references_bib_path}")
    else:
        print("\nNo new entries to add")

def main():
    """Main function to fetch BibTeX entries and update references.bib"""
    
    # Paths
    relatedworks_dir = "/home/yizhang/tech4HSE/relatedworks"
    references_bib_path = "/home/yizhang/tech4HSE/overleaf-folder/references.bib"
    
    print("Fetching BibTeX entries for papers in relatedworks folder...")
    print("=" * 60)
    
    # Fetch BibTeX entries
    bibtex_entries = fetch_bibtex_for_papers(relatedworks_dir)
    
    print(f"\nRetrieved {len(bibtex_entries)} BibTeX entries")
    
    # Update references.bib
    if bibtex_entries:
        update_references_bib(bibtex_entries, references_bib_path)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
