#!/usr/bin/env python3
"""
Improved script to fetch BibTeX entries for papers in relatedworks folder
Uses multiple APIs and better error handling
"""

import requests
import json
import re
import time
from pathlib import Path

def get_bibtex_from_doi(doi, timeout=15):
    """
    Fetch BibTeX entry from CrossRef API using DOI
    
    Args:
        doi: Digital Object Identifier
        timeout: Request timeout in seconds
        
    Returns:
        BibTeX entry as string or None
    """
    try:
        url = f"https://api.crossref.org/works/{doi}/transform/application/x-bibtex"
        response = requests.get(url, timeout=timeout)
        
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error fetching DOI {doi}: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error fetching DOI {doi}: {e}")
        return None

def search_paper_crossref(title, author=None, year=None, timeout=15):
    """
    Search for a paper using CrossRef API
    
    Args:
        title: Paper title
        author: Author name (optional)
        year: Publication year (optional)
        timeout: Request timeout in seconds
        
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
            params['filter'] = f'from-pub-date:{year},until-pub-date:{year}'
        
        url = "https://api.crossref.org/works"
        response = requests.get(url, params=params, timeout=timeout)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('message', {}).get('items', [])
        else:
            print(f"Error searching CrossRef: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error searching CrossRef: {e}")
        return []

def search_arxiv(title, timeout=15):
    """
    Search for paper on arXiv
    
    Args:
        title: Paper title
        timeout: Request timeout in seconds
        
    Returns:
        List of matching papers
    """
    try:
        # Clean title for search
        clean_title = re.sub(r'[^\w\s]', ' ', title).strip()
        
        params = {
            'search_query': f'ti:"{clean_title}"',
            'max_results': 5
        }
        
        url = "http://export.arxiv.org/api/query"
        response = requests.get(url, params=params, timeout=timeout)
        
        if response.status_code == 200:
            # Parse XML response (simplified)
            content = response.text
            if 'entry' in content:
                return [{'title': clean_title, 'source': 'arxiv'}]  # Simplified
            return []
        else:
            print(f"Error searching arXiv: HTTP {response.status_code}")
            return []
            
    except Exception as e:
        print(f"Error searching arXiv: {e}")
        return []

def get_known_bibtex_entries():
    """
    Return known BibTeX entries for papers we can identify
    
    Returns:
        Dictionary mapping titles to BibTeX entries
    """
    known_entries = {
        # SANTEXT paper
        "Differential Privacy for Text Analytics via Natural Text Sanitization": """@inproceedings{yue2021differential,
  title={Differential Privacy for Text Analytics via Natural Text Sanitization},
  author={Yue, Xiang and Du, Minxin and Wang, Tianhao and Li, Yaliang and Sun, Huan and Chow, Sherman SM},
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  pages={3853--3867},
  year={2021}
}""",
        
        # PRIV-QA paper (example - you may need to find the actual entry)
        "PRIV-QA: Privacy-Preserving Question Answering for Cloud Large Language Models": """@inproceedings{privqa2024,
  title={PRIV-QA: Privacy-Preserving Question Answering for Cloud Large Language Models},
  author={[Authors to be filled]},
  booktitle={[Conference to be filled]},
  pages={[Pages to be filled]},
  year={2024}
}""",
        
        # Tree of Attacks paper
        "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically": """@inproceedings{treeofattacks2024,
  title={Tree of Attacks: Jailbreaking Black-Box LLMs Automatically},
  author={[Authors to be filled]},
  booktitle={[Conference to be filled]},
  pages={[Pages to be filled]},
  year={2024}
}""",
        
        # Privacy Preserving Prompt Engineering Survey
        "Privacy Preserving Prompt Engineering: A Survey": """@article{privacy2024,
  title={Privacy Preserving Prompt Engineering: A Survey},
  author={[Authors to be filled]},
  journal={[Journal to be filled]},
  volume={[Volume to be filled]},
  pages={[Pages to be filled]},
  year={2024}
}""",
        
        # Graph of Attacks paper
        "Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation": """@inproceedings{graph2024,
  title={Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation},
  author={[Authors to be filled]},
  booktitle={[Conference to be filled]},
  pages={[Pages to be filled]},
  year={2024}
}"""
    }
    
    return known_entries

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
    
    # Handle special cases
    if 'SANTEXT-paper.pdf' in filename:
        return {
            'title': 'Differential Privacy for Text Analytics via Natural Text Sanitization',
            'year': '2021',
            'filename': filename,
            'known_title': True
        }
    
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
        'filename': filename,
        'known_title': False
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
    known_entries = get_known_bibtex_entries()
    
    # Get all PDF files
    pdf_files = list(relatedworks_path.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files in {relatedworks_dir}")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        # Extract paper info from filename
        paper_info = get_paper_info_from_filename(pdf_file.name)
        title = paper_info['title']
        year = paper_info['year']
        known_title = paper_info.get('known_title', False)
        
        print(f"  Title: {title}")
        if year:
            print(f"  Year: {year}")
        
        bibtex_entry = None
        
        # First, check if we have a known entry
        if known_title or title in known_entries:
            print("  Using known BibTeX entry")
            bibtex_entry = known_entries.get(title)
        else:
            # Try to extract DOI from PDF first
            doi = None  # Simplified for now
            
            if doi:
                print(f"  Found DOI: {doi}")
                bibtex_entry = get_bibtex_from_doi(doi)
            else:
                # Search CrossRef for the paper
                print("  Searching CrossRef...")
                papers = search_paper_crossref(title, year=year, timeout=20)
                
                if papers:
                    print(f"  Found {len(papers)} potential matches")
                    
                    # Try the first few matches
                    for i, paper in enumerate(papers[:3]):
                        paper_title = paper.get('title', [''])[0]
                        paper_doi = paper.get('DOI')
                        
                        print(f"    Match {i+1}: {paper_title[:100]}...")
                        
                        if paper_doi:
                            print(f"    DOI: {paper_doi}")
                            bibtex_entry = get_bibtex_from_doi(paper_doi, timeout=20)
                            
                            if bibtex_entry:
                                print("    ✓ Successfully retrieved BibTeX")
                                break
                            else:
                                print("    ✗ Failed to retrieve BibTeX")
                else:
                    print("  No matches found in CrossRef")
                    
                    # Try arXiv as fallback
                    print("  Trying arXiv...")
                    arxiv_papers = search_arxiv(title, timeout=20)
                    if arxiv_papers:
                        print("  Found potential arXiv matches")
                    else:
                        print("  No arXiv matches found")
        
        if bibtex_entry:
            bibtex_entries[pdf_file.name] = bibtex_entry
            print("  ✓ BibTeX entry retrieved")
        else:
            print("  ✗ No BibTeX entry found")
        
        # Be nice to the API
        time.sleep(2)
    
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
