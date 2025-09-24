#!/usr/bin/env python3
"""
Final script to fetch BibTeX entries using correct paper titles from text files
"""

import requests
import json
import re
import time
from pathlib import Path

def get_bibtex_from_doi(doi, timeout=15):
    """Fetch BibTeX entry from CrossRef API using DOI"""
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
    """Search for a paper using CrossRef API"""
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

def get_correct_paper_info():
    """Get correct paper information from text files"""
    paper_info = {
        "PRIV-QA- Privacy-Preserving Question Answering for Cloud Large Language Models.pdf": {
            "title": "PRIV-QA: Privacy-Preserving Question Answering for Cloud Large Language Models",
            "authors": ["Guangwei Li", "Yuansen Zhang", "Yinggui Wang", "Shoumeng Yan", "Lei Wang", "Tao Wei"],
            "year": "2025",
            "arxiv_id": "2502.13564"
        },
        "Tree of Attacks- Jailbreaking Black-Box LLMs Automatically.pdf": {
            "title": "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically",
            "authors": ["Anay Mehrotra", "Manolis Zampetakis", "Paul Kassianik", "Blaine Nelson", "Hyrum Anderson", "Yaron Singer", "Amin Karbasi"],
            "year": "2024",
            "venue": "NeurIPS"
        },
        "Graph of Attacks with Pruning- Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation.pdf": {
            "title": "Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation",
            "authors": ["[Authors to be found]"],
            "year": "2024"
        }
    }
    return paper_info

def get_known_bibtex_entries():
    """Return known BibTeX entries for papers we can identify"""
    known_entries = {
        # SANTEXT paper (already in references.bib)
        "Differential Privacy for Text Analytics via Natural Text Sanitization": """@inproceedings{yue2021differential,
  title={Differential Privacy for Text Analytics via Natural Text Sanitization},
  author={Yue, Xiang and Du, Minxin and Wang, Tianhao and Li, Yaliang and Sun, Huan and Chow, Sherman SM},
  booktitle={Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  pages={3853--3867},
  year={2021}
}""",
        
        # PRIV-QA paper (from arXiv)
        "PRIV-QA: Privacy-Preserving Question Answering for Cloud Large Language Models": """@misc{li2025privqa,
  title={PRIV-QA: Privacy-Preserving Question Answering for Cloud Large Language Models},
  author={Guangwei Li and Yuansen Zhang and Yinggui Wang and Shoumeng Yan and Lei Wang and Tao Wei},
  year={2025},
  eprint={2502.13564},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}""",
        
        # Tree of Attacks paper (NeurIPS 2024)
        "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically": """@inproceedings{mehrotra2024tree,
  title={Tree of Attacks: Jailbreaking Black-Box LLMs Automatically},
  author={Anay Mehrotra and Manolis Zampetakis and Paul Kassianik and Blaine Nelson and Hyrum Anderson and Yaron Singer and Amin Karbasi},
  booktitle={Advances in Neural Information Processing Systems},
  year={2024}
}""",
        
        # Graph of Attacks paper (placeholder - need to find actual venue)
        "Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation": """@inproceedings{graph2024,
  title={Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation},
  author={[Authors to be filled]},
  booktitle={[Conference to be filled]},
  pages={[Pages to be filled]},
  year={2024}
}"""
    }
    
    return known_entries

def fetch_bibtex_for_papers(relatedworks_dir):
    """Fetch BibTeX entries for all papers in relatedworks directory"""
    relatedworks_path = Path(relatedworks_dir)
    bibtex_entries = {}
    known_entries = get_known_bibtex_entries()
    correct_info = get_correct_paper_info()
    
    # Get all PDF files
    pdf_files = list(relatedworks_path.glob("*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files in {relatedworks_dir}")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        
        # Get correct paper info if available
        if pdf_file.name in correct_info:
            paper_info = correct_info[pdf_file.name]
            title = paper_info["title"]
            authors = paper_info.get("authors", [])
            year = paper_info.get("year")
            
            print(f"  Title: {title}")
            print(f"  Authors: {', '.join(authors[:3])}{'...' if len(authors) > 3 else ''}")
            if year:
                print(f"  Year: {year}")
        else:
            # Fallback to filename parsing
            title = pdf_file.name.replace('.pdf', '').replace('-', ' ').replace('_', ' ')
            year = None
            authors = []
            print(f"  Title: {title}")
        
        bibtex_entry = None
        
        # Check if we have a known entry
        if title in known_entries:
            print("  Using known BibTeX entry")
            bibtex_entry = known_entries[title]
        else:
            # Try to search CrossRef
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
        
        if bibtex_entry:
            bibtex_entries[pdf_file.name] = bibtex_entry
            print("  ✓ BibTeX entry retrieved")
        else:
            print("  ✗ No BibTeX entry found")
        
        # Be nice to the API
        time.sleep(2)
    
    return bibtex_entries

def update_references_bib(bibtex_entries, references_bib_path):
    """Update references.bib file with new BibTeX entries"""
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
