#!/usr/bin/env python3
"""
SANTEXT+ Integration for QA Experiments
Integrates SANTEXT+ as a privacy mechanism alongside PhraseDP and InferDPT
"""

import numpy as np
import random
import re
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import torch
import os
import json

class SanTextPlusMechanism:
    """
    SANTEXT+ implementation for QA experiments
    """
    
    def __init__(self, epsilon=2.0, p=0.3, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize SANTEXT+ mechanism
        
        Args:
            epsilon: Privacy parameter
            p: Probability of sanitizing non-sensitive words
            embedding_model: Sentence transformer model for word embeddings
        """
        self.epsilon = epsilon
        self.p = p
        self.embedding_model_name = embedding_model
        
        # Load sentence transformer model
        print(f"Loading SANTEXT+ model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)
        
        # Vocabulary and embeddings (built dynamically)
        self.vocab_words = []
        self.word_embeddings = None
        self.word2id = {}
        self.prob_matrix = None
        
        # Sensitive words (can be customized)
        self.sensitive_words = set()
        
        # Statistics
        self.stats = {
            'total_words': 0,
            'sensitive_words_sanitized': 0,
            'non_sensitive_words_sanitized': 0,
            'out_of_vocab_words': 0
        }

    def load_vocabulary_from_files(self, vocab_path: str, embeddings_path: str):
        """Load a GLOBAL vocabulary and embedding matrix from disk.
        This avoids building vocab from test data and matches SANTEXT paper setup.
        """
        import json, os
        if not (os.path.exists(vocab_path) and os.path.exists(embeddings_path)):
            print(f"[SANTEXT+] Global vocab files not found: {vocab_path}, {embeddings_path}")
            return False

    def load_global_glove_embeddings(self, glove_path: str, max_words: int = 50000) -> bool:
        """Load global vocabulary and embeddings from a GloVe file (SanText-style)."""
        try:
            if not os.path.exists(glove_path):
                print(f"[SANTEXT+] GloVe file not found: {glove_path}")
                return False
            print(f"[SANTEXT+] Loading GloVe embeddings from: {glove_path}")
            vocab_words = []
            embeddings = []
            count = 0
            with open(glove_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(' ')
                    if len(parts) < 10:
                        continue
                    word = parts[0]
                    try:
                        vec = [float(x) for x in parts[1:]]
                    except Exception:
                        continue
                    vocab_words.append(word)
                    embeddings.append(vec)
                    count += 1
                    if max_words and count >= max_words:
                        break
            if not embeddings:
                print("[SANTEXT+] No embeddings loaded from GloVe.")
                return False
            self.vocab_words = vocab_words
            self.word_embeddings = np.array(embeddings, dtype=np.float32)
            self.word2id = {w: i for i, w in enumerate(self.vocab_words)}
            self._calculate_probability_matrix()
            print(f"[SANTEXT+] Loaded GloVe global embeddings: {len(self.vocab_words)} words")
            return True
        except Exception as e:
            print(f"[SANTEXT+] Failed to load GloVe embeddings: {e}")
            return False

    def load_global_bert_embeddings(self, model_name: str = "bert-base-uncased", max_tokens: int = 30000) -> bool:
        """Load global vocabulary and embeddings from a BERT model (SanText-style)."""
        try:
            print(f"[SANTEXT+] Loading BERT embeddings: {model_name}")
            from transformers import BertTokenizer, BertForMaskedLM
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = BertForMaskedLM.from_pretrained(model_name)
            embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
            vocab_words = []
            embeddings = []
            for i, token in enumerate(tokenizer.vocab):
                vocab_words.append(token)
                embeddings.append(embedding_matrix[tokenizer.convert_tokens_to_ids(token)])
                if max_tokens and len(vocab_words) >= max_tokens:
                    break
            if not embeddings:
                print("[SANTEXT+] No embeddings loaded from BERT.")
                return False
            self.vocab_words = vocab_words
            self.word_embeddings = np.array(embeddings, dtype=np.float32)
            self.word2id = {w: i for i, w in enumerate(self.vocab_words)}
            self._calculate_probability_matrix()
            print(f"[SANTEXT+] Loaded BERT global embeddings: {len(self.vocab_words)} tokens")
            return True
        except Exception as e:
            print(f"[SANTEXT+] Failed to load BERT embeddings: {e}")
            return False
        try:
            with open(vocab_path, 'r', encoding='utf-8') as f:
                self.vocab_words = json.load(f)
            self.word2id = {w: i for i, w in enumerate(self.vocab_words)}
            self.word_embeddings = np.load(embeddings_path)
            if self.word_embeddings.shape[0] != len(self.vocab_words):
                print("[SANTEXT+] Mismatch between vocab size and embeddings rows")
                return False
            # Build probability matrix with current epsilon
            self._calculate_probability_matrix()
            print(f"[SANTEXT+] Loaded global vocab: {len(self.vocab_words)} words")
            return True
        except Exception as e:
            print(f"[SANTEXT+] Failed to load global vocab: {e}")
            return False
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from a collection of texts
        
        Args:
            texts: List of text strings
        """
        print("Building SANTEXT+ vocabulary...")
        
        # Extract all words from texts
        all_words = set()
        for text in texts:
            words = self._tokenize(text)
            all_words.update(words)
        
        # Add common words that might be missing
        common_words = [
            'what', 'is', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'under', 'over',
            'and', 'or', 'but', 'so', 'yet', 'nor', 'for', 'as', 'if', 'when',
            'where', 'why', 'how', 'who', 'which', 'that', 'this', 'these',
            'those', 'a', 'an', 'some', 'any', 'all', 'every', 'each', 'both',
            'either', 'neither', 'one', 'two', 'three', 'first', 'second',
            'last', 'next', 'other', 'another', 'such', 'same', 'different',
            'new', 'old', 'good', 'bad', 'big', 'small', 'large', 'little',
            'high', 'low', 'long', 'short', 'wide', 'narrow', 'deep', 'shallow',
            'fast', 'slow', 'quick', 'early', 'late', 'young', 'old', 'hot',
            'cold', 'warm', 'cool', 'dry', 'wet', 'clean', 'dirty', 'full',
            'empty', 'open', 'closed', 'free', 'busy', 'easy', 'hard', 'soft',
            'heavy', 'light', 'strong', 'weak', 'rich', 'poor', 'happy', 'sad',
            'beautiful', 'ugly', 'important', 'useful', 'necessary', 'possible',
            'impossible', 'true', 'false', 'right', 'wrong', 'correct', 'incorrect'
        ]
        all_words.update(common_words)
        
        # Convert to list and sort for consistency
        self.vocab_words = sorted(list(all_words))
        
        # Get embeddings for all words
        print(f"Getting embeddings for {len(self.vocab_words)} words...")
        self.word_embeddings = self.model.encode(self.vocab_words)
        
        # Create word to ID mapping
        self.word2id = {word: i for i, word in enumerate(self.vocab_words)}
        
        # Calculate probability matrix
        self._calculate_probability_matrix()
        
        print(f"SANTEXT+ vocabulary built: {len(self.vocab_words)} words")
    
    def _calculate_probability_matrix(self):
        """Calculate the probability matrix for word replacements"""
        print("Calculating SANTEXT+ probability matrix...")
        
        # Calculate distances between all word embeddings
        distances = euclidean_distances(self.word_embeddings, self.word_embeddings)
        similarities = -distances
        
        # Apply exponential mechanism
        self.prob_matrix = softmax(self.epsilon * similarities / 2, axis=1)
        
        print("SANTEXT+ probability matrix calculated")
    
    def _tokenize(self, text):
        """
        Simple tokenization for SANTEXT+
        
        Args:
            text: Input text string
            
        Returns:
            List of tokens
        """
        # Convert to lowercase and split on whitespace
        tokens = text.lower().split()
        
        # Clean tokens (remove punctuation, keep alphanumeric)
        cleaned_tokens = []
        for token in tokens:
            # Remove punctuation but keep the word
            clean_token = re.sub(r'[^\w]', '', token)
            if clean_token:  # Only keep non-empty tokens
                cleaned_tokens.append(clean_token)
        
        return cleaned_tokens
    
    def set_sensitive_words(self, sensitive_words):
        """
        Set sensitive words that should always be sanitized
        
        Args:
            sensitive_words: Set or list of sensitive words
        """
        self.sensitive_words = set(sensitive_words)
        print(f"SANTEXT+ sensitive words set: {len(self.sensitive_words)} words")
    
    def sanitize_text(self, text):
        """
        Sanitize a single text using SANTEXT+
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text
        """
        if not self.vocab_words:
            raise ValueError("Vocabulary not built. Call build_vocabulary() first.")
        
        # Tokenize input text
        tokens = self._tokenize(text)
        sanitized_tokens = []
        
        for token in tokens:
            self.stats['total_words'] += 1
            
            if token in self.word2id:
                # In vocabulary
                if token in self.sensitive_words:
                    # Sensitive word - always sanitize
                    word_idx = self.word2id[token]
                    sampling_prob = self.prob_matrix[word_idx]
                    sampling_idx = np.random.choice(len(sampling_prob), 1, p=sampling_prob)[0]
                    sanitized_tokens.append(self.vocab_words[sampling_idx])
                    self.stats['sensitive_words_sanitized'] += 1
                else:
                    # Non-sensitive word - sanitize with probability p
                    if random.random() <= self.p:
                        word_idx = self.word2id[token]
                        sampling_prob = self.prob_matrix[word_idx]
                        sampling_idx = np.random.choice(len(sampling_prob), 1, p=sampling_prob)[0]
                        sanitized_tokens.append(self.vocab_words[sampling_idx])
                        self.stats['non_sensitive_words_sanitized'] += 1
                    else:
                        # Keep original
                        sanitized_tokens.append(token)
            else:
                # Out of vocabulary - random replacement
                random_word = random.choice(self.vocab_words)
                sanitized_tokens.append(random_word)
                self.stats['out_of_vocab_words'] += 1
        
        return ' '.join(sanitized_tokens)
    
    def get_stats(self):
        """Get sanitization statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'total_words': 0,
            'sensitive_words_sanitized': 0,
            'non_sensitive_words_sanitized': 0,
            'out_of_vocab_words': 0
        }

def create_santext_mechanism(epsilon=2.0, p=0.3, embedding_model="all-MiniLM-L6-v2"):
    """
    Factory function to create a SANTEXT+ mechanism
    
    Args:
        epsilon: Privacy parameter
        p: Probability of sanitizing non-sensitive words
        embedding_model: Sentence transformer model
        
    Returns:
        SanTextPlusMechanism instance
    """
    return SanTextPlusMechanism(epsilon=epsilon, p=p, embedding_model=embedding_model)

def demo_santext_integration():
    """Demo function to show SANTEXT+ integration"""
    
    print("SANTEXT+ Integration Demo")
    print("=" * 50)
    
    # Sample questions for vocabulary building
    sample_questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet in our solar system?",
        "When did World War II end?",
        "What is the chemical symbol for gold?"
    ]
    
    # Create SANTEXT+ mechanism
    santext = create_santext_mechanism(epsilon=2.0, p=0.3)
    
    # Build vocabulary from sample questions
    santext.build_vocabulary(sample_questions)
    
    # Set some sensitive words
    sensitive_words = {'france', 'paris', 'capital', 'romeo', 'juliet', 'shakespeare'}
    santext.set_sensitive_words(sensitive_words)
    
    # Test question
    test_question = "What is the capital of France?"
    print(f"\nOriginal: {test_question}")
    
    # Generate multiple sanitized versions
    print("\nSanitized versions:")
    for i in range(5):
        sanitized = santext.sanitize_text(test_question)
        print(f"  Version {i+1}: {sanitized}")
    
    # Show statistics
    stats = santext.get_stats()
    print(f"\nStatistics:")
    print(f"  Total words processed: {stats['total_words']}")
    print(f"  Sensitive words sanitized: {stats['sensitive_words_sanitized']}")
    print(f"  Non-sensitive words sanitized: {stats['non_sensitive_words_sanitized']}")
    print(f"  Out-of-vocab words: {stats['out_of_vocab_words']}")

if __name__ == "__main__":
    demo_santext_integration()
