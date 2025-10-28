# imports_and_init.py
import os
import re
import json
import pickle
import numpy as np
import torch
import spacy
# need this for google API



from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    GPT2Tokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
)
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from openai import OpenAI



# Optional: load environment and spacy model here
load_dotenv()
nlp = spacy.load("en_core_web_sm")

# Expose things you want available in the main script
__all__ = [
    "os", "re", "json", "pickle", "np", "torch", "spacy", "cosine_similarity",
    "AutoTokenizer", "AutoModel", "AutoModelForCausalLM", "GPT2Tokenizer",
    "LlamaTokenizer", "LlamaForCausalLM", "SentenceTransformer", "load_dataset",
    "OpenAI", "load_dotenv", "nlp", "defaultdict"
]


## load the API keys 
load_dotenv()
DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
hugging_face_api = os.getenv("HUGGING_FACE")
Gemini_API = os.getenv("GEMINI_API")

# specify which GPUS I will use
# os.environ["CUDA_VISIBLE_DEVICES"] = "02,3"