import mauve
import json
import os
import torch
import math
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity

# Assuming imports_and_init contains necessary imports
from imports_and_init import *
from dp_sanitizer import get_embedding, load_sentence_bert, compute_similarity, differentially_private_replacement

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Load GPT-2 model and tokenizer
gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").eval()

# Load Sentence-BERT model
sbert_model = load_sentence_bert()

# Function to extract tokens
def get_tokens(text, start=0, count=50):
    """Extract tokens using GPT-2 tokenizer."""
    tokens = gpt2tokenizer.tokenize(text)
    selected_tokens = tokens[start:start+count]
    tokenized_string = gpt2tokenizer.convert_tokens_to_string(selected_tokens)
    return tokenized_string

# Function to count tokens
def count_tokens(text):
    """Return the number of GPT-2 tokens in the text."""
    return len(gpt2tokenizer.encode(text, add_special_tokens=False))

# Function to compute perplexity for a single prompt and continuation
def compute_perplexity(prompt, continuation):
    """Compute perplexity of continuation given prompt using GPT-2."""
    # Tokenize prompt and continuation
    prompt_tokens = gpt2tokenizer(prompt, return_tensors="pt")["input_ids"]
    continuation_tokens = gpt2tokenizer(continuation, return_tensors="pt")["input_ids"]
    input_ids = torch.cat([prompt_tokens, continuation_tokens], dim=-1)

    # Get model outputs
    with torch.no_grad():
        outputs = gpt2_model(input_ids, labels=input_ids)
        log_probs = outputs.logits

    # Compute log probabilities for continuation tokens
    continuation_len = continuation_tokens.size(-1)
    log_probs = torch.log_softmax(log_probs, dim=-1)
    token_log_probs = torch.gather(log_probs[:, :-1, :], dim=-1, index=input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    continuation_log_probs = token_log_probs[:, prompt_tokens.size(-1)-1:]

    # Sum log probabilities and compute perplexity
    sum_log_probs = continuation_log_probs.sum().item()
    if continuation_len == 0:
        return float('inf')  # Handle empty continuation
    avg_neg_log_likelihood = -sum_log_probs / continuation_len
    ppl = math.exp(avg_neg_log_likelihood)
    return ppl

def extract_refined_responses(file_path, max_responses=2000):
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    # Read JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path}: {e}")
    
    # Extract refined_response fields, up to max_responses
    refined_responses = [item.get("refined_response", "") for item in data[:max_responses]]
    # Filter out empty responses
    refined_responses = [r for r in refined_responses if r.strip()]
    return refined_responses

def extract_rantext_responses(file_path, max_responses=2000, field_name="refined_response"):
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    
    # Read JSON file
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path}: {e}")
    
    # Extract and clean responses
    refined_responses = [
        item[field_name]
        for item in data[:max_responses]
        if item[field_name].strip()
    ]
    return refined_responses

if __name__ == "__main__":

    num_articles = 20
    
    # Prepare the ground truth
    ground_truths = []
    prompts = []


    data_name = "imdb"

    print(f"Load {data_name} dataset")

    if data_name == "cnn_dailymail" :
        dataset = load_dataset(data_name, "3.0.0", split="test")
    if data_name == "imdb" :
        dataset = load_dataset(data_name, split="test")


    for i in range(num_articles):
        article = ""
        if data_name == "cnn_dailymail" :
            article = dataset[i]["article"]
        if data_name == "imdb" :
            article = dataset[i]["text"]

        prompt = get_tokens(article, 0, 50)
        ground_truth = get_tokens(article, 50, 100)
        prompts.append(prompt)
        ground_truths.append(ground_truth)

    # for epsilon in [1, 2, 3, 4, 5]:

    model_name = "phi-4"

    for epsilon in [1]:
        print(f"\nepsilon = {epsilon}")
        
        # Extract responses

        file_name = f"./whole-replacement-results/all_{data_name}_{epsilon}_{model_name}_adv.json"


        phrasedp_responses = extract_rantext_responses(
            file_path=file_name,
            field_name="dp_refined_response", 
            max_responses=num_articles
        )

        rantext_responses = extract_rantext_responses(
            file_path=file_name,
            field_name="refined_rantext_response", 
            max_responses=num_articles
        )

        for i in range(num_articles):
            article = ""
            if data_name == "cnn_dailymail" :
                article = dataset[i]["article"]
            if data_name == "imdb" :
                article = dataset[i]["text"]
            prompt = prompts[i]

            ground_truth = ground_truths[i]
            phrasedp_response = phrasedp_responses[i]
            rantext_response = rantext_responses[i]

            # Truncate responses to 100 tokens
            # rantext_responses[i] = get_tokens(rantext_responses[i], 0, 100)
            # phrasedp_responses[i] = get_tokens(phrasedp_responses[i], 0, 100)

            rantext_responses[i] = get_tokens(rantext_responses[i], 0, 50)
            phrasedp_responses[i] = get_tokens(phrasedp_responses[i], 0, 50)

            prompt_tokens = count_tokens(prompt)
            mine_tokens = count_tokens(phrasedp_response)
            rantext_tokens = count_tokens(rantext_response)
            ground_truth_tokens = count_tokens(ground_truth)

            print("prompt_tokens = ", prompt_tokens)
            print("my response _tokens = ", mine_tokens)
            print("rantext_tokens = ", rantext_tokens)
            print("ground_truth_tokens = ", ground_truth_tokens)

            print("\n\n")



        # Get embeddings for all texts
        embeddings_A = sbert_model.encode(ground_truths)
        embeddings_B = sbert_model.encode(rantext_responses)
        embeddings_C = sbert_model.encode(phrasedp_responses)
        embeddings_D = sbert_model.encode(prompts)

        # Calculate cosine similarity and coherence
        similarities = []
        similarities_ = []
        coherence_mine = []
        coherence_ran = []
        for i in range(len(ground_truths)):
            embedding_A = embeddings_A[i].reshape(1, -1)
            embedding_B = embeddings_B[i].reshape(1, -1)
            embedding_C = embeddings_C[i].reshape(1, -1)
            embedding_D = embeddings_D[i].reshape(1, -1)

            sim = cosine_similarity(embedding_A, embedding_B)[0][0]
            similarities.append(sim)
            sim = cosine_similarity(embedding_A, embedding_C)[0][0]
            similarities_.append(sim)
            
            coherence = cosine_similarity(embedding_C, embedding_D)[0][0]
            coherence_mine.append(coherence)
            coherence = cosine_similarity(embedding_B, embedding_D)[0][0]
            coherence_ran.append(coherence)

        # Print similarity and coherence results
        avg_similarity = np.mean(similarities)
        print(f"\tAVG sim (truth, Rantext): {avg_similarity:.4f}")
        avg_similarity = np.mean(similarities_)
        print(f"\tAVG sim (truth, mine): {avg_similarity:.4f}")
        print(f"\tAVG coherence (prompt, Rantext): {np.mean(coherence_ran):.4f}")
        print(f"\tAVG coherence (prompt, mine): {np.mean(coherence_mine):.4f}")

        # Compute perplexity for rantext_responses
        rantext_ppls = []
        for prompt, response in zip(prompts, rantext_responses):
            ppl = compute_perplexity(prompt, response)
            if not math.isinf(ppl):  # Skip invalid perplexities
                rantext_ppls.append(ppl)
        
        # Compute perplexity for phrasedp_responses
        phrasedp_ppls = []
        for prompt, response in zip(prompts, phrasedp_responses):
            ppl = compute_perplexity(prompt, response)
            if not math.isinf(ppl):  # Skip invalid perplexities
                phrasedp_ppls.append(ppl)

        # Print average perplexity
        print(f"\tAVG Perplexity (Rantext): {np.mean(rantext_ppls):.4f}")
        print(f"\tAVG Perplexity (Phrase DP): {np.mean(phrasedp_ppls):.4f}")


