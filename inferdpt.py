import json
import string
import tiktoken
import random
from sklearn.metrics.pairwise import euclidean_distances
import tqdm
from decimal import getcontext
import numpy as np
from transformers import GPT2Tokenizer
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import argparse

# use this for multi-threading
from multiprocessing import Pool, cpu_count



# Load environment variables
load_dotenv()
DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
HUGGING_FACE_API = os.getenv("HUGGING_FACE")

# Set precision for decimal calculations
getcontext().prec = 100

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-4")
    parser.add_argument("--eps", type=float, default=1.0)
    return parser

def get_first_50_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.tokenize(text)
    first_50_tokens = tokens[:50]
    tokenized_string = tokenizer.convert_tokens_to_string(first_50_tokens)
    return tokenized_string

def get_first_100_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.tokenize(text)
    first_100_tokens = tokens[:100]
    tokenized_string = tokenizer.convert_tokens_to_string(first_100_tokens)
    return tokenized_string

def calculate_distance(i, j, vector_matrix, pb):
    distance = euclidean_distances(
        vector_matrix[i].reshape(1, -1).astype(np.longdouble),
        vector_matrix[j].reshape(1, -1).astype(np.longdouble)
    )
    pb.update(1)
    return i, j, distance[0, 0]

def generate_tasks(n_vectors):
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            yield (i, j)

def initialize_embeddings(epsilon):
    """Initialize embeddings and related data structures if not already present."""
    base_path = "InferDPT/data"
    embeddings_path = f"{base_path}/cl100_embeddings.json"
    sorted_embeddings_path = f"{base_path}/sorted_cl100_embeddings.json"
    sensitivity_path = f"{base_path}/sensitivity_of_embeddings.json"
    temp_distance_path = f"{base_path}/temp_distance_json_path.json"

    # Load token-to-vector dictionary
    try:
        with open(embeddings_path, 'r') as f:
            cl100_emb = json.load(f)
            vector_data_json = {k: cl100_emb[k] for k in list(cl100_emb.keys())[:11000]}
            token_to_vector_dict = {token: np.array(vector) for token, vector in vector_data_json.items()}
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        raise

    print("# Initialize sorted embeddings if not present")
    if not os.path.exists(sorted_embeddings_path):
        word_list = list(vector_data_json.keys())
        vector_matrix = np.array(list(vector_data_json.values()))
        n_vectors = len(word_list)

        # Compute distance matrix
        distance_matrix = np.zeros((n_vectors, n_vectors))
        total_tasks = (n_vectors * (n_vectors - 1)) // 2
        results = [None] * total_tasks
        if not os.path.exists(temp_distance_path):
            with tqdm.tqdm(total=total_tasks) as pb:
                pb.set_description('Computing distance matrix')
                tasks = list(generate_tasks(n_vectors))
                for index, task in enumerate(tasks):
                    try:
                        results[index] = calculate_distance(task[0], task[1], vector_matrix, pb)
                    except Exception as e:
                        print(f"Task at index {index} failed with exception {e}")
                for i, j, distance in results:
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance

            # Create temporary distance dictionary
            temp_distance_dict_matrix = {}
            for i, word1 in enumerate(word_list):
                for j, word2 in enumerate(word_list):
                    pair = tuple(sorted([word1, word2]))
                    if pair in temp_distance_dict_matrix:
                        continue
                    temp_distance_dict_matrix[str(pair)] = float(distance_matrix[i, j])
            with open(temp_distance_path, 'w') as f:
                json.dump(temp_distance_dict_matrix, f)

        # Load temporary distance matrix and create sorted embeddings
        with open(temp_distance_path, 'r') as f:
            temp_distance_dict_matrix = json.load(f)
        word_to_index = {word: idx for idx, word in enumerate(word_list)}
        n = len(word_list)
        temp_distance_matrix = np.zeros((n, n))
        with tqdm.tqdm(total=len(temp_distance_dict_matrix)) as pbm:
            pbm.set_description('Building distance matrix')
            for key, value in temp_distance_dict_matrix.items():
                word1, word2 = tuple(key.strip("()").split(", "))
                i = word_to_index[word1.strip("'")]
                j = word_to_index[word2.strip("'")]
                temp_distance_matrix[i, j] = value
                temp_distance_matrix[j, i] = value
                pbm.update(1)

        # Create sorted distance dictionary
        sorted_distance_dict_matrix = {}
        with tqdm.tqdm(total=n) as pbm:
            pbm.set_description('Sorting distances')
            for i, word in enumerate(word_list):
                sorted_indices = np.argsort(temp_distance_matrix[i])
                sorted_words = [(word_list[j], temp_distance_matrix[i, j]) for j in sorted_indices]
                sorted_distance_dict_matrix[word] = sorted_words
                pbm.update(1)

        with open(sorted_embeddings_path, 'w') as f:
            json.dump(sorted_distance_dict_matrix, f)
    else:
        with open(sorted_embeddings_path, 'r') as f:
            sorted_distance_dict_matrix = json.load(f)

    print("# Initialize sensitivity data if not present")
    if not os.path.exists(sensitivity_path):
        vector_matrix = np.array(list(vector_data_json.values()))
        n_dimensions = vector_matrix.shape[1]
        delta_f_new = np.zeros(n_dimensions)
        for dim in tqdm.trange(n_dimensions):
            dim_data = vector_matrix[:, dim]
            sorted_dim_data = np.sort(dim_data)
            delta_f_new[dim] = sorted_dim_data[-1] - sorted_dim_data[0]
        with open(sensitivity_path, 'w') as f:
            json.dump(delta_f_new.tolist(), f)
    else:
        with open(sensitivity_path, 'r') as f:
            delta_f_new = np.array(json.load(f))

    return token_to_vector_dict, sorted_distance_dict_matrix, delta_f_new


### old implementation
### sent = cnn_dm_prompt
def perturb_sentence(sent, epsilon, model="gpt-4", 
    token_to_vector_dict=None, 
    sorted_distance_data=None, 
    delta_f_new=None):
    """Perturb a sentence by adding noise to token embeddings and selecting close tokens."""
    # Initialize embeddings if not provided

    if token_to_vector_dict is None or sorted_distance_data is None or delta_f_new is None:
        print("Initializing the embeddings for InferDPT.")
        token_to_vector_dict, sorted_distance_data, delta_f_new = initialize_embeddings(epsilon)

    enc = tiktoken.encoding_for_model(model)
    tokens_b = enc.encode(sent)
    tokens = [(enc.decode_single_token_bytes(t)).decode('Latin-1') for t in tokens_b]
    new_tokens = []
    Delta_u = 1.0
    exp_factor = epsilon / (2 * Delta_u)

    for origin_token in tokens:
        if origin_token.isnumeric():
            new_tokens.append(str(random.randint(1, 1000)))
            continue
        if origin_token[0] == ' ':
            origin_token = origin_token[1:]
        origin_embed = token_to_vector_dict.get(origin_token)
        if origin_embed is None:
            continue
        noise_embed = add_laplace_noise_to_vector(origin_embed, epsilon, delta_f_new)
        distance = np.linalg.norm(origin_embed - noise_embed)
        sorted_distances_for_token = sorted_distance_data.get(origin_token)
        if sorted_distances_for_token is None:
            continue
        distances_only = np.array([item[1] for item in sorted_distances_for_token])
        index = np.searchsorted(distances_only, distance)
        close_tokens = [item[0] for item in sorted_distances_for_token[:index]]
        close_distances = np.array([item[1] for item in sorted_distances_for_token[:index]])
        if not close_tokens:
            continue
        unnormalized_probabilities = np.exp(exp_factor * ((distance - close_distances) / distance))
        total_unnormalized_prob = np.sum(unnormalized_probabilities)
        probabilities = unnormalized_probabilities / total_unnormalized_prob
        selected_token = np.random.choice(close_tokens, p=probabilities)
        new_tokens.append(selected_token)

    sanitized_sent = ' '.join(new_tokens)
    return sanitized_sent



def add_laplace_noise_to_vector(vector, epsilon, delta_f_new=None):
    """Add Laplace noise to a vector for differential privacy."""
    vector = np.asarray(vector, dtype=np.longdouble)
    if delta_f_new is None:
        raise ValueError("delta_f_new must be provided")
    
    tt = 0
    if (epsilon * 19.064721649556482 - 38.1294334077209) > 0:
        tt = 0.01658160142016071 * np.log(epsilon * 19.064721649556482 - 38.1294334077209) + 9.311083811697406
    if epsilon < 2:
        beta_values = delta_f_new / epsilon
    else:
        beta_values = delta_f_new / tt
    beta_values = beta_values.astype(np.longdouble)
    noisy_vector = np.zeros_like(vector, dtype=np.longdouble)
    for dim in range(len(vector)):
        noise = np.random.laplace(0, beta_values[dim])
        noisy_vector[dim] = vector[dim] + noise
    return noisy_vector.astype(float)

def text_generation_with_black_box_LLMs(prompt, tem=1.0, max_tokens=150):
    """Generate text using either OpenAI or DeepSeek based on the provider."""
    system_prompt = (
        "You are a helpful assistant. When asked to continue a text, return only the continuation. "
        "Do not explain or preface your response."
    )
    provider = "deepseek"

    if provider == "openai":
        client = OpenAI(api_key=OPEN_AI_KEY)
        model = "gpt-4o-mini"
    elif provider == "deepseek":
        client = OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")
        model = "deepseek-chat"
    else:
        raise ValueError("Unsupported provider. Use 'openai' or 'deepseek'.")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=tem,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error with {provider} API: {e}")
        return "API Error"



# if __name__ == "__main__":
# parser = get_parser()
# args = parser.parse_args()
print("Loading inferdpt.py")
print("Embeddings will be loaded when perturb_sentence is called")