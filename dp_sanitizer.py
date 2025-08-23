# text_processing.py

from imports_and_init import *

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained model
def load_sentence_bert(model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    return model

# Get embedding for a given text
def get_embedding(model, text):
    return model.encode(text, convert_to_tensor=True)

def compute_similarity(model, text1, text2):
    """Compute cosine similarity between two texts."""
    embedding1 = get_embedding(model, text1).cpu().numpy().reshape(1, -1)
    embedding2 = get_embedding(model, text2).cpu().numpy().reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

def differentially_private_replacement(
    target_phrase, epsilon, 
    candidate_phrases, 
    candidate_embeddings,
    sbert_model):

    # Compute embedding only for the target phrase
    target_embedding = get_embedding(sbert_model, target_phrase).cpu().numpy()


    # Ensure target_embedding is 2D (1, n_features)
    if target_embedding.ndim == 1:
        target_embedding = target_embedding.reshape(1, -1)

    # Stack precomputed candidate embeddings
    candidate_embeddings_matrix = np.vstack([candidate_embeddings[phrase] for phrase in candidate_phrases])

    # Compute cosine similarity
    similarities = cosine_similarity(target_embedding, candidate_embeddings_matrix)[0]
    
    # ANSI color codes
    COLOR_PHRASE = "\033[94m"   # Blue
    COLOR_SIM = "\033[92m"      # Green
    COLOR_RESET = "\033[0m"

    # Print all candidate phrases with similarity scores
    print("Candidates and similarities:")
    for phrase, sim in zip(candidate_phrases, similarities):
        print(f"  {COLOR_PHRASE}{phrase}{COLOR_RESET} \t\t similarity = {COLOR_SIM}{sim:.4f}{COLOR_RESET}")


    # Convert similarity to distance
    distances = 1 - similarities
    
    # Apply the exponential mechanism
    p_unnorm = np.exp(-epsilon * distances)
    p_norm = p_unnorm / np.sum(p_unnorm)  # Normalize to make it a probability distribution

    # Sample a replacement
    return np.random.choice(candidate_phrases, p=p_norm)