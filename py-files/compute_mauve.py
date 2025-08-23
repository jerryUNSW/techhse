import mauve
import json
import os
from imports_and_init import *
from dp_sanitizer import get_embedding, load_sentence_bert, compute_similarity, differentially_private_replacement

# read from responses of my algorithm: 

# this function is more flexible

gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Load Sentence-BERT model
sbert_model = load_sentence_bert()

### okay, these are start and count. 
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

def extract_refined_responses(
    file_path, 
    max_responses=2000):

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
    # refined_responses = [item.get("noisy_response", "") for item in data[:max_responses]]

    
    
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
    
    ground_truths = []
    prompts = []

    # data_name = "imdb"

    data_name = "cnn_dailymail"

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


    model_name = "phi-4"

    # epsilon = 1
    # for epsilon in [1, 2, 3, 4, 5]:
    for epsilon in [1]:

        print("epsilon  = ", epsilon)

        file_name = f"./whole-replacement-results/all_{data_name}_{epsilon}_{model_name}.json"

        # file_name = f"./whole-replacement-results/all_{data_name}_{epsilon}_{model_name}_adv.json"
        
        print("looking at: ", file_name)

        # pretending noisy response as my response
        phrasedp_responses = extract_rantext_responses(
            file_path= file_name, 
            field_name = "dp_refined_response", 
            max_responses =num_articles
        )

        # getting RANTEXT results 
        rantext_responses = extract_rantext_responses(
            file_path= file_name, 
            field_name = "refined_rantext_response", 
            max_responses = num_articles
        )
        # print(f"Extracted {len(rantext_responses)} responses from RANText.")

        # dataset = load_dataset("cnn_dailymail", "3.0.0", split="test")
        for i in range(num_articles):

            article = ""
            if data_name == "cnn_dailymail" :
                article = dataset[i]["article"]
            if data_name == "imdb" :
                article = dataset[i]["text"]

            # article = dataset[i]["article"]

            prompt = prompts[i]

            ground_truth = ground_truths[i]

            phrasedp_response = phrasedp_responses[i]

            # grab 100 from rantext
            # rantext_responses[i] = get_tokens(rantext_responses[i], 0, 100)
            # phrasedp_responses[i] = get_tokens(phrasedp_responses[i], 0, 100)


            Len = 100
            rantext_responses[i] = get_tokens(rantext_responses[i], 0, Len)
            phrasedp_responses[i] = get_tokens(phrasedp_responses[i], 0, Len)

            rantext_response = rantext_responses[i]


            prompt_tokens = count_tokens(prompt)
            mine_tokens = count_tokens(phrasedp_response)
            rantext_tokens = count_tokens(rantext_response)
            ground_truth_tokens = count_tokens(ground_truth)

            # print(f"{'Prompt':<12} = \033[35m{prompt}\033[0m (Tokens: {prompt_tokens})")
            # print(f"{'Raw':<12} = \033[32m{phrasedp_response}\033[0m (Tokens: {mine_tokens})")
            # print(f"{'RANTEXT':<12} = \033[36m{rantext_response}\033[0m (Tokens: {rantext_tokens})")
            # print(f"{'Ground Truth':<12} = \033[33m{ground_truth}\033[0m (Tokens: {ground_truth_tokens})")
            # print("\033[34m" + "═" * 80 + "\033[0m\n")
            # print("\n")

        # continue
        # exit(0)

        # Get embeddings for all texts
        embeddings_A = sbert_model.encode(ground_truths)
        embeddings_B = sbert_model.encode(rantext_responses)
        embeddings_C = sbert_model.encode(phrasedp_responses)
        embeddings_D = sbert_model.encode(prompts)

        # Calculate cosine similarity for each pair
        similarities = []
        similarities_ = []

        coherence_mine = []
        coherence_ran = []
        for i in range(len(ground_truths)):
            # Reshape embeddings for cosine_similarity function
            embedding_A = embeddings_A[i].reshape(1, -1)
            embedding_B = embeddings_B[i].reshape(1, -1)
            embedding_C = embeddings_C[i].reshape(1, -1)
            embedding_D = embeddings_D[i].reshape(1, -1)

            # Calculate similarity
            sim = cosine_similarity(embedding_A, embedding_B)[0][0]
            similarities.append(sim)
            # print(f"sim(truth, Rantext): {sim:.4f}")

            # Calculate similarity
            sim = cosine_similarity(embedding_A, embedding_C)[0][0]
            similarities_.append(sim)
            # print(f"sim(truth, Raw): {sim:.4f}")    

            # coherence of mine 
            coherence = cosine_similarity(embedding_C, embedding_D)[0][0]
            coherence_mine.append(coherence)

            coherence = cosine_similarity(embedding_B, embedding_D)[0][0]
            coherence_ran.append(coherence)
            # coherence of rantext   

        # Calculate and print average similarity
        avg_similarity = np.mean(similarities)
        print(f"\tAVG sim (truth, Rantext): {avg_similarity:.4f}")

        avg_similarity = np.mean(similarities_)
        print(f"\tAVG sim (truth, mine): {avg_similarity:.4f}")

        print("\n")
        print(f"\tAVG coherence (prompt, Rantext): {np.mean(coherence_ran):.4f}")

        print(f"\tAVG coherence (prompt, mine): {np.mean(coherence_mine):.4f}")

        # # Compute MAUVE
        mauve_output = mauve.compute_mauve(
            p_text=rantext_responses,
            # q_text=ground_truths,
            q_text = prompts,
            # device_id=-1,  # Use CPU; change to 0 if you want to force GPU
            verbose=False  # Add this line to suppress progress bars
        )
        print(f"RANTEXT MAUVE Score: {mauve_output.mauve:.4f}")

        # # Compute Raw
        mauve_output = mauve.compute_mauve(
            p_text=phrasedp_responses,
            # q_text=ground_truths,
            q_text = prompts,
            # device_id=-1,  # Use CPU; change to 0 if you want to force GPU
            verbose=False  # Add this line to suppress progress bars
        )
        print(f"Phrase DP MAUVE Score: {mauve_output.mauve:.4f}")

# Recommendation (Practical combo):
# If you want automatic + interpretable:

# Compute NSP score (BERT or RoBERTa-based)

# Measure cosine similarity using SBERT embeddings

# Check entity repetition using spaCy