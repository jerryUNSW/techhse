import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embedding(phrase, tokenizer, model):
    """Generate BERT embedding for a given phrase."""
    inputs = tokenizer(phrase, return_tensors="pt", 
        padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_states = outputs.last_hidden_state
    attention_mask = inputs["attention_mask"].unsqueeze(-1)
    return (hidden_states * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

def compute_similarity(sentence1, sentence2, model_name="bert-base-uncased"):
    """Compute cosine similarity between two sentences using BERT embeddings."""
    # Load BERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    
    # Get embeddings for both sentences
    embedding1 = get_embedding(sentence1, tokenizer, model).cpu().numpy()
    embedding2 = get_embedding(sentence2, tokenizer, model).cpu().numpy()
    
    # Compute cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    return similarity

if __name__ == "__main__":
    # Specify your sentences here
    s1 = "that could send Vick to prison for his role in a dogfighting operation. Prosecutors allege that Vick funded and participated in the illegal enterprise, which involved the training and execution of underperforming dogs."

    # with CNN removed 
    s2= "that includes Vick’s admission of guilt in the federal dogfighting case. The suspension comes amid growing public backlash, and the league has emphasized its commitment to upholding personal conduct standards."
    # with 

    # with person name removed
    s3 = "that could result in prison time. The case centers around allegations of involvement in an illegal dogfighting operation, which has drawn widespread media attention and public scrutiny."

    # with person name removed
    s4 = "Friday following his indictment on federal dogfighting charges. The case has sparked public outrage and drawn scrutiny from animal rights groups, putting Vick’s football career in jeopardy."


    # Optionally specify a different model
    model_name = "bert-base-uncased"
    
    print(f"Running on device: {device}")
    print(f"Sentence 1: {s1}")
    print(f"Sentence 2: {s2}")
    print(f"Sentence 3: {s3}")
    print(f"Sentence 4: {s4}")

    similarity = compute_similarity(s1, s2, model_name)
    
    print(f"Cosine Similarity: {similarity:.4f}")

    similarity = compute_similarity(s1, s3, model_name)
    
    print(f"Cosine Similarity: {similarity:.4f}")

    similarity = compute_similarity(s1, s4, model_name)
    
    print(f"Cosine Similarity: {similarity:.4f}")
    # issue: person name appears twice in the sentence 
    # organization only appeared once in the sentence. 
    # 