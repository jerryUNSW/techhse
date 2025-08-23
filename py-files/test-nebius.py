import os
from openai import OpenAI

# Your existing NEBIUS_API key
NEBIUS_API = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDEwMzcxNjk3NzMxOTc3MDM3ODAxNyIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwNjA5NjQwNywidXVpZCI6ImQ4ZTAxOTkzLTlhNzUtNDcxOC04YzZjLWQxZDg2NTBmN2E2YSIsIm5hbWUiOiJZaXpoYW5nIiwiZXhwaXJlc19hdCI6IjIwMzAtMDUtMjdUMDc6MTM6MjcrMDAwMCJ9.CkkdaZMDtIvODgWgjUv_Vit1Ay6rBdWRgJPNr35JOxY"

# Initialize Nebius client
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=NEBIUS_API,
)

def find_available_qwen_model():
    """Find available Qwen models on Nebius"""
    try:
        models = nebius_client.models.list()
        qwen_models = [model.id for model in models.data if 'qwen' in model.id.lower() or 'Qwen' in model.id]
        
        # Preferred model names in order of preference
        preferred_models = [
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct", 
            "qwen/qwen2.5-coder-7b-instruct",
            "qwen2.5-coder-7b-instruct"
        ]
        
        # Try to find preferred model
        for preferred in preferred_models:
            if preferred in [model.id for model in models.data]:
                print(f"Using model: {preferred}")
                return preferred
        
        # If no preferred model found, use first available Qwen model
        if qwen_models:
            print(f"Using available Qwen model: {qwen_models[0]}")
            return qwen_models[0]
        
        # Fallback to any available model
        if models.data:
            fallback_model = models.data[0].id
            print(f"No Qwen models found, using fallback: {fallback_model}")
            return fallback_model
            
        raise Exception("No models available")
        
    except Exception as e:
        print(f"Error finding models: {e}")
        # Default fallback
        return "Qwen/Qwen2.5-Coder-7B-Instruct"

# Get the working model
QWEN_MODEL = find_available_qwen_model()

print("the working model is: ", QWEN_MODEL)

def extraction_module_nebius(prefix_text, perturbed_generation, max_tokens=150):
    """
    Nebius API version of extraction_module function.
    Replaces the local QWEN model with Nebius API call.
    
    Args:
        prefix_text (str): The original prefix text
        perturbed_generation (str): The generated text that needs refinement
        max_tokens (int): Maximum tokens to generate
    
    Returns:
        str: Refined text that seamlessly continues the prefix
    """
    
    input_prompt = f'''
Your task is to refine 'Perturbed Generation' such that it is a seamless continuation of the 'Prefix Text'. 
'Prefix Text': {prefix_text} 
'Perturbed Generation': {perturbed_generation}
Provide the refined text only. 
Refined text: '''

    try:
        response = nebius_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that refines text to create seamless continuations."},
                {"role": "user", "content": input_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95
        )
        
        extended_text = response.choices[0].message.content
        print("extended_text = ", extended_text)
        
        # Use your existing slice_refined_text function
        refined_text = slice_refined_text(extended_text, prefix_text)
        
        return refined_text
        
    except Exception as e:
        print(f"Error with Nebius API: {e}")
        return perturbed_generation  # Return original if API fails

def generate_named_entity_replacements_nebius(entity, entity_label, context_text):

    
    candidate_replacement_prompt = f"""
Your task is to generate **20 replacement named entities that match the entity '{entity}' (label: {entity_label}) found in the text below.

### Output rules:
- Output ONLY real-world named entities of the same type.
- DO NOT output full sentences, descriptions, or commentary.
- Each line must contain ONLY one named entity (e.g., "AirAsia", "Richard Branson").
- Do NOT include punctuation, extra whitespace, or explanatory text.
- The replacements must be of similar length to '{entity}'.

### Diversity requirements:
- Use names from a variety of countries and cultural contexts.
- Avoid repeating similar styles or variants of '{entity}'.

### Example:
Text: 'LAS VEGAS is known for its vibrant nightlife.'
Entity: LAS VEGAS (label: LOCATION)
Replacements:
Miami
New Orleans
Atlantic City
Los Angeles
Orlando
Phoenix
Tokyo
Singapore
Barcelona
Sydney
Amsterdam
Dubai
Cape Town
Rio de Janeiro
Reykjavik
Lima
Kathmandu
Nairobi
Ulaanbaatar
Nuuk

### Task:
Text: '{context_text}'
Entity: '{entity}' (label: {entity_label})
Replacements:
"""

    try:
        response = nebius_client.chat.completions.create(
            model=QWEN_MODEL,
            messages=[
                {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that generates diverse named entity replacements."},
                {"role": "user", "content": candidate_replacement_prompt}
            ],
            max_tokens=200,
            temperature=0.7,
            top_p=0.95
        )
        
        replacement_text = response.choices[0].message.content
        replacement_list = replacement_text.strip().split('\n')
        replacement_list = [r.strip() for r in replacement_list if r.strip()]
        
        # Ensure original entity is included
        replacement_list.append(entity)
        
        return replacement_list
        
    except Exception as e:
        print(f"Error generating replacements with Nebius API: {e}")
        return [entity]  # Return original entity if API fails

# Modified phrase_PD_perturbation function to use Nebius API
def phrase_PD_perturbation_nebius(cnn_dm_prompt, epsilon):
    """
    Modified version using Nebius API instead of local model
    """
    doc = nlp(cnn_dm_prompt)
    entity_replacements = defaultdict(str)
    
    # for each extracted entity in the document.
    for ent in doc.ents:
        print("ent = ", ent.text, " label = ", ent.label_)
        entity = ent.text 
        
        # Call Nebius API to obtain diverse candidates for the target entity.
        replacement_list = generate_named_entity_replacements_nebius(
            entity=entity, 
            entity_label=ent.label_,
            context_text=cnn_dm_prompt
        )
        
        # Prepare candidate_embeddings
        candidate_embeddings = {}
        
        for phrase in replacement_list:
            similarity = compute_similarity(sbert_model, entity, phrase)
            print("\t", phrase, "sim score = ", similarity)
            candidate_embeddings[phrase] = get_embedding(sbert_model, phrase).cpu().numpy()
            
        dp_replacement = differentially_private_replacement(
            target_phrase=entity,
            epsilon=epsilon,
            candidate_phrases=replacement_list,
            candidate_embeddings=candidate_embeddings,
            sbert_model=sbert_model
        )
        
        print("DP replacement = ", dp_replacement)
        entity_replacements[entity] = dp_replacement

    sanitized_prompt = cnn_dm_prompt
    for original_entity, new_entity in entity_replacements.items():
        sanitized_prompt = sanitized_prompt.replace(original_entity, new_entity)
    
    return sanitized_prompt

# Test function to verify the API works
def test_nebius_api():
    """Test function to verify Nebius API is working"""
    print("Testing Nebius API...")
    
    test_prompt = "The quick brown fox"
    test_perturbed = "The slow red cat jumps over the lazy dog"
    
    result = extraction_module_nebius(test_prompt, test_perturbed)
    print(f"Test result: {result}")
    
    # Test entity replacement
    test_entities = generate_named_entity_replacements_nebius("Apple", "ORG", "Apple is a technology company")
    print(f"Test entities: {test_entities}")

if __name__ == "__main__":
    test_nebius_api()