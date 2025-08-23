import os
from openai import OpenAI

NEBIUS_API = "eyJhbGciOiJIUzI1NiIsImtpZCI6IlV6SXJWd1h0dnprLVRvdzlLZWstc0M1akptWXBvX1VaVkxUZlpnMDRlOFUiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiJnb29nbGUtb2F1dGgyfDEwMzcxNjk3NzMxOTc3MDM3ODAxNyIsInNjb3BlIjoib3BlbmlkIG9mZmxpbmVfYWNjZXNzIiwiaXNzIjoiYXBpX2tleV9pc3N1ZXIiLCJhdWQiOlsiaHR0cHM6Ly9uZWJpdXMtaW5mZXJlbmNlLmV1LmF1dGgwLmNvbS9hcGkvdjIvIl0sImV4cCI6MTkwNjA5NjQwNywidXVpZCI6ImQ4ZTAxOTkzLTlhNzUtNDcxOC04YzZjLWQxZDg2NTBmN2E2YSIsIm5hbWUiOiJZaXpoYW5nIiwiZXhwaXJlc19hdCI6IjIwMzAtMDUtMjdUMDc6MTM6MjcrMDAwMCJ9.CkkdaZMDtIvODgWgjUv_Vit1Ay6rBdWRgJPNr35JOxY"



# Set up your API key (either set environment variable or replace with your key)
API_KEY = NEBIUS_API

# Initialize client
client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=API_KEY,
)

# Simple text generation function
def generate_text(prompt, max_tokens=500, temperature=0.7):
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-7B-Instruct",  # Try alternatives if this fails
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

# Function to list available models (for debugging)
def list_models():
    try:
        models = client.models.list()
        print("Available models:")
        for model in models.data:
            print(f"- {model.id}")
        return [model.id for model in models.data]
    except Exception as e:
        print(f"Error listing models: {e}")
        return []

# Common model name variations to try
MODEL_ALTERNATIVES = [
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B", 
    "qwen/qwen2.5-coder-7b-instruct",
    "qwen2.5-coder-7b-instruct",
    "Qwen2.5-Coder-7B-Instruct",
    "Qwen2.5-Coder-7B"
]

def find_working_model():
    available_models = list_models()
    
    for model_name in MODEL_ALTERNATIVES:
        if model_name in available_models:
            print(f"Found working model: {model_name}")
            return model_name
    
    # If no exact match, look for any Qwen model
    qwen_models = [m for m in available_models if 'qwen' in m.lower() or 'Qwen' in m]
    if qwen_models:
        print(f"Found Qwen models: {qwen_models}")
        return qwen_models[0]
    
    print("No Qwen models found. Available models:")
    for model in available_models:
        print(f"- {model}")
    return None

# Example usage
if __name__ == "__main__":
    # First, find the correct model name
    print("=== Finding Available Models ===")
    working_model = find_working_model()
    
    if not working_model:
        print("No suitable model found. Please check available models.")
        exit(1)
    
    # Update the generate_text function to use the found model
    def generate_text_with_model(prompt, max_tokens=500, temperature=0.7):
        try:
            response = client.chat.completions.create(
                model=working_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    print(f"\nUsing model: {working_model}")
    print("\n" + "="*50 + "\n")
    # Example 1: Code generation
    print("=== Code Generation Example ===")
    prompt1 = "Write a Python function to reverse a string"
    result1 = generate_text_with_model(prompt1)
    print(result1)
    print("\n" + "-"*50 + "\n")
    
    # Example 2: Explanation
    print("=== Explanation Example ===")
    prompt2 = "Explain what is a REST API in simple terms"
    result2 = generate_text_with_model(prompt2)
    print(result2)
    print("\n" + "-"*50 + "\n")
    
    # Example 3: Problem solving
    print("=== Problem Solving Example ===")
    prompt3 = "How to fix 'ModuleNotFoundError' in Python?"
    result3 = generate_text_with_model(prompt3)
    print("Problem Solving:")
    print(result3)