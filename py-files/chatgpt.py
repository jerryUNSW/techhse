from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

client = OpenAI(api_key=OPEN_AI_KEY)

# Set your API key

# Make a request to the ChatGPT API
response = client.chat.completions.create(
    model="gpt-3.5-turbo",  
    # or "gpt-4" if you have access
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me the capital of France?"}
    ],
    max_tokens=50,  # Limit the response length
    # temperature=0.7  # Controls randomness (0.0 to 1.0)
)

# Print the response
print(response.choices[0].message.content)