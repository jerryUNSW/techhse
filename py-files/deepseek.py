from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY")

client = OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Tell me about interesting places in Berlin."},
    ],
    max_tokens=100,  # Limit the response length
    stream=False
)

print(response.choices[0].message.content)