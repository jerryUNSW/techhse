#!/usr/bin/env python3
"""
Test script for Lightning API
"""

from openai import OpenAI

def test_lightning_api():
    """Test the Lightning API connection"""
    try:
        client = OpenAI(
            base_url="https://lightning.ai/api/v1/",
            api_key="41efc720-1181-4916-896a-7c4ea4da0bc0/jerrystat2017/vision-model",
        )

        completion = client.chat.completions.create(
            model="openai/gpt-5",
            messages=[
              {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, world!"}]
              },
            ],
        )

        print(completion.choices[0].message.content)
        return True
        
    except Exception as e:
        print(f"Error testing Lightning API: {e}")
        return False

if __name__ == "__main__":
    print("Testing Lightning API...")
    success = test_lightning_api()
    if success:
        print("Lightning API test completed successfully!")
    else:
        print("Lightning API test failed!")
