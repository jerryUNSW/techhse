#!/usr/bin/env python3
"""
Debug script to test Nebius API and list available models
"""

import os
from dotenv import load_dotenv
from openai import OpenAI
import yaml

# Load environment variables
load_dotenv()

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def test_nebius_connection():
    """Test connection to Nebius API"""
    api_key = os.getenv("NEBIUS")
    if not api_key:
        print("❌ NEBIUS API key not found in environment")
        return None

    base_url = "https://api.studio.nebius.ai/v1/"
    client = OpenAI(base_url=base_url, api_key=api_key)

    print(f"✅ Nebius client created with base_url: {base_url}")
    return client

def test_model(client, model_name):
    """Test if a specific model works with Nebius"""
    try:
        print(f"Testing model: {model_name}")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0.0
        )
        print(f"✅ {model_name} - WORKS")
        return True
    except Exception as e:
        print(f"❌ {model_name} - FAILED: {e}")
        return False

def test_all_configured_models(client):
    """Test all models from config"""
    print("\n=== Testing models from config ===")

    # Test main local model
    main_model = config.get('local_model')
    if main_model:
        print(f"\nTesting main local_model from config:")
        test_model(client, main_model)

    # Test all local models
    local_models = config.get('local_models', [])
    if local_models:
        print(f"\nTesting all local_models from config:")
        working_models = []
        for model in local_models:
            if test_model(client, model):
                working_models.append(model)

        print(f"\n✅ Working models: {working_models}")
        print(f"❌ Total models tested: {len(local_models)}")
        print(f"✅ Working models count: {len(working_models)}")

        return working_models

    return []

def test_common_model_names(client):
    """Test some common model name variations"""
    print("\n=== Testing common model name variations ===")

    common_models = [
        "microsoft/phi-4",
        "phi-4",
        "google/gemma-2-9b-it",
        "google/gemma-2-9b-it-fast",
        "gemma-2-9b-it",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Meta-Llama-3.1-8B-Instruct",
        "llama-3.1-8b-instruct",
        "Qwen/Qwen2.5-Coder-7B",
        "qwen2.5-coder-7b",
        "mistralai/Mistral-Nemo-Instruct-2407",
        "mistral-nemo-instruct"
    ]

    working_common = []
    for model in common_models:
        if test_model(client, model):
            working_common.append(model)

    return working_common

if __name__ == "__main__":
    print("🔍 Debugging Nebius API Model Availability")
    print("=" * 50)

    # Test connection
    client = test_nebius_connection()
    if not client:
        exit(1)

    # Test configured models
    working_config_models = test_all_configured_models(client)

    # Test common variations
    working_common_models = test_common_model_names(client)

    print("\n" + "=" * 50)
    print("🏆 SUMMARY")
    print("=" * 50)
    print(f"✅ Working models from config: {working_config_models}")
    print(f"✅ Working common model variations: {working_common_models}")

    if working_config_models or working_common_models:
        all_working = list(set(working_config_models + working_common_models))
        print(f"\n🎯 RECOMMENDED MODEL TO USE: {all_working[0]}")
        print(f"Update config.yaml with: local_model: {all_working[0]}")
    else:
        print("\n❌ No working models found. Check Nebius API key and model availability.")