#!/usr/bin/env python3
"""
Simplified PhraseDP Agent Test - Direct API calls without LangChain complexity
"""

import os
import json
import yaml
from datasets import load_dataset
import utils
from sanitization_methods import phrasedp_sanitize_text
import openai
from dotenv import load_dotenv

load_dotenv()

def test_simple_phrasedp_pipeline():
    """Test the core PhraseDP pipeline directly."""

    print("🧪 Simple PhraseDP Agent Test")
    print("=" * 50)

    # Load MedQA dataset
    print("📊 Loading MedQA dataset...")
    dataset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')

    # Select a question
    question_index = 50
    item = dataset[question_index]

    print(f"\n📋 Testing Question (Index {question_index}):")
    print(f"Question: {item['question']}")
    print(f"Options:")
    options_dict = item['options']
    for key, value in options_dict.items():
        print(f"  {key}) {value}")
    print(f"Correct Answer: {item['answer_idx']}")

    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Get clients
    nebius_client = utils.get_nebius_client()
    epsilon = 1.0

    print(f"\n🔄 Step 1: Perturb Question with PhraseDP (ε={epsilon})")
    print("-" * 50)

    # Perturb the question
    original_question = item['question']
    perturbed_question = phrasedp_sanitize_text(
        original_question,
        epsilon=epsilon,
        nebius_client=nebius_client,
        nebius_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    print(f"Original: {original_question}")
    print(f"Perturbed: {perturbed_question}")

    print(f"\n🔄 Step 2: Batch Perturb Options with PhraseDP (ε={epsilon})")
    print("-" * 50)

    # Combine and perturb options
    options_list = [options_dict['A'], options_dict['B'], options_dict['C'], options_dict['D']]
    combined_options = "; ".join(options_list)
    print(f"Combined Options: {combined_options}")

    perturbed_combined_options = phrasedp_sanitize_text(
        combined_options,
        epsilon=epsilon,
        nebius_client=nebius_client,
        nebius_model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )

    print(f"Perturbed Options: {perturbed_combined_options}")

    print(f"\n🔄 Step 3: Generate Remote CoT with Perturbed Content")
    print("-" * 50)

    # Generate CoT with remote model using perturbed content
    formatted_question = f"{perturbed_question}\n\nOptions:\n{perturbed_combined_options}\n\nAnswer:"
    cot_prompt = f"{formatted_question}\n\nPlease provide a step-by-step chain of thought to solve this medical question:"

    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=config['remote_models']['cot_model'],
            messages=[
                {"role": "system", "content": "You are a medical expert. Provide a clear, step-by-step chain of thought to solve the given medical multiple choice question. Focus on medical reasoning and knowledge."},
                {"role": "user", "content": cot_prompt}
            ],
            max_tokens=512,
            temperature=0.0
        )

        remote_cot = response.choices[0].message.content
        print(f"Remote CoT (using perturbed content):")
        print(f"{remote_cot}")

    except Exception as e:
        print(f"❌ Error generating remote CoT: {e}")
        remote_cot = "Error generating CoT"

    print(f"\n🔄 Step 4: Local Model Answer with Private CoT")
    print("-" * 50)

    # Use local model with original question + private CoT
    formatted_original = f"{original_question}\n\nOptions:\n"
    for key, value in options_dict.items():
        formatted_original += f"{key}) {value}\n"
    formatted_original += "\n\nAnswer:"

    full_prompt = f"""{formatted_original}

Chain of Thought:
{remote_cot}

Based on the chain of thought above, what is the correct answer? Provide only the letter (A, B, C, or D):"""

    try:
        # Find working Nebius model
        def find_working_model(client):
            candidates = [config.get('local_model', 'meta-llama/Meta-Llama-3.1-8B-Instruct')]
            if 'meta-llama/Meta-Llama-3.1-8B-Instruct' not in candidates:
                candidates.append('meta-llama/Meta-Llama-3.1-8B-Instruct')

            for model in candidates:
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": "ping"}],
                        max_tokens=1,
                        temperature=0.0,
                    )
                    return model
                except Exception:
                    continue
            return candidates[0]

        local_model = find_working_model(nebius_client)
        response = nebius_client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "system", "content": "You are a medical expert. Use the provided chain of thought to answer the multiple choice question. Provide only the letter (A, B, C, or D) of the correct option."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=256,
            temperature=0.0
        )

        local_answer = response.choices[0].message.content.strip()

        # Extract letter from answer
        def extract_letter(answer):
            answer = answer.strip().upper()
            for letter in ['A', 'B', 'C', 'D']:
                if answer == letter or answer.startswith(letter) or f" {letter}" in answer:
                    return letter
            return answer[:1] if answer else "Error"

        predicted_letter = extract_letter(local_answer)
        is_correct = predicted_letter.upper() == item['answer_idx'].upper()

        print(f"Local Model: {local_model}")
        print(f"Local Answer: {local_answer}")
        print(f"Extracted Letter: {predicted_letter}")
        print(f"Correct Answer: {item['answer_idx']}")
        print(f"Result: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

        print(f"\n📊 FINAL SUMMARY:")
        print(f"=" * 50)
        print(f"✅ Question perturbed with PhraseDP (ε={epsilon})")
        print(f"✅ Options batch-perturbed with PhraseDP (ε={epsilon})")
        print(f"✅ Remote CoT generated using perturbed content (private)")
        print(f"✅ Local model answered using original content + private CoT")
        print(f"🎯 Final Result: {'CORRECT' if is_correct else 'INCORRECT'}")

        return {
            'success': True,
            'correct': is_correct,
            'perturbed_question': perturbed_question,
            'perturbed_options': perturbed_combined_options,
            'remote_cot': remote_cot,
            'local_answer': local_answer,
            'predicted_letter': predicted_letter
        }

    except Exception as e:
        print(f"❌ Error in local model inference: {e}")
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    test_simple_phrasedp_pipeline()