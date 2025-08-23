import os
import re
from dotenv import load_dotenv
from difflib import SequenceMatcher
from openai import OpenAI
from prompt_loader import load_system_prompt, load_user_prompt_template, format_user_prompt

def is_nonsensical_question(question):
    """
    Detect nonsensical questions that are automatically "yes" or too obvious.
    These defeat the purpose of privacy protection.
    """
    question_lower = question.lower()
    
    # Only filter out the most obvious tautologies
    obvious_tautologies = [
        "is it true that two professionals from the same country share a nationality",
        "do people from the same place have the same origin",
        "are two people from the same country from the same country",
        "does the same thing belong to the same category",
        "is the same thing part of the same group",
        "are the same items in the same location",
        "do the same people come from the same place",
        "is it correct that same things are the same",
        "would it be accurate that same things are the same",
        "can it be said that same things are the same",
        "is it possible that same things are the same",
        "could it be that same things are the same",
        "might same things be the same",
        "would same things be the same"
    ]
    
    for tautology in obvious_tautologies:
        if tautology in question_lower:
            return True
    
    return False

# Load environment variables
load_dotenv()

# Get API keys from environment variables
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY")
HUGGING_FACE_API = os.getenv("HUGGING_FACE")
NEBIUS_API = os.getenv("NEBIUS")

def extract_final_answer_from_cot(text):
    """
    Extracts the final, concise answer from a Chain-of-Thought (CoT) response.
    """
    final_answer_marker = "Final Answer:"
    if final_answer_marker in text:
        return text.split(final_answer_marker)[-1].strip()
    sentences = text.split('.')
    if len(sentences) > 1:
        return sentences[-2].strip() if sentences[-2].strip() else text.strip()
    return text.strip()

def llm_judge_answer(client, question, ground_truth, model_answer):
    """
    Uses an LLM as a judge to determine if the model's answer is correct.
    """
    judge_prompt = (
        f"You are an expert evaluator. Your task is to determine if a model's "
        f"answer to a question is semantically correct, based on a given ground truth. "
        f"You must respond with only 'Correct' or 'Incorrect'.\n\n"
        f"Question: {question}\n"
        f"Ground Truth: {ground_truth}\n"
        f"Model Answer: {model_answer}\n\n"
        f"Is the model's answer semantically correct? Your answer must be a single word: Correct or Incorrect."
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Using REMOTE_JUDGE_MODEL as default, configurable via config
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": judge_prompt}
            ],
            max_tokens=10,
            temperature=0.0
        )
        verdict = response.choices[0].message.content.strip().lower()
        return verdict == "correct"
    except Exception as e:
        print(f"\033[91mError with Judge LLM API: {e}\033[0m")
        return False

def get_remote_llm_client(provider):
    """Get the appropriate client for the remote LLM API."""
    if provider == "openai":
        if not OPEN_AI_KEY:
            raise ValueError("OPEN_AI_KEY not found. Please set it in your .env file.")
        return OpenAI(api_key=OPEN_AI_KEY)
    elif provider == "deepseek":
        if not DEEP_SEEK_KEY:
            raise ValueError("DEEP_SEEK_KEY not found. Please set it in your .env file.")
        return OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unsupported remote LLM provider: {provider}")

def generate_sentence_replacements_with_nebius(nebius_client, nebius_model_name, input_sentence, num_return_sequences=10, max_tokens=150, num_api_calls=5):
    """
    Generates diverse, generalized, and anonymized paraphrases of an input sentence
    using a Nebius LLM via the Nebius API.
    """
    # Load prompts from external files
    system_prompt = load_system_prompt()
    user_prompt_template = load_user_prompt_template()
    user_prompt = format_user_prompt(user_prompt_template, input_sentence=input_sentence)
    
    try:
        all_paraphrases = []
        
        # Make multiple API calls to get more diverse candidates
        for call_num in range(num_api_calls):
            print(f"API call {call_num + 1}/{num_api_calls}...")
            
            response = nebius_client.chat.completions.create(
                model=nebius_model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.9,
                top_p=0.95,
                n=num_return_sequences
            )

            for choice in response.choices:
                content = choice.message.content.strip()
                if content and content.lower() != input_sentence.lower():
                    # Split by lines and extract individual paraphrases
                    lines = content.split('\n')
                    for line in lines:
                        clean_line = line.strip()
                        # Remove numbering (1., 2., etc.) and bullet points (-, •, etc.)
                        clean_line = re.sub(r'^\d+\.\s*', '', clean_line)  # Remove "1. ", "2. ", etc.
                        clean_line = re.sub(r'^[-•*]\s*', '', clean_line)  # Remove "- ", "• ", "* "
                        clean_line = clean_line.strip()
                        
                        # Minimal filtering - only basic quality checks, NO content filtering
                        if (clean_line and
                            len(clean_line) > 10 and  # Minimum length
                            clean_line.lower() != input_sentence.lower() and
                            not clean_line.startswith("Generate") and
                            not clean_line.startswith("Output") and
                            not clean_line.startswith("CRITICAL") and
                            not "paraphrase" in clean_line.lower() and
                            clean_line.endswith('?')):  # Should be a question
                            all_paraphrases.append(clean_line)
                            break  # Only take the first valid paraphrase from each completion

        print(f"Generated {len(all_paraphrases)} total candidates from {num_api_calls} API calls")
        return all_paraphrases

    except Exception as e:
        print(f"\033[91mError with Nebius API for paraphrase generation: {e}\033[0m")
        return []

def phrase_DP_perturbation(nebius_client, nebius_model_name, cnn_dm_prompt, epsilon, sbert_model):
    """
    Applies differential privacy perturbation to the question using the user's provided logic.
    """
    print(f"\033[92mApplying differential privacy perturbation with epsilon={epsilon}...\033[0m")

    # Step 1: Generate candidate sentence-level replacements using the Nebius model
    candidate_sentences = generate_sentence_replacements_with_nebius(
        nebius_client,
        nebius_model_name,
        input_sentence=cnn_dm_prompt,
        num_return_sequences=10,
    )

    if not candidate_sentences:
        raise ValueError("No candidate sentences were generated. Check the Nebius API call for paraphrase generation.")

    # Step 2: Precompute embeddings (assuming get_embedding and differentially_private_replacement are in dp_sanitizer)
    from dp_sanitizer import get_embedding, differentially_private_replacement
    candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in candidate_sentences}

    # Step 3: Select a replacement using exponential mechanism
    dp_replacement = differentially_private_replacement(
        target_phrase=cnn_dm_prompt,
        epsilon=epsilon,
        candidate_phrases=candidate_sentences,
        candidate_embeddings=candidate_embeddings,
        sbert_model=sbert_model
    )

    print("DP replacement selected:", dp_replacement)

    return dp_replacement

def phrase_DP_perturbation_with_candidates(nebius_client, nebius_model_name, cnn_dm_prompt, epsilon, sbert_model):
    """
    Applies differential privacy perturbation to the question and returns both the selected replacement and all candidates.
    """
    print(f"\033[92mApplying differential privacy perturbation with epsilon={epsilon}...\033[0m")

    # Step 1: Generate candidate sentence-level replacements using the Nebius model
    candidate_sentences = generate_sentence_replacements_with_nebius(
        nebius_client,
        nebius_model_name,
        input_sentence=cnn_dm_prompt,
        num_return_sequences=10,
    )

    if not candidate_sentences:
        raise ValueError("No candidate sentences were generated. Check the Nebius API call for paraphrase generation.")

    # Step 2: Precompute embeddings (assuming get_embedding and differentially_private_replacement are in dp_sanitizer)
    from dp_sanitizer import get_embedding, differentially_private_replacement
    candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in candidate_sentences}

    # Step 3: Select a replacement using exponential mechanism
    dp_replacement = differentially_private_replacement(
        target_phrase=cnn_dm_prompt,
        epsilon=epsilon,
        candidate_phrases=candidate_sentences,
        candidate_embeddings=candidate_embeddings,
        sbert_model=sbert_model
    )

    print("DP replacement selected:", dp_replacement)

    return dp_replacement, candidate_sentences