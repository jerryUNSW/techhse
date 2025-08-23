import os
from dotenv import load_dotenv
from difflib import SequenceMatcher
from openai import OpenAI

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

def generate_sentence_replacements_with_nebius(nebius_client, nebius_model_name, input_sentence, num_return_sequences=10, max_tokens=150):
    """
    Generates diverse, generalized, and anonymized paraphrases of an input sentence
    using a Nebius LLM via the Nebius API.
    """
    system_prompt = """
    You are a highly skilled AI assistant specializing in natural language processing.
    Your primary goal is to rephrase user questions in a generalized manner,
    preserving their core logical structure and intent, but explicitly removing
    or generalizing any sensitive personal identifying information (PII) or
    unique details that could lead to re-identification.

    Focus on transforming specific entities (like names, exact locations, unique events,
    or precise dates) into generic descriptions or placeholders.
    The output should be a semantically diverse and coherent paraphrase.
    """

    user_prompt = f"""
    Generalize and paraphrase the following sentence.
    Ensure the logical structure and core meaning are maintained, but remove or generalize
    any sensitive personal information (like names of people, specific unique places,
    exact dates, or highly specific event details).

    ### Output Rules:
    - Output ONLY the paraphrased and generalized sentence.
    - Do NOT repeat the original sentence or use near-identical phrasing.
    - No numbering, bullet points, or commentary.
    - Do NOT include introductory or explanatory text.
    - Ensure the sentence is grammatically correct and semantically coherent.

    ### Generalization and Anonymization Examples:
    Original: "Was John Smith, born on October 26, 1970, in London, the first CEO of ExampleCorp?"
    Generalized: "Was an individual, born in a specific city, the first CEO of a certain corporation?"

    Original: "Did Sarah visit the Eiffel Tower on her trip to Paris last summer?"
    Generalized: "Did a person visit a famous landmark during their trip to a major European city recently?"

    Original: "Were Scott Derrickson and Ed Wood of the same nationality?"
    Generalized: "Did the two filmmakers share the same nationality?"

    ### Task:
    Original: {input_sentence}
    Paraphrase:
    """
    
    try:
        response = nebius_client.chat.completions.create(
            model=nebius_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.98,
            n=num_return_sequences
        )

        paraphrases = set()
        for choice in response.choices:
            clean_line = choice.message.content.strip()
            if clean_line and clean_line.lower() != input_sentence.lower():
                if not (clean_line.startswith("Generalize and paraphrase") or "Output ONLY the paraphrased" in clean_line):
                    paraphrases.add(clean_line)

        return list(paraphrases)

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