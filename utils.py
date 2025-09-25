import os
import re
from dotenv import load_dotenv
from difflib import SequenceMatcher
from openai import OpenAI
from prompt_loader import load_system_prompt, load_user_prompt_template, format_user_prompt


# Load environment variables
load_dotenv()

# Get API keys from environment variables
# Prefer OPENAI_API_KEY; fall back to legacy OPEN_AI_KEY for compatibility
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_AI_KEY")
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
    """
    Get the appropriate client for commercial remote LLM APIs (e.g., OpenAI, DeepSeek).

    IMPORTANT:
    - This helper is ONLY for remote/commercial providers used for CoT and judging.
    - It is NOT intended for local/edge model providers (e.g., Nebius) used inside
      PhraseDP candidate generation. For PhraseDP, use the Nebius-specific client
      getter (e.g., `get_nebius_client()`) to avoid misconfiguration.
    """
    if provider == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY (or legacy OPEN_AI_KEY) in .env")
        return OpenAI(api_key=OPENAI_API_KEY)
    elif provider == "deepseek":
        if not DEEP_SEEK_KEY:
            raise ValueError("DEEP_SEEK_KEY not found. Please set it in your .env file.")
        return OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")
    else:
        raise ValueError(f"Unsupported remote LLM provider: {provider}")


def get_nebius_client(config_model_fallback: str = None):
    """
    Return an OpenAI-compatible Nebius client for local/edge models used in PhraseDP.

    Reads credentials and endpoint from environment variables loaded via .env:
    - NEBIUS_API or NEBIUS_API_KEY: API key
    - NEBIUS_BASE_URL (optional): Base URL (defaults to https://api.studio.nebius.ai/v1/)

    The caller can choose a model name separately; a common pattern is to use
    `config['local_model']` as the default Nebius model.
    """
    api_key = os.getenv("NEBIUS_API") or os.getenv("NEBIUS_API_KEY") or NEBIUS_API
    if not api_key:
        raise ValueError("Nebius API key not found. Set NEBIUS_API or NEBIUS_API_KEY in .env")

    base_url = os.getenv("NEBIUS_BASE_URL") or "https://api.studio.nebius.ai/v1/"
    return OpenAI(base_url=base_url, api_key=api_key)

def generate_sentence_replacements_with_nebius(nebius_client, nebius_model_name, 
    input_sentence, num_return_sequences=10, max_tokens=150, num_api_calls=5):
    """
    DEPRECATED: Use `generate_sentence_replacements_with_nebius_diverse` instead.
    """
    raise NotImplementedError("generate_sentence_replacements_with_nebius is deprecated. Use the *_diverse variant.")

    # Historical implementation (commented out):
    # system_prompt = load_system_prompt()
    # user_prompt_template = load_user_prompt_template()
    # user_prompt = format_user_prompt(user_prompt_template, input_sentence=input_sentence)
    # try:
    #     all_paraphrases = []
    #     for call_num in range(num_api_calls):
    #         response = nebius_client.chat.completions.create(
    #             model=nebius_model_name,
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": user_prompt}
    #             ],
    #             max_tokens=max_tokens,
    #             temperature=0.9,
    #             top_p=0.95,
    #             n=num_return_sequences
    #         )
    #         for choice in response.choices:
    #             content = choice.message.content.strip()
    #             if content and content.lower() != input_sentence.lower():
    #                 lines = content.split('\n')
    #                 for line in lines:
    #                     clean_line = line.strip()
    #                     clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
    #                     clean_line = re.sub(r'^[-•*]\s*', '', clean_line)
    #                     clean_line = clean_line.strip()
    #                     if (clean_line and len(clean_line) > 10 and clean_line.lower() != input_sentence.lower()
    #                         and not clean_line.startswith("Generate") and not clean_line.startswith("Output")
    #                         and not clean_line.startswith("CRITICAL") and not "paraphrase" in clean_line.lower()
    #                         and clean_line.endswith('?')):
    #                         all_paraphrases.append(clean_line)
    #                         break
    #     return all_paraphrases
    # except Exception:
    #     return []

def generate_sentence_replacements_with_nebius_diverse(nebius_client, nebius_model_name, 
    input_sentence, num_return_sequences=10, max_tokens=150, num_api_calls=5,
    enforce_similarity_filter=True, filter_margin=0.05,
    low_band_quota_boost=True,
    refill_underfilled_bands=True,
    max_refill_retries=2,
    equal_band_target=None,
    global_equalize_max_loops=5,
    verbose=False):
    """
    NEW: Generates diverse candidates with targeted similarity levels across 5 API calls.
    Each API call targets a different similarity range for better exponential mechanism effectiveness.
    
    This is the improved version that generates candidates with wide similarity range (0.1-0.9)
    instead of the narrow range (0.59-0.85) from the original implementation.
    """
    # Load base prompts from external files
    system_prompt = load_system_prompt()
    user_prompt_template = load_user_prompt_template()
    
    # Define 5 different prompts for different similarity levels
    similarity_prompts = [
        {
            'level': 'band_0.0-0.1',
            'target': '0.0-0.1',
            'description': 'Extreme abstraction, preserve only core concept',
            'prompt': """
            Generate 5 paraphrases with EXTREMELY LOW similarity to the original.
            Target cosine similarity to the original between 0.0 and 0.1.
            Use maximum abstraction and very general terms.
            Replace all specific details with broad, generic terms.
            Focus on preserving only the core concept and question structure.
            Make the paraphrases extremely different from the original while maintaining the essential meaning.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        },
        {
            'level': 'band_0.1-0.2',
            'target': '0.1-0.2',
            'description': 'Very heavy abstraction, preserve core concept',
            'prompt': """
            Generate 5 paraphrases with VERY LOW similarity to the original.
            Target cosine similarity to the original between 0.1 and 0.2.
            Use very heavy abstraction and generalization.
            Replace most specific details with broad, generic terms.
            Focus on preserving only the core concept and question structure.
            Make the paraphrases very different from the original while maintaining the essential meaning.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        },
        {
            'level': 'band_0.2-0.3',
            'target': '0.2-0.3',
            'description': 'Heavy abstraction, preserve main concept',
            'prompt': """
            Generate 5 paraphrases with LOW similarity to the original.
            Target cosine similarity to the original between 0.2 and 0.3.
            Use heavy abstraction and generalization.
            Replace most specific details with broad, generic terms.
            Focus on preserving the main concept and question structure.
            Make the paraphrases quite different from the original while maintaining the essential meaning.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        },
        {
            'level': 'band_0.3-0.4',
            'target': '0.3-0.4',
            'description': 'Moderate-heavy abstraction, preserve main concept',
            'prompt': """
            Generate 5 paraphrases with MODERATE-LOW similarity to the original.
            Target cosine similarity to the original between 0.3 and 0.4.
            Use moderate-heavy abstraction and generalization.
            Replace many specific details with general terms.
            Preserve the main concept and some context.
            Make the paraphrases noticeably different from the original.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        },
        {
            'level': 'band_0.4-0.5',
            'target': '0.4-0.5',
            'description': 'Moderate abstraction, preserve main concept',
            'prompt': """
            Generate 5 paraphrases with MODERATE similarity to the original.
            Target cosine similarity to the original between 0.4 and 0.5.
            Use moderate abstraction and generalization.
            Replace many specific details with general terms.
            Preserve the main concept and some context.
            Make the paraphrases moderately different from the original.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        },
        {
            'level': 'band_0.5-0.6',
            'target': '0.5-0.6',
            'description': 'Light-moderate abstraction, preserve most context',
            'prompt': """
            Generate 5 paraphrases with MODERATE-HIGH similarity to the original.
            Target cosine similarity to the original between 0.5 and 0.6.
            Use light-moderate abstraction and some generalization.
            Replace some specific details with general terms.
            Preserve most context and details.
            Make the paraphrases somewhat different from the original.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        },
        {
            'level': 'band_0.6-0.7',
            'target': '0.6-0.7',
            'description': 'Light abstraction, preserve most context',
            'prompt': """
            Generate 5 paraphrases with HIGH similarity to the original.
            Target cosine similarity to the original between 0.6 and 0.7.
            Use light abstraction and some generalization.
            Replace some specific details with general terms.
            Preserve most context and details.
            Make the paraphrases somewhat different from the original.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        },
        {
            'level': 'band_0.7-0.8',
            'target': '0.7-0.8',
            'description': 'Minimal abstraction, preserve most details',
            'prompt': """
            Generate 5 paraphrases with HIGH similarity to the original.
            Target cosine similarity to the original between 0.7 and 0.8.
            Use minimal abstraction and generalization.
            Replace only the most specific details.
            Preserve most context, details, and terminology.
            Make the paraphrases similar to the original with minor changes.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        },
        {
            'level': 'band_0.8-0.9',
            'target': '0.8-0.9',
            'description': 'Very minimal changes, preserve almost everything',
            'prompt': """
            Generate 5 paraphrases with VERY HIGH similarity to the original.
            Target cosine similarity to the original between 0.8 and 0.9.
            Use minimal abstraction and generalization.
            Replace only PII (names, locations, organizations).
            Preserve almost all context, details, and terminology.
            Make the paraphrases very similar to the original with minimal changes.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        },
        {
            'level': 'band_0.9-1.0',
            'target': '0.9-1.0',
            'description': 'Minimal changes, preserve everything',
            'prompt': """
            Generate 5 paraphrases with EXTREMELY HIGH similarity to the original.
            Target cosine similarity to the original between 0.9 and 1.0.
            Use minimal abstraction and generalization.
            Replace only PII (names, locations, organizations).
            Preserve all context, details, and terminology.
            Make the paraphrases extremely similar to the original with minimal changes.
            Return each paraphrase on a new line, ending each with a question mark.
            """
        }
    ]
    
    try:
        all_paraphrases = []
        seen = set()
        # Prepare SBERT for optional post-filtering
        if enforce_similarity_filter:
            from sentence_transformers import SentenceTransformer
            from dp_sanitizer import compute_similarity
            sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        def _parse_range(rng: str):
            try:
                a, b = rng.split('-')
                return float(a), float(b)
            except Exception:
                return None
        
        # Make 10 API calls with different similarity-targeted prompts
        for call_num in range(min(num_api_calls, 10)):
            prompt_config = similarity_prompts[call_num]
            
            print(f"API call {call_num + 1}/{min(num_api_calls, 10)}: {prompt_config['level']} (target: {prompt_config['target']})")
            print(f"  Description: {prompt_config['description']}")
            
            # Create the full user prompt combining base template with similarity-specific instructions
            base_user_prompt = format_user_prompt(user_prompt_template, input_sentence=input_sentence)
            full_user_prompt = f"""
            {base_user_prompt}
            
            {prompt_config['prompt']}
            """
            
            # Increase quota for lower bands if requested
            n_per_call = num_return_sequences
            if low_band_quota_boost:
                if prompt_config['level'] in ('very_low_similarity', 'low_similarity'):
                    n_per_call = max(num_return_sequences, 40)
                elif prompt_config['level'] == 'medium_similarity':
                    n_per_call = max(num_return_sequences, 25)
                elif prompt_config['level'] == 'high_similarity':
                    n_per_call = max(num_return_sequences, 20)
                else:  # very_high
                    n_per_call = max(num_return_sequences, 15)

            def make_request(n):
                return nebius_client.chat.completions.create(
                    model=nebius_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": full_user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.9,
                    top_p=0.95,
                    n=n
                )

            response = make_request(n_per_call)

            band_added = 0
            target_band = _parse_range(prompt_config['target'])
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
                            # Optional similarity band filtering per targeted call
                            if enforce_similarity_filter and target_band is not None:
                                sim = compute_similarity(sbert_model, input_sentence, clean_line)
                                lo = max(0.0, target_band[0] - filter_margin)
                                hi = min(1.0, target_band[1] + filter_margin)
                                if lo <= sim <= hi:
                                    if clean_line not in seen:
                                        all_paraphrases.append(clean_line)
                                        seen.add(clean_line)
                                    band_added += 1
                                    break
                                else:
                                    # Skip this candidate; try next line in the same choice
                                    continue
                            else:
                                if clean_line not in seen:
                                    all_paraphrases.append(clean_line)
                                    seen.add(clean_line)
                                band_added += 1
                                break
            print(f"    Accepted {band_added} candidates in band {prompt_config['target']} (±{filter_margin})")
            
            # Show actual candidates if verbose
            if verbose and band_added > 0:
                print(f"    Generated candidates:")
                # Show the last band_added candidates from all_paraphrases
                recent_candidates = all_paraphrases[-band_added:]
                for i, candidate in enumerate(recent_candidates, 1):
                    sim = compute_similarity(sbert_model, input_sentence, candidate)
                    print(f"      {i:2d}. [{sim:.3f}] {candidate}")
                print()

            # Refill logic if a targeted band is underfilled
            if (enforce_similarity_filter and refill_underfilled_bands and target_band is not None):
                # Define soft targets per band to improve low-end diversity
                soft_targets = {
                    'very_low_similarity': 30,
                    'low_similarity': 30,
                    'medium_similarity': 25,
                    'high_similarity': 20,
                    'very_high_similarity': 15,
                }
                desired = (
                    int(equal_band_target) if (equal_band_target is not None) else soft_targets.get(prompt_config['level'], 20)
                )
                retries = 0
                while band_added < desired and retries < max_refill_retries:
                    retries += 1
                    extra_n = max(20, desired - band_added)
                    resp2 = make_request(extra_n)
                    for choice in resp2.choices:
                        content = choice.message.content.strip()
                        if content and content.lower() != input_sentence.lower():
                            lines = content.split('\n')
                            for line in lines:
                                clean_line = line.strip()
                                clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                                clean_line = re.sub(r'^[-•*]\s*', '', clean_line)
                                clean_line = clean_line.strip()
                                if (clean_line and len(clean_line) > 10 and clean_line.lower() != input_sentence.lower()
                                    and not clean_line.startswith("Generate") and not clean_line.startswith("Output")
                                    and not clean_line.startswith("CRITICAL") and not "paraphrase" in clean_line.lower()
                                    and clean_line.endswith('?')):
                                    sim = compute_similarity(sbert_model, input_sentence, clean_line) if enforce_similarity_filter else None
                                    lo = max(0.0, target_band[0] - filter_margin)
                                    hi = min(1.0, target_band[1] + filter_margin)
                                    if (not enforce_similarity_filter) or (lo <= sim <= hi):
                                        if clean_line in seen:
                                            continue
                                        all_paraphrases.append(clean_line)
                                        seen.add(clean_line)
                                        band_added += 1
                                        if band_added >= desired:
                                            break
                            if band_added >= desired:
                                break
                if retries:
                    print(f"    Refilled band {prompt_config['target']} -> now {band_added}/{desired}")

        # After all targeted calls, if equal_band_target specified: enforce equalization with while loop
        if enforce_similarity_filter and equal_band_target is not None:
            # Helper to count by band
            def count_by_band(cands):
                counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                bands = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
                for cand in cands:
                    sim = compute_similarity(sbert_model, input_sentence, cand)
                    for i, (blo, bhi) in enumerate(bands):
                        lo = max(0.0, blo - filter_margin)
                        hi = min(1.0, bhi + filter_margin)
                        if lo <= sim <= hi:
                            counts[i] += 1
                            break
                return counts

            def request_band_fill(band_index, needed):
                prompt_config = similarity_prompts[band_index]
                base_user_prompt = format_user_prompt(user_prompt_template, input_sentence=input_sentence)
                full_user_prompt = f"""
                {base_user_prompt}

                {prompt_config['prompt']}
                """
                resp = make_request(max(20, needed))
                added = 0
                tband = _parse_range(prompt_config['target'])
                for choice in resp.choices:
                    content = choice.message.content.strip()
                    if content and content.lower() != input_sentence.lower():
                        lines = content.split('\n')
                        for line in lines:
                            clean_line = line.strip()
                            clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
                            clean_line = re.sub(r'^[-•*]\s*', '', clean_line)
                            clean_line = clean_line.strip()
                            if (clean_line and len(clean_line) > 10 and clean_line.lower() != input_sentence.lower()
                                and not clean_line.startswith("Generate") and not clean_line.startswith("Output")
                                and not clean_line.startswith("CRITICAL") and not "paraphrase" in clean_line.lower()
                                and clean_line.endswith('?')):
                                sim = compute_similarity(sbert_model, input_sentence, clean_line)
                                lo = max(0.0, tband[0] - filter_margin)
                                hi = min(1.0, tband[1] + filter_margin)
                                if lo <= sim <= hi and clean_line not in seen:
                                    all_paraphrases.append(clean_line)
                                    seen.add(clean_line)
                                    added += 1
                                    if added >= needed:
                                        break
                        if added >= needed:
                            break
                return added

            loops = 0
            while loops < global_equalize_max_loops:
                loops += 1
                counts = count_by_band(all_paraphrases)
                needs = [max(0, equal_band_target - c) for c in counts]
                if sum(needs) == 0:
                    break
                for i, need in enumerate(needs):
                    if need > 0:
                        added = request_band_fill(i, need)
                        if added:
                            print(f"    Equalize loop {loops}: filled band {i} +{added} (target {equal_band_target})")

        print(f"Generated {len(all_paraphrases)} total candidates from {num_api_calls} API calls (after filtering={enforce_similarity_filter})")
        print(f"Expected similarity range: 0.1-0.9 (vs original 0.59-0.85)")
        return all_paraphrases

    except Exception as e:
        print(f"\033[91mError with Nebius API for diverse paraphrase generation: {e}\033[0m")
        return []


def phrase_DP_perturbation_diverse(nebius_client, nebius_model_name, cnn_dm_prompt, epsilon, sbert_model):
    """
    NEW: Applies differential privacy perturbation using the diverse candidate generation approach.
    This version should provide better exponential mechanism effectiveness due to wider similarity range.
    """
    print(f"\033[92mApplying DIVERSE differential privacy perturbation with epsilon={epsilon}...\033[0m")

    # Step 1: Generate diverse candidate sentence-level replacements using the new approach
    candidate_sentences = generate_sentence_replacements_with_nebius_diverse(
        nebius_client,
        nebius_model_name,
        input_sentence=cnn_dm_prompt,
        num_return_sequences=10,
    )

    if not candidate_sentences:
        raise ValueError("No candidate sentences were generated. Check the Nebius API call for diverse paraphrase generation.")

    # Step 2: Precompute embeddings (same as original)
    from dp_sanitizer import get_embedding, differentially_private_replacement
    candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in candidate_sentences}

    # Step 3: Select a replacement using exponential mechanism (same as original)
    dp_replacement = differentially_private_replacement(
        target_phrase=cnn_dm_prompt,
        epsilon=epsilon,
        candidate_phrases=candidate_sentences,
        candidate_embeddings=candidate_embeddings,
        sbert_model=sbert_model
    )

    print("Diverse DP replacement selected:", dp_replacement)
    return dp_replacement

def phrase_DP_perturbation_with_candidates_diverse(nebius_client, nebius_model_name, cnn_dm_prompt, epsilon, sbert_model):
    """
    NEW: Applies diverse differential privacy perturbation and returns both the selected replacement and all candidates.
    This version should provide better exponential mechanism effectiveness due to wider similarity range.
    """
    print(f"\033[92mApplying DIVERSE differential privacy perturbation with epsilon={epsilon}...\033[0m")

    # Step 1: Generate diverse candidate sentence-level replacements using the new approach
    candidate_sentences = generate_sentence_replacements_with_nebius_diverse(
        nebius_client,
        nebius_model_name,
        input_sentence=cnn_dm_prompt,
        num_return_sequences=10,
    )

    if not candidate_sentences:
        raise ValueError("No candidate sentences were generated. Check the Nebius API call for diverse paraphrase generation.")

    # Step 2: Precompute embeddings (same as original)
    from dp_sanitizer import get_embedding, differentially_private_replacement
    candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in candidate_sentences}

    # Step 3: Select a replacement using exponential mechanism (same as original)
    dp_replacement = differentially_private_replacement(
        target_phrase=cnn_dm_prompt,
        epsilon=epsilon,
        candidate_phrases=candidate_sentences,
        candidate_embeddings=candidate_embeddings,
        sbert_model=sbert_model
    )

    print("Diverse DP replacement selected:", dp_replacement)
    return dp_replacement, candidate_sentences

# def phrase_DP_perturbation(nebius_client, nebius_model_name, cnn_dm_prompt, epsilon, sbert_model):
#     """
#     DEPRECATED: Use `phrase_DP_perturbation_diverse` instead.
#     """
#     raise NotImplementedError("phrase_DP_perturbation is deprecated. Use phrase_DP_perturbation_diverse.")

    # Historical implementation (commented out):
    # print(f"\033[92mApplying differential privacy perturbation with epsilon={epsilon}...\033[0m")
    # candidate_sentences = generate_sentence_replacements_with_nebius(
    #     nebius_client,
    #     nebius_model_name,
    #     input_sentence=cnn_dm_prompt,
    #     num_return_sequences=10,
    # )
    # if not candidate_sentences:
    #     raise ValueError("No candidate sentences were generated. Check the Nebius API call for paraphrase generation.")
    # from dp_sanitizer import get_embedding, differentially_private_replacement
    # candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in candidate_sentences}
    # dp_replacement = differentially_private_replacement(
    #     target_phrase=cnn_dm_prompt,
    #     epsilon=epsilon,
    #     candidate_phrases=candidate_sentences,
    #     candidate_embeddings=candidate_embeddings,
    #     sbert_model=sbert_model
    # )
    # print("DP replacement selected:", dp_replacement)
    # return dp_replacement

# def phrase_DP_perturbation_with_candidates(nebius_client, nebius_model_name, cnn_dm_prompt, epsilon, sbert_model):
#     """
#     DEPRECATED: Use `phrase_DP_perturbation_with_candidates_diverse` instead.
#     """
#     raise NotImplementedError("phrase_DP_perturbation_with_candidates is deprecated. Use phrase_DP_perturbation_with_candidates_diverse.")

    # Historical implementation (commented out):
    # print(f"\033[92mApplying differential privacy perturbation with epsilon={epsilon}...\033[0m")
    # candidate_sentences = generate_sentence_replacements_with_nebius(
    #     nebius_client,
    #     nebius_model_name,
    #     input_sentence=cnn_dm_prompt,
    #     num_return_sequences=10,
    # )
    # if not candidate_sentences:
    #     raise ValueError("No candidate sentences were generated. Check the Nebius API call for paraphrase generation.")
    # from dp_sanitizer import get_embedding, differentially_private_replacement
    # candidate_embeddings = {sent: get_embedding(sbert_model, sent).cpu().numpy() for sent in candidate_sentences}
    # dp_replacement = differentially_private_replacement(
    #     target_phrase=cnn_dm_prompt,
    #     epsilon=epsilon,
    #     candidate_phrases=candidate_sentences,
    #     candidate_embeddings=candidate_embeddings,
    #     sbert_model=sbert_model
    # )
    # print("DP replacement selected:", dp_replacement)
    # return dp_replacement, candidate_sentences