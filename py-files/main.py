#!/usr/bin/env python3
import numpy as np
import time
from imports_and_init import *
from dp_sanitizer import get_embedding, load_sentence_bert, compute_similarity, differentially_private_replacement
import google.generativeai as genai

from difflib import SequenceMatcher


def normalize_answer(ans):
    return ans.lower().strip()

def exact_match(pred, gold):
    return normalize_answer(pred) == normalize_answer(gold)

def fuzzy_match(pred, gold):
    return SequenceMatcher(None, normalize_answer(pred), normalize_answer(gold)).ratio()


LLM_PROVIDER = "deepseek"

# ~/.cache/huggingface/hub

print("loading the API keys ")
load_dotenv()
DEEP_SEEK_KEY = os.getenv("DEEP_SEEK_KEY")
OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")
hugging_face_api = os.getenv("HUGGING_FACE")
Gemini_API = os.getenv("GEMINI_API")

NEBIUS_API = os.getenv("NEBIUS")

# ANSI color codes
RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


QWEN_MODEL_CANDIDATES = [
    "Qwen/Qwen2.5-Coder-7B-fast",
    "Qwen/Qwen2.5-Coder-7B",
    "Qwen/Qwen3-14B",  # pretty bad
    "meta-llama/Llama-3.3-70B-Instruct",  # great results
    "microsoft/phi-4",  # works well, 14B
    "meta-llama/Meta-Llama-3.1-8B-Instruct",  # refuses to refine
    "google/gemma-2-2b-it",
    "google/gemma-2-9b-it-fast",  # works good enough
    "mistralai/Mistral-Nemo-Instruct-2407"  # works alright
]

LOCAL_MODEL = "microsoft/phi-4" # this works well, it has 14B.

# Initialize Nebius client
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=NEBIUS_API,
)


from inferdpt import *

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print("GPU is available:", torch.cuda.get_device_name(device))
else:
    print("Using CPU")

# Load Sentence-BERT model
sbert_model = load_sentence_bert()

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load GPT-2 tokenizer (for consistency with your earlier request)
gpt2tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# this function is more flexible
def get_tokens(text, start=0, count=50):
    """Extract tokens using GPT-2 tokenizer."""
    tokens = gpt2tokenizer.tokenize(text)
    selected_tokens = tokens[start:start+count]
    tokenized_string = gpt2tokenizer.convert_tokens_to_string(selected_tokens)
    return tokenized_string

def get_first_tokens(text, N):
    """Extract the first tokens using GPT-2 tokenizer."""
    tokens = gpt2tokenizer.tokenize(text)
    first_50_tokens = tokens[:N]
    tokenized_string = gpt2tokenizer.convert_tokens_to_string(first_50_tokens)
    return tokenized_string

def slice_refined_text(text, prefix_text):
    # Find the index where "Refined text:" appears
    refined_start_index = text.find("Refined text:")
    
    if refined_start_index != -1:
        # Slice the text starting right after "Refined text:"
        refined_text = text[refined_start_index + len("Refined text:"):].strip()

        # Check if prefix_text appears in the refined_text and remove it if it does
        if refined_text.startswith(prefix_text):
            refined_text = refined_text[len(prefix_text):].strip()

        return refined_text
    else:
        # If "Refined text:" is not found, return the original text
        return text

# try google gemini 
def get_response_from_remote_LLM(user_prompt, max_tokens=200):
    """Get response from OpenAI/DeepSeek/Gemini based on the provider."""

    system_prompt = (
    "You are a helpful assistant. When asked to continue a text, return only the direct continuation. "
    "Do not include phrases like 'Here is the continuation:', 'Sure:', or any explanation. "
    "Return only the next sentence or paragraph that follows naturally."
    )

    user_prompt = f"Please help me to extend the following text: {user_prompt}"

    provider = LLM_PROVIDER
    # provider = "gemini"

    if provider == "openai":
        client = OpenAI(api_key=OPEN_AI_KEY)
        model = "gpt-4o-mini"
    elif provider == "deepseek":
        client = OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")
        model = "deepseek-chat"
    elif provider == "gemini":
        genai.configure(api_key=Gemini_API)
        # model_name = 'models/gemini-2.5-pro-preview-05-06'
        model_name = 'models/gemini-1.5-flash'
        print("using ", model_name)
    else:
        raise ValueError("Unsupported provider. Use 'openai' or 'deepseek'.")

    try:
        if provider=="gemini":
            model = genai.GenerativeModel(model_name)

            combined_prompt = f"""
            You are a helpful assistant. When asked to continue a text, return only the direct continuation. 
            Do not include phrases like 'Here is the continuation:', 'Sure:', or any explanation. "
            Return only the next sentence or paragraph that follows naturally.

            Prefix text: "{user_prompt}"

            Continuaton: 
            """
            
            # system_prompt + " " + user_prompt

            print("combined_prompt = ", combined_prompt)
            # response = model.generate_content(combined_prompt)

            response = model.generate_content(
                combined_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=150,  # You can increase this for longer outputs
                    temperature=0.9,        # Higher values = more diverse / longer responses
                    top_p=0.95,             # Top-p sampling
                    top_k=40                # (optional) limit to top-k tokens
                )
            )

            response_text = response.text.strip()

            return response_text
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content
    except Exception as e:
        print(f"Error with {provider} API: {e}")
        return "API Error"

def generate_named_entity_replacements(qwen_model, qwen_tokenizer, entity, entity_label, context_text):
    """
    Generate a list of named entity replacements for the given entity using QWEN.

    Args:
        qwen_model: The Qwen model instance.
        qwen_tokenizer: The tokenizer for the Qwen model.
        entity (str): The named entity to replace.
        entity_label (str): The entity label (e.g., PERSON, LOCATION).
        context_text (str): The full text containing the entity.
    Returns:
        List[str]: A list of candidate replacement phrases, including the original entity.
    """

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

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": candidate_replacement_prompt}
    ]

    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(qwen_model.device)
    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=128,  # shorter
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.2,
        eos_token_id=qwen_tokenizer.eos_token_id
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    replacement = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    replacement_list = replacement.strip().split('\n')

    replacement_list = [r.strip() for r in replacement_list if r.strip()]

    # Ensure original entity is included
    replacement_list.append(entity)

    return replacement_list

## see if repetition_penalty=1.2,  # Add this to discourage repetition
## check whether this can penalize the model from repeating itself.
def extraction_module(model, tokenizer, prefix_text, perturbed_generation):

    input_prompt = f'''
    Your taks is to refine 'Perturbed Generation' such that it is a seamless continuation of the 'Prefix Text'. 
    'Prefix Text': {prefix_text} \n 
    'Perturbed Generation': {perturbed_generation}\n
    Provide the refined text only. 
    Refined text: 
    '''

    # Tokenize the input prompt
    inputs = tokenizer(input_prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        pad_token_id=tokenizer.eos_token_id,  # prevent warning
        do_sample=True,
        top_p=0.95,
        temperature=0.7
    )
    extended_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("extended_text = ", extended_text)

    refined_text = slice_refined_text(extended_text, prefix_text)

    return refined_text

def generate_sentence_replacements_with_qwen(
    model,
    tokenizer,
    input_sentence,
    num_return_sequences=100,
    max_length=128,
    diversity_level="mixed"):

    prompt = f"""
    Your task is to generate diverse paraphrases of the sentence below.

    ### Output rules:
    - Output ONLY the paraphrased sentence.
    - Avoid repeating the original sentence.
    - No numbering, bullet points, or commentary.
    - Do NOT include introductory or explanatory text.
    - Sentence must be grammatically correct and semantically coherent.

    ### Diversity Requirements:
    - Include both close and distant semantic variants.
    - Vary structure, word choices, and phrasing.

    ### Example:
    Original: The quick brown fox jumps over the lazy dog.
    Paraphrase: A nimble brown fox leaps over a sleepy dog.

    ### Task:
    Original: {input_sentence}
    Paraphrase:"""

    # Control generation diversity using sampling params
    if diversity_level == "low":
        temperature = 0.7
        top_p = 0.9
    elif diversity_level == "high":
        temperature = 1.5
        top_p = 0.95
    else:  # mixed
        temperature = 1.0
        top_p = 0.92

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_length,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode and post-process
    candidates = tokenizer.batch_decode(output, skip_special_tokens=True)

    # Split output into individual paraphrases, remove prompt prefix and unwanted text
    cleaned = set()
    for text in candidates:
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            # Remove "Paraphrase:" prefix (case-insensitive) and other unwanted prefixes
            if line.lower().startswith("paraphrase:"):
                line = line[10:].strip()  # Remove "Paraphrase:" and any leading/trailing spaces
            elif line.startswith("###"):
                continue  # Skip lines starting with markdown headers
            elif line in ["sentence:", ""]:  # Skip irrelevant or empty lines
                continue
            # Only add non-empty lines that differ from the input sentence
            if line and line != input_sentence:
                cleaned.add(line)

    return list(cleaned)

# def generate_sentence_replacements_with_nebius(
#     local_model, 
#     input_sentence, 
#     num_return_sequences=100, 
#     max_tokens=150, 
#     diversity_level="high"):
#     system_prompt = "You are a helpful assistant that generates diverse paraphrases of given sentences."
    
#     if diversity_level == "low":
#         temperature = 0.7
#         top_p = 0.9
#     elif diversity_level == "high":
#         temperature = 1.5
#         top_p = 0.95
#     else:  # mixed
#         temperature = 1.0
#         top_p = 0.92
    
#     user_prompt = f"""
#     Your task is to generate a diverse paraphrase of the sentence below.

#     ### Output rules:
#     - Output ONLY the paraphrased sentence.
#     - Avoid repeating the original sentence.
#     - No numbering, bullet points, or commentary.
#     - Do NOT include introductory or explanatory text.
#     - The sentence must be grammatically correct and semantically coherent.

#     ### Diversity Requirements:
#     - Vary structure, word choices, and phrasing.

#     ### Example:
#     Original: The quick brown fox jumps over the lazy dog.
#     Paraphrase: A nimble brown fox leaps over a sleepy dog.

#     ### Task:
#     Original: {input_sentence}
#     Paraphrase:"""

    
#     try:
#         response = nebius_client.chat.completions.create(
#             model=local_model,
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_prompt}
#             ],
#             max_tokens=max_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             n=num_return_sequences
#         )
        
#         paraphrases = set()
#         for choice in response.choices:
#             clean_line = choice.message.content.strip()
#             if clean_line and clean_line != input_sentence:
#                 paraphrases.add(clean_line)
        
#         return list(paraphrases)
    
#     except Exception as e:
#         print(f"Error with Nebius API: {e}")
#         return []

def generate_sentence_replacements_with_nebius(
    local_model,
    input_sentence,
    num_return_sequences=100,
    max_tokens=150
):
    system_prompt = """
    You are a creative assistant specializing in generating paraphrases with diverse semantic interpretations. Your goal is to rephrase the input sentence in varied ways, altering structure, word choice, and meaning while preserving the core intent. Avoid literal rephrasing; instead, explore different perspectives, contexts, or expressions.
    """

    user_prompt = f"""
    Your task is to generate a diverse paraphrase of the sentence below, emphasizing varied semantic meanings.

    ### Output Rules:
    - Output ONLY the paraphrased sentence.
    - Do NOT repeat the original sentence or use near-identical phrasing.
    - No numbering, bullet points, or commentary.
    - Do NOT include introductory or explanatory text.
    - Ensure the sentence is grammatically correct and semantically coherent.

    ### Diversity Requirements:
    - Vary sentence structure, vocabulary, and perspective.
    - Explore alternative contexts or interpretations of the original meaning.
    - Avoid minor word substitutions; aim for creative re-expressions.

    ### Example:
    Original: The quick brown fox jumps over the lazy dog.
    Paraphrases:
    - A swift fox vaults over a resting hound.
    - The energetic brown fox hops past the idle dog.
    - A nimble fox clears the lounging canine in a bound.

    ### Task:
    Original: {input_sentence}
    Paraphrase:
    """

    try:
        response = nebius_client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=1.7,
            top_p=0.98,
            n=num_return_sequences
        )

        paraphrases = set()
        for choice in response.choices:
            clean_line = choice.message.content.strip()
            if clean_line and clean_line.lower() != input_sentence.lower():
                paraphrases.add(clean_line)

        # Basic semantic diversity filter: remove paraphrases too similar to each other
        filtered_paraphrases = []
        for paraphrase in paraphrases:
            if not any(
                len(set(paraphrase.lower().split()) & set(existing.lower().split())) > 0.7 * len(paraphrase.lower().split())
                for existing in filtered_paraphrases
            ):
                filtered_paraphrases.append(paraphrase)

        return filtered_paraphrases[:num_return_sequences]

    except Exception as e:
        print(f"Error with Nebius API: {e}")
        return []
        
def phrase_PD_perturbation(local_model, cnn_dm_prompt, epsilon):


    candidate_sentences = generate_sentence_replacements_with_nebius(
        local_model, 
        input_sentence=cnn_dm_prompt,
        num_return_sequences=10,
    )

    # Step 2: Precompute embeddings
    candidate_embeddings = {
        sent: get_embedding(sbert_model, sent).cpu().numpy()
        for sent in candidate_sentences
    }

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

def extraction_module_nebius_no_system_prompt(local_model, 
    prefix_text, perturbed_generation, 
    max_tokens=150):
    input_prompt = f'''
    Your task is to refine 'Perturbed Generation' such that it is a seamless continuation of the 'Prefix Text'. 
    'Prefix Text': {prefix_text} 
    'Perturbed Generation': {perturbed_generation}
    Provide the refined text only and don't provide additional editorial artefacts. 
    Don't include the 'Prefix Text' in the response. 
    Refined text: 
    '''

    try:
        response = nebius_client.chat.completions.create(
            model=local_model,
            messages=[
                # {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant that refines text to create seamless continuations."},
                {"role": "user", "content": input_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95
        )
        
        extended_text = response.choices[0].message.content
        # print("raw response = ", extended_text)
        
        # Use your existing slice_refined_text function
        # refined_text = slice_refined_text(extended_text, prefix_text)
        
        # return refined_text

        return extended_text
        
    except Exception as e:
        print(f"Error with Nebius API: {e}")
        return perturbed_generation  # Return original if API fails


def extraction_module_nebius_no_system_prompt_qa(local_model, context, question, noisy_answer, max_tokens=100):
    input_prompt = f'''
    Your task is to refine the 'Noisy Answer' so that it directly and concisely answers the 'Question' based on the provided 'Context'. 
    Do not repeat the question or context in your answer. 
    Only provide the refined answer.

    Context: {context}
    Question: {question}
    Noisy Answer: {noisy_answer}

    Refined Answer:
    '''

    try:
        response = nebius_client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "user", "content": input_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9
        )
        
        refined_answer = response.choices[0].message.content.strip()
        return refined_answer
        
    except Exception as e:
        print(f"Error with Nebius API: {e}")
        return noisy_answer  # fallback

def extraction_module_nebius(local_model, prefix_text, perturbed_generation, 
    max_tokens=250):
    system_prompt = (
        "You are a helpful assistant that refines noisy text so it flows naturally after a given prefix. "
        "Only return the continuation. Do not repeat or include the prefix."
    )
    
    user_prompt = (
        f"Refine the following continuation so it aligns smoothly with the prefix.\n\n"
        f"Prefix Text: {prefix_text}\n"
        f"Perturbed Generation: {perturbed_generation}\n\n"
        f"Refined Text:"
    )

    try:
        response = nebius_client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error with Nebius API: {e}")
        return perturbed_generation

def extraction_module_nebius_adv(local_model, prefix_text, perturbed_generation, 
    max_tokens=250):
    system_prompt = (
        "You are a helpful assistant that refines text continuations to align seamlessly with a given prefix."
        "Output only the refined continuation as plain text."
        "Do not include explanatory text, prefixes, commentary, or any formatting like JSON."
        "Ensure the output is grammatically correct, coherent, and matches the tone, style, and sentiment of the prefix."
    )
    
    user_prompt = (
        f"Your task is to refine the 'Perturbed Generation' to flow naturally from the 'Prefix Text', closely matching its wording, language, style, tone, and sentiment. Improve the perturbed generation's coherence and alignment with the prefix without repeating the perturbed text verbatim. Output only the refined continuation as plain text, with no prefixes, commentary, or additional formatting.\n\n"
        f"Prefix Text: {prefix_text}\n"
        f"Perturbed Generation: {perturbed_generation}\n\n"
        f"Refined Text:"
    )

    try:
        response = nebius_client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error with Nebius API: {e}")
        return perturbed_generation

def process_article(i, local_model, epsilon, prefix_):

    # rantext responses
    rantext_perturbed_prefix  = perturb_sentence(prefix_, epsilon)
    rantext_noisy_response = get_response_from_remote_LLM(rantext_perturbed_prefix)
    refined_rantext_response = extraction_module_nebius_no_system_prompt(local_model, 
        prefix_, 
        rantext_noisy_response
    )
    refined_rantext_response = get_tokens(refined_rantext_response, 0, 100)



    # phrase DP:
    dp_perturbed_prefix  =  phrase_PD_perturbation(local_model, 
        prefix_, 
        epsilon
    )
    dp_noisy_response = get_response_from_remote_LLM(dp_perturbed_prefix)

    # 
    refined_dp_response = extraction_module_nebius(local_model, 
        prefix_, dp_noisy_response
    )

    # adv way:
    # refined_dp_response = extraction_module_nebius_adv(local_model, 
    #     prefix_, dp_noisy_response
    # )
    # generate 250, grab 100
    refined_dp_response = get_tokens(refined_dp_response, 0, 100)

    # print(f"{CYAN}original Prompt:{RESET}\n{prefix_}\n")
    # ## RANTEXT 
    # print(f"{BLUE}rantext_noisy_response:{RESET}\n{rantext_noisy_response}\n")
    # print(f"{BLUE}refined_rantext_response:{RESET}\n{refined_rantext_response}\n")
    # ## PhraseLDP
    # print(f"{YELLOW}dp_noisy_response:{RESET}\n{dp_noisy_response}\n")
    # print(f"{YELLOW}refined_dp_response :{RESET}\n{refined_dp_response}\n")
    # build result item with everything
    result_item = {
        "article_id": i,
        "old_prompt": prefix_,

        "rantext_perturbed_prefix": rantext_perturbed_prefix, 
        "rantext_noisy_response": rantext_noisy_response,
        "refined_rantext_response": refined_rantext_response,
        
        "dp_perturbed_prefix": dp_perturbed_prefix,
        "dp_noisy_response": dp_noisy_response,
        "dp_refined_response": refined_dp_response
    }
    return result_item

def unit_test():

    epsilon = 1

    prefix_ = "Isaac Florentine has made some of the best western Martial Arts action movies ever produced. In particular US Seals 2, Cold Harvest, Special Forces and Undisputed 2 are all action classics. You can tell Isaac has a real passion for"

    # testing each model candidates in nebius
    for local_model in [LOCAL_MODEL]:

        print("using model: ", local_model)
        dp_perturbed_prefix  =  phrase_PD_perturbation(local_model, prefix_, epsilon)

        dp_noisy_response = get_response_from_remote_LLM(dp_perturbed_prefix)

        refined_dp_response = extraction_module_nebius(local_model, prefix_, dp_noisy_response)

        print(f"{CYAN}original Prompt:{RESET}\n{prefix_}\n")

        print(f"{BLUE}DP Prompt:{RESET}\n{dp_perturbed_prefix}\n")



        print(f"{YELLOW}dp_noisy_response:{RESET}\n{dp_noisy_response}\n")

        print(f"{YELLOW}refined_dp_response:{RESET}\n{refined_dp_response}\n")

        ## PhraseLDP


        rantext_perturbed_prefix  = perturb_sentence(prefix_, epsilon)
        rantext_noisy_response = get_response_from_remote_LLM(rantext_perturbed_prefix)
        refined_rantext_response = extraction_module_nebius_no_system_prompt(local_model, 
            prefix_, 
            rantext_noisy_response
        )
        refined_rantext_response = get_tokens(refined_rantext_response, 0, 100)
        



        ## RANTEXT 
        print(f"{BLUE}rantext_noisy_response:{RESET}\n{rantext_noisy_response}\n")

        print(f"{BLUE}refined_rantext_response:{RESET}\n{refined_rantext_response}\n")

    ## this is a hard replacement. 
    ## should we ask local language model to replace it instead? 
    # reversed_dp_response = refined_dp_response
    # for original_entity, new_entity in entity_replacements.items():
    #     if not new_entity.strip():
    #         continue
    #     reversed_dp_response = reversed_dp_response.replace(new_entity, original_entity)
    # new_refined_response_with_reverse_rp = extraction_module_nebius_with_entity_replacement(
    #     prefix_, dp_noisy_response, entity_replacements
    # )


    # print(f"{CYAN}original Prompt:{RESET}\n{prefix_}\n")


    # ## RANTEXT 
    # # print(f"{BLUE}rantext_noisy_response:{RESET}\n{rantext_noisy_response}\n")

    # # print(f"{BLUE}refined_rantext_response:{RESET}\n{refined_rantext_response}\n")


    # ## PhraseLDP
    # print(f"{YELLOW}dp_noisy_response:{RESET}\n{dp_noisy_response}\n")

    # print(f"{YELLOW}refined_dp_response:{RESET}\n{refined_dp_response}\n")

    # print(f"{YELLOW}reversed_dp_response:{RESET}\n{reversed_dp_response}\n")

    # print(f"{YELLOW}new reversed_dp_response:{RESET}\n{new_refined_response_with_reverse_rp}\n")
    
    # issue: dp replacement should probably also use a small model.


def get_answer_from_LLM(context, question, max_tokens=100):
    """
    Get a short answer to a question given a context using the selected LLM provider.
    """

    system_prompt = (
        "You are a helpful assistant. Answer the user's question using only the given context. "
        "Be concise. If the answer is not in the context, respond with 'I don't know'."
    )

    user_prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"

    provider = LLM_PROVIDER

    if provider == "openai":
        client = OpenAI(api_key=OPEN_AI_KEY)
        model = "gpt-4o-mini"
    elif provider == "deepseek":
        client = OpenAI(api_key=DEEP_SEEK_KEY, base_url="https://api.deepseek.com")
        model = "deepseek-chat"
    elif provider == "gemini":
        genai.configure(api_key=Gemini_API)
        model_name = "models/gemini-1.5-flash"
        print("Using", model_name)
    else:
        raise ValueError("Unsupported provider: must be 'openai', 'deepseek', or 'gemini'.")

    try:
        if provider == "gemini":
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                f"{system_prompt}\n\n{user_prompt}",
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=0.5,
                    top_p=0.95,
                    top_k=40
                )
            )
            return response.text.strip()

        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                stream=False
            )
            return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error with {provider} API: {e}")
        return "API Error"

def extraction_module_nebius_qa(local_model, context, question, noisy_answer, max_tokens=50):
    system_prompt = (
        "You are a helpful assistant that refines noisy answers to be direct and concise. "
        "Your job is to rewrite the answer so that it *directly* answers the question using only the given context. "
        "Do not include the context or question in the output. Just return the clean answer."
    )

    user_prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"Noisy Answer: {noisy_answer}\n\n"
        f"Refined Answer:"
    )

    try:
        response = nebius_client.chat.completions.create(
            model=local_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
            top_p=0.9
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"Error with Nebius API: {e}")
        return noisy_answer




def unit_test_qa():
    qa_examples = [
        {
            "context": "The capital of France is Paris, which is known for landmarks such as the Eiffel Tower and the Louvre Museum.",
            "question": "What is the capital of France?",
            "answer": "Paris"
        },
        {
            "context": "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            "question": "At what temperature does water boil at standard pressure?",
            "answer": "100 degrees Celsius"
        },
        {
            "context": "Isaac Newton formulated the laws of motion and universal gravitation in the 17th century.",
            "question": "Who formulated the laws of motion?",
            "answer": "Isaac Newton"
        },
        {
            "context": "The Amazon rainforest is the largest tropical rainforest in the world, spanning multiple South American countries.",
            "question": "What is the largest tropical rainforest in the world?",
            "answer": "The Amazon rainforest"
        },
        {
            "context": "The mitochondria in cells are responsible for producing ATP through the process of cellular respiration.",
            "question": "What do mitochondria produce?",
            "answer": "ATP"
        },
        {
            "context": "Mount Everest, located in the Himalayas, is the highest mountain peak on Earth.",
            "question": "Where is Mount Everest located?",
            "answer": "The Himalayas"
        },
        {
            "context": "The human heart has four chambers: two atria and two ventricles.",
            "question": "How many chambers does the human heart have?",
            "answer": "Four"
        },
        {
            "context": "J.K. Rowling is best known for writing the Harry Potter series, which has been translated into over 80 languages.",
            "question": "What is J.K. Rowling known for?",
            "answer": "Writing the Harry Potter series"
        },
        {
            "context": "The Sun is approximately 93 million miles away from Earth and is the center of our solar system.",
            "question": "How far is the Sun from Earth?",
            "answer": "93 million miles"
        },
        {
            "context": "Python is a popular programming language known for its simplicity and readability.",
            "question": "Why is Python popular?",
            "answer": "Its simplicity and readability"
        }
    ]
    qa_2 = [
        {
            "context": "Mercury is the closest planet to the Sun, but Venus is the hottest planet in our solar system due to its thick atmosphere.",
            "question": "Which planet is the hottest?",
            "answer": "Venus"
        },
        {
            "context": "There are 12 apples in a basket. If 5 are taken out, how many apples remain?",
            "question": "How many apples are left in the basket?",
            "answer": "7"
        },
        {
            "context": "The Declaration of Independence was signed in 1776, and the American Civil War began in 1861.",
            "question": "How many years passed between the Declaration of Independence and the Civil War?",
            "answer": "85 years"
        },
        {
            "context": "John left the party early because he was feeling unwell.",
            "question": "Why did John leave the party?",
            "answer": "Because he was feeling unwell"
        },
        {
            "context": "The Nobel Prize in Literature was awarded to Kazuo Ishiguro in 2017 for his novels.",
            "question": "Who won the Nobel Prize in Literature in 2017?",
            "answer": "Kazuo Ishiguro"
        },
        {
            "context": "Alice gave the book to Mary because she wanted to help her.",
            "question": "Who wanted to help whom?",
            "answer": "Alice wanted to help Mary"
        },
        {
            "context": "Because of heavy rain, the football match was postponed.",
            "question": "Why was the football match postponed?",
            "answer": "Because of heavy rain"
        },
        {
            "context": "The restaurant is famous for its seafood, but it does not serve fish.",
            "question": "Does the restaurant serve fish?",
            "answer": "No"
        },
        {
            "context": "Mary's brother is a doctor. He works at the city hospital.",
            "question": "Where does Mary's brother work?",
            "answer": "At the city hospital"
        },
        {
            "context": "The sky is not green but blue during the day.",
            "question": "Is the sky green during the day?",
            "answer": "No"
        }
    ]

    epsilon = 1
    local_model = LOCAL_MODEL

    phrase_dp_correct = 0
    rantext_correct = 0

    print("Using model:", local_model)

    # for i, qa in enumerate(qa_examples):
    for i, qa in enumerate(qa_2):  
        context = qa["context"]
        question = qa["question"]
        answer = qa["answer"]

        print(f"\n--- QA Example {i+1} ---")
        print(f"Q: {question}")
        print(f"Ground Truth: {answer}\n")

        ### PhraseLDP
        dp_context = phrase_PD_perturbation(local_model, context, epsilon)

        print("dp_context = ", dp_context)

        dp_question = phrase_PD_perturbation(local_model, question, epsilon)

        print("dp_question = ", dp_question)

        dp_response = get_answer_from_LLM(dp_context, dp_question)

        # dp_refined = extraction_module_nebius(local_model, context, dp_response)
        dp_refined = extraction_module_nebius_qa(local_model, context, question, dp_response)

        print(f"{YELLOW}PhraseLDP Noisy Answer:{RESET} {dp_response}")
        print(f"{YELLOW}PhraseLDP Refined Answer:{RESET} {dp_refined}")
        if exact_match(dp_refined, answer):
            phrase_dp_correct += 1


        exit(1)

        ### RanText
        rantext_context = perturb_sentence(context, epsilon)

        rantext_question = perturb_sentence(question, epsilon)

        rantext_response = get_answer_from_LLM(rantext_context, rantext_question)
        # rantext_refined = extraction_module_nebius_no_system_prompt(local_model, context, rantext_response)
        
        rantext_refined = extraction_module_nebius_no_system_prompt_qa(local_model, context, 
            question, 
            rantext_response
        )
        
        rantext_refined = get_tokens(rantext_refined, 0, 100)

        print(f"{BLUE}RanText Noisy Answer:{RESET} {rantext_response}")
        print(f"{BLUE}RanText Refined Answer:{RESET} {rantext_refined}")
        if exact_match(rantext_refined, answer):
            rantext_correct += 1

        

    total = len(qa_examples)
    print("\n--- QA Evaluation Summary ---")
    print(f"PhraseLDP Accuracy: {phrase_dp_correct}/{total} = {phrase_dp_correct / total:.2f}")
    print(f"RanText Accuracy:   {rantext_correct}/{total} = {rantext_correct / total:.2f}")

if __name__ == "__main__":

    # unit_test_qa()
    # exit(0)


    # Load CNN/DM dataset (for example usage)

    data_name = "cnn_dailymail"

    # data_name = "imdb"

    num_articles = 50

    print(f"Load {data_name} dataset")

    if data_name == "cnn_dailymail" :
        dataset = load_dataset(data_name, "3.0.0", split="test")

    if data_name == "imdb" :
        dataset = load_dataset(data_name, split="test")
        
    print("Size of dataset = ", len(dataset))

    ## deepSeek related code: 
    ##  we are using deepseek as the remote LLM
    client = OpenAI(api_key=DEEP_SEEK_KEY, 
        base_url="https://api.deepseek.com"
    )

    # for epsilon  in [1, 2, 3, 4, 5]:
    for epsilon  in [1]:

        args.eps = epsilon
    
        results = []

        results_dir = "./whole-replacement-results"

        os.makedirs(results_dir, exist_ok=True)

        model_name = LOCAL_MODEL.split("/")[-1]
        results_file_path = os.path.join(results_dir, f"all_{data_name}_{epsilon}_{model_name}.json")

        # results_file_path = os.path.join(results_dir, f"all_{data_name}_{epsilon}_{model_name}_adv.json")

        # Initialize results list or load existing data
        if os.path.exists(results_file_path):
            with open(results_file_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        else:
            results = []

        
        for i in range(num_articles):

            print("processing article: ", i)

            article = ""
            if data_name == "cnn_dailymail" :
                article = dataset[i]["article"]
            if data_name == "imdb" :
                article = dataset[i]["text"]

            # need to vary this
            cnn_dm_prompt = get_tokens(article, 0, 50)

            result_item = process_article(i, LOCAL_MODEL, epsilon, cnn_dm_prompt)


            results.append(result_item)
            
            # Write to file after each article is processed
            with open(results_file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Print progress message
            print(f"Article {i} processed and saved to {results_file_path}")

# my advantage: 
# I will use system prompt instead of everything in the user prompt 
# I will use better designed prompt, is this okay? 
# my framework is multi-agent. everything is modularized. easy to handle different remote LLM models. the existing work can only handle OpenAI models (is this true?)

# export CUDA_VISIBLE_DEVICES=2,3 