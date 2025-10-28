#!/usr/bin/env python3
"""Run a 500-question CUAD extractive/yes-no test using a LLaMA 8B-style model.

This script downloads the CUAD dataset (Hugging Face), samples N questions (default 500),
and runs them through a local or HF-hosted LLaMA-8B-family model. Results are written to a
timestamped JSON file whose name explains the experiment.

Usage examples:
  python run_cuad_500_llama8b_test.py --num_questions 500
  python run_cuad_500_llama8b_test.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --num_questions 500

Notes:
 - The script attempts to load the model specified by --model via Transformers. Adjust
   device options (CUDA/CPU) through environment (CUDA_VISIBLE_DEVICES) or edit the code.
 - If you don't have the model locally, Transformers will try to download it (ensure HF access).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List

import random
from datasets import load_dataset
from tqdm import tqdm

import utils


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def safe_get_text_field(dataset_item: Dict[str, Any]) -> str:
    """Return the most likely text/context field from a CUAD dataset item."""
    for candidate_key in ("context", "text", "contract_text", "document", "passage"):
        if candidate_key in dataset_item and dataset_item[candidate_key]:
            return dataset_item[candidate_key]
    # fallback: join all string fields
    pieces = [str(v) for v in dataset_item.values() if isinstance(v, str) and v.strip()]
    return "\n".join(pieces)


def safe_get_question_field(dataset_item: Dict[str, Any]) -> str:
    for candidate_key in ("question", "query", "q"):
        if candidate_key in dataset_item and dataset_item[candidate_key]:
            return dataset_item[candidate_key]
    # try structured QA fields (common in CUAD-like datasets)
    if "qas" in dataset_item and isinstance(dataset_item["qas"], list) and dataset_item["qas"]:
        first_qa = dataset_item["qas"][0]
        return first_qa.get("question", first_qa.get("q", ""))
    return ""


def build_prompt(contract_text: str, question_text: str) -> str:
    """Construct a concise prompt instructing the model to answer extractively when possible."""
    prompt = (
        "You are a contract-question answering assistant.\n"
        "Read the contract context carefully and then answer the question. If the answer is explicitly stated, return the exact span. "
        "If the answer is not present, reply 'NOT IN TEXT'. Keep answers concise.\n\n"
        f"Contract context:\n{contract_text}\n\nQuestion: {question_text}\nAnswer:"
    )
    return prompt


def get_gold_from_item(item: Dict[str, Any]) -> str:
    """Extract CUAD gold answer from a dataset item if available."""
    # CUAD uses 'answer_text' and 'answer_start'
    if isinstance(item, dict):
        if "answer_text" in item and item["answer_text"]:
            return item["answer_text"].strip()
        if "answerText" in item and item["answerText"]:
            return item["answerText"].strip()
        # fallback to SQuAD-like structure
        if "qas" in item and item["qas"]:
            qa = item["qas"][0]
            if "answers" in qa and qa["answers"]:
                a = qa["answers"][0]
                if isinstance(a, dict) and "text" in a:
                    return a["text"].strip()
                if isinstance(a, str):
                    return a.strip()
    return ""


def generate_answer_nebius(nebius_model_name: str, prompt: str, max_tokens: int = 256) -> str:
    """Call Nebius (OpenAI-compatible client returned by utils.get_nebius_client) to get an answer."""
    nebius_client = utils.get_nebius_client()
    system_msg = {
        "role": "system",
        "content": "You are a contract-question answering assistant. Answer concisely; return the exact span when present or 'NOT IN TEXT' if absent.",
    }
    user_msg = {"role": "user", "content": prompt}
    try:
        resp = nebius_client.chat.completions.create(
            model=nebius_model_name,
            messages=[system_msg, user_msg],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logging.exception("Nebius generation error: %s", e)
        return "ERROR"


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Run CUAD 500-question test with LLaMA-8B-family model")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="(deprecated) Transformers model name for LLaMA-8B-style model")
    parser.add_argument("--nebius_model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Nebius model name/ID to call for generation")
    parser.add_argument("--num_questions", type=int, default=500, help="Number of QA instances to test")
    parser.add_argument("--dataset", type=str, default="Nadav-Timor/CUAD", help="Hugging Face dataset identifier for CUAD")
    parser.add_argument("--split", type=str, default="train", help="Which split to sample from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path (optional)")
    args = parser.parse_args()

    random_generator = random.Random(args.seed)

    logging.info("Loading dataset %s (split=%s)", args.dataset, args.split)
    dataset = load_dataset(args.dataset, split=args.split)
    dataset_length = len(dataset)
    logging.info("Dataset contains %d instances", dataset_length)

    num_to_sample = min(args.num_questions, dataset_length)
    sampled_indices = random_generator.sample(range(dataset_length), k=num_to_sample)

    # Use Nebius client for generation (local LLaMA via Nebius)
    nebius_model_name = args.nebius_model

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output_filename = f"cuad_{num_to_sample}_questions_llama8b_test_{timestamp}.json"
    output_path = args.output or default_output_filename

    logging.info("Running %d questions through model %s", num_to_sample, args.model)

    results: List[Dict[str, Any]] = []

    for idx in tqdm(sampled_indices, desc="CUAD questions"):
        item = dataset[idx]
        contract_text = safe_get_text_field(item)
        question_text = safe_get_question_field(item)
        prompt = build_prompt(contract_text, question_text)
        model_answer = generate_answer_nebius(nebius_model_name, prompt)

        gold_answer = get_gold_from_item(item)
        result_entry = {
            "dataset_index": idx,
            "question": question_text,
            "contract_text_snippet": contract_text[:1000],
            "gold": gold_answer,
            "model": args.model,
            "answer": model_answer,
        }
        results.append(result_entry)

    logging.info("Saving results to %s", output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": args.model, "num_questions": num_to_sample, "timestamp": timestamp}, "results": results}, f, indent=2, ensure_ascii=False)

    logging.info("Done. Results written to %s", output_path)


if __name__ == "__main__":
    main()


