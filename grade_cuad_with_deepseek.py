#!/usr/bin/env python3
"""Grade CUAD run results using DeepSeek (DeepSeek chat) as a judge.

This script reads a run JSON produced by `run_cuad_500_llama8b_test.py`, loads the
corresponding CUAD dataset to obtain gold answers, and asks DeepSeek to judge each
model answer as 'Correct' or 'Incorrect'. Outputs a graded JSON and a small summary.

Usage:
  python grade_cuad_with_deepseek.py --results /path/to/cuad_500_...json

Note: Requires `priv-env` with repo dependencies and DEEP_SEEK_KEY set in .env.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

from datasets import load_dataset
from tqdm import tqdm

import utils


def get_gold_answer_from_item(item: Dict[str, Any]) -> str:
    # Try common CUAD / SQuAD-like shapes
    try:
        # CUAD uses 'answer_text' and 'answer_start' fields
        if "answer_text" in item and item["answer_text"]:
            return item["answer_text"].strip()
        if "answerText" in item and item["answerText"]:
            return item["answerText"].strip()
        if "qas" in item and item["qas"]:
            qa = item["qas"][0]
            if "answers" in qa and qa["answers"]:
                ans = qa["answers"][0]
                if isinstance(ans, dict) and "text" in ans:
                    return ans["text"].strip()
                if isinstance(ans, str):
                    return ans.strip()
        if "answers" in item and item["answers"]:
            a0 = item["answers"][0]
            if isinstance(a0, dict) and "text" in a0:
                return a0["text"].strip()
            if isinstance(a0, str):
                return a0.strip()
        for key in ("answer", "gold", "label", "solution"):
            if key in item and item[key]:
                return str(item[key]).strip()
    except Exception:
        pass
    return ""


def deepseek_judge(client, model_name: str, question: str, gold: str, model_answer: str) -> str:
    system = {"role": "system", "content": "You are an expert grader. Reply with only 'Correct' or 'Incorrect'."}
    user = {
        "role": "user",
        "content": (
            f"Question: {question}\n\nGround truth: {gold}\n\nModel answer: {model_answer}\n\n"
            "Is the model answer semantically correct compared to the ground truth? Reply with only 'Correct' or 'Incorrect'."
        ),
    }
    resp = client.chat.completions.create(
        model=model_name,
        messages=[system, user],
        max_tokens=8,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to run results JSON")
    parser.add_argument("--dataset", default="Nadav-Timor/CUAD", help="Hugging Face dataset id")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--deepseek_model", default="deepseek-chat", help="DeepSeek model id")
    parser.add_argument("--output", default=None, help="Output graded JSON path (optional)")
    args = parser.parse_args()

    with open(args.results, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    results = run_data.get("results", [])
    if not results:
        print("No results found in the provided JSON.")
        return

    print("Loading dataset", args.dataset, "split", args.split)
    dataset = load_dataset(args.dataset, split=args.split)

    # Build a quick mapping from normalized question text -> gold answer for robust lookup
    def norm(s: str) -> str:
        return " ".join(s.strip().split()).lower() if s else ""

    question_to_gold: Dict[str, str] = {}
    for i, it in enumerate(dataset):
        q = str(it.get("question", ""))
        a = get_gold_answer_from_item(it)
        if q and a:
            question_to_gold[norm(q)] = a

    client = utils.get_remote_llm_client("deepseek")

    graded = []
    correct = 0
    total_judged = 0

    for entry in tqdm(results, desc="Grading with DeepSeek"):
        idx = entry.get("dataset_index")
        # ensure idx is an int when possible (results may store it as string)
        try:
            if idx is not None:
                idx = int(idx)
        except Exception:
            idx = None
        question = entry.get("question", "")
        model_answer = entry.get("answer", "")

        gold = ""
        try:
            if idx is not None and 0 <= idx < len(dataset):
                gold = get_gold_answer_from_item(dataset[idx])
        except Exception:
            gold = ""

        # If gold still empty, try normalized-question lookup
        if not gold and question:
            nq = norm(question)
            if nq in question_to_gold:
                gold = question_to_gold[nq]

        if not gold:
            verdict = "NO_GOLD"
        else:
            try:
                verdict_raw = deepseek_judge(client, args.deepseek_model, question, gold, model_answer)
                verdict = verdict_raw.strip().lower()
                if verdict.startswith("correct"):
                    verdict = "Correct"
                    correct += 1
                else:
                    verdict = "Incorrect"
                total_judged += 1
            except Exception as e:
                verdict = f"ERROR: {e}"

        out = dict(entry)
        out["gold"] = gold
        out["verdict"] = verdict
        graded.append(out)

        # small pause to avoid bursting the API
        time.sleep(0.15)

    summary = {
        "total": len(results),
        "total_judged": total_judged,
        "correct": correct,
        "accuracy": (correct / total_judged) if total_judged else None,
    }

    out_path = args.output or (args.results.replace('.json', '') + '_graded.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"meta": {"deepseek_model": args.deepseek_model}, "summary": summary, "graded": graded}, f, indent=2, ensure_ascii=False)

    print("Grading complete. Summary:", summary)
    print("Graded results written to", out_path)


if __name__ == '__main__':
    main()


