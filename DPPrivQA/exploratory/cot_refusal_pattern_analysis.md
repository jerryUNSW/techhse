# CoT Refusal Pattern Analysis

## Key Finding

**The "I can't share step-by-step reasoning" pattern is UNIVERSAL, not specific to failures.**

## Statistics (MMLU Professional Law - 100 questions)

- **Total Local+CoT cases**: 100
- **Cases with refusal pattern**: 97 (97%)
- **Failed cases (is_correct=0)**: 28 total
  - With refusal pattern: 25 (89%)
- **Successful cases (is_correct=1)**: 72 total
  - With refusal pattern: 72 (100%)

## Pattern Examples

All these variations appear in both successful AND failed cases:

```
"I can't provide the step-by-step chain-of-thought you requested, but here's a concise, high-level reasoning and conclusion."
"I can't share step-by-step chain-of-thought reasoning, but here's a concise answer and brief explanation."
"I can't share my step-by-step internal reasoning, but here's a concise answer and brief explanation:"
"I can't share my step-by-step chain-of-thought, but here's a concise answer and reasoning summary."
"I can't share my step-by-step chain-of-thought. Here's a concise answer and key points instead:"
"I can't share my step-by-step internal chain-of-thought, but here's a concise, outcome-focused explanation."
"Sorry, I can't share step-by-step chain-of-thought reasoning. Here's a concise answer instead:"
```

## Root Cause

**GPT-5 has a built-in safety/refusal mechanism** that prevents it from sharing "internal reasoning" or "step-by-step chain-of-thought" when explicitly requested. Instead, it provides:

- "Concise summaries"
- "Key points"
- "High-level reasoning"
- "Brief explanations"

## Current Prompt

```python
prompt_lines = [
    "Here is the question:",
    question,
    "",
    "Please provide a clear, step-by-step chain-of-thought reasoning to solve this question. Do NOT provide the final answer; provide only the reasoning steps."
]

system_message = "You are an expert reasoner. Provide a clear, step-by-step chain of thought to analyze the given question. Focus on domain-appropriate reasoning and knowledge."
```

## Implications

1. **The refusal pattern is NOT the cause of failure** - it appears in 100% of successful cases
2. **The quality of the "concise explanation" varies** - some are helpful, some are not
3. **GPT-5 is refusing to follow the explicit instruction** to provide step-by-step reasoning
4. **The local model may benefit differently** from "concise summaries" vs true step-by-step reasoning

## Questions to Investigate

1. What distinguishes helpful "concise explanations" from unhelpful ones?
2. Would a different prompt format avoid the refusal?
3. Are there patterns in the content quality that correlate with success/failure?
4. Should we modify the prompt to work with GPT-5's refusal pattern rather than against it?

## Next Steps

1. Analyze the actual content quality of CoT text (not just the opening phrase)
2. Compare successful vs failed cases' CoT content structure
3. Consider prompt engineering to work with GPT-5's behavior rather than requesting what it refuses to provide
4. Test alternative prompts that don't trigger the refusal mechanism


