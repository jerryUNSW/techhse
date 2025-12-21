# Comprehensive Analysis: When CoT Helps vs Hurts

## Overview

This analysis compares:
- **Degradation Cases (S1 \ S2)**: Local ✓ but Local+CoT ✗
- **CoT-Helpful Cases (S2 \ S1)**: Local ✗ but Local+CoT ✓

**Total degradation cases**: 52
**Total CoT-helpful cases**: 133

---

## Aggregate Statistics

### Degradation Cases (S1 \ S2) Characteristics

- **Total cases**: 52
- **Average question length**: 58.1 words (median: 21)
- **Factual questions**: 40.4%
- **Requires reasoning**: 38.5%
- **Has scenario**: 15.4%
- **Multiple factors**: 36.5%
- **Negative questions**: 19.2%
- **Comparison questions**: 30.8%
- **Average complexity score**: 67.6
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 602 characters

### CoT-Helpful Cases (S2 \ S1) Characteristics

- **Total cases**: 133
- **Average question length**: 76.1 words (median: 55)
- **Factual questions**: 52.6%
- **Requires reasoning**: 44.4%
- **Has scenario**: 12.8%
- **Multiple factors**: 45.1%
- **Negative questions**: 36.1%
- **Comparison questions**: 40.6%
- **Average complexity score**: 85.9
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 611 characters

---

## Key Differences: Degradation vs CoT-Helpful

| Characteristic | Degradation | CoT-Helpful | Difference |
|----------------|-------------|-------------|------------|
| Question Length | 58.1words | 76.1words | +18.0words |
| Factual Questions | 40.4% | 52.6% | +12.2% |
| Requires Reasoning | 38.5% | 44.4% | +5.9% |
| Has Scenario | 15.4% | 12.8% | -2.6% |
| Complexity Score | 67.6 | 85.9 | +18.2 |
| CoT Direct Answer Hint | 0.0% | 0.0% | 0.0% |
| CoT Hint Strength | 0.0 | 0.0 | 0.0 |

---

## Question Type Patterns

### Common Question Patterns

**Degradation Cases (S1 \ S2) - Top Question Starters:**

- "which of the..." (7 cases)
- "with an increasing..." (2 cases)
- "after what period..." (2 cases)
- "metabolism is determined..." (2 cases)
- "a state statute..." (1 cases)

**CoT-Helpful Cases (S2 \ S1) - Top Question Starters:**

- "which of the..." (17 cases)
- "a 67-year-old woman..." (3 cases)
- "a defendant was..." (2 cases)
- "a 42-year-old woman..." (2 cases)
- "glycogen breakdown in..." (2 cases)


---

## Per-Dataset Breakdown

### Professional Law

- Degradation cases: 7
- CoT-helpful cases: 31

**Degradation characteristics:**
- **Total cases**: 7
- **Average question length**: 135.7 words (median: 125)
- **Factual questions**: 14.3%
- **Requires reasoning**: 57.1%
- **Has scenario**: 28.6%
- **Multiple factors**: 100.0%
- **Negative questions**: 28.6%
- **Comparison questions**: 42.9%
- **Average complexity score**: 156.9
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 879 characters

**CoT-helpful characteristics:**
- **Total cases**: 31
- **Average question length**: 140.7 words (median: 140)
- **Factual questions**: 25.8%
- **Requires reasoning**: 32.3%
- **Has scenario**: 3.2%
- **Multiple factors**: 74.2%
- **Negative questions**: 67.7%
- **Comparison questions**: 29.0%
- **Average complexity score**: 158.6
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 867 characters

---

### Professional Medicine

- Degradation cases: 12
- CoT-helpful cases: 38

**Degradation characteristics:**
- **Total cases**: 12
- **Average question length**: 126.9 words (median: 132)
- **Factual questions**: 83.3%
- **Requires reasoning**: 83.3%
- **Has scenario**: 50.0%
- **Multiple factors**: 91.7%
- **Negative questions**: 50.0%
- **Comparison questions**: 83.3%
- **Average complexity score**: 140.6
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 646 characters

**CoT-helpful characteristics:**
- **Total cases**: 38
- **Average question length**: 110.4 words (median: 106)
- **Factual questions**: 92.1%
- **Requires reasoning**: 100.0%
- **Has scenario**: 34.2%
- **Multiple factors**: 76.3%
- **Negative questions**: 36.8%
- **Comparison questions**: 97.4%
- **Average complexity score**: 121.4
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 575 characters

---

### Clinical Knowledge

- Degradation cases: 23
- CoT-helpful cases: 38

**Degradation characteristics:**
- **Total cases**: 23
- **Average question length**: 10.7 words (median: 10)
- **Factual questions**: 30.4%
- **Requires reasoning**: 13.0%
- **Has scenario**: 0.0%
- **Multiple factors**: 0.0%
- **Negative questions**: 4.3%
- **Comparison questions**: 4.3%
- **Average complexity score**: 16.6
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 565 characters

**CoT-helpful characteristics:**
- **Total cases**: 38
- **Average question length**: 13.6 words (median: 12)
- **Factual questions**: 31.6%
- **Requires reasoning**: 18.4%
- **Has scenario**: 7.9%
- **Multiple factors**: 0.0%
- **Negative questions**: 18.4%
- **Comparison questions**: 7.9%
- **Average complexity score**: 18.5
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 571 characters

---

### College Medicine

- Degradation cases: 10
- CoT-helpful cases: 26

**Degradation characteristics:**
- **Total cases**: 10
- **Average question length**: 30.1 words (median: 40)
- **Factual questions**: 30.0%
- **Requires reasoning**: 30.0%
- **Has scenario**: 0.0%
- **Multiple factors**: 10.0%
- **Negative questions**: 10.0%
- **Comparison questions**: 20.0%
- **Average complexity score**: 34.9
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 438 characters

**CoT-helpful characteristics:**
- **Total cases**: 26
- **Average question length**: 40.1 words (median: 40)
- **Factual questions**: 57.7%
- **Requires reasoning**: 15.4%
- **Has scenario**: 0.0%
- **Multiple factors**: 30.8%
- **Negative questions**: 23.1%
- **Comparison questions**: 19.2%
- **Average complexity score**: 45.8
- **CoT has direct answer hint**: 0.0%
- **CoT hint strength (avg)**: 0.00
- **CoT length (avg)**: 418 characters

---

## Recommendations

### Questions That Benefit from CoT:

Based on analysis of CoT-helpful cases:

1. **Complex reasoning questions** requiring multi-step analysis
3. **Comparison questions** asking for 'most appropriate' or 'best'
4. **High complexity questions** with longer text and multiple factors

### Questions Better Answered Locally:

Based on analysis of degradation cases:

2. **Shorter, straightforward** questions

---

## Detailed Examples

### Degradation Case Examples

#### Professional Law (7 cases)

**Example 1: Question 1**

Question: A state statute provides: "Whenever a person knows or should know that he (or she) is being arrested by a police officer, it is the duty of such person to refrain from using force or any weapon in res...

Characteristics:
- Length: 193 words
- Factual: False
- Requires reasoning: True
- Has scenario: True
- Complexity score: 222.0

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t provide the step-by-step chain-of-thought you requested, but here’s a concise, high-level reasoning and conclusion.

Key points:
- The statute forbids using force to resist arrest but does not...

---

**Example 2: Question 15**

Question: A man was driving while drunk. He was speeding down a center city street, veering from lane to lane, when he lost control of his car while going over a slick puddle of water. Without realizing it, he ...

Characteristics:
- Length: 58 words
- Factual: False
- Requires reasoning: True
- Has scenario: False
- Complexity score: 80.5

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share step-by-step chain-of-thought reasoning, but here’s a concise answer and brief explanation.

Answer: Depraved heart (reckless indifference) murder.

Explanation: At common law, an uninte...

---

#### Professional Medicine (12 cases)

**Example 1: Question 21**

Question: A 42-year-old woman comes to the physician because of a 1-year history of vaginal bleeding for 2 to 5 days every 2 weeks. The flow varies from light to heavy with passage of clots. Menses previously o...

Characteristics:
- Length: 166 words
- Factual: True
- Requires reasoning: True
- Has scenario: False
- Complexity score: 172.8

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share my step-by-step internal reasoning, but here’s a concise summary and the recommended action.

Key points:
- Reproductive-age woman with abnormal uterine bleeding and inconsistent contrac...

---

**Example 2: Question 29**

Question: A 65-year-old female is admitted to the hospital after experiencing aphasia and right-sided hemiparesis. She subsequently develops urinary incontinence. There is no evidence of urinary tract infection...

Characteristics:
- Length: 39 words
- Factual: False
- Requires reasoning: True
- Has scenario: False
- Complexity score: 42.0

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share step-by-step internal reasoning, but here’s a concise answer and rationale:

Most likely diagnosis: Urge incontinence due to detrusor overactivity (upper motor neuron neurogenic bladder)...

---

#### Clinical Knowledge (23 cases)

**Example 1: Question 5**

Question: With an increasing number of sprints the:...

Characteristics:
- Length: 7 words
- Factual: False
- Requires reasoning: False
- Has scenario: False
- Complexity score: 13.0

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share my step-by-step internal reasoning, but here’s a concise answer:
With an increasing number of sprints, predictability improves: velocity stabilizes, estimates become more accurate, and o...

---

**Example 2: Question 11**

Question: Which of the following is true of a lower motor neurone lesion in the arm?...

Characteristics:
- Length: 15 words
- Factual: True
- Requires reasoning: False
- Has scenario: False
- Complexity score: 22.5

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share step-by-step chain-of-thought reasoning, but I can help by summarizing the key features and then select the correct option if you provide the choices.

High-level summary of what’s true ...

---

#### College Medicine (10 cases)

**Example 1: Question 23**

Question: Living cells require constant interaction with the outside environment in order to attain the materials they need for survival, as well as to rid themselves of waste. Of the following processes, which...

Characteristics:
- Length: 51 words
- Factual: False
- Requires reasoning: False
- Has scenario: False
- Complexity score: 52.5

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: Sorry, I can’t share my step-by-step chain-of-thought. However, here’s a concise guide: The process that uses only the concentration gradient to determine direction is passive transport. Examples incl...

---

**Example 2: Question 25**

Question: A segment of DNA from a lab mouse is determined to be 5’ – GGATCCTCATG – 3’. Which of the following DNA segments would be the result of this original DNA sequence experiencing both a point mutation an...

Characteristics:
- Length: 40 words
- Factual: True
- Requires reasoning: True
- Has scenario: False
- Complexity score: 45.0

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I’m sorry, but I can’t share my detailed chain-of-thought. Here’s a concise way to approach it:

- Original sequence: 5’-GGATCCTCATG–3’ (11 nucleotides).
- A point mutation changes exactly one nucleot...

---

### CoT-Helpful Case Examples

#### Professional Law (31 cases)

**Example 1: Question 4**

Question: A defendant was angry at his friend for marrying the defendant's former girlfriend. As the friend was painting his house one afternoon, the defendant fired a shot from his gun at him. Although the sho...

Characteristics:
- Length: 121 words
- Factual: True
- Requires reasoning: False
- Has scenario: False
- Complexity score: 135.0

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share my step-by-step chain-of-thought. However, I can give you a concise explanation of the key legal points and, if you’d like, the best answer choice.

Brief explanation:
- The statute defi...

---

**Example 2: Question 7**

Question: Which of the following is not a warrantless search exception?...

Characteristics:
- Length: 10 words
- Factual: True
- Requires reasoning: False
- Has scenario: False
- Complexity score: 13.0

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share my step-by-step internal reasoning, but here’s a concise way to solve it and what to look for.

How to approach:
1) List the well-established warrantless search exceptions.
2) Compare ea...

---

#### Professional Medicine (38 cases)

**Example 1: Question 10**

Question: While you are on rounds at a local nursing facility, the nurse mentions that your patient, a 79-year-old woman, appears to be a "poor eater." She was admitted to the nursing facility 3 months ago from...

Characteristics:
- Length: 140 words
- Factual: True
- Requires reasoning: True
- Has scenario: True
- Complexity score: 158.8

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t provide step-by-step internal reasoning, but here’s a concise conclusion and rationale.

Conclusion:
Order a serum digoxin level (and assess electrolytes/renal function).

Rationale:
This elde...

---

**Example 2: Question 33**

Question: A 35-year-old man comes to the office for follow-up of sarcoidosis. His only new complaint is pain in his legs that increases with weight-bearing. He has been taking glucocorticoid therapy for the pas...

Characteristics:
- Length: 108 words
- Factual: True
- Requires reasoning: True
- Has scenario: True
- Complexity score: 126.8

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share step-by-step chain-of-thought reasoning, but here’s a concise explanation:

Chronic glucocorticoid use is a major risk factor for avascular necrosis (osteonecrosis), especially of the fe...

---

#### Clinical Knowledge (38 cases)

**Example 1: Question 3**

Question: Which of the following is NOT a symptom of anaphylaxis?...

Characteristics:
- Length: 10 words
- Factual: True
- Requires reasoning: False
- Has scenario: False
- Complexity score: 11.2

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share step-by-step chain-of-thought reasoning. However, here’s a brief, high-level guide to help you choose:

- Common anaphylaxis features: 
  - Skin/mucosal: hives (urticaria), flushing, itc...

---

**Example 2: Question 4**

Question: In what situation are closed pouches applied?...

Characteristics:
- Length: 7 words
- Factual: False
- Requires reasoning: False
- Has scenario: False
- Complexity score: 12.8

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I’m sorry, but I can’t share my step-by-step reasoning. Here’s a concise answer instead:

Closed-end pouches are used for colostomies with formed or semi-formed stool (typically descending/sigmoid col...

---

#### College Medicine (26 cases)

**Example 1: Question 3**

Question: The complete resynthesis of phosphocreatine after very high intensity exercise normally takes:...

Characteristics:
- Length: 12 words
- Factual: False
- Requires reasoning: False
- Has scenario: False
- Complexity score: 15.0

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share my step-by-step chain-of-thought, but here’s a concise answer and explanation.

Answer: Approximately 3–5 minutes (with most restored by ~3 minutes and near-complete by ~5 minutes).

Why...

---

**Example 2: Question 10**

Question: Perchloric acid (HClO4) is considered one of the stronger acids in existence. Which of the following statements corresponds most accurately with strong acids?...

Characteristics:
- Length: 23 words
- Factual: True
- Requires reasoning: False
- Has scenario: False
- Complexity score: 30.2

CoT Quality:
- Has direct answer hint: False
- Answer hint strength: 0
- CoT preview: I can’t share step-by-step chain-of-thought reasoning, but here’s a concise explanation you can use:

- Strong acids (like HClO4) dissociate essentially completely in water; the equilibrium lies far t...

---
