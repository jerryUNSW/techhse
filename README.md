Tech4HSE: Privacy-Preserving Multi-Hop Question Answering

A research framework for privacy-preserving question answering using differential privacy mechanisms.

PRIVACY MECHANISMS
------------------
- PhraseDP: Phrase-level differential privacy with medical mode
- PhraseDP+: PhraseDP with customized prompt based on background metadata (e.g., field-specific context)  
- InferDPT: Inference-based differential privacy for text
- SANTEXT+: Sanitized text plus mechanism
- CluSanT: Clustering-based sanitization technique

OVERVIEW
--------
This project implements privacy-preserving question answering by combining local and remote language models with phrase-level differential privacy techniques. It supports multiple inference scenarios and is specialized for medical QA datasets.

EXPERIMENTS
----------------
This project conducts four main experiments:

1. PII PROTECTION EXPERIMENTS
   - Test privacy protection effectiveness using labeled Kaggle PII dataset
   - Evaluate protection of 4 PII types: emails, phone numbers, addresses, names
   - Test 5 privacy mechanisms: PhraseDP, InferDPT, SANTEXT+, CluSanT, CusText+
   - Measure binary protection rates (leaked vs protected) across epsilon values 1.0-3.0
   - Use exact substring matching to detect PII leakage in sanitized text
   - Results will generate Figure 1 and Figure 2 for the Overleaf project

2. QA on MedQA-USMLE
   - Medical question answering dataset
   - Test accuracy and performance of privacy-preserving QA systems
   - Compare 7 inference strategies: Local, InferDPT, SANTEXT+, PhraseDP, PhraseDP+, Local + CoT, Remote
   - Results will generate Figure 3 for the Overleaf project
   - Evaluate privacy-utility trade-offs across different epsilon values

3. QA on MedMCQA
   - Medical multiple choice question answering dataset  
   - Test accuracy and performance of privacy-preserving QA systems
   - Compare 7 inference strategies: Local, InferDPT, SANTEXT+, PhraseDP, PhraseDP+, Local + CoT, Remote
   - Results will generate Figure 4 for the Overleaf project
   - Evaluate privacy-utility trade-offs across different epsilon values

4. QA on HSE-bench
   - Health, Safety, and Environment benchmark dataset
   - Test accuracy and performance of privacy-preserving QA systems
   - Compare 7 inference strategies: Local, InferDPT, SANTEXT+, PhraseDP, PhraseDP+, Local + CoT, Remote
   - Results will generate Figure 5 for the Overleaf project
   - Evaluate privacy-utility trade-offs across different epsilon values

PRIVACY-PRESERVING QA WORKFLOW
-------------------------------
The following flowchart demonstrates our privacy-preserving question answering process:

    Question with Context
         |
         +---> question -----> DP Sanitizer
         |                           |
         |                           v
         |                    Sanitized Question
         |                           |
         |                           v
         |                    LLM Server (Remote)
         |                           |
         |                           v
         |                    Chain-of-thought from 
         |                    remote LLM for solving
         |                    the sanitized question
         |                           |
         |                           v
         |                    Local LLM
         |                    (receives both context
         |                     and remote CoT)
         |                           |
         |                           v
         +---> context -------> Answer to the 
                                 original question

WORKFLOW DETAILS:
-----------------
1. Question with Context: Input medical question and relevant context
2. DP Sanitizer: Apply differential privacy mechanisms (PhraseDP, PhraseDP+, etc.)
3. LLM Server: Remote model processes sanitized question and generates reasoning
4. Local LLM: Combines original context with remote reasoning to produce final answer
5. Privacy Protection: Only sanitized question sent to remote server, original context stays local

WORKFLOW EXPLANATION:
---------------------
1. Input: Medical questions from datasets (MedQA-USMLE, MedMCQA, HSE-bench)
2. Privacy Mechanisms: Apply different sanitization methods with various epsilon values
3. Model Inference: Use local (Llama 8B) or remote (GPT-4o) models for answer generation
4. Evaluation: Compare accuracy and privacy protection across mechanisms
5. Analysis: Measure privacy-utility trade-offs and mechanism effectiveness

MAIN SCRIPTS
------------
- test-qa-1.py: Main script for Experiment 2 (QA on MedQA-USMLE)
- exp/ppi-protection/run_ppi_protection_experiment.py: Main script for Experiment 1 (PII protection)
- utils.py: Core utility functions and PhraseDP implementation
- sanitization_methods.py: Unified privacy mechanism interface
- create_results_database.py: Database management for results

QUICK START
-----------
# Install dependencies
conda create -n priv-env python=3.9
conda activate priv-env
pip install -r requirements.txt

# Run MedQA experiment
conda run -n priv-env python test-qa-1.py --epsilon 2.0 --index 0

# Run PII protection experiment
conda run -n priv-env python exp/ppi-protection/run_ppi_protection_experiment.py --rows 100

PROJECT STRUCTURE
-----------------
tech4HSE/
├── test-qa-1.py              # Main script for Experiment 2 (QA on MedQA-USMLE)
├── utils.py                         # Core utilities & PhraseDP
├── sanitization_methods.py          # Privacy mechanism interface
├── create_results_database.py       # Database management
├── ner_pii_privacy_evaluation.py    # PII protection evaluation
├── exp/                             # Experiment results
│   ├── medqa-ume-results/          # MedQA-USMLE QA results
│   ├── medmcqa/                    # MedMCQA QA results
│   ├── hse-bench/                  # HSE-bench QA results
│   └── ppi-protection/             # PII protection experiments
├── sanitization-methods/            # External privacy methods
│   ├── CluSanT/                    # CluSanT implementation
│   ├── InferDPT/                   # InferDPT implementation
│   ├── SanText/                    # SanText implementation
│   └── external/                   # CusText dependencies
├── reports-and-logs/               # Analysis, reports, logs
├── plots/                          # Generated visualizations
└── overleaf-folder/               # LaTeX documents

-POST-MEETING TO-DO
--------------------
- **Meeting date**: Tuesday, October 7, 2025
- **Add granularity to PPI protection experiment**: current evaluation uses exact substring matching to detect PII leakage; investigate and implement more granular detection (token-level / partial match / edit-distance / entity-level), update `exp/ppi-protection/` and `ner_pii_privacy_evaluation.py`, and verify that varying `epsilon` affects protection rates as expected.
- **Analyze PhraseDP strengths/weaknesses by question and dataset**: run per-question diagnostics across datasets (MedQA-USMLE, MedMCQA, HSE-bench, CUAD) to identify which question types PhraseDP performs well on or fails; save per-question reports in `exp/` and generate summary plots.
- **Add ablation studies and document them**: implement planned ablations (e.g., PhraseDP vs PhraseDP+, candidate pool size/band refill, fixed vs dynamic pools, CoT inclusion), run experiments, add results/plots to `overleaf-folder/` and document methodology and findings in this README.

Next steps:
- Run the PPI protection experiment with improved matching on a small sample to validate changes.
- Produce per-question PhraseDP diagnostics and draft ablation scripts.
- Update `overleaf-folder/` with resulting figures and a short write-up for the ablation section.



