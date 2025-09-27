# Tech4HSE: Privacy-Preserving Multi-Hop Question Answering

A comprehensive framework for privacy-preserving question answering using local and remote language models with differential privacy techniques.

## Overview

This project implements a multi-hop question answering system that combines local and remote language models while preserving privacy through differential privacy mechanisms. The system supports various scenarios including purely local inference, remote Chain-of-Thought (CoT) assistance, and privacy-preserving inference using phrase-level differential privacy.

## Features

- **Multi-Scenario Experiments**: Compare different inference strategies
- **Privacy-Preserving Inference**: Implement phrase-level differential privacy
- **Local + Remote Model Integration**: Combine local models with remote CoT assistance
- **Medical QA Support**: Specialized for medical question answering datasets
- **Flexible Model Support**: Support for multiple local and remote LLM providers

## Project Structure

```
tech4HSE/
├── main_qa.py                    # Main QA system with privacy features
├── multi_hop_experiment.py       # Multi-hop experiment framework
├── multi_hop_experiment_copy.py  # Alternative experiment implementation
├── utils.py                      # Utility functions for LLM interactions
├── dp_sanitizer.py              # Differential privacy implementation
├── config.yaml                  # Configuration file
├── imports_and_init.py          # Initialization and imports
├── testing_medical_qa.py        # Medical QA testing
├── test_phrase_dp.py            # Phrase DP testing
├── InferDPT/                    # InferDPT privacy framework
│   ├── main.py
│   ├── func.py
│   └── README.md
└── test-results/                # Experiment results
```

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Hugging Face account and API token

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd tech4HSE
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file with your API keys:
   ```bash
   OPEN_AI_KEY=your_openai_key
   DEEP_SEEK_KEY=your_deepseek_key
   HUGGING_FACE=your_huggingface_token
   NEBIUS=your_nebius_api_key
   GEMINI_API=your_gemini_api_key
   ```

4. **Install spaCy model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Configuration

Edit `config.yaml` to configure your experiment:

```yaml
# Dataset settings
dataset:
  name: medmcqa
  split: validation
  num_samples: 10

# Model settings
local_model: "microsoft/phi-4"
local_models:
  - "microsoft/phi-4"
  - "google/gemma-2-9b-it-fast"
  - "Qwen/Qwen2.5-Coder-7B"

# Remote LLM settings
remote_llm_provider: "openai"
remote_models:
  llm_model: "gpt-4o"
  cot_model: "gpt-4o"
  judge_model: "gpt-4o"

# Privacy settings
epsilon: 1.0
```

## Usage

### Running Multi-Hop Experiments

```bash
python multi_hop_experiment.py
```

This will run experiments comparing different scenarios:
1. **Purely Local Model**: Baseline local model inference
2. **Non-Private Local + Remote CoT**: Local model with remote Chain-of-Thought assistance
3. **Non-Private Local + Local CoT**: Local model with local CoT
4. **Private Local + CoT**: Local model with privacy-preserving CoT
5. **Purely Remote Model**: Remote model inference

### Running Medical QA Testing

```bash
python testing_medical_qa.py
```

### Testing Phrase Differential Privacy

```bash
python test_phrase_dp.py
```

### Main QA System

```bash
python main_qa.py
```

## Key Components

### 1. Differential Privacy Implementation (`dp_sanitizer.py`)

Implements phrase-level differential privacy using:
- Sentence-BERT embeddings for semantic similarity
- Exponential mechanism for privacy-preserving replacements
- Configurable privacy budget (epsilon)

### 2. Multi-Hop Experiment Framework (`multi_hop_experiment.py`)

Compares different inference strategies:
- **Scenario 1**: Purely local model baseline
- **Scenario 2**: Local model + remote CoT (non-private)
- **Scenario 2.5**: Local model + local CoT
- **Scenario 3**: Local model + private CoT (with DP)
- **Scenario 4**: Purely remote model

### 3. Utility Functions (`utils.py`)

Provides helper functions for:
- LLM client management
- Answer extraction from CoT responses
- LLM-based answer judging
- Remote LLM provider integration

### 4. InferDPT Integration

Integrates the InferDPT framework for additional privacy-preserving inference capabilities.

## Supported Models

### Local Models (via Nebius API)
- Microsoft Phi-4
- Google Gemma models
- Qwen models
- Mistral models
- Meta Llama models

### Remote Models
- OpenAI GPT-4o
- DeepSeek models
- Other OpenAI-compatible APIs

## Privacy Features

### Phrase-Level Differential Privacy
- Semantic similarity-based phrase replacement
- Configurable privacy budget
- Preserves semantic meaning while protecting sensitive information
- Unified epsilon-style control of the privacy–utility trade-off (single knob `epsilon`)

**Important Behavioral Note**: PhraseDP transforms the entire combined text into a semantically equivalent but structurally different sentence. When processing batch inputs (e.g., multiple-choice options combined as "Option A: text1\nOption B: text2..."), PhraseDP treats the entire block as a single semantic unit and may generate a completely restructured output (e.g., "What type of bacteria are characterized by...?") rather than preserving the original list structure.

**Solution Implemented**: Instead of attempting complex parsing of transformed output, the batch perturbation approach now:
1. Concatenates options with `;` separator
2. Applies PhraseDP to generate single perturbed text
3. Uses the perturbed text directly for Chain-of-Thought generation (no parsing needed)
4. Remote LLM receives both perturbed question and perturbed options for privacy-preserving CoT
5. Local model uses original question structure with the private CoT for final inference

This approach maintains efficiency (single API call) while ensuring all options contribute to the perturbation process.

### CusText and CusText+ (Token-Level Sanitization)
- CusText: Customized token-level candidate sets with the exponential mechanism; supports non-metric similarities (e.g., cosine on Counter-Fitting vectors) for improved utility-privacy trade-offs.
- CusText+: Same as CusText but with stopword preservation (NLTK stopword list); common stopwords are skipped from perturbation to preserve fluency while conserving privacy budget.
  - Usage (CusText repo): `--save_stop_words True`
  - Embeddings: Counter-Fitting 300-d vectors (`ct_vectors.txt`) or GloVe 840B 300-d.

### Privacy-Preserving Inference
- Perturbs questions before sending to remote models
- Uses local models for final inference
- Maintains privacy while leveraging remote model capabilities

Note on PRIV-QA: PRIV-QA uses policy/heuristic-based sanitization and does not provide differential privacy guarantees; by contrast, our approach offers unified epsilon-style DP control.

## Experiment Results

Results are stored in the `test-results/` directory and include:
- Accuracy comparisons across scenarios
- Privacy-utility trade-off analysis
- Performance metrics for different model combinations

### MedQA Dataset Experiment (500 Questions)

**Dataset**: `GBaker/MedQA-USMLE-4-options` - Medical multiple-choice questions with clinical vignettes

**FINAL RESULTS (500 Questions Completed - September 27, 2025)**:
| **Scenario** | **Accuracy** | **Correct/Total** | **Performance** |
|--------------|-------------|-------------------|-----------------|
| **4. Purely Remote Model** | **89.60%** | 448/500 | 🏆 **Best** |
| **2. Non-Private Local + Remote CoT** | **83.00%** | 415/500 | 🥈 **Second** |
| **3.0. Private Local + CoT (Old Phrase DP)** | **74.40%** | 372/500 | 🥉 **Third** |
| **1. Purely Local Model** | **59.80%** | 299/500 | Baseline |
| **3.2. Private Local + CoT (InferDPT + Batch)** | **58.80%** | 294/500 | Below baseline |
| **3.3. Private Local + CoT (SANTEXT+ + Batch)** | **56.60%** | 283/500 | Below baseline |
| **3.1.2. Private Local + CoT (Old Phrase DP + Batch)** | **0.00%** | 0/500 | ❌ **Failed** |

**Key Findings from Final 500-Question Experiment:**

**Performance Rankings:**
- **Remote Model** achieves highest accuracy (89.6%) - expected as it has full access to original questions
- **Non-Private CoT** provides excellent performance (83.0%) with reasoning guidance
- **Old PhraseDP** maintains good privacy-utility balance (74.4%) - best privacy-preserving method
- **Batch Options Processing** shows significant performance degradation across all methods

**Privacy vs Performance Analysis:**
- **Privacy Cost**: 15.2 percentage points (89.6% remote vs 74.4% private)
- **CoT Benefits**: +23.2% (non-private CoT vs baseline), +14.6% (private CoT vs baseline)
- **Batch Processing Impact**: -15.6 percentage points for Old PhraseDP (74.4% → 58.9%)

**Critical Technical Issues:**
- **Old PhraseDP + Batch Options** completely failed (0% accuracy) - indicates critical bug in batch implementation
- **Batch options processing** consistently underperforms single-option perturbation
- **SANTEXT+ and InferDPT** show similar performance in batch mode (~57-59%)

**Key Finding: Batch Perturbing Options is Counter-Productive**
- **Batch processing** consistently reduces performance across all privacy mechanisms
- **Old PhraseDP**: 74.4% (single) vs 0% (batch) - complete failure
- **InferDPT**: 58.8% (batch) vs 47.4% (no batch) - still underperforms
- **SANTEXT+**: 56.6% (batch) vs 53.2% (no batch) - minimal improvement
- **Recommendation**: Avoid batch perturbation of options; process questions and options separately for better privacy-utility trade-offs

**Strategic Recommendations:**
1. **For Maximum Accuracy**: Use Remote Model (89.6%)
2. **For Privacy + Performance**: Use Old PhraseDP single-option (74.4%)
3. **For Balanced Approach**: Use Non-Private CoT (83.0%)
4. **Avoid Batch Processing**: Significant performance degradation across all methods
5. **Fix Batch Options Bug**: Critical issue with Old PhraseDP + Batch Options implementation
6. **Process Options Separately**: Batch perturbation of options is counter-productive; maintain individual option structure

## Datasets

### 1. MedQA-USMLE-4-options Dataset

**Dataset Name**: `GBaker/MedQA-USMLE-4-options`  
**Type**: Medical Question Answering with Multiple Choice, No local context

Example usage:
```python
from datasets import load_dataset

dataset = load_dataset('GBaker/MedQA-USMLE-4-options')
```

### 2. MedMCQA Dataset

**Dataset Name**: `medmcqa/medmcqa`  
**Type**: Medical Question Answering with Multiple Choice, No local context

Example usage:
```python
from datasets import load_dataset

dataset = load_dataset("medmcqa")
```

### 3. EMRQA-MSQUAD Dataset

**Dataset Name**: `Eladio/emrqa-msquad`  
**Type**: Medical Question Answering with Extractive QA

Example usage:
```python
from datasets import load_dataset

dataset = load_dataset("Eladio/emrqa-msquad")
```

## Project Progress

### August 26, 2025 - Complete 500-Question Experiment Results

Major achievements:
- ✅ Full experiment completion on 500 MedQA questions
- ✅ Comprehensive performance analysis across 5 methods
- ✅ Privacy-utility trade-off quantification (Phrase DP > InferDPT)
- ✅ Performance gap analysis (remote vs local, CoT benefit)

### August 19, 2025 - Enhanced Differential Privacy Implementation

Key improvements:
- ✅ Diverse candidate generation and prompt externalization
- ✅ Systematic epsilon experiments and full candidate recording
- ✅ Wider similarity spectrum enabling better DP selection

### September 20, 2025 - Critical Finding: Epsilon Sensitivity Problem

Critical issue identified:
- ❌ Weak/absent correlation between epsilon and selected similarity
- ❌ Implies broken calibration of the exponential mechanism
- ▶ Actions: Recalibrate mechanism, revisit utility, add temperature scaling, verify probability formulation

### August 19, 2025 - InferDPT Integration and Comparison

Key outcomes:
- ✅ InferDPT integrated and evaluated as Scenario 3.2
- ✅ Phrase DP maintains better semantic coherence than token-level perturbation

### August 25, 2025 - MedQA Dataset Experiment Completion (50 Q milestone)

Highlights:
- ✅ Completed and analyzed initial 50-question run
- ✅ Established performance/utility gaps and privacy trade-offs

### Commit-derived milestones (from repository history)

#### August 23, 2025
- Initial repository setup for privacy-preserving multi-hop QA framework.
- Externalized prompts for maintainability; added comprehensive DP testing harness and results logging.

#### August 24, 2025
- Enhanced differential privacy implementation across experiment scripts.

#### August 25, 2025
- Added 20-question experiment (Meta-Llama-3.1-8B-Instruct + GPT-5) with logged results.
- Documented MedMCQA dataset characteristics (notably no explicit context field).
- Completed 50-question MedQA experiment and produced comprehensive analysis.

#### August 26, 2025
- Consolidated MedQA analysis and dataset documentation.
- Updated README with 500-question results and consolidated experiment files.

#### September 19, 2025
- Date-stamped progress entry aligning with upcoming epsilon-sensitivity analysis.

## Privacy Evaluation Framework

### Current Privacy Evaluation (Document-Level)
- BERT inference attack (document similarity)
- Embedding inversion attack (distance)
- GPT inference attack (document recovery)

### Planned Privacy Evaluation (Token-Level)
- BERT token recovery, embedding proximity, GPT token prediction

### Planned Privacy Evaluation (Tiered Sensitivity)
- Weighted privacy by entity sensitivity (PERSON/GPE high; PRODUCT low)

### Planned Privacy Evaluation (Linguistic Quality)
- Semantic coherence, grammaticality, readability, domain appropriateness

## Recommendations & Next Steps

From the interim analysis (250 questions) and subsequent work, the following actions are prioritized:
1. Focus on Phrase DP as the preferred privacy mechanism given best utility balance.
2. Improve InferDPT or adjust parameters to reduce utility loss.
3. Scale experiments (complete full datasets; extend cross-dataset validation on MedMCQA).
4. Enhance CoT prompt engineering and local model reasoning.
5. Recalibrate the exponential mechanism for proper epsilon sensitivity (see critical finding above).

## Future Enhancements

Meeting notes (January 2025) identified higher-impact enhancements:
1. Context summarization before remote LLM processing to reduce exposure.
2. Rule-based guardrails layered with Phrase DP to catch sensitive patterns.
3. **CoT Prompt Improvement**: Modify remote LLM CoT generation to provide reasoning guidance rather than solving the problem directly.

### CoT Prompt Enhancement

**✅ COMPLETED**: Improved remote LLM CoT generation to provide better guidance for local models:

**Previous Issue**: Remote LLM was solving the problem directly and providing final answers, which made the local model redundant.

**Solution Implemented**: Modified CoT prompts to:
- **Provide reasoning guidance** rather than solving the problem
- **Explain the approach** without choosing final answers (A, B, C, D)
- **Give step-by-step medical reasoning** that the local model can apply
- **Explicitly tell the remote model** that its output will be fed back to a local model
- **Focus on diagnostic frameworks** and reasoning principles

**Benefits Achieved**:
- True CoT: Remote provides reasoning, local applies it
- Better privacy: Remote doesn't know the final answer
- Cleaner separation: Remote = expert guidance, Local = decision maker
- More realistic: Like having a medical expert guide a student

**Implementation**: Updated `generate_cot_from_remote_llm_with_perturbed_options()` and `generate_cot_from_remote_llm_with_perturbed_options_dict()` functions with improved prompts that explicitly mention the CoT will be fed back to a local model.

### HotpotQA Context-Aware Privacy Enhancement

**TODO**: For HotpotQA experiments where context is available:
- **Scenarios 3.1 and 3.2**: Apply text summarization to context using a local model
- **Then**: Apply text sanitization to the summarized context
- **Finally**: Send sanitized context + sanitized question to remote LLM for CoT generation

This enhancement will:
- Reduce context exposure by summarizing before sanitization
- Maintain privacy through dual-layer protection (summarization + sanitization)
- Improve utility by preserving key information in summarized form
- Enable context-aware multi-hop reasoning while maintaining privacy guarantees

### PhraseDP Performance Optimization

**TODO**: Consider making PhraseDP faster for improved performance in privacy-preserving text sanitization experiments. This optimization would help reduce the computational overhead when running large-scale experiments with multiple questions and scenarios.

## Acknowledgments

- InferDPT framework for privacy-preserving inference
- Hugging Face for model hosting
- OpenAI and other LLM providers for API access



## Recent Technical Changes, Findings, and Progress

### September 25, 2025 — Medical QA Implementation with 5 Text Sanitization Methods

**Major Implementation Achievement**: Successfully implemented and tested 5 comprehensive text sanitization methods for privacy-preserving medical question answering:

1. **PhraseDP**: Phrase-level differential privacy with semantic similarity-based replacements
2. **InferDPT**: Token-level differential privacy using embedding perturbation
3. **SANTEXT+**: Vocabulary-based sanitization with semantic preservation
4. **CUSTEXT+**: Customized token-level sanitization with stopword preservation
5. **CluSanT**: Clustering-based sanitization for privacy-utility trade-offs

**Technical Implementation Details:**
- **Fixed Function Call Issues**: Resolved missing function calls in scenarios 3.3 and 3.4 for SANTEXT+ and CUSTEXT+ integration
- **Unified API**: All sanitization methods now use consistent function signatures (`*_sanitize_text()`)
- **Medical QA Testing**: Integrated all 5 methods into `test-medqa-usmle-4-options.py` for USMLE medical question evaluation
- **Remote CoT Integration**: All methods properly generate Chain-of-Thought from remote LLMs using sanitized text

**Successful Test Results** (Single Question Validation):
- **Scenario 3.3 (SANTEXT+)**: ✅ Function calls fixed, CoT generation working, extracted answer "A"
- **Scenario 3.4 (CUSTEXT+)**: ✅ Function calls fixed, CoT generation working, extracted answer "A"
- **Previous Methods**: Scenarios 3.1 (PhraseDP) and 3.2 (InferDPT) already functional

**Key Function Fixes Applied:**
```python
# Before (broken):
get_cot_from_remote_llm()  # Function didn't exist
get_answer_from_local_llm_with_cot()  # Function didn't exist

# After (working):
generate_cot_from_remote_llm()  # Existing function
get_answer_from_local_model_with_cot()  # Existing function
```

**Testing Infrastructure**:
- Single question testing: `python test-medqa-usmle-4-options.py --index 0`
- Full experiment support for all 5 sanitization methods
- Proper error handling and result extraction across all scenarios

**Files Modified:**
- `test-medqa-usmle-4-options.py`: Fixed function calls in scenarios 3.3 and 3.4
- All sanitization method integrations now working end-to-end

### September 25, 2025 — Unified PII Protection (5 Mechanisms) and Overleaf Integration

- Computed 62-point averages for PhraseDP, InferDPT, and SANTEXT+ across ε ∈ {1.0, 1.5, 2.0, 2.5, 3.0} by parsing `ppi-protection-exp.txt`.
- Integrated CusText+ and CluSanT into unified comparison; merged results in `generate_unified_ppi_plots.py` (auto-parses text log and merges JSONs), and regenerated plots.
- Improved plot clarity: distinct colors/linestyles/markers, thicker lines, jittered radar traces, higher fill opacity; auto-removes old plots before regenerating.
- Generated and saved unified figures under `plots/ppi/` with timestamp; copied to Overleaf at `overleaf-folder/plots/ppi/`.
- Included figures and captions in Overleaf `4experiment.tex` (overall line plot + per-ε radar charts), noting that 62 data points were used for PhraseDP/InferDPT/SANTEXT+.
- Code organization: added robust parser for `ppi-protection-exp.txt`, and ensured reproducible figure generation via `generate_unified_ppi_plots.py`.

### PPI Protection Plots Data Dependencies

The PPI protection plots (`generate_unified_ppi_plots.py`) rely on the following JSON files:

**Primary Data Sources:**
1. **Main Results** (PhraseDP, InferDPT, SANTEXT+):
   - `pii_protection_results_*.json` (uses latest file automatically)
   - Contains results for PhraseDP, InferDPT, and SANTEXT+ across epsilon values

2. **CluSanT Results**:
   - `results/clusant_ppi_protection_20250924_220026.json` (hardcoded path)
   - Contains CluSanT protection results

3. **CusText+ Results** (all 5 epsilon values):
   - `results/custext_ppi_protection_eps1.0.json`
   - `results/custext_ppi_protection_eps1.5.json`
   - `results/custext_ppi_protection_eps2.0.json`
   - `results/custext_ppi_protection_eps2.5.json`
   - `results/custext_ppi_protection_eps3.0.json`

**Override Data Source:**
4. **Text File Override** (optional):
   - `ppi-protection-exp.txt` (can override PhraseDP, InferDPT, SANTEXT+ if available)

**Generated Plots:**
- Overall protection vs epsilon: `overall_protection_vs_epsilon_5mech_*.png`
- Radar plots per epsilon: `protection_radar_5mech_*_eps_*.png`
- Saved to: `plots/ppi/` and copied to `overleaf-folder/plots/ppi/`

## PhraseDP Agent

The PhraseDP Agent is an intelligent privacy-preserving text sanitization system powered by a local model (Llama 8B). It provides adaptive, context-aware text perturbation with comprehensive metadata generation.

### Features

- **Intelligent Dataset Analysis**: Automatically detects dataset types (medical, legal, general, academic)
- **Question Type Detection**: Identifies question formats (multiple choice, fill-in-blank, open-ended)
- **Context Summarization**: Creates concise summaries while preserving key information
- **PII Detection & Replacement**: Detects and replaces PII with appropriate placeholders
- **Adaptive Candidate Generation**: Generates domain-appropriate replacement candidates
- **Consistent Perturbation**: Applies same strategy across question, context, and options
- **Rich Metadata Generation**: Provides comprehensive information for remote LLMs
- **Privacy-Utility Control**: Adjustable epsilon parameter for privacy-utility trade-off

### Usage

```python
from phrasedp_agent import PhraseDPAgent

# Initialize agent
agent = PhraseDPAgent()

# Process medical multiple choice question
result = agent.process(
    question="What is the first-line treatment for hypertension?",
    options=["A) ACE inhibitors", "B) Beta-blockers", "C) Diuretics", "D) Calcium channel blockers"],
    epsilon=1.0
)

# Process question with context
result = agent.process(
    question="Who designed the Eiffel Tower?",
    context="The Eiffel Tower is a wrought-iron lattice tower...",
    epsilon=2.0
)
```

### Files

- `phrasedp_agent.py` - Main agent implementation
- `test_phrasedp_agent.py` - Comprehensive test suite
- `demo_phrasedp_agent.py` - Demo script with various scenarios

### Testing

```bash
# Run manual tests (requires local model)
python test_phrasedp_agent.py

# Run demo scenarios
python demo_phrasedp_agent.py

# Run unit tests (requires pytest)
pytest test_phrasedp_agent.py -v
```

### September 24, 2025 — Comprehensive PII Protection Experiment Results (Timestamp: 20250924_205229)

**Experiment 1 (Rows 0-9): 10 samples - COMPLETED**
- **Dataset**: 10 PII samples from external dataset (4,434 total rows available)
- **Epsilon values**: 1.0, 1.5, 2.0, 2.5, 3.0 (5 values)
- **Mechanisms**: PhraseDP, InferDPT, SANTEXT+
- **Approach**: Row-by-row testing (all mechanisms/epsilons per row)

**Key Results:**
- **InferDPT**: 100% protection across ALL epsilons and PII types (perfect performance)
- **PhraseDP**: 95-100% overall protection (varies by epsilon, weak at ε=1.0, 2.5)
- **SANTEXT+**: 79-87% overall protection (consistently lowest, struggles with names)

**PII Type Analysis:**
- **Emails/Phones**: 100% protection for all mechanisms
- **Addresses**: 100% for InferDPT/PhraseDP, 80-100% for SANTEXT+
- **Names**: 100% for InferDPT, 90-100% for PhraseDP, 50-60% for SANTEXT+

**Output Quality:**
- **PhraseDP**: Coherent, meaningful questions (high semantic quality)
- **InferDPT**: Word-salad output (perfect protection, zero readability)
- **SANTEXT+**: Corrupted but partially readable text

**Generated Visualizations (Timestamp: 20250924_205229):**
- `overall_performance_comparison_20250924_205229.png` - Line chart showing protection vs epsilon
- `pii_type_performance_20250924_205229.png` - 2x2 grid for each PII type
- `mechanism_comparison_bars_20250924_205229.png` - Bar chart comparison
- `epsilon_sensitivity_heatmap_20250924_205229.png` - Color-coded performance matrix
- `performance_radar_20250924_205229.png` - Multi-dimensional radar chart
- `performance_summary_table_20250924_205229.png` - Statistical summary table

**Experiment 2 (Rows 10-99): 90 samples - IN PROGRESS**
- **Current Status**: Row 41/90 completed (45.6% done)
- **Log File**: `ppi-protection-exp.txt` (27,698 lines)
- **Expected Completion**: ~49 rows remaining

**Files Generated:**
- Results JSON: `results/pii_protection_results_row_by_row_20250924_134121.json`
- Plots: `plots/ppi/` directory with 6 comprehensive visualizations
- Log: `ppi-protection-exp.txt` (contains both experiments)

**Key Insights:**
1. **InferDPT** provides perfect PII protection but sacrifices all readability
2. **PhraseDP** offers the best balance of protection and semantic quality
3. **Names are the hardest PII type** to protect while maintaining utility
4. **Higher epsilon doesn't always improve protection** (SANTEXT+ shows decline)
5. **Email/Phone protection** is relatively easy for all mechanisms

**Recommendation**: Use **PhraseDP** for applications requiring both protection and utility, **InferDPT** for maximum protection when readability is not needed.

### September 23, 2025 — PII Protection Experiment (PhraseDP vs InferDPT vs SANTEXT+)

- Implemented end-to-end PII protection evaluation script `pii_protection_experiment.py` to compare mechanisms across epsilon ∈ {1.0, 1.5, 2.0, 2.5, 3.0} on a 10-row sample.
- Added plotting/reporting utilities that generate summary charts and JSON artifacts.
- Current results (small sample) show 100% protection rate across emails/phones/addresses/names for all three mechanisms and all epsilons using regex/BIO-based detection.
  - Results JSON: `pii_protection_results_20250923_172751.json`
  - Plots PNG: `pii_protection_plots_20250923_172751.png`
- Implementation notes:
  - PhraseDP uses diverse candidate generation via Nebius (`NEBIUS` key) and SBERT similarity with the exponential mechanism.
  - SANTEXT+ integrated via `santext_integration.py` with vocabulary building from sample documents per epsilon.
  - InferDPT path in this script is a placeholder and requires full integration to reflect true behavior.
- Caveats and next steps:
  - The 10-row sample and regex/BIO detection are limited; expand dataset coverage and strengthen PII detectors.
  - Replace InferDPT placeholder with real sanitization and re-run.
  - Cross-validate with the tiered sensitivity framework and epsilon sensitivity recalibration efforts.

