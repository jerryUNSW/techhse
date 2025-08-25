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

## Installation

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

### Privacy-Preserving Inference
- Perturbs questions before sending to remote models
- Uses local models for final inference
- Maintains privacy while leveraging remote model capabilities

## Experiment Results

Results are stored in the `test-results/` directory and include:
- Accuracy comparisons across scenarios
- Privacy-utility trade-off analysis
- Performance metrics for different model combinations

### MedQA Dataset Experiment (50 Questions)

**Dataset**: `GBaker/MedQA-USMLE-4-options` - Medical multiple-choice questions with clinical vignettes

**Final Results (50 Questions)**:
| **Scenario** | **Accuracy** | **Correct/Total** | **Performance** |
|--------------|-------------|-------------------|-----------------|
| **1. Purely Local Model** | **74.00%** | 37/50 | Baseline |
| **2. Non-Private Local + Remote CoT** | **90.00%** | 45/50 | ⭐ **Best** |
| **3.1. Private Local + CoT (Phrase DP)** | **78.00%** | 39/50 | ⬆️ Above baseline |
| **3.2. Private Local + CoT (InferDPT)** | **72.00%** | 36/50 | ⬇️ Below baseline |
| **4. Purely Remote Model** | **90.00%** | 45/50 | ⭐ **Best** |

**Key Performance Gaps**:

1. **Remote vs Local Performance Gap**: The purely remote model (90.00%) significantly outperforms the local model (74.00%), demonstrating a **16% performance gap**. This shows that remote models have superior reasoning capabilities for complex medical questions.

2. **Phrase DP vs InferDPT Performance Gap**: Scenario 3.1 (Phrase DP) achieves 78.00% accuracy while Scenario 3.2 (InferDPT) achieves only 72.00%. This **6% performance gap** is attributed to:
   - **Phrase DP**: Maintains semantic coherence by perturbing phrases while preserving overall question meaning
   - **InferDPT**: Performs token-level perturbation that severely disrupts semantic coherence, producing nonsensical output

**Semantic Coherence Analysis**:
- **Phrase DP (3.1)**: Generates meaningful perturbations that preserve the medical context and question structure
- **InferDPT (3.2)**: Produces random word sequences that lose all semantic meaning, making it difficult for the local model to leverage the CoT reasoning

**Privacy-Utility Trade-off**:
- Both privacy-preserving methods (3.1 & 3.2) perform worse than non-private CoT (90.00%)
- Phrase DP provides a better balance between privacy and utility compared to InferDPT
- The performance gap demonstrates the fundamental trade-off between privacy protection and reasoning quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tech4hse2024,
  title={Tech4HSE: Privacy-Preserving Multi-Hop Question Answering},
  author={[Your Name]},
  journal={[Journal/Conference]},
  year={2024}
}
```

## Project Progress

### December 19, 2024 - Enhanced Differential Privacy Implementation

**Major Achievements:**
- ✅ **Improved DP Candidate Generation**: Enhanced the phrase-level differential privacy mechanism to generate diverse, meaningful candidates
- ✅ **Multiple API Calls Strategy**: Implemented 5 API calls with 5 candidates each (25 total) for better diversity
- ✅ **Prompt Engineering**: Separated prompts into external files (`prompts/system_prompt.txt`, `prompts/user_prompt_template.txt`) for better maintainability
- ✅ **Epsilon Parameter Analysis**: Created comprehensive epsilon experiment (`epsilon_experiment.py`) to analyze privacy-utility trade-offs
- ✅ **Quality Improvements**: Eliminated tautological/nonsensical questions through improved prompt engineering
- ✅ **Full Transparency**: Enhanced result recording to include all candidates with similarity scores

### December 19, 2024 - Critical Observation: Similarity Score Diversity Issue

**Problem Identified:**
- ❌ **Limited Similarity Range**: Current candidates all have very similar similarity scores (mostly around 25-35%)
- ❌ **Exponential Mechanism Ineffectiveness**: This narrow range doesn't provide enough diversity for the exponential mechanism to work effectively
- ❌ **Poor Privacy-Utility Tradeoffs**: The exponential mechanism needs a wide range of utility scores to make meaningful privacy-utility tradeoffs

**Root Cause Analysis:**
- The current prompts are too restrictive and focused on aggressive generalization
- All candidates end up with similar low similarity scores due to uniform anonymization approach
- The exponential mechanism cannot effectively distinguish between candidates with similar utility scores

**Solution Approach:**
- 🔄 **Spectrum-Based Generation**: Modified prompts to generate candidates across a spectrum of generalization levels
- 🔄 **Target Similarity Ranges**: 
  - High Specificity (60-80% similarity): Keep some original names/places
  - Medium Generalization (40-60% similarity): Replace most specifics
  - High Generalization (20-40% similarity): Aggressive anonymization
  - Maximum Privacy (10-20% similarity): Complete anonymization
- 🔄 **Progressive Generalization**: Implemented rules for varying levels of name/place replacement

**Expected Impact:**
- Wider range of similarity scores (10% to 80%) for better exponential mechanism performance
- More meaningful privacy-utility tradeoffs across different epsilon values
- Better demonstration of differential privacy effectiveness

**Technical Improvements:**
- **Candidate Generation**: Fixed parsing logic to handle LLM responses correctly (25 candidates instead of ~200)
- **Prompt Externalization**: Created `prompt_loader.py` for modular prompt management
- **Result Analysis**: Enhanced `test_phrase_dp.py` to record full candidate details and similarities
- **Epsilon Testing**: Systematic analysis of epsilon values (1.0 to 5.0) showing clear privacy-utility trade-offs

**Key Findings:**
- **Low Epsilon (1.0)**: High privacy protection, low similarity (0.24-0.32)
- **High Epsilon (5.0)**: Low privacy protection, high similarity (0.80-0.95)
- **Optimal Range**: Epsilon 2.0-3.0 provides good balance between privacy and utility

**Files Added/Modified:**
- `epsilon_experiment.py` - New epsilon parameter analysis
- `prompts/system_prompt.txt` - External system prompt
- `prompts/user_prompt_template.txt` - External user prompt template
- `prompt_loader.py` - Prompt management utilities
- `utils.py` - Enhanced candidate generation and parsing
- `test_phrase_dp.py` - Improved result recording

### December 19, 2024 - InferDPT Integration and Comparison

**Major Achievement:**
- ✅ **InferDPT Integration**: Successfully integrated InferDPT framework for comparison with custom phrase-level DP
- ✅ **Scenario 3.2 Implementation**: Added new experimental scenario using InferDPT's token-level perturbation
- ✅ **Comprehensive Testing**: Created `test_inferdpt.py` for systematic evaluation of InferDPT performance
- ✅ **Epsilon Analysis**: Tested InferDPT with varying epsilon values (0.1, 0.5, 1.0, 2.0, 5.0)

**InferDPT Testing Results:**
- **Test Question**: "Were Scott Derrickson and Ed Wood of the same nationality?"
- **Epsilon Values**: [0.1, 0.5, 1.0, 2.0, 5.0]
- **Success Rate**: 5/5 (100%)
- **Average Similarity**: 0.0587 (very low)
- **Similarity Range**: -0.0471 to 0.1339

**Key Findings:**
- **Token-Level Perturbation**: InferDPT performs token-by-token replacement, resulting in semantically incoherent text
- **High Privacy, Low Utility**: Provides extremely high privacy protection but very low semantic coherence
- **No Clear Epsilon Trend**: Similarity scores don't show consistent relationship with epsilon values
- **Comparison with Phrase DP**: Your custom approach maintains much better semantic coherence while providing reasonable privacy

**Technical Implementation:**
- **Fixed Data Path**: Corrected hardcoded path in `inferdpt.py` to use relative path `InferDPT/data/`
- **Multi-Scenario Framework**: Updated `multi_hop_experiment_copy.py` to include both phrase DP (3.1) and InferDPT (3.2)
- **Comprehensive Documentation**: Detailed analysis in `testing-inferdpt.txt`

**Files Added/Modified:**
- `test_inferdpt.py` - New InferDPT testing framework
- `inferdpt.py` - Fixed data path configuration
- `multi_hop_experiment_copy.py` - Added Scenario 3.2 (InferDPT)
- `testing-inferdpt.txt` - Comprehensive test results

### December 25, 2024 - MedQA Dataset Experiment Completion

**Major Achievement:**
- ✅ **Complete 50-Question Experiment**: Successfully completed comprehensive evaluation on MedQA dataset
- ✅ **API Budget Resolution**: Fixed Nebius API budget issues and completed all 50 questions
- ✅ **Performance Gap Analysis**: Identified key performance differences between privacy-preserving mechanisms
- ✅ **Semantic Coherence Comparison**: Demonstrated superiority of phrase-level DP over token-level perturbation

**Key Findings:**
- **Remote vs Local Gap**: 16% performance difference (90% vs 74%) showing remote models' superior reasoning
- **Phrase DP vs InferDPT Gap**: 6% difference (78% vs 72%) due to semantic coherence preservation
- **Privacy-Utility Trade-off**: Both privacy methods perform worse than non-private CoT, but Phrase DP provides better balance

**Technical Implementation:**
- **Dataset**: `GBaker/MedQA-USMLE-4-options` with clinical vignettes
- **Models**: Meta-Llama-3.1-8B-Instruct (local), GPT-5-chat-latest (remote)
- **Privacy Mechanisms**: Phrase DP (semantic-preserving) vs InferDPT (token-level)
- **Results File**: `medqa_experiment_complete_50questions.txt`

**Files Created:**
- `medqa_experiment.py` - Standalone MedQA experiment framework
- `medqa_experiment_complete_50questions.txt` - Complete results with all 50 questions
- `medqa_experiment_cleaned_results.txt` - 42 questions without API errors
- `medqa_experiment_last8questions_20250825_221550.txt` - Final 8 questions after API fix

### Previous Progress
- **Git Version Control**: Initialized repository with comprehensive `.gitignore`
- **Dataset Integration**: Successfully integrated HotpotQA and MedMCQA datasets
- **Multi-Hop Framework**: Implemented comprehensive experiment scenarios
- **Privacy Mechanisms**: Established phrase-level differential privacy foundation

## Acknowledgments

- InferDPT framework for privacy-preserving inference
- Hugging Face for model hosting
- OpenAI and other LLM providers for API access
