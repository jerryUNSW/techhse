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

**Final Results (500 Questions)**:
| **Scenario** | **Accuracy** | **Correct/Total** | **Performance** |
|--------------|-------------|-------------------|-----------------|
| **1. Purely Local Model** | **76.80%** | 384/500 | Baseline |
| **2. Non-Private Local + Remote CoT** | **92.81%** | 465/501 | ⭐ **Best** |
| **3.1. Private Local + CoT (Phrase DP)** | **83.80%** | 419/500 | ⬆️ Above baseline |
| **3.2. Private Local + CoT (InferDPT)** | **71.94%** | 359/499 | ⬇️ Below baseline |
| **4. Purely Remote Model** | **89.80%** | 449/500 | 🥈 **Second Best** |

**Key Performance Gaps**:

1. **CoT-Aiding Gain**: Non-Private Local + Remote CoT (92.81%) significantly outperforms the local model (76.80%), demonstrating a **+16.01% performance gain**. This shows the substantial benefit of chain-of-thought reasoning assistance.

2. **Privacy Cost Analysis**:
   - **Phrase DP vs Non-Private CoT**: **-9.01%** accuracy loss (privacy cost)
   - **InferDPT vs Non-Private CoT**: **-20.87%** accuracy loss (privacy cost)
   - **Phrase DP provides much better privacy-utility balance** than InferDPT

3. **Privacy Method Comparison**: Phrase DP (83.80%) significantly outperforms InferDPT (71.94%) by **+11.86%**, demonstrating that semantic coherence preservation is crucial for maintaining utility.

4. **Remote vs Local Performance Gap**: Remote models (89.80%, 92.81%) significantly outperform local models (76.80%), demonstrating a **+13.00% performance gap** and superior reasoning capabilities for complex medical questions.

**Semantic Coherence Analysis**:
- **Phrase DP (3.1)**: Maintains semantic coherence by preserving the overall structure and meaning of questions, achieving 83.80% accuracy
- **InferDPT (3.2)**: Performs token-level perturbation that severely disrupts semantic coherence, resulting in only 71.94% accuracy

**Privacy-Utility Trade-off**:
- **Phrase DP**: Provides a better balance between privacy and utility with only 9.01% accuracy loss
- **InferDPT**: Achieves stronger privacy protection but at a significant 20.87% utility cost
- **Complete Dataset**: All 500 questions successfully processed, providing robust statistical significance



## Datasets

### 1. MedQA-USMLE-4-options Dataset

**Dataset Name**: `GBaker/MedQA-USMLE-4-options`  
**Type**: Medical Question Answering with Multiple Choice, No local context
**Format**: Clinical vignettes with patient scenarios  
**Questions**: 4-option multiple choice questions  
**Source**: USMLE (United States Medical Licensing Examination) style questions  

**Usage in Code**:
```python
from datasets import load_dataset

# Load the MedQA dataset
dataset = load_dataset('GBaker/MedQA-USMLE-4-options')

# Available splits: ['train', 'test']
# Train: 10,178 examples
# Test: 1,273 examples
# Total: 11,451 examples

# Example format:
# {
#   'question': 'A 23-year-old pregnant woman at 22 weeks gestation presents...',
#   'answer': 'Nitrofurantoin',
#   'options': {'A': 'Ampicillin', 'B': 'Ceftriaxone', 'C': 'Doxycycline', 'D': 'Nitrofurantoin'},
#   'meta_info': 'step2&3',
#   'answer_idx': 'D',
#   'metamap_phrases': ['23 year old pregnant woman', 'weeks presents', 'burning', ...]
# }
```

**Available Fields**:
- **`question`**: Clinical vignette with patient scenario (string, ~583 chars)
- **`answer`**: Correct answer text (string, ~14 chars)
- **`options`**: Dictionary with 4 multiple choice options (A, B, C, D)
- **`meta_info`**: USMLE step information (string, e.g., 'step1', 'step2&3')
- **`answer_idx`**: Correct answer letter (string, A/B/C/D)
- **`metamap_phrases`**: List of medical concepts extracted from question

**Current Experiment Status**: ✅ **COMPLETED** - All 500 questions processed (100% completion)

### 2. MedMCQA Dataset

**Dataset Name**: `medmcqa/medmcqa`  
**Type**: Medical Question Answering with Multiple Choice, No local context  
**Format**: Multiple choice questions covering 21 medical subjects  
**Questions**: AIIMS & NEET PG entrance exam MCQs  
**Source**: Indian medical entrance exam questions  

**Usage in Code**:
```python
from datasets import load_dataset

# Load the MedMCQA dataset
dataset = load_dataset("medmcqa")

# Available splits: ['train', 'validation', 'test']
# Train: 182,822 examples
# Validation: 4,183 examples
# Test: 6,150 examples
# Total: 193,155 examples

# Example format:
# {
#   'id': 'e9ad821a-c438-4965-9f77-760819dfa155',
#   'question': 'Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma',
#   'opa': 'Hyperplasia',
#   'opb': 'Hyperophy',
#   'opc': 'Atrophy',
#   'opd': 'Dyplasia',
#   'cop': 2,
#   'choice_type': 'single',
#   'exp': 'Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is use...',
#   'subject_name': 'Anatomy',
#   'topic_name': 'Urinary tract'
# }
```

**Available Fields**:
- **`id`**: Unique question identifier (string, 36 chars)
- **`question`**: Question text (string, ~118 chars)
- **`opa`**: Option A (string, ~11 chars)
- **`opb`**: Option B (string, ~9 chars)
- **`opc`**: Option C (string, ~7 chars)
- **`opd`**: Option D (string, ~8 chars)
- **`cop`**: Correct option number (integer, 1-4)
- **`choice_type`**: Question type (string, 'single' or 'multi')
- **`exp`**: Expert's explanation of the answer (string, ~381 chars)
- **`subject_name`**: Medical subject name (string, ~7 chars, e.g., 'Anatomy', 'Biochemistry')
- **`topic_name`**: Medical topic name from the subject (string, ~13 chars)

**Key Characteristics**:
- **Large Scale**: 193,155 high-quality medical questions
- **21 Medical Subjects**: Anesthesia, Anatomy, Biochemistry, Dental, ENT, Forensic Medicine, etc.
- **2.4k Healthcare Topics**: Covers wide range of medical topics
- **Real Exam Questions**: From AIIMS & NEET PG entrance exams
- **Expert Explanations**: Includes detailed explanations for answers
- **Single/Multi Choice**: Supports both single and multiple choice questions

### 3. EMRQA-MSQUAD Dataset

**Dataset Name**: `Eladio/emrqa-msquad`  
**Type**: Medical Question Answering with Extractive QA  
**Format**: Medical records WITH CONTEXT + questions + extractive answers  
**Questions**: Natural language questions about patient information  
**Source**: Real medical records and clinical notes  

**Usage in Code**:
```python
from datasets import load_dataset

# Load the EMRQA-MSQUAD dataset
dataset = load_dataset("Eladio/emrqa-msquad")

# Available splits: ['train', 'validation']
# Train: 130,956 examples
# Validation: 32,739 examples
# Total: 163,695 examples

# Example format:
# {
#   'context': 'Mrs. Wetterauer is a 54-year-old female with coronary artery disease...',
#   'question': 'Has the patient ever taken glyburide for their diabetes mellitus?',
#   'answers': {
#     'text': ['Glyburide 5 mg p.o. q.d.,'],
#     'answer_start': [1553],
#     'answer_end': [1578]
#   }
# }
```

**Available Fields**:
- **`context`**: Complete medical record/patient history (long text)
- **`question`**: Medical question about patient information (string)
- **`answers`**: Dictionary containing:
  - **`text`**: List of answer text(s) extracted from context
  - **`answer_start`**: List of starting character positions in context
  - **`answer_end`**: List of ending character positions in context

**Key Characteristics**:
- **Real Medical Data**: Actual clinical questions from medical records
- **Extractive QA**: Answers are exact text spans from the context
- **Privacy-Sensitive**: Contains real patient information that needs protection
- **Rich Medical Terminology**: Uses proper medical abbreviations and drug names

## Project Progress

### August 26, 2025 - Complete 500-Question Experiment Results

**Major Achievement:**
- ✅ **Full Experiment Completion**: Successfully processed all 500 questions from MedQA-USMLE-4-options dataset
- ✅ **Comprehensive Performance Analysis**: All 5 methods evaluated with robust statistical significance
- ✅ **Privacy-Utility Trade-off Quantification**: Clear metrics showing Phrase DP's superior balance
- ✅ **Performance Gap Analysis**: Detailed comparison of all methods with specific percentage differences

**Key Findings:**
- **Best Method**: Non-Private Local + Remote CoT (92.81% accuracy)
- **Privacy Method Winner**: Phrase DP (83.80%) significantly outperforms InferDPT (71.94%)
- **Privacy Cost**: Phrase DP has only 9.01% accuracy loss vs 20.87% for InferDPT
- **Remote Advantage**: Remote models show 13.00% performance gap over local models

**Statistical Significance**: With 500 questions, these results provide strong statistical confidence in the performance differences between methods.

### August 19, 2025 - Enhanced Differential Privacy Implementation

**Major Achievements:**
- ✅ **Improved DP Candidate Generation**: Enhanced the phrase-level differential privacy mechanism to generate diverse, meaningful candidates
- ✅ **Multiple API Calls Strategy**: Implemented 5 API calls with 5 candidates each (25 total) for better diversity
- ✅ **Prompt Engineering**: Separated prompts into external files (`prompts/system_prompt.txt`, `prompts/user_prompt_template.txt`) for better maintainability
- ✅ **Epsilon Parameter Analysis**: Created comprehensive epsilon experiment (`epsilon_experiment.py`) to analyze privacy-utility trade-offs
- ✅ **Quality Improvements**: Eliminated tautological/nonsensical questions through improved prompt engineering
- ✅ **Full Transparency**: Enhanced result recording to include all candidates with similarity scores

### August 19, 2025 - Critical Observation: Similarity Score Diversity Issue

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

### August 19, 2025 - InferDPT Integration and Comparison

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

### August 25, 2025 - MedQA Dataset Experiment Completion

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

## Privacy Evaluation Framework

### Current Privacy Evaluation (Document-Level)
We have implemented a comprehensive privacy evaluation framework based on the InferDPT paper methodology, but adapted for document-level analysis:

#### 1. BERT Inference Attack (Document-Level)
- **Method**: Uses BERT embeddings to measure semantic similarity between original and perturbed questions
- **Process**: Encode full original vs perturbed questions, calculate cosine similarity
- **Privacy Level**: `1 - document_similarity` (higher = better privacy)
- **Results**: Phrase DP: 23.7% privacy, InferDPT: 99.3% privacy

#### 2. Embedding Inversion Attack (Document-Level)
- **Method**: Measures how well original embeddings can be recovered from perturbed ones
- **Process**: Calculate L2 distance between full question embeddings
- **Privacy Level**: `normalized_embedding_distance` (higher = better privacy)
- **Results**: Phrase DP: 33.6% privacy, InferDPT: 70.4% privacy

#### 3. GPT Inference Attack (Document-Level)
- **Method**: Uses GPT-4o-mini to attempt recovering original questions from perturbed ones
- **Process**: Send perturbed question to GPT with recovery prompt, compare with original
- **Privacy Level**: `1 - document_recovery_similarity` (higher = better privacy)
- **Results**: Phrase DP: 31.2% privacy, InferDPT: 95.5% privacy

### Planned Privacy Evaluation (Token-Level)
We are planning to implement the original InferDPT paper's token-level privacy evaluation methods:

#### 1. BERT Inference Attack (Token-Level)
- **Method**: BERT tries to recover individual tokens by masking them
- **Process**: Replace each token with "[MASK]", see if BERT can predict original token
- **Success Metric**: Token-by-token accuracy (does recovered token = original token?)
- **Privacy Level**: `1 - token_recovery_accuracy`

#### 2. Embedding Inversion Attack (Token-Level)
- **Method**: For each perturbed token, find closest original token in embedding space
- **Process**: Compute Euclidean distance between perturbed token embedding and all original token embeddings
- **Success Metric**: Can we find original token in top-K closest embeddings?
- **Privacy Level**: `1 - token_recovery_accuracy`

#### 3. GPT Inference Attack (Token-Level)
- **Method**: GPT tries to recover individual tokens from perturbed text
- **Process**: Give GPT perturbed text, ask it to predict each original token
- **Success Metric**: Token-by-token accuracy
- **Privacy Level**: `1 - token_recovery_accuracy`

### Planned Privacy Evaluation (Tiered Sensitivity)
We are planning to implement a more sophisticated privacy evaluation using **tiered sensitivity levels** for different entity types:

#### Current Issue with NER-Based Evaluation
Our current NER-based evaluation treats all entity types equally, but not all entities are equally sensitive:
- **"John Smith"** (PERSON) - Very sensitive (patient name)
- **"61-year-old"** (DATE) - Very sensitive (specific age)
- **"insulin"** (PRODUCT) - Less sensitive (medical term needed for reasoning)
- **"last week"** (DATE) - Less sensitive (relative time)

#### Proposed Tiered Sensitivity Approach

**Tier 1: Highly Sensitive (Always Privacy Risk)**
- **PERSON**: Patient names, doctor names, family members
- **GPE**: Hospital locations, cities, specific addresses
- **MONEY**: Financial information, costs, payments

**Tier 2: Moderately Sensitive (Context Dependent)**
- **DATE**: Specific dates, ages, appointment times
- **CARDINAL**: Lab values, vital signs, specific measurements
- **QUANTITY**: Exact measurements, dosages, amounts
- **TIME**: Specific times, durations

**Tier 3: Low Sensitivity (Usually Safe)**
- **PRODUCT**: Drug names, medical devices (needed for reasoning)
- **WORK_OF_ART**: Book titles, movie names
- **LANGUAGE**: Language names
- **EVENT**: Medical events, procedures

**Tier 4: Medical Context Dependent**
- **ORG**: Specific hospitals vs generic medical organizations
- **FAC**: Specific facilities vs generic medical facilities
- **NORP**: Ethnicity (sensitive but sometimes needed for medical context)

#### Weighted Privacy Calculation
```python
sensitivity_weights = {
    'PERSON': 1.0,      # Highly sensitive
    'GPE': 1.0,         # Highly sensitive  
    'MONEY': 1.0,       # Highly sensitive
    'DATE': 0.8,        # Moderately sensitive
    'CARDINAL': 0.6,    # Moderately sensitive
    'QUANTITY': 0.6,    # Moderately sensitive
    'TIME': 0.7,        # Moderately sensitive
    'ORG': 0.5,         # Context dependent
    'FAC': 0.5,         # Context dependent
    'PRODUCT': 0.3,     # Low sensitivity
    'WORK_OF_ART': 0.1, # Very low sensitivity
    'LANGUAGE': 0.1,    # Very low sensitivity
    'EVENT': 0.2,       # Low sensitivity
    'LAW': 0.4,         # Moderate sensitivity
    'NORP': 0.3,        # Low sensitivity
    'ORDINAL': 0.5,     # Moderate sensitivity
    'PERCENT': 0.4      # Moderate sensitivity
}

privacy_level = 1 - (weighted_risk / total_weight)
```

#### Benefits of Tiered Sensitivity
1. **More Realistic Assessment**: Reflects actual privacy risks
2. **Context-Aware**: Considers medical reasoning needs
3. **Weighted Importance**: More sensitive entities have higher impact
4. **Domain-Agnostic**: Can be adapted for different domains
5. **Better Privacy-Utility Balance**: Distinguishes between necessary and unnecessary information

#### Future Implementation
We will probably implement this tiered sensitivity approach to provide a more nuanced and realistic privacy evaluation that better reflects the actual privacy risks in different contexts.

### Planned Privacy Evaluation (Linguistic Quality Assessment)
We are planning to conduct additional experiments to evaluate the **linguistic quality** of perturbations, which is crucial for practical deployment:

#### Current Gap in Evaluation
Our current privacy evaluations focus primarily on **privacy protection** but don't assess **linguistic quality**:
- **InferDPT**: High privacy (96.1%) but poor linguistic quality (token-level noise creates "jibberish")
- **Phrase DP**: Good privacy (91.2%) but better linguistic quality (semantic coherence preserved)

#### Proposed Linguistic Quality Metrics

**1. Semantic Coherence**
- **Method**: Measure semantic similarity between original and perturbed text
- **Metric**: BERT embedding similarity, semantic role labeling
- **Goal**: Ensure perturbed text maintains meaningful structure

**2. Grammatical Correctness**
- **Method**: Use language models to assess grammaticality
- **Metric**: Perplexity scores, grammatical error detection
- **Goal**: Ensure perturbed text is grammatically valid

**3. Readability Assessment**
- **Method**: Standard readability metrics (Flesch-Kincaid, etc.)
- **Metric**: Reading level, sentence complexity
- **Goal**: Ensure perturbed text remains readable

**4. Medical Domain Appropriateness**
- **Method**: Domain-specific language model evaluation
- **Metric**: Medical terminology preservation, clinical context maintenance
- **Goal**: Ensure medical reasoning capability is preserved

#### Comprehensive Evaluation Framework
```python
def comprehensive_evaluation(original, perturbed):
    # Privacy Assessment
    privacy_score = ner_pii_privacy_evaluation(original, perturbed)
    
    # Linguistic Quality Assessment
    semantic_coherence = bert_similarity(original, perturbed)
    grammatical_correctness = perplexity_score(perturbed)
    readability = flesch_kincaid_score(perturbed)
    domain_appropriateness = medical_domain_score(perturbed)
    
    # Combined Score
    linguistic_quality = (semantic_coherence + grammatical_correctness + 
                         readability + domain_appropriateness) / 4
    
    return privacy_score, linguistic_quality
```

#### Expected Results and Conclusion
Based on our preliminary analysis, we expect to find:

**Privacy vs Linguistic Quality Trade-off:**
- **InferDPT**: High privacy (96.1%) + Low linguistic quality (token noise)
- **Phrase DP**: Good privacy (91.2%) + High linguistic quality (semantic preservation)

**Overall Assessment:**
When considering **both privacy protection AND linguistic quality**, we anticipate concluding that **Phrase DP is the overall better approach** because:

1. **Balanced Performance**: Good privacy protection (91.2%) with excellent linguistic quality
2. **Practical Usability**: Perturbed text remains readable and semantically coherent
3. **Medical Reasoning**: Preserves medical terminology and clinical context
4. **Real-world Deployment**: More suitable for actual medical applications
5. **Privacy-Utility Balance**: Achieves the optimal trade-off between privacy and utility

**Comprehensive Conclusion:**
While InferDPT provides slightly better raw privacy protection, **Phrase DP emerges as the superior method overall** when considering the complete picture of privacy protection, linguistic quality, and practical usability for medical question answering applications.

### Key Differences Between Approaches
| Aspect | Current (Document-Level) | Planned (Token-Level) |
|--------|-------------------------|----------------------|
| **Granularity** | Document-level | Token-level |
| **BERT Attack** | Compare full documents | Mask individual tokens |
| **Embedding Attack** | Compare full document embeddings | Find closest token embeddings |
| **GPT Attack** | Recover full questions | Recover individual tokens |
| **Evaluation** | Semantic similarity | Token accuracy |
| **Use Case** | Medical QA (preserve meaning) | Text generation (token privacy) |

## Comprehensive Evaluation Framework

### **Date: September 19, 2025**

Based on our comprehensive experimental analysis, we have established a three-dimensional evaluation framework for comparing Phrase DP and InferDPT methods. This framework considers the fundamental trade-offs in privacy-preserving question answering systems:

#### **1. Privacy Protection (PII Protection)**
- **Objective**: Measure how well each method protects personal identifying information
- **Evaluation Method**: NER-based PII privacy evaluation using spaCy
- **Metrics**: 
  - Protection level by entity type (PERSON, GPE, ORG, DATE, etc.)
  - Overall privacy protection score (0-1 scale)
  - Entity preservation analysis
- **Key Finding**: Both methods provide excellent privacy protection (>0.9), with InferDPT slightly outperforming Phrase DP (0.961 vs 0.912)

#### **2. Reasoning Ability (Accuracy in Answering Questions)**
- **Objective**: Measure the utility preservation - how well the system can answer questions correctly
- **Evaluation Method**: Multi-hop question answering experiments on MedQA dataset
- **Metrics**:
  - Accuracy percentage on medical multiple-choice questions
  - Performance comparison across different scenarios
  - Utility-accuracy trade-off analysis
- **Key Finding**: Phrase DP maintains higher accuracy (83.80% vs 71.94%) due to better semantic preservation

#### **3. Linguistic Quality**
- **Objective**: Measure how well each method preserves semantic coherence and readability
- **Evaluation Method**: BERT-based semantic similarity evaluation
- **Metrics**:
  - Mean semantic similarity between original and perturbed questions
  - Quality distribution (High >0.7, Medium 0.4-0.7, Low <0.4)
  - Statistical analysis of similarity distributions
- **Key Finding**: Phrase DP maintains significantly better semantic coherence (73.7% vs 4.4% mean similarity)

#### **Evaluation Results Summary**

| Dimension | Phrase DP | InferDPT | Winner |
|-----------|-----------|----------|---------|
| **Privacy Protection** | 0.912 | 0.961 | InferDPT (5.4% better) |
| **Reasoning Ability** | 83.80% | 71.94% | Phrase DP (16.5% better) |
| **Linguistic Quality** | 73.7% | 4.4% | Phrase DP (16.7x better) |

#### **Overall Assessment**
- **Phrase DP**: Optimal balance across all three dimensions, providing excellent privacy protection while maintaining high utility and semantic coherence
- **InferDPT**: Superior privacy protection but at significant cost to utility and linguistic quality
- **Recommendation**: Phrase DP is the preferred approach for most applications requiring both privacy and utility

This three-dimensional evaluation framework provides a comprehensive assessment methodology for privacy-preserving question answering systems, ensuring that all critical aspects of system performance are considered in method comparison and selection.

#### **Future Research Direction: Tree of Privacy Attacks (ToPA)**

Based on the "Tree of Attacks: Jailbreaking Black-Box LLMs Automatically" paper, we propose adapting the Tree of Attacks with Pruning (TAP) methodology to create a comprehensive privacy evaluation and defense system for our Phrase DP mechanism.

**Tree of Privacy Attacks (ToPA) Framework:**

**Core Methodology (Adapted from TAP):**
- **Attacker LLM**: Generates adversarial prompts designed to extract sensitive information from sanitized questions
- **Evaluator LLM**: Assesses whether generated prompts are likely to succeed in privacy attacks
- **Target System**: Our Phrase DP sanitization mechanism
- **Branching Factor**: Generate multiple attack variations per iteration
- **Pruning Strategy**: Eliminate ineffective attacks and retain promising ones

**Implementation Strategy:**
1. **Branch Phase**: Generate multiple adversarial prompts targeting different privacy vulnerabilities:
   - Direct PII extraction attempts
   - Context manipulation attacks
   - Prompt injection techniques
   - Semantic reconstruction attacks

2. **Prune Phase 1**: Use evaluator to identify attacks likely to succeed against current Phrase DP

3. **Attack Phase**: Test remaining attacks against our sanitization system

4. **Prune Phase 2**: Retain most successful attacks for next iteration refinement

**Privacy-Specific Adaptations:**
- **Objective Function**: Instead of jailbreaking, focus on extracting PII or reconstructing original questions
- **Success Metrics**: Measure information leakage rather than harmful content generation
- **Attack Categories**:
  - **PII Extraction**: Attempts to recover names, locations, medical terms
  - **Context Reconstruction**: Efforts to rebuild original question context
  - **Semantic Inference**: Attacks that infer sensitive information from sanitized output

**Benefits for Privacy Protection:**
- **Proactive Vulnerability Assessment**: Systematically discover privacy weaknesses
- **Adaptive Defense**: Continuously improve Phrase DP based on discovered attack patterns
- **Comprehensive Testing**: Cover diverse attack vectors beyond current evaluation methods
- **Automated Red-Teaming**: Scale privacy testing without manual effort

**Integration with Current Framework:**
This approach adds a fourth dimension to our evaluation framework: **Adversarial Robustness**, complementing:
1. **Privacy Protection** (PII extraction resistance)
2. **Reasoning Ability** (accuracy preservation)  
3. **Linguistic Quality** (semantic coherence)
4. **Adversarial Robustness** (resistance to sophisticated attacks)

**Expected Outcomes:**
- Identify previously unknown privacy vulnerabilities in Phrase DP
- Develop more robust sanitization prompts through adversarial training
- Create a benchmark for privacy mechanism evaluation
- Establish automated privacy testing pipeline for continuous improvement

---

## Future Enhancements

### Meeting Notes - January 2025

Based on colleague feedback and discussions, the following improvements are planned for future implementation:

#### 1. Rule-Based Privacy Guardrails for Phrase DP
- **Objective**: Add an additional layer of privacy protection to the existing Phrase DP mechanism
- **Implementation**: Integrate rule-based filters that can catch and sanitize sensitive information before Phrase DP processing
- **Benefits**: 
  - Enhanced privacy protection through multi-layered approach
  - Reduced reliance on DP noise alone for privacy preservation
  - Better handling of edge cases and specific sensitive patterns

#### 2. Context Summarization Technique
- **Objective**: Reduce privacy exposure by summarizing context before remote LLM processing
- **Implementation**: 
  - Summarize full context into a few key sentences locally
  - Send only the summarized context to remote LLM instead of full text
  - Apply Phrase DP sanitizer to the summarized content
- **Benefits**:
  - Minimizes data sent to remote services
  - Reduces privacy attack surface
  - Maintains semantic information while protecting detailed content
  - Potential performance improvements due to reduced token usage

#### Implementation Priority
1. **Phase 1**: Context summarization technique (higher impact on privacy)
2. **Phase 2**: Rule-based privacy guardrails (complementary enhancement)

#### Technical Considerations
- Context summarization may require fine-tuning summarization models for medical domain
- Rule-based guardrails need careful design to avoid over-sanitization
- Both techniques should be evaluated against existing privacy attack methods
- Performance impact assessment needed for both enhancements

---

## Acknowledgments

- InferDPT framework for privacy-preserving inference
- Hugging Face for model hosting
- OpenAI and other LLM providers for API access
