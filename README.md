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

## Acknowledgments

- InferDPT framework for privacy-preserving inference
- Hugging Face for model hosting
- OpenAI and other LLM providers for API access
