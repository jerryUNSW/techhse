# PII Protection Experiment

A comprehensive experiment framework for evaluating Privacy-Preserving Information (PPI) protection mechanisms across multiple differential privacy methods.

## Overview

This experiment tests how well different privacy mechanisms protect Personally Identifiable Information (PII) in text data. It evaluates 5 state-of-the-art sanitization methods across various epsilon values and provides detailed analysis of protection rates.

## Features

### 🚀 **Optimized Performance**
- **Embedding Cache System**: Load embeddings once and reuse across all experiments
- **Incremental Saving**: Save results after each mechanism/epsilon completion
- **Resume Capability**: Continue experiments from where they left off
- **Memory Efficient**: Minimal memory footprint with smart caching

### 🔒 **Privacy Methods Tested**
1. **PhraseDP**: Phrase-level differential privacy with semantic similarity
2. **InferDPT**: Token-level DP with embedding perturbation
3. **SANTEXT+**: Vocabulary-based sanitization with global embeddings
4. **CusText+**: Customized token-level with stopword preservation
5. **CluSanT**: Clustering-based sanitization with metric DP

### 📊 **Comprehensive Evaluation**
- **PII Types**: Emails, phone numbers, addresses, names
- **Protection Metrics**: Binary protection rates (leaked vs protected)
- **Epsilon Values**: 1.0, 1.5, 2.0, 2.5, 3.0
- **Visualization**: Automated plot generation and analysis

## Requirements

```bash
# Core dependencies
pip install pandas numpy matplotlib seaborn sentence-transformers
pip install transformers sklearn tqdm openai python-dotenv nltk

# Method-specific dependencies
# - InferDPT: Custom implementation in inferdpt.py
# - SANTEXT+: Custom implementation in santext_integration.py
# - CusText+: External/CusText directory with embeddings
# - CluSanT: CluSanT directory with clustering data
```

## Dataset Format

The experiment expects a CSV file at `/home/yizhang/tech4HSE/pii_external_dataset.csv` with columns:
- `document`: Original text containing PII
- `tokens`: Tokenized text (list format)
- `trailing_whitespace`: Whitespace information (list format)
- `labels`: BIO tags for PII identification (list format)

## Usage

### Basic Usage

```bash
# Run experiment on first 10 rows
python pii_protection_experiment.py

# Custom range and size
python pii_protection_experiment.py --start 50 --rows 20

# Resume from existing results
python pii_protection_experiment.py --resume /path/to/results.json
```

### Command Line Arguments

- `--start`: Start index for dataset rows (default: 0)
- `--rows`: Number of rows to process (default: 10)
- `--resume`: Path to existing results file to resume from

### Environment Variables

Required environment variables:
```bash
# For PhraseDP candidate generation
NEBIUS_API=your_nebius_api_key
NEBIUS_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct  # optional

# Email notifications (optional)
# Configure in /data1/yizhangh/email_config.json or /home/yizhang/tech4HSE/email_config.json
```

## Architecture

### EmbeddingCache System

The `EmbeddingCache` class provides efficient embedding management:

```python
cache = EmbeddingCache()

# Load once, reuse everywhere
sbert_model = cache.get_sbert_model()
ct_emb_matrix, ct_idx2word, ct_word2idx, ct_stop_set = cache.get_custext_embeddings()
token_dict, sorted_data, delta_f = cache.get_inferdpt_embeddings()
clus_embeddings, clusant_root = cache.get_clusant_embeddings()
```

### Incremental Results

Results are saved automatically after each mechanism/epsilon completion:

```json
{
  "PhraseDP": {
    "1.0": {
      "overall": 0.85,
      "emails": 0.90,
      "phones": 0.80,
      "addresses": 0.85,
      "names": 0.90,
      "samples": [{"row": 0, "original": "...", "sanitized": "..."}]
    }
  }
}
```

### Resume Mechanism

The experiment automatically:
1. Loads existing results if `--resume` is specified
2. Skips completed mechanism/epsilon combinations
3. Continues from the next incomplete combination
4. Preserves all existing data and samples

## Output Files

### Results
- **Location**: `/home/yizhang/tech4HSE/pii_protection_results_TIMESTAMP.json`
- **Format**: Nested JSON with protection rates and sample texts
- **Content**: Per-mechanism, per-epsilon protection rates and examples

### Plots
- **Location**: `/home/yizhang/tech4HSE/pii_protection_plots_TIMESTAMP.png`
- **Content**:
  - Overall protection rate vs epsilon
  - PII type-specific protection (epsilon=2.0)
  - Protection rate heatmap
  - Mechanism comparison chart

### Email Notification
Automatic email with results summary and attachments (if configured).

## Performance Optimizations

### Memory Usage
- **Before**: ~2-4GB per mechanism (repeated loading)
- **After**: ~500MB-1GB total (cached embeddings)

### Runtime
- **Before**: 5-10 minutes initialization per mechanism
- **After**: 30 seconds one-time initialization

### Robustness
- **Incremental saving**: No work lost on crashes
- **Resume capability**: Continue multi-hour experiments seamlessly
- **Skip completed**: Avoid duplicate work

## Privacy Evaluation Methodology

### PII Detection
1. **Ground Truth**: Uses BIO tags from dataset labels
2. **Extraction**: Extracts exact PII spans from tokens + whitespace
3. **Sentence Filtering**: Processes only sentences containing PII

### Protection Measurement
1. **Normalization**: Case-insensitive, punctuation-stripped comparison
2. **Leak Detection**: Check if original PII substrings appear in sanitized text
3. **Binary Scoring**: 1 if protected, 0 if leaked
4. **Aggregation**: Average across PII types present in original text

### Metrics
- **Email Protection**: Preserves '@' during normalization
- **Phone Protection**: Standard punctuation normalization
- **Address Protection**: Standard punctuation normalization
- **Name Protection**: Standard punctuation normalization
- **Overall Protection**: Mean of protection rates for PII types present

## Troubleshooting

### Common Issues

**1. Missing Dependencies**
```bash
# Install missing packages
pip install sentence-transformers transformers sklearn
```

**2. InferDPT Initialization Errors**
```bash
# Ensure InferDPT data directory exists
mkdir -p InferDPT/data
# Check embeddings path in inferdpt.py
```

**3. CusText+ Path Issues**
```bash
# Verify embeddings path
ls /home/yizhang/tech4HSE/external/CusText/CusText/embeddings/ct_vectors.txt
```

**4. Memory Issues**
- Reduce `--rows` parameter
- Ensure sufficient RAM (8GB+ recommended)
- Close other memory-intensive applications

**5. Resume Not Working**
- Verify results file path exists
- Check JSON file is valid (not corrupted)
- Ensure write permissions to results directory

### Debug Mode

For debugging, you can modify the script to:
```python
# Enable verbose output
verbose=True  # In candidate generation

# Reduce epsilon values for testing
epsilon_values = [1.0, 2.0]  # Instead of full range

# Process fewer rows
num_rows = 2  # For quick testing
```

## Advanced Usage

### Custom Epsilon Values
Modify the epsilon list in the script:
```python
epsilon_values = [0.5, 1.0, 2.0, 4.0, 8.0]  # Custom range
```

### Method Selection
Run specific methods by modifying:
```python
mechanisms = ['PhraseDP', 'InferDPT']  # Subset only
```

### Batch Processing
For large datasets, use batch processing:
```bash
# Process dataset in chunks
for i in {0..900..100}; do
    python pii_protection_experiment.py --start $i --rows 100
    sleep 60  # Cool down between batches
done
```

## Citation

If you use this experiment framework in your research, please cite:

```bibtex
@software{pii_protection_experiment,
  title={PII Protection Experiment Framework},
  year={2025},
  note={Privacy-preserving text sanitization evaluation toolkit}
}
```

## Contact

For issues, questions, or contributions:
- Check existing issues and documentation
- Verify environment setup and dependencies
- Test with smaller datasets first (`--rows 2`)