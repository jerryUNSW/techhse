# PII Protection Experiment - Quick Reference

## 🚀 Quick Start

```bash
# Basic run (first 10 rows)
python pii_protection_experiment.py

# Custom range
python pii_protection_experiment.py --start 100 --rows 50

# Resume interrupted experiment
python pii_protection_experiment.py --resume pii_protection_results_20250927_120000.json
```

## 📁 File Structure

```
experiment_results/ppi-protection/
├── pii_protection_experiment.py    # Main experiment script
├── README.md                       # User documentation
├── ARCHITECTURE.md                 # Technical architecture
└── QUICK_REFERENCE.md             # This file

Required dependencies (parent directories):
├── inferdpt.py                    # InferDPT implementation
├── santext_integration.py         # SANTEXT+ integration
├── dp_sanitizer.py               # PhraseDP DP selection
├── utils.py                       # PhraseDP candidate generation
├── external/CusText/              # CusText+ embeddings
└── CluSanT/                      # CluSanT clustering data
```

## 🔧 Command Line Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--start` | int | 0 | Starting row index in dataset |
| `--rows` | int | 10 | Number of rows to process |
| `--resume` | str | None | Path to existing results file |

## 📊 Output Files

### Results JSON
```
/home/yizhang/tech4HSE/pii_protection_results_YYYYMMDD_HHMMSS.json
```

### Plots
```
/home/yizhang/tech4HSE/pii_protection_plots_YYYYMMDD_HHMMSS.png
```

## 🔍 Monitoring Progress

### Console Output
```
=== Mechanism: PhraseDP ===
-- Epsilon: 1.0
  [Row 0] Starting...
  [Row 0] Original (PII highlighted): John lives at 123 Main Street
  [PhraseDP][eps=1.0][row=0] Person resides at 456 Oak Avenue
  [Row 0] Protection (binary): {'emails': 1, 'phones': 1, 'addresses': 1, 'names': 1}, overall=1.0
  [PhraseDP][eps=1.0] Completed row 1/10 (total elapsed for this eps 15.2s)
```

### Key Progress Indicators
- `[Init]`: One-time embedding loading
- `[Row X]`: Per-row processing
- `Protection (binary)`: 1=protected, 0=leaked
- `total elapsed`: Time for current epsilon

## 🛠️ Troubleshooting

### Common Errors & Solutions

#### 1. Module Import Errors
```bash
# Error: ModuleNotFoundError: No module named 'sentence_transformers'
pip install sentence-transformers transformers torch

# Error: No module named 'inferdpt'
# Solution: Ensure inferdpt.py is in parent directory
```

#### 2. Missing Data Files
```bash
# Error: PII dataset not found
# Solution: Verify dataset path
ls /home/yizhang/tech4HSE/pii_external_dataset.csv

# Error: CusText+ vectors not found
# Solution: Check embedding file path
ls /home/yizhang/tech4HSE/external/CusText/CusText/embeddings/ct_vectors.txt
```

#### 3. Memory Issues
```bash
# Error: Out of memory
# Solution: Reduce batch size
python pii_protection_experiment.py --rows 5

# Or process in smaller chunks
for i in {0..100..10}; do
    python pii_protection_experiment.py --start $i --rows 10
done
```

#### 4. Resume Issues
```bash
# Error: Cannot resume, file not found
# Solution: Check file path exists
ls pii_protection_results_*.json

# Error: JSON decode error
# Solution: File may be corrupted, restart without resume
python pii_protection_experiment.py --start X --rows Y
```

## ⚡ Performance Tips

### Optimize for Speed
```bash
# Use fewer epsilon values (edit script)
epsilon_values = [1.0, 2.0]  # Instead of [1.0, 1.5, 2.0, 2.5, 3.0]

# Test subset of methods (edit script)
mechanisms = ['PhraseDP', 'InferDPT']  # Instead of all 5

# Use smaller dataset chunks
python pii_protection_experiment.py --rows 20
```

### Batch Processing Large Datasets
```bash
#!/bin/bash
# Process 1000 rows in batches of 50
for start in {0..950..50}; do
    echo "Processing rows $start to $((start+49))"
    python pii_protection_experiment.py --start $start --rows 50

    # Optional: wait between batches to cool down
    sleep 30
done
```

### Resume Long Experiments
```bash
# Start experiment
nohup python pii_protection_experiment.py --start 0 --rows 1000 > experiment.log 2>&1 &

# If interrupted, resume with latest results file
latest=$(ls -t pii_protection_results_*.json | head -1)
nohup python pii_protection_experiment.py --start 0 --rows 1000 --resume "$latest" >> experiment.log 2>&1 &
```

## 📈 Results Analysis

### Key Metrics
- **overall**: Average protection across all PII types present
- **emails**: Email protection rate (0.0 = all leaked, 1.0 = all protected)
- **phones**: Phone number protection rate
- **addresses**: Address protection rate
- **names**: Name protection rate

### Interpreting Results
```json
{
  "PhraseDP": {
    "1.0": {
      "overall": 0.85,     // 85% overall protection
      "emails": 0.90,      // 90% emails protected
      "phones": 0.80,      // 80% phones protected
      "samples": [...]     // Example original/sanitized pairs
    }
  }
}
```

### Best Practices
1. **Compare overall**: Higher overall protection is better
2. **Check samples**: Review original vs sanitized text quality
3. **Consider epsilon**: Higher epsilon = less privacy, more utility
4. **Method selection**: Choose based on protection + utility tradeoff

## 🔄 Workflow Examples

### Development/Testing
```bash
# Quick test with 2 rows
python pii_protection_experiment.py --rows 2

# Test specific range
python pii_protection_experiment.py --start 10 --rows 5

# Check if resume works
# Ctrl+C to interrupt, then:
python pii_protection_experiment.py --rows 5 --resume pii_protection_results_*.json
```

### Production Runs
```bash
# Full dataset processing
python pii_protection_experiment.py --start 0 --rows 500

# Distributed processing across multiple machines
# Machine 1:
python pii_protection_experiment.py --start 0 --rows 250

# Machine 2:
python pii_protection_experiment.py --start 250 --rows 250
```

### Analysis Pipeline
```bash
# 1. Run experiment
python pii_protection_experiment.py --start 0 --rows 100

# 2. Check results
ls -la pii_protection_results_*.json
ls -la pii_protection_plots_*.png

# 3. Review console output for any warnings
grep -i "warning\|error" experiment.log
```

## 🚨 Important Notes

### Data Requirements
- Dataset must have BIO-tagged PII labels
- Minimum 1GB RAM per 10 rows processed
- ~2GB disk space for full results and plots

### Privacy Considerations
- Original PII data appears in results JSON
- Sanitized examples are stored for analysis
- Email notifications may contain PII samples

### Performance Expectations
- **Small test (2 rows)**: 2-3 minutes
- **Medium test (20 rows)**: 15-30 minutes
- **Large test (100 rows)**: 2-4 hours
- **Full dataset (500+ rows)**: 8+ hours

### Resource Requirements
- **CPU**: Multi-core recommended for embedding operations
- **Memory**: 8GB+ recommended, 4GB minimum
- **Storage**: 1GB+ for results, embeddings, and intermediate files
- **Network**: Required for PhraseDP (Nebius API), email notifications