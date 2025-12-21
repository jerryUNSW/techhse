# Missing MedQA-USMLE Experiments - Run Scripts

This directory contains scripts to run the 5 missing MedQA-USMLE experiments identified in `todo-experiment.md`.

## Overview

The following experiments need to be completed:

1. **PhraseDP (ε=1.0)**: Questions 100-499 (400 questions)
2. **PhraseDP (ε=2.0)**: Questions 100-499 (400 questions)
3. **PhraseDP (ε=3.0)**: Questions 100-499 (400 questions)
4. **PhraseDP+ (ε=1.0)**: Questions 358-499 (142 questions)
5. **PhraseDP+ (ε=3.0)**: Questions 358-499 (142 questions)

## Prerequisites

1. **Environment Setup**:
   - Python environment with all required packages installed
   - API keys configured in `.env` file:
     - `NEBIUS` or `NEBIUS_API` or `NEBIUS_API_KEY` for local model (Nebius)
     - `OPENAI_API_KEY` for remote model (GPT-4o-mini)

2. **Dependencies**:
   - Sentence-BERT model (loaded automatically)
   - MetaMap phrases (pre-extracted in dataset)
   - Database: `tech4hse_results.db` (created automatically)

## Individual Scripts

### PhraseDP Experiments

Run each PhraseDP experiment individually:

```bash
# PhraseDP (ε=1.0)
./scripts/run_phrasedp_eps1.0_q100-499.sh

# PhraseDP (ε=2.0)
./scripts/run_phrasedp_eps2.0_q100-499.sh

# PhraseDP (ε=3.0)
./scripts/run_phrasedp_eps3.0_q100-499.sh
```

**What these scripts do:**
- Run PhraseDP (normal mode, without MetaMap phrases)
- Skip epsilon-independent scenarios (Local, Local+CoT, Remote)
- Skip PhraseDP+ variants
- Process 400 questions (indices 100-499)
- Save results to `exp/new-exp/`
- Log to `exp/new-exp/logs/`

### PhraseDP+ Experiments

Run each PhraseDP+ experiment individually:

```bash
# PhraseDP+ (ε=1.0)
./scripts/run_phrasedp_plus_eps1.0_q358-499.sh

# PhraseDP+ (ε=3.0)
./scripts/run_phrasedp_plus_eps3.0_q358-499.sh
```

**What these scripts do:**
- Run PhraseDP+ (medical mode with MetaMap phrases)
- Skip epsilon-independent scenarios
- Skip PhraseDP normal mode
- Skip PhraseDP+ few-shot variant
- Process 142 questions (indices 358-499)
- Save results to `exp/new-exp/`
- Log to `exp/new-exp/logs/`

## Run All Experiments

To run all 5 experiments in sequence:

```bash
./scripts/run_all_missing_experiments.sh
```

**What this script does:**
- Runs all 5 experiments one after another
- Continues even if one experiment fails
- Creates a master log file with all results
- Provides a summary at the end

## Output Files

### Results Files
- Location: `exp/new-exp/`
- Format: JSON files with timestamp
- Naming: `medqa_usmle_*_eps{epsilon}_*_{timestamp}.json`

### Log Files
- Location: `exp/new-exp/logs/`
- Format: Text logs with timestamps
- Naming: `{mechanism}_eps{epsilon}_q{start}-{end}_{timestamp}.log`

### Database
- Location: `tech4hse_results.db`
- Tables: `medqa_results`, `medqa_detailed_results`

## Monitoring Progress

Each script:
- Prints progress to console
- Saves incremental results after each question
- Logs all operations to timestamped log files
- Shows real-time accuracy updates

## Error Handling

- Scripts use `set -e` to exit on errors
- Failed experiments are logged with error messages
- The master script continues even if individual experiments fail
- Check log files for detailed error information

## Estimated Runtime

- **PhraseDP experiments**: ~400 questions × ~5-10 seconds/question = ~30-60 minutes each
- **PhraseDP+ experiments**: ~142 questions × ~5-10 seconds/question = ~12-24 minutes each
- **Total for all 5 experiments**: ~2.5-5 hours

*Note: Runtime depends on API latency, network conditions, and system load.*

## Troubleshooting

### API Errors
- Check `.env` file for valid API keys
- Verify API quotas/limits
- Check network connectivity

### Model Loading Issues
- Ensure Sentence-BERT model can be downloaded/loaded
- Check disk space for model cache

### Database Errors
- Ensure write permissions for `tech4hse_results.db`
- Check disk space

### Script Permissions
If scripts are not executable:
```bash
chmod +x scripts/run_*.sh
```

## Configuration

Default settings in scripts:
- **Local Model**: `meta-llama/Meta-Llama-3.1-8B-Instruct` (Nebius)
- **Remote Model**: `gpt-4o-mini` (OpenAI)

To change models, edit the script variables:
```bash
LOCAL_MODEL="your-local-model"
REMOTE_MODEL="your-remote-model"
```

## Notes

- All scripts automatically create necessary directories
- Results are saved incrementally (after each question)
- Logs include full workflow details per the logging requirements
- Experiments can be interrupted and resumed (results are saved incrementally)

