# Run CluSanT PII Protection Experiment (100 samples)

## Quick Start

```bash
cd /home/yizhang/tech4HSE/experiment_results/ppi-protection
conda run -n priv-env python run_ppi_protection_experiment.py --rows 100
```

## What This Does

1. **Runs CluSanT** on 100 PII samples across 5 epsilon values (1.0, 1.5, 2.0, 2.5, 3.0)
2. **Saves results** to:
   - JSON file: `pii_protection_results_TIMESTAMP.json`
   - SQLite database: `/home/yizhang/tech4HSE/tech4hse_results.db`
   - Plot file: `pii_protection_plots_TIMESTAMP.png`
3. **Sends email** with results summary and attachments

## Command Options

```bash
# Default: 100 samples, write to database
python run_ppi_protection_experiment.py --rows 100

# Start from a different row index
python run_ppi_protection_experiment.py --start 100 --rows 100

# Resume from existing JSON file
python run_ppi_protection_experiment.py --resume pii_protection_results_20250930_120000.json

# Skip database write (JSON only)
python run_ppi_protection_experiment.py --rows 100 --no-db

# Use custom database path
python run_ppi_protection_experiment.py --rows 100 --db-path /path/to/custom.db
```

## Estimated Runtime

- **Per sample**: ~2-5 seconds (depends on text complexity)
- **100 samples × 5 epsilon values**: ~16-40 minutes total
- **Progress updates**: Printed after each row completion

## Output Files

### JSON Results
```json
{
  "CluSanT": {
    "1.0": {
      "overall": 0.095,
      "emails": 0.21,
      "phones": 0.41,
      "addresses": 0.4,
      "names": 0.06,
      "samples": [
        {"row": 0, "original": "...", "sanitized": "..."},
        ...
      ]
    },
    ...
  }
}
```

### Database Tables
- **experiments**: Experiment metadata
- **pii_protection_results**: Aggregated protection rates per mechanism/epsilon
- **pii_protection_samples**: Individual perturbation samples (original + sanitized text)

## Verification

After the experiment completes, verify the database:

```bash
cd /home/yizhang/tech4HSE
sqlite3 tech4hse_results.db "SELECT mechanism, epsilon, num_samples, overall_protection FROM pii_protection_results WHERE mechanism = 'CluSanT' ORDER BY epsilon;"
```

Expected output:
```
CluSanT|1.0|100|<protection_rate>
CluSanT|1.5|100|<protection_rate>
CluSanT|2.0|100|<protection_rate>
CluSanT|2.5|100|<protection_rate>
CluSanT|3.0|100|<protection_rate>
```

## Troubleshooting

### Embedding Errors
If you see "CluSanT embedding init failed":
- Check that `/home/yizhang/tech4HSE/CluSanT` exists
- Verify embeddings are in `/home/yizhang/tech4HSE/CluSanT/embeddings/`

### API Errors
If you see "NEBIUS API key not found":
- This is expected - CluSanT doesn't use LLM APIs
- The experiment will proceed normally

### Database Errors
If database write fails:
- Check that `/home/yizhang/tech4HSE/tech4hse_results.db` exists
- Run with `--no-db` to skip database writing
- Results will still be saved to JSON

## Next Steps

After generating CluSanT results:

1. **Generate plots** with all 5 mechanisms:
   ```bash
   cd /home/yizhang/tech4HSE/experiment_results/ppi-protection
   conda run -n priv-env python plot_ppi_protection_from_db.py
   ```

2. **View results** in database browser:
   ```bash
   cd /home/yizhang/tech4HSE
   conda run -n priv-env python db_browser.py
   ```

3. **Compare with other mechanisms** using the comprehensive plot scripts.

