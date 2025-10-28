# CluSanT 100-Sample PII Protection Experiment Setup

## Summary

Modified `run_ppi_protection_experiment.py` to generate 100-sample CluSanT results and write them directly to the SQLite database.

## Changes Made

### 1. Database Integration
- Added `sqlite3` import
- Created `write_results_to_db()` function to write results directly to `tech4hse_results.db`
- Function creates/updates:
  - `experiments` table entry
  - `pii_protection_results` table (aggregated rates)
  - `pii_protection_samples` table (individual perturbations)

### 2. Updated Function Signature
```python
def run_pii_protection_experiment(
    start_idx: int = 0, 
    num_rows: int = 10, 
    resume_file: str = None,
    write_to_db: bool = True,  # NEW
    db_path: str = '/home/yizhang/tech4HSE/tech4hse_results.db'  # NEW
)
```

### 3. Command-Line Arguments
```bash
--rows         # Number of samples (default: 100)
--start        # Starting row index (default: 0)
--resume       # Resume from JSON file (optional)
--no-db        # Skip database write (optional)
--db-path      # Custom database path (optional)
```

### 4. Experiment Flow
```
1. Load PII dataset (100 samples)
2. Run CluSanT for each epsilon (1.0, 1.5, 2.0, 2.5, 3.0)
3. Save incrementally to JSON after each epsilon
4. Write final results to SQLite database
5. Generate plots
6. Send email with results
```

## Current State

### Database Status
- ✅ CluSanT data **removed** from database (old 10-sample data)
- ⏳ Ready for new 100-sample CluSanT experiment
- ✅ 4 mechanisms remain: PhraseDP, InferDPT, SANTEXT+, CusText+ (100 samples each)

### JSON Files
- ✅ `pii_protection_results_20250927_220805.json` (4 mechanisms + CluSanT eps=1.0)
- ❌ `pii_protection_results_20250929_071204.json` (removed - was 10-sample CluSanT)
- ✅ `comprehensive_ppi_protection_results_20250927_164033_backup.json` (summary backup)

## How to Run

### Full 100-Sample Experiment
```bash
cd /home/yizhang/tech4HSE/experiment_results/ppi-protection
conda run -n priv-env python run_ppi_protection_experiment.py --rows 100
```

### Test Run (10 samples)
```bash
conda run -n priv-env python run_ppi_protection_experiment.py --rows 10
```

### Resume from Existing File
```bash
conda run -n priv-env python run_ppi_protection_experiment.py --resume pii_protection_results_20250930_120000.json
```

## Expected Runtime
- **Per sample per epsilon**: ~2-5 seconds
- **100 samples × 5 epsilon values**: ~16-40 minutes total
- **With incremental saving**: Progress preserved after each epsilon

## Verification After Run

### 1. Check Database
```bash
cd /home/yizhang/tech4HSE
sqlite3 tech4hse_results.db "SELECT mechanism, epsilon, num_samples, overall_protection FROM pii_protection_results WHERE mechanism = 'CluSanT' ORDER BY epsilon;"
```

Expected:
```
CluSanT|1.0|100|<rate>
CluSanT|1.5|100|<rate>
CluSanT|2.0|100|<rate>
CluSanT|2.5|100|<rate>
CluSanT|3.0|100|<rate>
```

### 2. Check Sample Count
```bash
sqlite3 tech4hse_results.db "SELECT pr.mechanism, pr.epsilon, COUNT(ps.id) as sample_count FROM pii_protection_results pr LEFT JOIN pii_protection_samples ps ON pr.id = ps.protection_result_id WHERE pr.mechanism = 'CluSanT' GROUP BY pr.mechanism, pr.epsilon ORDER BY pr.epsilon;"
```

Expected:
```
CluSanT|1.0|100
CluSanT|1.5|100
CluSanT|2.0|100
CluSanT|2.5|100
CluSanT|3.0|100
```

### 3. Generate Final Plots
```bash
cd /home/yizhang/tech4HSE/experiment_results/ppi-protection
conda run -n priv-env python plot_ppi_protection_from_db.py
```

This will create:
- **5 radar plots** (one per epsilon): `protection_radar_5mech_final_20250929_eps_*.png/pdf`
- **4 individual PII plots**: `final_pii_protection_by_type_*.png/pdf`

All plots read directly from the database and include all 5 mechanisms.

## Files Modified

1. **`experiment_results/ppi-protection/run_ppi_protection_experiment.py`**
   - Added database integration
   - Updated default `--rows` to 100
   - Added `--no-db` and `--db-path` options
   
2. **`experiment_results/ppi-protection/plot_ppi_protection_from_db.py`**
   - Already reads from database (no changes needed)
   - Will automatically include CluSanT once data is in DB

## Files Created

- **`experiment_results/ppi-protection/RUN_CLUSANT_EXPERIMENT.md`**: User guide
- **`CLUSANT_EXPERIMENT_SETUP.md`**: This technical summary

## Next Steps

1. **Run the experiment**:
   ```bash
   cd /home/yizhang/tech4HSE/experiment_results/ppi-protection
   conda run -n priv-env python run_ppi_protection_experiment.py --rows 100
   ```

2. **Verify results** in database

3. **Generate all plots** from database

4. **Update paper figures** with complete 5-mechanism comparison

## Notes

- Script is currently configured to run **CluSanT only** (line 337: `for mechanism_name in ['CluSanT']:`)
- To run all mechanisms, change line 337 to:
  ```python
  for mechanism_name in ['PhraseDP', 'InferDPT', 'SANTEXT+', 'CusText+', 'CluSanT']:
  ```
- Database uses `INSERT OR REPLACE` so re-running won't create duplicates
- JSON files serve as backup; database is the primary data source for plotting
