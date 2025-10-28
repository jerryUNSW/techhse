# Directory Cleanup Suggestions

## Overview
This document outlines suggestions for cleaning up and organizing the `/home/yizhang/tech4HSE` directory to improve maintainability and navigation.

## 1. Categorize and Group Files

### Create Subdirectories
- **`scripts/`** - for all the Python scripts
- **`data/`** - for datasets, logs, and other data files
- **`reports/`** - for analysis reports, summaries, and documentation
- **`plots/`** - for generated plots and visualizations
- **`config/`** - for configuration files like `config.yaml` and `email_config.json`
- **`external/`** - for third-party libraries or dependencies

## 2. Clean Up Unused/Obsolete Files

### Files to Review for Deletion
- `test.txt`
- `debug_*.py`
- Old experiment result files
- Temporary files

## 3. Organize Experiment-Related Files

### Create `experiments/` Subdirectory
Move the following files:
- `test-medqa-usmle-4-options.py`
- `monitor_epsilon2_progress.py`
- Experiment result files (e.g., `test-500-new-epsilon-2.txt`)

## 4. Improve Naming Conventions

### Rename Files to Follow Consistent Naming Scheme
- `experiment_report_epsilon2.md`
- `privacy_mechanism_comparison.py`
- `medqa_dataset_analysis.ipynb`

## 5. Separate Utility Scripts

### Move General Utility Scripts
- Move `utils.py` to a `utils/` subdirectory
- This helps keep the main directory clean and makes it easier to find and maintain these shared scripts

## 6. Organize Documentation

### Create `docs/` Subdirectory
Store all the Markdown files, reports, and other documentation:
- `README.md`
- `ARCHITECTURE.md`
- `EXPERIMENT_RESULTS.md`

## 7. Handle Temporary/Backup Files

### Create `tmp/` or `backup/` Directory
- Store any temporary or backup files
- This keeps the main directory clean and organized

## 8. Automate Cleanup

### Create a Cleanup Script
- Create a script (e.g., `cleanup.py`) that can automatically perform some of these cleanup tasks:
  - Deleting unused files
  - Moving files to appropriate subdirectories
  - Renaming files to follow conventions

## Conclusion
This structured approach will help keep the `/home/yizhang/tech4HSE` directory clean, organized, and easier to navigate. Implement these suggestions gradually to avoid disrupting ongoing work.
