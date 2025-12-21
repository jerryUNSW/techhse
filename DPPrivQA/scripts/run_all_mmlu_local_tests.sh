#!/bin/bash
# Shell script to run Local model tests on all MMLU datasets
# Skips questions 0-4 for Professional Law (already tested)

set -e  # Exit on error

# Set Nebius API credentials
export NEBIUS_API="v1.CmQKHHN0YXRpY2tleS1lMDB0eGphaDV5c3h2MjVjcmoSIXNlcnZpY2VhY2NvdW50LWUwMGo0NjlyNHdkbjB4ZDhxeTIMCLmVjcoGEPGQz5QDOgwIuJillQcQgP6-gQJAAloDZTAw.AAAAAAAAAAHiuTiwVEUCOO2oduqPLAyAu670KcUdmwRe7vU2eKcGXgxoK8O_l_vqBkknUz2_yzguwTX8Q1iUkFdF2yRqyL0O"
export NEBIUS_BASE_URL="https://api.studio.nebius.ai/v1/"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "MMLU Local Model Testing Script"
echo "=========================================="
echo ""

# Test 1: MMLU Professional Law (skip questions 0-4)
echo "=========================================="
echo "Test 1/4: MMLU Professional Law"
echo "Starting from question index 5 (skipping 0-4)"
echo "=========================================="
python3 experiments/test_mmlu_local_only.py \
    --subset professional_law \
    --start-index 5

echo ""
echo "=========================================="
echo "Test 1/4 completed: MMLU Professional Law"
echo "=========================================="
echo ""

# Test 2: MMLU Professional Medicine
echo "=========================================="
echo "Test 2/4: MMLU Professional Medicine"
echo "Testing all 272 questions"
echo "=========================================="
python3 experiments/test_mmlu_local_only.py \
    --subset professional_medicine \
    --start-index 0

echo ""
echo "=========================================="
echo "Test 2/4 completed: MMLU Professional Medicine"
echo "=========================================="
echo ""

# Test 3: MMLU Clinical Knowledge
echo "=========================================="
echo "Test 3/4: MMLU Clinical Knowledge"
echo "Testing all 265 questions"
echo "=========================================="
python3 experiments/test_mmlu_local_only.py \
    --subset clinical_knowledge \
    --start-index 0

echo ""
echo "=========================================="
echo "Test 3/4 completed: MMLU Clinical Knowledge"
echo "=========================================="
echo ""

# Test 4: MMLU College Medicine
echo "=========================================="
echo "Test 4/4: MMLU College Medicine"
echo "Testing all 173 questions"
echo "=========================================="
python3 experiments/test_mmlu_local_only.py \
    --subset college_medicine \
    --start-index 0

echo ""
echo "=========================================="
echo "Test 4/4 completed: MMLU College Medicine"
echo "=========================================="
echo ""

echo "=========================================="
echo "ALL TESTS COMPLETED!"
echo "=========================================="
echo ""
echo "Results stored in: exp-results/results.db"
echo "Log files stored in: exp-results/logs/"
echo ""
echo "To view results, run:"
echo "  sqlite3 exp-results/results.db \"SELECT * FROM experiments ORDER BY created_at DESC;\""


