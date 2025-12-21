#!/bin/bash
# Test all questions for three MMLU datasets: Professional Medicine, Clinical Knowledge, College Medicine

set -e  # Exit on error

export NEBIUS_API="v1.CmQKHHN0YXRpY2tleS1lMDB0eGphaDV5c3h2MjVjcmoSIXNlcnZpY2VhY2NvdW50LWUwMGo0NjlyNHdkbjB4ZDhxeTIMCLmVjcoGEPGQz5QDOgwIuJillQcQgP6-gQJAAloDZTAw.AAAAAAAAAAHiuTiwVEUCOO2oduqPLAyAu670KcUdmwRe7vU2eKcGXgxoK8O_l_vqBkknUz2_yzguwTX8Q1iUkFdF2yRqyL0O"
export NEBIUS_BASE_URL="https://api.studio.nebius.ai/v1/"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Testing Three MMLU Datasets with Local Model"
echo "=========================================="
echo ""

# Test 1: MMLU Professional Medicine (272 questions)
echo "=========================================="
echo "Test 1/3: MMLU Professional Medicine"
echo "Testing all 272 questions"
echo "=========================================="
python3 experiments/test_mmlu_local_only.py \
    --subset professional_medicine \
    --start-index 0 \
    --num-questions 10000

echo ""
echo "=========================================="
echo "Test 1/3 completed: MMLU Professional Medicine"
echo "=========================================="
echo ""

# Test 2: MMLU Clinical Knowledge (265 questions)
echo "=========================================="
echo "Test 2/3: MMLU Clinical Knowledge"
echo "Testing all 265 questions"
echo "=========================================="
python3 experiments/test_mmlu_local_only.py \
    --subset clinical_knowledge \
    --start-index 0 \
    --num-questions 10000

echo ""
echo "=========================================="
echo "Test 2/3 completed: MMLU Clinical Knowledge"
echo "=========================================="
echo ""

# Test 3: MMLU College Medicine (173 questions)
echo "=========================================="
echo "Test 3/3: MMLU College Medicine"
echo "Testing all 173 questions"
echo "=========================================="
python3 experiments/test_mmlu_local_only.py \
    --subset college_medicine \
    --start-index 0 \
    --num-questions 10000

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETED!"
echo "=========================================="
echo ""
echo "Results stored in: exp-results/results.db"
echo "Log files stored in: exp-results/logs/"
echo ""
echo "Database tables:"
echo "  - mmlu_professional_medicine_epsilon_independent_results"
echo "  - mmlu_clinical_knowledge_epsilon_independent_results"
echo "  - mmlu_college_medicine_epsilon_independent_results"
echo ""
echo "To view results, run:"
echo "  sqlite3 exp-results/results.db \"SELECT * FROM experiments ORDER BY created_at DESC LIMIT 3;\""


