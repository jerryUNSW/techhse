#!/bin/bash
# Run MMLU experiments with InferDPT and SANTEXT+ (epsilon=2.0)
# MedQA-USMLE Dataset

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
EPSILON=2.0
LOG_DIR="exp/mmlu-results/logs"
mkdir -p "$LOG_DIR"

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/mmlu_inferdpt_eps${EPSILON}_${TIMESTAMP}.log"

echo "=========================================="
echo "MMLU Experiments: InferDPT + QA"
echo "Epsilon: ${EPSILON}"
echo "Log file: ${LOG_FILE}"
echo "=========================================="
echo ""
echo "Datasets:"
echo "  - MMLU Professional Law: first 200 questions"
echo "  - MMLU Professional Medicine: all 272 questions"
echo "  - MMLU Clinical Knowledge: all 265 questions"
echo "  - MMLU College Medicine: all 173 questions"
echo ""

# Run experiment with comprehensive logging
python test-mmlu-inferdpt-santext.py 2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✅ Experiments completed successfully!"
    echo "Results saved to: exp/mmlu-results/"
    echo "Log saved to: ${LOG_FILE}"
else
    echo ""
    echo "❌ Experiments failed with exit code: ${EXIT_CODE}"
    echo "Check log file: ${LOG_FILE}"
    exit ${EXIT_CODE}
fi

