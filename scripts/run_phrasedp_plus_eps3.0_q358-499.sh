#!/bin/bash
# Run PhraseDP+ experiment for epsilon=3.0, questions 358-499 (142 questions)
# MedQA-USMLE Dataset

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
EPSILON=3.0
START_INDEX=358
NUM_SAMPLES=142
LOCAL_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
REMOTE_MODEL="gpt-4o-mini"

# Create log directory
LOG_DIR="exp/new-exp/logs"
mkdir -p "$LOG_DIR"

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/phrasedp_plus_eps${EPSILON}_q${START_INDEX}-$((START_INDEX+NUM_SAMPLES-1))_${TIMESTAMP}.log"

echo "=========================================="
echo "PhraseDP+ Experiment (ε=${EPSILON})"
echo "Questions: ${START_INDEX}-$((START_INDEX+NUM_SAMPLES-1)) (${NUM_SAMPLES} questions)"
echo "Log file: ${LOG_FILE}"
echo "=========================================="
echo ""

# Run experiment with comprehensive logging
python test-qa-1-new.py \
    --start-index ${START_INDEX} \
    --num-samples ${NUM_SAMPLES} \
    --epsilons "${EPSILON}" \
    --local-model "${LOCAL_MODEL}" \
    --remote-model "${REMOTE_MODEL}" \
    --skip-epsilon-independent \
    --skip-phrasedp-normal \
    --skip-phrasedp-plus-fewshot \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}

if [ ${EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✅ Experiment completed successfully!"
    echo "Results saved to: exp/new-exp/"
    echo "Log saved to: ${LOG_FILE}"
else
    echo ""
    echo "❌ Experiment failed with exit code: ${EXIT_CODE}"
    echo "Check log file: ${LOG_FILE}"
    exit ${EXIT_CODE}
fi

