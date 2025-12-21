#!/bin/bash
# Run all 5 missing MedQA-USMLE experiments in sequence
# This script runs all missing experiments one after another

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Create log directory
LOG_DIR="exp/new-exp/logs"
mkdir -p "$LOG_DIR"

# Master log file
MASTER_LOG="$LOG_DIR/all_missing_experiments_$(date +"%Y%m%d_%H%M%S").log"

echo "=========================================="
echo "Running All Missing MedQA-USMLE Experiments"
echo "Master log: ${MASTER_LOG}"
echo "=========================================="
echo ""

# Function to run experiment and log results
run_experiment() {
    local script_name=$1
    local description=$2
    
    echo "" >> "${MASTER_LOG}"
    echo "========================================" >> "${MASTER_LOG}"
    echo "$(date): Starting $description" >> "${MASTER_LOG}"
    echo "========================================" >> "${MASTER_LOG}"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $description"
    echo "Script: $script_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    
    if bash "$SCRIPT_DIR/$script_name" >> "${MASTER_LOG}" 2>&1; then
        echo "✅ $description completed successfully" | tee -a "${MASTER_LOG}"
        return 0
    else
        echo "❌ $description failed" | tee -a "${MASTER_LOG}"
        return 1
    fi
}

# Start master log
echo "All Missing Experiments Run Started: $(date)" > "${MASTER_LOG}"
echo "========================================" >> "${MASTER_LOG}"

# Track results
TOTAL_EXPERIMENTS=5
SUCCESSFUL=0
FAILED=0

# Run experiments in sequence
echo "Experiment 1/5: PhraseDP (ε=1.0), Questions 100-499"
if run_experiment "run_phrasedp_eps1.0_q100-499.sh" "PhraseDP (ε=1.0), Questions 100-499"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
    echo "⚠️  Continuing with next experiment..."
fi

echo ""
echo "Experiment 2/5: PhraseDP (ε=2.0), Questions 100-499"
if run_experiment "run_phrasedp_eps2.0_q100-499.sh" "PhraseDP (ε=2.0), Questions 100-499"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
    echo "⚠️  Continuing with next experiment..."
fi

echo ""
echo "Experiment 3/5: PhraseDP (ε=3.0), Questions 100-499"
if run_experiment "run_phrasedp_eps3.0_q100-499.sh" "PhraseDP (ε=3.0), Questions 100-499"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
    echo "⚠️  Continuing with next experiment..."
fi

echo ""
echo "Experiment 4/5: PhraseDP+ (ε=1.0), Questions 358-499"
if run_experiment "run_phrasedp_plus_eps1.0_q358-499.sh" "PhraseDP+ (ε=1.0), Questions 358-499"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
    echo "⚠️  Continuing with next experiment..."
fi

echo ""
echo "Experiment 5/5: PhraseDP+ (ε=3.0), Questions 358-499"
if run_experiment "run_phrasedp_plus_eps3.0_q358-499.sh" "PhraseDP+ (ε=3.0), Questions 358-499"; then
    ((SUCCESSFUL++))
else
    ((FAILED++))
fi

# Final summary
echo ""
echo "=========================================="
echo "All Experiments Completed"
echo "=========================================="
echo "Successful: ${SUCCESSFUL}/${TOTAL_EXPERIMENTS}"
echo "Failed: ${FAILED}/${TOTAL_EXPERIMENTS}"
echo "Master log: ${MASTER_LOG}"
echo "=========================================="

echo "" >> "${MASTER_LOG}"
echo "========================================" >> "${MASTER_LOG}"
echo "All Experiments Completed: $(date)" >> "${MASTER_LOG}"
echo "Successful: ${SUCCESSFUL}/${TOTAL_EXPERIMENTS}" >> "${MASTER_LOG}"
echo "Failed: ${FAILED}/${TOTAL_EXPERIMENTS}" >> "${MASTER_LOG}"
echo "========================================" >> "${MASTER_LOG}"

if [ ${FAILED} -eq 0 ]; then
    echo "✅ All experiments completed successfully!"
    exit 0
else
    echo "⚠️  Some experiments failed. Check individual logs for details."
    exit 1
fi

