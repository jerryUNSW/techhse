#!/bin/bash

# MedQA USMLE Comprehensive Comparison Test (Efficient Epsilon Handling)
# ====================================================================
# 
# This script runs comprehensive tests comparing:
# - Local Model (Llama 8B) - Epsilon Independent ✅
# - InferDPT - Epsilon Dependent ❌
# - SANTEXT+ - Epsilon Dependent ❌
# - PhraseDP (Normal Mode) - Epsilon Dependent ❌
# - PhraseDP+ (Medical Mode) - Epsilon Dependent ❌
# - Local + CoT (Non-private) - Epsilon Independent ✅
# - Remote Model (GPT-4o) - Epsilon Independent ✅
#
# EFFICIENCY: Epsilon-independent mechanisms run once and are reused
# Across epsilon values: 1.0, 2.0, 3.0
# On first 100 questions (indices 0-99)

echo "🚀 Starting MedQA USMLE Comprehensive Comparison Test (Efficient)"
echo "==============================================================="
echo "Testing: Local, InferDPT, SANTEXT+, PhraseDP, PhraseDP+, Local+CoT, Remote"
echo "Efficiency: Epsilon-independent mechanisms cached and reused"
echo "Epsilon values: 1.0, 2.0, 3.0"
echo "Questions: First 20 (indices 0-19)"
echo "Local Model: meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "Remote Model: gpt-4o"
echo "==============================================================="
echo ""

# Set common parameters
START_INDEX=0
NUM_SAMPLES=20
PHRASEDP_MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
ANSWER_MODEL="gpt-4o"

# Function to run test for a specific epsilon
run_epsilon_test() {
    local epsilon=$1
    echo "🔄 Running comprehensive tests for Epsilon = $epsilon"
    echo "---------------------------------------------------"
    
    # Run comprehensive comparison (all 7 mechanisms)
    echo "📊 Testing all 7 mechanisms for epsilon $epsilon..."
    echo "   ⚡ Epsilon-independent mechanisms will be cached and reused"
    echo "   🎯 Epsilon-dependent mechanisms will run for this epsilon"
    
    conda run -n priv-env python test-medqa-usmle-phrasedp-comparison.py \
        --epsilon $epsilon \
        --phrasedp-model "$PHRASEDP_MODEL" \
        --answer-model "$ANSWER_MODEL" \
        --start-index $START_INDEX \
        --num-samples $NUM_SAMPLES
    
    if [ $? -eq 0 ]; then
        echo "✅ Epsilon $epsilon comprehensive test completed successfully"
    else
        echo "❌ Epsilon $epsilon comprehensive test failed"
    fi
    echo ""
}

# Run comprehensive test for all epsilon values in a single process
echo "🎯 Starting comprehensive test for all epsilon values..."
echo "⚡ Running all epsilons in a single process for maximum efficiency"
echo ""

# Single comprehensive test across all epsilon values
echo "📊 Running comprehensive test for epsilons 1.0, 2.0, 3.0..."
conda run -n priv-env python test-medqa-usmle-phrasedp-comparison.py \
    --epsilons "1.0,2.0,3.0" \
    --phrasedp-model "$PHRASEDP_MODEL" \
    --answer-model "$ANSWER_MODEL" \
    --start-index $START_INDEX \
    --num-samples $NUM_SAMPLES

if [ $? -eq 0 ]; then
    echo "✅ Comprehensive test for all epsilon values completed successfully"
else
    echo "❌ Comprehensive test failed"
fi

echo "🏁 Comprehensive test completed!"
echo "==============================================================="
echo "📁 Check the following result files:"
echo "   - medqa_usmle_efficient_eps1.0_2.0_3.0_*_FINAL_*.json"
echo ""
echo "📊 Each file contains results for all 7 mechanisms:"
echo "   ✅ Local Model (Epsilon Independent)"
echo "   ❌ InferDPT (Epsilon Dependent)"
echo "   ❌ SANTEXT+ (Epsilon Dependent)"
echo "   ❌ PhraseDP (Normal Mode, Epsilon Dependent)"
echo "   ❌ PhraseDP+ (Medical Mode, Epsilon Dependent)"
echo "   ✅ Local + CoT (Epsilon Independent)"
echo "   ✅ Remote Model (Epsilon Independent)"
echo ""
echo "⚡ EFFICIENCY GAIN: Epsilon-independent mechanisms cached and reused"
echo "   - 3x fewer API calls for Local Model, Local + CoT, Remote Model"
echo "   - Significant time and cost savings"
echo "==============================================================="
