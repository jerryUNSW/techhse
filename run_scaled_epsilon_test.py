#!/usr/bin/env python3
"""
Runner script for scaled epsilon comparison test.
"""

import subprocess
import sys

print("🔬 Running Scaled Epsilon Comparison Test...")
print("=" * 80)
print("Test Parameters:")
print("- 20 questions (vs 5 previously)")
print("- Old method: 10 API calls × 10 candidates = 100 candidates")
print("- New method: 5 API calls × 20 candidates = 100 candidates")
print("- Epsilon values: 0.5, 1.0, 1.5, 2.0")
print("- Total tests: 20 questions × 4 epsilons × 2 methods = 160 tests")
print("=" * 80)

try:
    # Use sys.executable to ensure the script runs with the current Python interpreter
    subprocess.run([sys.executable, "test_epsilon_comparison_scaled.py"], check=True)
    print("\n✅ Scaled epsilon comparison test completed successfully!")
    print("📊 Check the generated plots and JSON results for analysis.")
    
except subprocess.CalledProcessError as e:
    print(f"\n❌ Error running scaled epsilon comparison test: {e}")
    sys.exit(1)

