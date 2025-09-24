#!/usr/bin/env python3
"""
Simple runner script for epsilon comparison test.
"""

import subprocess
import sys

print("🔬 Running Epsilon Comparison Test...")
print("Testing Old vs New Phrase DP across epsilon values: 0.5, 1.0, 1.5, 2.0")
print("=" * 70)

try:
    # Use sys.executable to ensure the script runs with the current Python interpreter
    subprocess.run([sys.executable, "test_epsilon_comparison.py"], check=True)
    print("\n✅ Epsilon comparison test completed successfully!")
    print("📊 Check the generated plots and JSON results for analysis.")
    
except subprocess.CalledProcessError as e:
    print(f"\n❌ Error running epsilon comparison test: {e}")
    sys.exit(1)

