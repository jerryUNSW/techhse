#!/usr/bin/env python3
"""
Simple script to run the Phrase DP comparison test.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test_phrase_dp_comparison import test_phrase_dp_comparison

if __name__ == "__main__":
    print("Running Phrase DP Comparison Test...")
    print("This will compare old vs new Phrase DP implementations")
    print("and generate similarity distribution analysis.")
    print()
    
    try:
        old_results, new_results = test_phrase_dp_comparison()
        print("\n✅ Comparison test completed successfully!")
        print("\nCheck the following files for results:")
        print("- phrase_dp_comparison_results.txt (detailed results)")
        print("- similarity_distributions.png (visualization)")
        print("- phrase_dp_comparison.json (structured data)")
        
    except Exception as e:
        print(f"\n❌ Error during comparison test: {e}")
        import traceback
        traceback.print_exc()
