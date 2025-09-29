#!/usr/bin/env python3
"""
Monitor Experiment Progress
Monitors the running experiment and automatically processes results when complete
"""

import time
import os
import glob
import subprocess
from datetime import datetime

def check_experiment_status():
    """Check if the experiment is still running."""
    try:
        # Check for any running Python processes with pii_protection_experiment
        result = subprocess.run(['pgrep', '-f', 'pii_protection_experiment.py'],
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def find_latest_results():
    """Find the latest experiment results."""
    pattern = "pii_protection_results_*.json"
    files = glob.glob(pattern)

    if files:
        return max(files, key=os.path.getctime)
    return None

def wait_for_completion():
    """Wait for experiment to complete and then process results."""
    print("🔍 Monitoring experiment progress...")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    last_results_file = find_latest_results()
    last_results_time = os.path.getctime(last_results_file) if last_results_file else 0

    while True:
        # Check if experiment is still running
        is_running = check_experiment_status()

        # Check for new results file
        current_results_file = find_latest_results()
        current_results_time = os.path.getctime(current_results_file) if current_results_file else 0

        if not is_running and current_results_time > last_results_time:
            print("✅ Experiment completed!")
            print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return current_results_file

        # Print status update
        current_time = datetime.now().strftime('%H:%M:%S')
        status = "RUNNING" if is_running else "CHECKING"
        print(f"[{current_time}] Status: {status} | Latest results: {os.path.basename(current_results_file) if current_results_file else 'None'}")

        # Wait before next check
        time.sleep(30)  # Check every 30 seconds

def main():
    """Main monitoring function."""
    print("=== PII Protection Experiment Monitor ===")

    # Wait for experiment to complete
    results_file = wait_for_completion()

    if results_file:
        print(f"\n📊 Processing results from: {os.path.basename(results_file)}")

        # Process results and generate plots
        try:
            os.chdir('/home/yizhang/tech4HSE/experiment_results/ppi-protection')
            result = subprocess.run(['python', 'process_experiment_results.py'],
                                  capture_output=True, text=True)

            if result.returncode == 0:
                print("✅ Results processed successfully!")
                print(result.stdout)

                # Optionally send email (if configured)
                print("\n📧 Checking email configuration...")
                email_result = subprocess.run(['python', 'email_results.py'],
                                            capture_output=True, text=True)
                print(email_result.stdout)

            else:
                print("❌ Error processing results:")
                print(result.stderr)

        except Exception as e:
            print(f"❌ Error: {e}")

    print("\n🎉 Monitoring complete!")

if __name__ == "__main__":
    main()