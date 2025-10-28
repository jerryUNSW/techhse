#!/usr/bin/env python3
"""
HSE-bench Progress Monitor
Monitors the progress of HSE-bench experiments and sends hourly email reports.
"""

import os
import json
import datetime
import smtplib
import socket
import glob
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# --- Configuration ---
EMAIL_CONFIG_PATH = "/home/yizhang/tech4HSE/email_config.json"
RESULTS_DIR = "/home/yizhang/tech4HSE/QA-results/hse-bench/"

# --- Helper Functions ---
def load_email_config():
    """Load email configuration from a JSON file."""
    try:
        with open(EMAIL_CONFIG_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error loading email config: {EMAIL_CONFIG_PATH} not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {EMAIL_CONFIG_PATH}.")
        return None

def get_progress_from_files():
    """Check progress from all HSE-bench result files."""
    if not os.path.exists(RESULTS_DIR):
        return {"error": "Results directory not found"}, "Unknown"

    # Find all result files, prioritize recent ones
    pattern = os.path.join(RESULTS_DIR, "hse_bench_results_*.json")
    files = glob.glob(pattern)
    
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)

    if not files:
        return {"error": "No result files found"}, "Unknown"

    progress_data = {}
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Extract experiment info
            model_name = data.get('model_name', 'Unknown')
            num_samples = data.get('num_samples', 0)
            epsilon_values = data.get('epsilon_values', [])
            start_time_str = data.get('start_time')
            
            # Calculate accuracies for shared results
            shared_results = data.get('shared_results', {})
            shared_accuracies = calculate_shared_accuracies(shared_results)
            
            # Calculate accuracies for each epsilon
            epsilon_accuracies = {}
            results = data.get('results', {})
            for epsilon in epsilon_values:
                if str(epsilon) in results:
                    epsilon_accuracies[epsilon] = calculate_epsilon_accuracies(results[str(epsilon)])

            if start_time_str:
                start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
                runtime = datetime.datetime.now() - start_time
                runtime_str = str(runtime).split('.')[0]  # Remove microseconds
            else:
                runtime_str = "N/A"

            # Only show files with actual results (not empty/error files)
            if num_samples > 0 or shared_results or any(epsilon_accuracies.values()):
                progress_data[os.path.basename(file_path)] = {
                    "model_name": model_name,
                    "num_samples": num_samples,
                    "epsilon_values": epsilon_values,
                    "runtime": runtime_str,
                    "start_time": start_time_str,
                    "shared_accuracies": shared_accuracies,
                    "epsilon_accuracies": epsilon_accuracies
                }
        except Exception as e:
            progress_data[f"error_{os.path.basename(file_path)}"] = str(e)
    
    return progress_data, "HSE-bench"

def calculate_shared_accuracies(shared_results):
    """Calculate accuracy for shared mechanisms."""
    if not shared_results:
        return {}
    
    total = shared_results.get('total_questions', 0)
    if total == 0:
        return {}
    
    local_correct = shared_results.get('local_alone_correct', 0)
    cot_correct = shared_results.get('non_private_cot_correct', 0)
    remote_correct = shared_results.get('purely_remote_correct', 0)
    
    return {
        'Local Alone': f"{local_correct}/{total} = {local_correct/total*100:.3f}%",
        'Non-Private CoT': f"{cot_correct}/{total} = {cot_correct/total*100:.3f}%",
        'Purely Remote': f"{remote_correct}/{total} = {remote_correct/total*100:.3f}%"
    }

def calculate_epsilon_accuracies(epsilon_results):
    """Calculate accuracy for epsilon-specific mechanisms."""
    if not epsilon_results:
        return {}
    
    total = epsilon_results.get('total_questions', 0)
    if total == 0:
        return {}
    
    phrasedp_correct = epsilon_results.get('old_phrase_dp_local_cot_correct', 0)
    inferdpt_correct = epsilon_results.get('inferdpt_local_cot_correct', 0)
    santext_correct = epsilon_results.get('santext_local_cot_correct', 0)
    
    return {
        'PhraseDP': f"{phrasedp_correct}/{total} = {phrasedp_correct/total*100:.3f}%",
        'InferDPT': f"{inferdpt_correct}/{total} = {inferdpt_correct/total*100:.3f}%",
        'SANTEXT+': f"{santext_correct}/{total} = {santext_correct/total*100:.3f}%"
    }

def send_progress_email(progress_data, experiment_type="HSE-bench"):
    """Send progress report via email."""
    email_config = load_email_config()
    if not email_config:
        print("Email configuration not loaded. Cannot send email.")
        return

    sender_email = email_config['from_email']
    sender_password = email_config['password']
    receiver_email = email_config['to_email']
    smtp_server = email_config['smtp_server']
    smtp_port = email_config['smtp_port']

    hostname = socket.gethostname()
    subject = f"HSE-bench Progress Report - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}"

    body = f"""
HSE-bench Progress Report
========================
Host: {hostname}
Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Progress Summary:
"""

    if "error" in progress_data:
        body += f"ERROR: {progress_data['error']}\n"
    else:
        for file_name, data in progress_data.items():
            if file_name.startswith("error_"):
                body += f"ERROR in {file_name}: {data}\n"
            else:
                body += f"""
File: {file_name}
Model: {data['model_name']}
Samples: {data['num_samples']}
Epsilon Values: {data['epsilon_values']}
Runtime: {data['runtime']}
Start Time: {data['start_time']}
"""
                # Shared results
                if data['shared_accuracies']:
                    body += "  Shared Results (Epsilon-Independent):\n"
                    for mech, acc in data['shared_accuracies'].items():
                        body += f"    {mech}: {acc:.1f}%\n"
                
                # Epsilon-specific results
                if data['epsilon_accuracies']:
                    for epsilon, accuracies in data['epsilon_accuracies'].items():
                        body += f"  Epsilon {epsilon} (Privacy Mechanisms):\n"
                        for mech, acc in accuracies.items():
                            body += f"    {mech}: {acc:.1f}%\n"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Use STARTTLS for Gmail
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_string())
        print("Progress email sent successfully to", receiver_email)
    except Exception as e:
        print(f"Error sending email: {e}")

def main():
    """Main function to check progress and send report."""
    print(f"Checking HSE-bench progress at {datetime.datetime.now()}")

    # Get progress data
    progress_data, experiment_type = get_progress_from_files()

    # Print to console
    print(f"\nProgress Summary ({experiment_type} experiment):")
    if "error" in progress_data:
        print(f"ERROR: {progress_data['error']}")
    else:
        for file_name, data in progress_data.items():
            if file_name.startswith("error_"):
                print(f"ERROR in {file_name}: {data}")
            else:
                print(f"File: {file_name}")
                print(f"  Model: {data['model_name']}")
                print(f"  Samples: {data['num_samples']}")
                print(f"  Runtime: {data['runtime']}")
                # Show questions processed
                if 'shared_results' in data and data['shared_results']:
                    total_questions = data['shared_results'].get('total_questions', 0)
                    print(f"  Questions Processed: {total_questions}")
                
                # Show shared accuracies (scenarios 1, 2, 4)
                if data['shared_accuracies']:
                    print("  Shared Results (Scenarios 1, 2, 4):")
                    for mech, acc in data['shared_accuracies'].items():
                        print(f"    {mech}: {acc}")
                
                # Show epsilon-specific accuracies (privacy mechanisms)
                if data['epsilon_accuracies']:
                    for epsilon_str, accuracies in data['epsilon_accuracies'].items():
                        if accuracies:  # Only show if there are results
                            print(f"  Epsilon {epsilon_str} (Privacy Mechanisms):")
                            for mech, acc in accuracies.items():
                                print(f"    {mech}: {acc}")

    # Send email report
    send_progress_email(progress_data, experiment_type)

if __name__ == "__main__":
    main()
