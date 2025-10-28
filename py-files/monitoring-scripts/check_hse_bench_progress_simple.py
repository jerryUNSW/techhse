#!/usr/bin/env python3
"""
Simple HSE-bench progress monitor - only shows current experiment.
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

def get_current_experiment():
    """Get progress from current HSE-bench experiment only."""
    if not os.path.exists(RESULTS_DIR):
        return {"error": "Results directory not found"}

    # Find only the current experiment files (with -1q and recent timestamp)
    pattern = os.path.join(RESULTS_DIR, "hse_bench_results_*_-1q_eps*.json")
    files = glob.glob(pattern)
    
    if not files:
        return {"error": "No current experiment files found"}

    # Get the most recent file
    files.sort(key=os.path.getmtime, reverse=True)
    file_path = files[0]
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Extract experiment info
        model_name = data.get('model_name', 'Unknown')
        num_samples = data.get('num_samples', 0)
        epsilon_values = data.get('epsilon_values', [])
        start_time_str = data.get('start_time')
        
        # Get shared results
        shared_results = data.get('shared_results', {})
        shared_total = shared_results.get('total_questions', 0)
        
        # Get epsilon results
        results = data.get('results', {})
        
        if start_time_str:
            start_time = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S.%f")
            runtime = datetime.datetime.now() - start_time
            runtime_str = str(runtime).split('.')[0]
        else:
            runtime_str = "N/A"

        return {
            "file_name": os.path.basename(file_path),
            "model_name": model_name,
            "num_samples": num_samples,
            "epsilon_values": epsilon_values,
            "runtime": runtime_str,
            "start_time": start_time_str,
            "shared_results": shared_results,
            "epsilon_results": results,
            "shared_total": shared_total
        }
        
    except Exception as e:
        return {"error": f"Error reading experiment file: {str(e)}"}

def send_progress_email(experiment_data):
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

    if "error" in experiment_data:
        body = f"HSE-bench Progress Report\nError: {experiment_data['error']}"
    else:
        # Build report
        body = f"""HSE-bench Progress Report
===============================================
Host: {hostname}
Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Current Experiment:
  Model: {experiment_data['model_name']}
  Samples: {experiment_data['num_samples']}
  Runtime: {experiment_data['runtime']}
  Questions Processed: {experiment_data['shared_total']}

Shared Results (Scenarios 1, 2, 4):
"""
        
        # Add shared results
        shared = experiment_data['shared_results']
        if shared and shared.get('total_questions', 0) > 0:
            total = shared['total_questions']
            local_correct = shared.get('local_alone_correct', 0)
            cot_correct = shared.get('non_private_cot_correct', 0)
            remote_correct = shared.get('purely_remote_correct', 0)
            
            body += f"  Local Alone: {local_correct}/{total} = {local_correct/total*100:.3f}%\n"
            body += f"  Non-Private CoT: {cot_correct}/{total} = {cot_correct/total*100:.3f}%\n"
            body += f"  Purely Remote: {remote_correct}/{total} = {remote_correct/total*100:.3f}%\n"
        else:
            body += "  No shared results yet.\n"
        
        # Add epsilon results
        for epsilon in experiment_data['epsilon_values']:
            epsilon_str = str(epsilon)
            if epsilon_str in experiment_data['epsilon_results']:
                eps_data = experiment_data['epsilon_results'][epsilon_str]
                total_eps = eps_data.get('total_questions', 0)
                if total_eps > 0:
                    body += f"\nEpsilon {epsilon} (Privacy Mechanisms):\n"
                    
                    phrasedp_correct = eps_data.get('old_phrase_dp_local_cot_correct', 0)
                    inferdpt_correct = eps_data.get('inferdpt_local_cot_correct', 0)
                    santext_correct = eps_data.get('santext_local_cot_correct', 0)
                    
                    body += f"  PhraseDP: {phrasedp_correct}/{total_eps} = {phrasedp_correct/total_eps*100:.3f}%\n"
                    body += f"  InferDPT: {inferdpt_correct}/{total_eps} = {inferdpt_correct/total_eps*100:.3f}%\n"
                    body += f"  SANTEXT+: {santext_correct}/{total_eps} = {santext_correct/total_eps*100:.3f}%\n"

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
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

    # Get current experiment data
    experiment_data = get_current_experiment()

    # Print to console
    if "error" in experiment_data:
        print(f"ERROR: {experiment_data['error']}")
    else:
        print(f"\nCurrent HSE-bench Experiment:")
        print(f"  Model: {experiment_data['model_name']}")
        print(f"  Samples: {experiment_data['num_samples']}")
        print(f"  Runtime: {experiment_data['runtime']}")
        print(f"  Questions Processed: {experiment_data['shared_total']}")
        
        # Show shared results
        shared = experiment_data['shared_results']
        if shared and shared.get('total_questions', 0) > 0:
            total = shared['total_questions']
            local_correct = shared.get('local_alone_correct', 0)
            cot_correct = shared.get('non_private_cot_correct', 0)
            remote_correct = shared.get('purely_remote_correct', 0)
            
            print(f"\n  Shared Results (Scenarios 1, 2, 4):")
            print(f"    Local Alone: {local_correct}/{total} = {local_correct/total*100:.3f}%")
            print(f"    Non-Private CoT: {cot_correct}/{total} = {cot_correct/total*100:.3f}%")
            print(f"    Purely Remote: {remote_correct}/{total} = {remote_correct/total*100:.3f}%")
        
        # Show epsilon results
        for epsilon in experiment_data['epsilon_values']:
            epsilon_str = str(epsilon)
            if epsilon_str in experiment_data['epsilon_results']:
                eps_data = experiment_data['epsilon_results'][epsilon_str]
                total_eps = eps_data.get('total_questions', 0)
                if total_eps > 0:
                    print(f"\n  Epsilon {epsilon} (Privacy Mechanisms):")
                    
                    phrasedp_correct = eps_data.get('old_phrase_dp_local_cot_correct', 0)
                    inferdpt_correct = eps_data.get('inferdpt_local_cot_correct', 0)
                    santext_correct = eps_data.get('santext_local_cot_correct', 0)
                    
                    print(f"    PhraseDP: {phrasedp_correct}/{total_eps} = {phrasedp_correct/total_eps*100:.3f}%")
                    print(f"    InferDPT: {inferdpt_correct}/{total_eps} = {inferdpt_correct/total_eps*100:.3f}%")
                    print(f"    SANTEXT+: {santext_correct}/{total_eps} = {santext_correct/total_eps*100:.3f}%")

    # Send email report
    send_progress_email(experiment_data)

if __name__ == "__main__":
    main()



