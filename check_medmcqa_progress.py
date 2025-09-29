#!/usr/bin/env python3
"""
MedMCQA Progress Reporter
========================

A script to check the progress of MedMCQA experiments and send email reports.
This script reads the JSON result files and reports the current status.

Author: Tech4HSE Team
Date: 2025-01-27
"""

import os
import json
import glob
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import socket

def load_email_config():
    """Load email configuration from the config file."""
    try:
        with open('/home/yizhang/tech4HSE/email_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading email config: {e}")
        return None

def get_progress_from_files():
    """Check progress from all MedMCQA result files."""
    results_dir = "/home/yizhang/tech4HSE/QA-results/medmcqa/"
    
    if not os.path.exists(results_dir):
        return {"error": "Results directory not found"}
    
    # Find all result files
    pattern = os.path.join(results_dir, "medmcqa_results_*_100q_eps*.json")
    files = glob.glob(pattern)
    
    if not files:
        return {"error": "No result files found"}
    
    progress_data = {}
    
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract epsilon from filename
            filename = os.path.basename(file_path)
            if "eps1.0" in filename:
                epsilon = "1.0"
            elif "eps2.0" in filename:
                epsilon = "2.0"
            elif "eps3.0" in filename:
                epsilon = "3.0"
            else:
                epsilon = "unknown"
            
            # Get progress info
            total_questions = data.get("results", {}).get("total_questions", 0)
            start_time = data.get("experiment_info", {}).get("start_time", "")
            
            # Calculate runtime
            if start_time:
                try:
                    start_dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    runtime = datetime.datetime.now() - start_dt
                    runtime_str = str(runtime).split('.')[0]  # Remove microseconds
                except:
                    runtime_str = "Unknown"
            else:
                runtime_str = "Unknown"
            
            progress_data[epsilon] = {
                "total_questions": total_questions,
                "start_time": start_time,
                "runtime": runtime_str,
                "file": filename
            }
            
        except Exception as e:
            progress_data[f"error_{os.path.basename(file_path)}"] = str(e)
    
    return progress_data

def send_progress_email(progress_data):
    """Send progress report via email."""
    email_config = load_email_config()
    if not email_config:
        print("No email config available")
        return False
    
    try:
        # Get hostname
        hostname = socket.gethostname()
        
        # Create email content
        subject = f"MedMCQA Progress Report - Host: {hostname}"
        
        # Build email body
        body = f"""
MedMCQA Experiment Progress Report
==================================

Host: {hostname}
Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Progress Summary:
"""
        
        if "error" in progress_data:
            body += f"ERROR: {progress_data['error']}\n"
        else:
            for epsilon, data in progress_data.items():
                if epsilon.startswith("error_"):
                    body += f"ERROR in {epsilon}: {data}\n"
                else:
                    body += f"""
Epsilon {epsilon}:
  Questions Processed: {data['total_questions']}/100
  Runtime: {data['runtime']}
  File: {data['file']}
  Start Time: {data['start_time']}
"""
        
        # Add overall status
        if "1.0" in progress_data and "2.0" in progress_data and "3.0" in progress_data:
            total_processed = sum(data.get('total_questions', 0) for data in progress_data.values() if isinstance(data, dict) and 'total_questions' in data)
            body += f"\nOverall Progress: {total_processed}/300 questions processed across all epsilon values\n"
        
        # Create email
        msg = MIMEMultipart()
        msg['From'] = email_config['from_email']
        msg['To'] = email_config['to_email']
        msg['Subject'] = subject
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['from_email'], email_config['password'])
        text = msg.as_string()
        server.sendmail(email_config['from_email'], email_config['to_email'], text)
        server.quit()
        
        print(f"Progress email sent successfully to {email_config['to_email']}")
        return True
        
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def main():
    """Main function to check progress and send report."""
    print(f"Checking MedMCQA progress at {datetime.datetime.now()}")
    
    # Get progress data
    progress_data = get_progress_from_files()
    
    # Print to console
    print("\nProgress Summary:")
    if "error" in progress_data:
        print(f"ERROR: {progress_data['error']}")
    else:
        for epsilon, data in progress_data.items():
            if epsilon.startswith("error_"):
                print(f"ERROR in {epsilon}: {data}")
            else:
                print(f"Epsilon {epsilon}: {data['total_questions']}/100 questions processed")
    
    # Send email report
    send_progress_email(progress_data)

if __name__ == "__main__":
    main()
