#!/usr/bin/env python3
"""
Monitor Experiment 2 Progress (Epsilon 2.0)
Sends hourly reports on all mechanisms being tested
"""

import os
import re
import json
import smtplib
import socket
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

def load_email_config():
    """Load email configuration from the main directory."""
    config_path = '/home/yizhang/tech4HSE/email_config.json'
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Failed to load email config: {e}")
        return None

def parse_progress_from_file(log_file):
    """Parse progress from the experiment log file."""
    if not os.path.exists(log_file):
        return None
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
    except Exception as e:
        print(f"Failed to read log file: {e}")
        return None
    
    # Initialize results dictionary
    results = {
        'total_questions': 0,
        'total_expected': 500,
        'mechanisms': {
            '1. Purely Local Model (Baseline)': 0,
            '2. Non-Private Local Model + Remote CoT': 0,
            '3.0. Private Local Model + CoT (Old Phrase DP)': 0,
            '3.2. Private Local Model + CoT (InferDPT)': 0,
            '3.3. Private Local Model + CoT (SANTEXT+)': 0,
            '3.4. Private Local Model + CoT (CUSTEXT+)': 0,
            '3.5. Private Local Model + CoT (CluSanT)': 0,
            '4. Purely Remote Model': 0
        }
    }

    # Split content into question blocks
    question_blocks = re.split(r'\[93m--- Question \d+/\d+', content)
    if len(question_blocks) > 1:
        results['total_questions'] = len(question_blocks) - 1  # Subtract 1 for the initial split

    # Define exact patterns for each mechanism's result
    patterns = {
        '1. Purely Local Model (Baseline)': (
            r'--- Scenario 1: Purely Local Model.*?'
            r'Result: (Correct|Incorrect)'
        ),
        '2. Non-Private Local Model + Remote CoT': (
            r'--- Scenario 2: Non-Private Local Model \+ Remote CoT.*?'
            r'Result: (Correct|Incorrect)'
        ),
        '3.0. Private Local Model + CoT (Old Phrase DP)': (
            r'--- Scenario 3: Private Local Model \+ CoT \(Phrase DP \(Old\)\).*?'
            r'Result: (Correct|Incorrect)'
        ),
        '3.2. Private Local Model + CoT (InferDPT)': (
            r'--- Scenario 3: Private Local Model \+ CoT \(InferDPT\).*?'
            r'Result: (Correct|Incorrect)'
        ),
        '3.3. Private Local Model + CoT (SANTEXT+)': (
            r'--- Scenario 3: Private Local Model \+ CoT \(SANTEXT\+\).*?'
            r'Result: (Correct|Incorrect)'
        ),
        '3.4. Private Local Model + CoT (CUSTEXT+)': (
            r'--- Scenario 3: Private Local Model \+ CoT \(CUSTEXT\+\).*?'
            r'Result: (Correct|Incorrect)'
        ),
        '3.5. Private Local Model + CoT (CluSanT)': (
            r'--- Scenario 3: Private Local Model \+ CoT \(CluSanT\).*?'
            r'Result: (Correct|Incorrect)'
        ),
        '4. Purely Remote Model': (
            r'--- Scenario 4: Purely Remote Model.*?'
            r'Result: (Correct|Incorrect)'
        )
    }

    # Process each question block
    for block in question_blocks[1:]:  # Skip the first split which is before first question
        for mechanism, pattern in patterns.items():
            match = re.search(pattern, block, re.DOTALL)
            if match and match.group(1) == 'Correct':
                results['mechanisms'][mechanism] += 1

    return results

def send_progress_email(config, results):
    """Send progress email with mechanism accuracies."""
    try:
        msg = MIMEMultipart()
        host = socket.gethostname()
        msg['From'] = config['from_email']
        msg['To'] = config['to_email']
        msg['Subject'] = f"Experiment Progress Report (Epsilon 3.0) | Host: {host}"
        
        # Create progress report
        total_questions = results.get('total_questions', 0)
        total_expected = results.get('total_expected', 500)
        progress_percent = (total_questions / total_expected * 100) if total_expected > 0 else 0
        
        body = f"""Experiment Progress Report (Epsilon 3.0)
Host: {host}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PROGRESS SUMMARY:
Questions Completed: {total_questions}/{total_expected} ({progress_percent:.1f}%)
Status: {'RUNNING' if total_questions < total_expected else 'COMPLETED'}

MECHANISM ACCURACIES:
"""
        
        # Add mechanism accuracies
        for mechanism, correct in results['mechanisms'].items():
            accuracy = (correct / total_questions * 100) if total_questions > 0 else 0
            performance = ""
            if accuracy >= 80:
                performance = "🏆 Excellent"
            elif accuracy >= 70:
                performance = "✅ Good"
            elif accuracy >= 60:
                performance = "✓ Decent"
            elif accuracy < 50:
                performance = "❌ Poor"
            
            body += f"{mechanism}: {correct}/{total_questions} = {accuracy:.1f}% {performance}\n"
        
        # Find best performing mechanism
        if total_questions > 0:
            best_mechanism = max(results['mechanisms'].items(), key=lambda x: x[1])
            best_accuracy = (best_mechanism[1] / total_questions * 100)
            body += f"\nBEST PERFORMER: {best_mechanism[0]} ({best_accuracy:.1f}%)\n"
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['from_email'], config['password'])
        server.sendmail(config['from_email'], config['to_email'], msg.as_string())
        server.quit()
        
        print("Progress email sent successfully")
        
        # Also print to console
        print("\nExperiment Progress Summary:")
        print(f"Questions completed: {total_questions}")
        for mechanism, correct in results['mechanisms'].items():
            accuracy = (correct / total_questions * 100) if total_questions > 0 else 0
            print(f"{mechanism}: {correct}/{total_questions} = {accuracy:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"Failed to send progress email: {e}")
        return False

def main():
    """Main monitoring function."""
    log_file = '/home/yizhang/tech4HSE/test-500-new-epsilon-2.txt'
    
    print(f"Monitoring Experiment progress...")
    print(f"Log file: {log_file}")
    
    # Load email config
    config = load_email_config()
    if not config:
        print("No email config found, skipping email report")
        return
    
    # Parse progress
    results = parse_progress_from_file(log_file)
    if not results:
        print("No progress data found")
        return
    
    # Send email report
    send_progress_email(config, results)

if __name__ == "__main__":
    main()


