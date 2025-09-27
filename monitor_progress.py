#!/usr/bin/env python3
"""
Progress monitoring script for MedQA experiment.
Sends hourly email updates with progress and accuracy.
"""

import os
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json

def get_hostname():
    """Get the machine hostname."""
    import socket
    return socket.gethostname()

def load_email_config():
    """Load email configuration from the config file."""
    try:
        with open('/home/yizhang/tech4HSE/email_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("Email config file not found. Using default settings.")
        return {
            "smtp_server": "smtp.gmail.com",
            "smtp_port": 587,
            "sender_email": "your-email@gmail.com",
            "sender_password": "your-app-password",
            "recipient_email": "your-email@gmail.com"
        }

def parse_progress_from_file(file_path):
    """Parse progress from the test-500-new.txt file."""
    if not os.path.exists(file_path):
        return None, None, None
    
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Count completed questions
        question_matches = re.findall(r'Question \d+/\d+ \(Dataset idx: \d+\)', content)
        questions_done = len(question_matches)
        
        # Extract final results if available
        final_results_match = re.search(r'FINAL RESULTS.*?(?=\n\n|\Z)', content, re.DOTALL)
        if final_results_match:
            results_text = final_results_match.group(0)
            
            # Parse accuracy results for each method
            method_accuracies = {}
            accuracy_pattern = r'([^:]+): (\d+)/(\d+) = ([\d.]+)%'
            accuracies = re.findall(accuracy_pattern, results_text)
            
            for method, correct, total, percentage in accuracies:
                method_accuracies[method.strip()] = {
                    'correct': int(correct),
                    'total': int(total),
                    'percentage': float(percentage)
                }
            
            return questions_done, method_accuracies, "COMPLETED"
        else:
            # Extract current progress from individual questions by method
            method_stats = {}
            
            # Parse results by looking for scenario headers and their corresponding results
            scenario_patterns = {
                'Baseline (No Privacy)': r'Scenario 1: Purely Local Model \(Baseline\)',
                'Non-Private CoT': r'Scenario 2: Non-Private Local Model \+ Remote CoT',
                'Old PhraseDP': r'Scenario 3: Private Local Model \+ CoT \(Phrase DP \(Old\)\)(?! with Batch Options)',
                'Old PhraseDP + Batch Options': r'Scenario 3: Private Local Model \+ CoT \(Phrase DP \(Old\) with Batch Options\)',
                'InferDPT + Batch Options': r'Scenario 3: Private Local Model \+ CoT \(InferDPT with Batch Options\)',
                'SANTEXT+ + Batch Options': r'Scenario 3: Private Local Model \+ CoT \(SANTEXT\+ with Batch Options\)',
                'Remote Model': r'Scenario 4: Purely Remote Model'
            }
            
            for method, pattern in scenario_patterns.items():
                # Find all occurrences of this scenario
                scenario_matches = list(re.finditer(pattern, content))
                correct_count = 0
                total_count = 0
                
                for match in scenario_matches:
                    # Look for the next "Result:" line after this scenario
                    start_pos = match.end()
                    remaining_content = content[start_pos:]
                    
                    # Find the next "Result:" line
                    result_match = re.search(r'Result: (Correct|Incorrect)', remaining_content)
                    if result_match:
                        total_count += 1
                        if result_match.group(1) == 'Correct':
                            correct_count += 1
                
                if total_count > 0:
                    accuracy = (correct_count / total_count) * 100
                    method_stats[method] = {
                        'correct': correct_count,
                        'total': total_count,
                        'percentage': accuracy
                    }
            
            if method_stats:
                return questions_done, method_stats, "IN_PROGRESS"
            else:
                return questions_done, "No results yet", "STARTING"
                
    except Exception as e:
        return None, f"Error parsing file: {str(e)}", "ERROR"

def send_progress_email(questions_done, accuracy_info, status):
    """Send progress email."""
    config = load_email_config()
    
    hostname = get_hostname()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create email content
    subject = f"MedQA Experiment Progress - {hostname} - {timestamp}"
    
    # Format accuracy information
    if isinstance(accuracy_info, dict):
        # Method-specific accuracies
        accuracy_text = "Method Accuracies:\n"
        best_method = None
        best_accuracy = 0
        
        for method, stats in accuracy_info.items():
            if isinstance(stats, dict) and 'percentage' in stats:
                accuracy_text += f"- {method}: {stats['correct']}/{stats['total']} = {stats['percentage']:.1f}%\n"
                if stats['percentage'] > best_accuracy:
                    best_accuracy = stats['percentage']
                    best_method = method
            else:
                accuracy_text += f"- {method}: {stats}\n"
        
        if best_method:
            accuracy_text += f"\n🏆 Best Performing Method: {best_method} ({best_accuracy:.1f}%)\n"
    else:
        # Simple accuracy string
        accuracy_text = str(accuracy_info)
    
    body = f"""
MedQA Experiment Progress Report
Host: {hostname}
Time: {timestamp}
Status: {status}

Progress Summary:
- Questions Completed: {questions_done}/500
- Progress: {(questions_done/500)*100:.1f}%

{accuracy_text}

Experiment Details:
- Testing 500 questions from MedQA dataset
- 7 active scenarios (1, 2, 3.0, 3.1.2, 3.2, 3.3, 4)
- Enhanced CoT generation with reasoning guidance
- No semicolon enforcement (simple concatenation)
- Quota error handling enabled

File Location: /home/yizhang/tech4HSE/test-500-new.txt

---
This is an automated progress report.
    """
    
    # Create message
    msg = MIMEMultipart()
    msg['From'] = config['from_email']
    msg['To'] = config['to_email']
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    # Send email
    try:
        server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
        server.starttls()
        server.login(config['from_email'], config['password'])
        text = msg.as_string()
        server.sendmail(config['from_email'], config['to_email'], text)
        server.quit()
        print(f"Progress email sent successfully at {timestamp}")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

def main():
    """Main monitoring function."""
    file_path = "/home/yizhang/tech4HSE/test-500-new.txt"
    
    questions_done, accuracy_info, status = parse_progress_from_file(file_path)
    
    if questions_done is None:
        print("Could not parse progress from file")
        return
    
    print(f"Questions done: {questions_done}")
    print(f"Status: {status}")
    print(f"Accuracy info: {accuracy_info}")
    
    # Send email
    send_progress_email(questions_done, accuracy_info, status)

if __name__ == "__main__":
    main()
