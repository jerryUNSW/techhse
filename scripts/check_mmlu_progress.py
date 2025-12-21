#!/usr/bin/env python3
"""
Check MMLU InferDPT QA experiment progress and send email report.
"""

import json
import glob
import os
import sys
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_progress_report():
    """Get current progress report."""
    files = glob.glob('exp/mmlu-results/mmlu_*_eps2.0_*.json')
    if not files:
        return "No result files found yet"
    
    # Get latest file for each dataset
    datasets = {}
    for f in files:
        if 'all_datasets' in f:
            continue
        parts = os.path.basename(f).split('_')
        if len(parts) >= 2:
            dataset = '_'.join(parts[1:-3])
            if dataset not in datasets or os.path.getmtime(f) > os.path.getmtime(datasets[dataset]):
                datasets[dataset] = f
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("InferDPT + QA Progress Report - MMLU Datasets (ε=2.0)")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    total_questions = 0
    total_correct = 0
    
    for dataset, filepath in sorted(datasets.items()):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            results = data.get('inferdpt_results', [])
            summary = data.get('summary', {}).get('inferdpt', {})
            
            num_questions = len(results)
            correct = summary.get('correct', 0)
            accuracy = summary.get('accuracy', 0)
            dataset_name = data.get('dataset', dataset)
            total_expected = data.get('num_questions', 0)
            
            progress_pct = (num_questions / total_expected * 100) if total_expected > 0 else 0
            status_icon = "✓" if num_questions == total_expected else "⏳"
            
            report_lines.append(f"{status_icon} {dataset_name}")
            report_lines.append(f"   Progress: {num_questions}/{total_expected} questions ({progress_pct:.1f}%)")
            report_lines.append(f"   Correct: {correct}/{num_questions}")
            report_lines.append(f"   Accuracy: {accuracy:.2f}%")
            report_lines.append("")
            
            total_questions += num_questions
            total_correct += correct
        except Exception as e:
            report_lines.append(f"Error reading {filepath}: {e}")
    
    report_lines.append("=" * 70)
    report_lines.append(f"📈 Overall Progress: {total_questions} questions processed")
    if total_questions > 0:
        overall_acc = (total_correct/total_questions)*100
        report_lines.append(f"   Overall Accuracy: {overall_acc:.2f}%")
        report_lines.append(f"   Total Expected: 910 questions (200 + 272 + 265 + 173)")
        overall_progress = (total_questions / 910) * 100
        report_lines.append(f"   Overall Completion: {overall_progress:.1f}%")
        remaining = 910 - total_questions
        report_lines.append(f"   Remaining: {remaining} questions")
    report_lines.append("=" * 70)
    
    # Check if process is running
    import subprocess
    result = subprocess.run(['pgrep', '-f', 'test-mmlu-inferdpt-santext.py'], 
                           capture_output=True, text=True)
    if result.returncode == 0:
        report_lines.append("")
        report_lines.append("✅ Experiment process is RUNNING")
    else:
        report_lines.append("")
        report_lines.append("⚠️  Experiment process is NOT RUNNING")
    
    return "\n".join(report_lines)

def send_email_report(report_text):
    """Send email report using email_config.json."""
    try:
        # Load email config - try multiple locations
        possible_paths = [
            '/data1/yizhangh/email_config.json',
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'configs', 'email_config.json'),
            os.path.expanduser('~/email_config.json'),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), 'email_config.json')
        ]
        
        config_path = None
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        
        if not config_path:
            print(f"Email config not found. Tried: {possible_paths}")
            return False
        
        with open(config_path, 'r') as f:
            email_config = json.load(f)
        
        smtp_server = email_config.get('smtp_server', 'smtp.gmail.com')
        smtp_port = email_config.get('smtp_port', 587)
        sender_email = email_config.get('sender_email')
        sender_password = email_config.get('sender_password')
        recipient_email = 'jerrystat2017@gmail.com'
        
        # Get hostname
        import socket
        hostname = socket.gethostname()
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f'MMLU InferDPT QA Progress Report - Host: {hostname}'
        
        body = f"""
MMLU InferDPT + QA Experiment Progress Report

{report_text}

---
Generated automatically by cron job
Host: {hostname}
"""
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False

def main():
    """Main function."""
    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # Get progress report
    report = get_progress_report()
    print(report)
    
    # Send email
    send_email_report(report)

if __name__ == "__main__":
    main()

