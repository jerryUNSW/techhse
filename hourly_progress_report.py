#!/usr/bin/env python3
import json
import os
import smtplib
import subprocess
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def get_latest_results_file():
    """Find the most recent PII protection results file."""
    results_dir = "/home/yizhang/tech4HSE"
    result_files = [f for f in os.listdir(results_dir) if f.startswith("pii_protection_results_") and f.endswith(".json")]
    if not result_files:
        return None

    # Sort by modification time
    result_files.sort(key=lambda x: os.path.getmtime(os.path.join(results_dir, x)), reverse=True)
    return os.path.join(results_dir, result_files[0])

def check_experiment_running():
    """Check if the experiment process is still running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'pii_protection_experiment.py'],
                              capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except:
        return False

def analyze_progress():
    """Analyze the current progress of the experiment."""
    results_file = get_latest_results_file()
    if not results_file:
        return "No results file found"

    try:
        with open(results_file, 'r') as f:
            data = json.load(f)

        # Count completed mechanisms and epsilons
        total_mechanisms = 5  # PhraseDP, InferDPT, SANTEXT+, CusText+, CluSanT
        total_epsilons = 5    # 1.0, 1.5, 2.0, 2.5, 3.0
        total_combinations = total_mechanisms * total_epsilons

        completed_combinations = 0
        current_status = "Unknown"
        latest_results = {}

        mechanisms = ["PhraseDP", "InferDPT", "SANTEXT+", "CusText+", "CluSanT"]
        epsilon_values = ["1.0", "1.5", "2.0", "2.5", "3.0"]

        for mechanism in mechanisms:
            if mechanism in data:
                for epsilon in epsilon_values:
                    if epsilon in data[mechanism]:
                        completed_combinations += 1
                        latest_results[f"{mechanism}_ε{epsilon}"] = {
                            'overall': data[mechanism][epsilon]['overall'],
                            'emails': data[mechanism][epsilon]['emails'],
                            'phones': data[mechanism][epsilon]['phones'],
                            'addresses': data[mechanism][epsilon]['addresses'],
                            'names': data[mechanism][epsilon]['names']
                        }
                        current_status = f"Completed {mechanism} ε={epsilon}"

        # Calculate progress percentage
        progress_percent = (completed_combinations / total_combinations) * 100

        # File info
        file_size = os.path.getsize(results_file)
        file_modified = datetime.fromtimestamp(os.path.getmtime(results_file))

        # Is experiment running?
        is_running = check_experiment_running()

        return {
            'progress_percent': progress_percent,
            'completed_combinations': completed_combinations,
            'total_combinations': total_combinations,
            'current_status': current_status,
            'latest_results': latest_results,
            'file_size': file_size,
            'file_modified': file_modified,
            'is_running': is_running,
            'results_file': results_file
        }

    except Exception as e:
        return f"Error analyzing progress: {str(e)}"

def format_progress_report(progress_data):
    """Format the progress data into a readable email."""
    if isinstance(progress_data, str):
        return f"Error: {progress_data}"

    report = f"""PII Protection Experiment - Hourly Progress Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

=== EXPERIMENT STATUS ===
Progress: {progress_data['progress_percent']:.1f}% ({progress_data['completed_combinations']}/{progress_data['total_combinations']} combinations)
Current Status: {progress_data['current_status']}
Experiment Running: {'Yes' if progress_data['is_running'] else 'No'}

=== FILE INFORMATION ===
Results File: {os.path.basename(progress_data['results_file'])}
File Size: {progress_data['file_size']:,} bytes ({progress_data['file_size']/1024:.1f} KB)
Last Modified: {progress_data['file_modified'].strftime('%Y-%m-%d %H:%M:%S')}

=== LATEST RESULTS ===
"""

    # Show latest completed results
    if progress_data['latest_results']:
        for combination, metrics in list(progress_data['latest_results'].items())[-5:]:  # Last 5
            report += f"{combination}:\n"
            report += f"  Overall: {metrics['overall']:.4f}\n"
            report += f"  Emails: {metrics['emails']:.4f} | Phones: {metrics['phones']:.4f}\n"
            report += f"  Addresses: {metrics['addresses']:.4f} | Names: {metrics['names']:.4f}\n"
            report += "\n"
    else:
        report += "No results available yet\n"

    report += f"""
=== NEXT STEPS ===
"""

    if progress_data['progress_percent'] >= 100:
        report += "✅ Experiment COMPLETE! All mechanisms and epsilon values processed.\n"
    elif not progress_data['is_running']:
        report += "⚠️  Experiment appears to have stopped. Check for errors or completion.\n"
    else:
        remaining = progress_data['total_combinations'] - progress_data['completed_combinations']
        report += f"🔄 Experiment in progress. {remaining} combinations remaining.\n"

    return report

def send_email_report(report_text):
    """Send the progress report via email."""
    try:
        # Load email configuration
        email_config_paths = [
            "/data1/yizhangh/email_config.json",
            "/home/yizhang/tech4HSE/email_config.json"
        ]

        email_config = None
        for config_path in email_config_paths:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    email_config = json.load(f)
                break

        if not email_config:
            print("No email configuration found")
            return False

        # Create message
        msg = MIMEMultipart()
        msg['From'] = email_config['from_email']
        msg['To'] = email_config['to_email']
        msg['Subject'] = f"PII Protection Experiment - Hourly Report [{datetime.now().strftime('%H:%M')}]"

        msg.attach(MIMEText(report_text, 'plain'))

        # Send email
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['from_email'], email_config['password'])

        text = msg.as_string()
        server.sendmail(email_config['from_email'], email_config['to_email'], text)
        server.quit()

        print(f"Hourly report sent successfully at {datetime.now()}")
        return True

    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

def main():
    """Main function to generate and send hourly progress report."""
    print(f"Generating hourly progress report at {datetime.now()}")

    # Analyze progress
    progress_data = analyze_progress()

    # Format report
    report_text = format_progress_report(progress_data)

    # Print to console for logging
    print("\n" + "="*60)
    print(report_text)
    print("="*60)

    # Send email
    email_sent = send_email_report(report_text)

    if not email_sent:
        print("Email sending failed, but report generated successfully")

if __name__ == "__main__":
    main()