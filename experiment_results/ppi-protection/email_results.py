#!/usr/bin/env python3
"""
Email Results Script
Automatically sends experiment results and plots via email
"""

import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import glob

def send_results_email(results_file=None, plot_files=None):
    """Send experiment results via email with attachments."""

    # Email configuration
    sender_email = "your_email@example.com"  # Update this
    sender_password = "your_password"        # Update this
    recipient_email = "yizhang@example.com"  # Update this

    # Create message
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"PPI Protection Experiment Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    # Email body
    body = """
    📊 PPI Protection Experiment Results

    The comprehensive experiment has completed successfully!

    ✅ Tested 5 mechanisms: PhraseDP, InferDPT, SANTEXT+, CusText+, CluSanT
    ✅ 100 samples from PII dataset
    ✅ 5 epsilon values: 1.0, 1.5, 2.0, 2.5, 3.0
    ✅ All 4 PII types: emails, phones, addresses, names

    Results and plots are attached.

    Generated automatically by PPI Protection Experiment System
    """

    msg.attach(MIMEText(body, 'plain'))

    # Attach results file
    if results_file and os.path.exists(results_file):
        with open(results_file, "rb") as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= {os.path.basename(results_file)}',
        )
        msg.attach(part)

    # Attach plot files
    if plot_files:
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                with open(plot_file, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())

                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(plot_file)}',
                )
                msg.attach(part)

    # Send email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        print("✅ Email sent successfully!")
        return True
    except Exception as e:
        print(f"❌ Error sending email: {e}")
        return False

def find_latest_results():
    """Find the latest experiment results and plots."""

    # Look for results files
    results_pattern = "pii_protection_results_*.json"
    results_files = glob.glob(results_pattern)

    if results_files:
        latest_results = max(results_files, key=os.path.getctime)
    else:
        latest_results = None

    # Look for plot files
    plot_patterns = [
        "final_overall_protection_*.png",
        "final_pii_protection_by_type_*.png",
        "final_protection_radar_*.png"
    ]

    plot_files = []
    for pattern in plot_patterns:
        plot_files.extend(glob.glob(pattern))

    return latest_results, plot_files

if __name__ == "__main__":
    print("Finding latest experiment results...")
    results_file, plot_files = find_latest_results()

    if results_file:
        print(f"Found results file: {results_file}")
    else:
        print("No results file found")

    if plot_files:
        print(f"Found {len(plot_files)} plot files:")
        for pf in plot_files:
            print(f"  - {pf}")
    else:
        print("No plot files found")

    # Send email (disabled by default - needs configuration)
    print("\n⚠️  Email sending disabled - please configure email settings first")
    # send_results_email(results_file, plot_files)