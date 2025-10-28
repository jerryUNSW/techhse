#!/usr/bin/env python3
"""
Daily Outdated Files Report
==========================

Identifies the most outdated files in the project and sends a daily email report.
"""

import os
import json
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
import glob

def load_email_config():
    """Load email configuration from file."""
    try:
        with open('email_config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ email_config.json not found")
        return None

def get_file_age_info(file_path):
    """Get file age information."""
    try:
        stat = os.stat(file_path)
        modified_time = datetime.fromtimestamp(stat.st_mtime)
        age = datetime.now() - modified_time
        return {
            'path': file_path,
            'modified': modified_time,
            'age_days': age.days,
            'age_hours': age.total_seconds() / 3600,
            'size': stat.st_size
        }
    except Exception as e:
        return None

def find_outdated_files(directory, max_age_days=7, file_patterns=None):
    """Find files older than max_age_days."""
    if file_patterns is None:
        file_patterns = [
            '*.py', '*.json', '*.md', '*.txt', '*.yaml', '*.yml',
            '*.csv', '*.log', '*.sql', '*.sh', '*.bat'
        ]
    
    outdated_files = []
    
    for pattern in file_patterns:
        for file_path in glob.glob(os.path.join(directory, '**', pattern), recursive=True):
            # Skip hidden files and common directories
            if any(skip in file_path for skip in ['.git', '__pycache__', '.pytest_cache', 'node_modules']):
                continue
                
            file_info = get_file_age_info(file_path)
            if file_info and file_info['age_days'] > max_age_days:
                outdated_files.append(file_info)
    
    # Sort by age (oldest first)
    outdated_files.sort(key=lambda x: x['age_days'], reverse=True)
    return outdated_files

def categorize_files(files):
    """Categorize files by type and importance."""
    categories = {
        'Critical Scripts': [],
        'Configuration Files': [],
        'Documentation': [],
        'Data Files': [],
        'Log Files': [],
        'Other': []
    }
    
    for file_info in files:
        file_path = file_info['path']
        filename = os.path.basename(file_path)
        
        if any(keyword in filename.lower() for keyword in ['test', 'main', 'run', 'start', 'execute']):
            categories['Critical Scripts'].append(file_info)
        elif any(ext in filename.lower() for ext in ['.json', '.yaml', '.yml', '.ini', '.cfg', '.conf']):
            categories['Configuration Files'].append(file_info)
        elif any(ext in filename.lower() for ext in ['.md', '.txt', '.rst', '.doc']):
            categories['Documentation'].append(file_info)
        elif any(ext in filename.lower() for ext in ['.csv', '.json', '.sql', '.db']):
            categories['Data Files'].append(file_info)
        elif any(ext in filename.lower() for ext in ['.log', '.out', '.err']):
            categories['Log Files'].append(file_info)
        else:
            categories['Other'].append(file_info)
    
    return categories

def generate_report(outdated_files, max_files=20):
    """Generate the daily report."""
    report_date = datetime.now().strftime('%Y-%m-%d')
    hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
    
    report = f"""
Daily Outdated Files Report
==========================
Date: {report_date}
Host: {hostname}
Total Outdated Files Found: {len(outdated_files)}

TOP {max_files} MOST OUTDATED FILES:
{'='*80}
"""
    
    # Show top N most outdated files
    for i, file_info in enumerate(outdated_files[:max_files], 1):
        report += f"""
{i:2d}. {os.path.basename(file_info['path'])}
     Path: {file_info['path']}
     Last Modified: {file_info['modified'].strftime('%Y-%m-%d %H:%M:%S')}
     Age: {file_info['age_days']} days ({file_info['age_hours']:.1f} hours)
     Size: {file_info['size']:,} bytes
"""
    
    # Categorized breakdown
    categories = categorize_files(outdated_files)
    report += f"""

CATEGORIZED BREAKDOWN:
{'='*50}
"""
    
    for category, files in categories.items():
        if files:
            report += f"\n{category} ({len(files)} files):\n"
            for file_info in files[:5]:  # Show top 5 per category
                report += f"  • {os.path.basename(file_info['path'])} ({file_info['age_days']} days old)\n"
            if len(files) > 5:
                report += f"  ... and {len(files) - 5} more\n"
    
    # Recommendations
    report += f"""

RECOMMENDATIONS:
{'='*30}
"""
    
    critical_old = [f for f in outdated_files if any(keyword in os.path.basename(f['path']).lower() 
                   for keyword in ['test', 'main', 'run', 'start']) and f['age_days'] > 30]
    
    if critical_old:
        report += f"⚠️  {len(critical_old)} critical scripts are over 30 days old - consider updating\n"
    
    config_old = [f for f in outdated_files if any(ext in os.path.basename(f['path']).lower() 
                  for ext in ['.json', '.yaml', '.yml']) and f['age_days'] > 14]
    
    if config_old:
        report += f"⚙️  {len(config_old)} configuration files are over 14 days old - review for updates\n"
    
    doc_old = [f for f in outdated_files if any(ext in os.path.basename(f['path']).lower() 
               for ext in ['.md', '.txt']) and f['age_days'] > 60]
    
    if doc_old:
        report += f"📚 {len(doc_old)} documentation files are over 60 days old - consider updating\n"
    
    if not critical_old and not config_old and not doc_old:
        report += "✅ No critical files need immediate attention\n"
    
    return report

def send_email_report(report, email_config):
    """Send the report via email."""
    try:
        msg = MIMEMultipart()
        msg['From'] = email_config['sender_email']
        msg['To'] = email_config['recipient_email']
        msg['Subject'] = f"Daily Outdated Files Report - {datetime.now().strftime('%Y-%m-%d')} - Host: {os.uname().nodename if hasattr(os, 'uname') else 'unknown'}"
        
        msg.attach(MIMEText(report, 'plain'))
        
        server = smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port'])
        server.starttls()
        server.login(email_config['sender_email'], email_config['password'])
        
        text = msg.as_string()
        server.sendmail(email_config['sender_email'], email_config['recipient_email'], text)
        server.quit()
        
        print(f"✅ Email report sent successfully to {email_config['recipient_email']}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

def main():
    """Main function."""
    print("🔍 Starting Daily Outdated Files Report...")
    
    # Load email configuration
    email_config = load_email_config()
    if not email_config:
        print("❌ Cannot proceed without email configuration")
        return
    
    # Find outdated files (older than 7 days)
    project_dir = os.getcwd()
    print(f"📁 Scanning directory: {project_dir}")
    
    outdated_files = find_outdated_files(project_dir, max_age_days=7)
    print(f"📊 Found {len(outdated_files)} outdated files")
    
    if not outdated_files:
        print("✅ No outdated files found!")
        return
    
    # Generate report
    report = generate_report(outdated_files, max_files=20)
    print("📝 Report generated")
    
    # Print report to console
    print("\n" + "="*80)
    print(report)
    print("="*80)
    
    # Send email
    if send_email_report(report, email_config):
        print("📧 Daily report sent successfully!")
    else:
        print("❌ Failed to send email report")
    
    # Save report to file
    report_file = f"daily_outdated_files_report_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"💾 Report saved to: {report_file}")

if __name__ == "__main__":
    main()

