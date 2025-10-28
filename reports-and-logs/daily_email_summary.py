#!/usr/bin/env python3
"""
Daily Email Summary Script for Tech4HSE Project
Sends a summary of recent progress, issues, and todos every morning at 9 AM
"""

import json
import smtplib
import os
import subprocess
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

class DailyEmailSummary:
    def __init__(self, config_file="email_config.json"):
        self.config_file = config_file
        self.load_config()
        self.project_root = Path(__file__).parent
        
    def load_config(self):
        """Load email configuration from JSON file"""
        try:
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Error: {self.config_file} not found!")
            exit(1)
    
    def get_git_status(self):
        """Get recent git commits and status"""
        try:
            # Get recent commits (last 3 days)
            result = subprocess.run([
                'git', 'log', '--oneline', '--since=3 days ago'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            recent_commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Get current branch
            branch_result = subprocess.run([
                'git', 'branch', '--show-current'
            ], capture_output=True, text=True, cwd=self.project_root)
            
            current_branch = branch_result.stdout.strip()
            
            return {
                'recent_commits': recent_commits[:5],  # Last 5 commits
                'current_branch': current_branch,
                'total_commits': len(recent_commits)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def get_project_files_status(self):
        """Get status of key project files"""
        key_files = [
            'README.md',
            'complete-550.txt',
            'ner_pii_privacy_results.json',
            'privacy_breach_analysis.txt',
            '68ad44193c5dfb725ec03fdb/main.tex'
        ]
        
        file_status = {}
        for file_path in key_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                stat = full_path.stat()
                file_status[file_path] = {
                    'exists': True,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                }
            else:
                file_status[file_path] = {'exists': False}
        
        return file_status
    
    def get_recent_issues_and_todos(self):
        """Extract recent issues and todos from project files"""
        issues = []
        todos = []
        
        # Check README for recent updates
        readme_path = self.project_root / 'README.md'
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                content = f.read()
                
                # Look for TODO markers
                if 'TODO' in content or 'FIXME' in content:
                    todos.append("Check README.md for TODO/FIXME items")
                
                # Look for recent updates
                if 'August 26, 2025' in content:
                    todos.append("Complete 500-question experiment results documented")
        
        # Check for experiment results
        results_file = self.project_root / 'ner_pii_privacy_results.json'
        if results_file.exists():
            todos.append("NER-based privacy evaluation completed (100 questions)")
        
        # Check for privacy analysis
        privacy_analysis = self.project_root / 'privacy_breach_analysis.txt'
        if privacy_analysis.exists():
            with open(privacy_analysis, 'r') as f:
                content = f.read()
                if 'EXCELLENT PRIVACY PROTECTION' in content:
                    todos.append("Privacy breach analysis completed - no breaches detected")
        
        # Check LaTeX paper status
        latex_file = self.project_root / '68ad44193c5dfb725ec03fdb/main.tex'
        if latex_file.exists():
            todos.append("Research paper (LaTeX) is being maintained")
        
        return issues, todos
    
    def generate_summary_content(self):
        """Generate the email summary content"""
        git_status = self.get_git_status()
        file_status = self.get_project_files_status()
        issues, todos = self.get_recent_issues_and_todos()
        
        # Get current date
        today = datetime.now().strftime('%Y-%m-%d')
        
        content = f"""
Tech4HSE Project Daily Summary - {today}

=== EXPERIMENT #2 PROGRESS ===
"""
        
        # Check for experiment #2 progress
        exp2_file = self.project_root / 'test-500-new-2.txt'
        if exp2_file.exists():
            stat = exp2_file.stat()
            content += f"✓ Experiment #2 Running: test-500-new-2.txt ({stat.st_size} bytes)\n"
            content += f"  Modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            # Try to get progress from the file
            try:
                with open(exp2_file, 'r') as f:
                    file_content = f.read()
                    import re
                    question_matches = re.findall(r'Question \d+/\d+ \(Dataset idx: \d+\)', file_content)
                    questions_done = len(question_matches)
                    content += f"  Progress: {questions_done}/500 questions completed ({(questions_done/500)*100:.1f}%)\n"
            except:
                content += "  Progress: Unable to parse current progress\n"
        else:
            content += "✗ Experiment #2: test-500-new-2.txt not found\n"
        
        # Git status
        if 'error' not in git_status:
            content += f"""
=== RECENT COMMITS ===
- Current Branch: {git_status['current_branch']}
- Recent Commits (last 3 days): {git_status['total_commits']}
"""
            if git_status['recent_commits']:
                content += "Recent commits:\n"
                for commit in git_status['recent_commits']:
                    content += f"  • {commit}\n"
        else:
            content += f"Git Status: Error - {git_status['error']}\n"
        
        # Key files status (focus on experiment files)
        content += "\n=== EXPERIMENT FILES STATUS ===\n"
        exp_files = ['test-500-new-2.txt', 'test-500-new.txt', 'README.md']
        for file_path in exp_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                stat = full_path.stat()
                content += f"✓ {file_path} ({stat.st_size} bytes, modified: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')})\n"
            else:
                content += f"✗ {file_path} (missing)\n"
        
        # Issues and TODOs
        content += "\n=== CURRENT ISSUES ===\n"
        if issues:
            for issue in issues:
                content += f"• {issue}\n"
        else:
            content += "No current issues identified.\n"
        
        content += "\n=== TODOS & NEXT STEPS ===\n"
        if todos:
            for todo in todos:
                content += f"• {todo}\n"
        else:
            content += "• Monitor Experiment #2 progress\n"
            content += "• Analyze results when complete\n"
        
        # Add standard todos
        content += """
• Continue linguistic quality evaluation experiments
• Implement tiered sensitivity approach for privacy evaluation
• Complete comprehensive privacy-utility trade-off analysis
• Prepare for paper submission
• Update documentation as needed
"""
        
        content += f"""
=== PROJECT OVERVIEW ===
This is an automated daily summary for the Tech4HSE privacy-preserving 
multi-hop question answering system project. The project focuses on 
comparing Phrase DP and InferDPT methods for medical QA applications.

Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return content
    
    def send_email(self, subject, body):
        """Send email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['from_email']
            msg['To'] = self.config['to_email']
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'plain'))
            
            # Connect to server and send
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['from_email'], self.config['password'])
            
            text = msg.as_string()
            server.sendmail(self.config['from_email'], self.config['to_email'], text)
            server.quit()
            
            print(f"Email sent successfully to {self.config['to_email']}")
            return True
            
        except Exception as e:
            print(f"Error sending email: {e}")
            return False
    
    def run_daily_summary(self):
        """Run the daily summary process"""
        print(f"Generating daily summary for {datetime.now().strftime('%Y-%m-%d')}")
        
        # Generate content
        content = self.generate_summary_content()
        
        # Create subject
        subject = f"Tech4HSE Daily Summary - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Send email
        success = self.send_email(subject, content)
        
        if success:
            print("Daily summary sent successfully!")
        else:
            print("Failed to send daily summary!")
        
        return success

def main():
    """Main function"""
    email_summary = DailyEmailSummary()
    email_summary.run_daily_summary()

if __name__ == "__main__":
    main()




