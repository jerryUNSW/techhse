#!/usr/bin/env python3
"""
MedQA Comprehensive Test Monitor
===============================

This script monitors the comprehensive test progress and provides hourly updates.
It tracks all three epsilon experiments (1.0, 2.0, 3.0) and provides detailed progress reports.

Author: Tech4HSE Team
Date: 2025-10-02
"""

import json
import os
import glob
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Color codes
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"

def find_latest_comprehensive_files() -> Dict[str, str]:
    """Find the latest comprehensive test files for each epsilon."""
    files = {}
    
    for epsilon in [1.0, 2.0, 3.0]:
        pattern = f"medqa_comprehensive_epsilon_{epsilon}_*_samples_100_*.json"
        matching_files = glob.glob(pattern)
        
        if matching_files:
            # Get the most recent file
            latest_file = max(matching_files, key=os.path.getmtime)
            files[f"epsilon_{epsilon}"] = latest_file
        else:
            files[f"epsilon_{epsilon}"] = None
    
    return files

def analyze_file_progress(filepath: str) -> Dict[str, Any]:
    """Analyze progress from a comprehensive test file."""
    if not filepath or not os.path.exists(filepath):
        return {
            'status': 'not_started',
            'total_questions': 0,
            'completed_questions': 0,
            'accuracy': {},
            'last_updated': None,
            'file_size': 0
        }
    
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if not data:
            return {
                'status': 'started',
                'total_questions': 100,
                'completed_questions': 0,
                'accuracy': {},
                'last_updated': None,
                'file_size': os.path.getsize(filepath)
            }
        
        # Get latest question data
        latest_question = data[-1]
        total_questions = 100  # Expected total
        completed_questions = len(data)
        
        # Calculate current accuracies
        accuracy = {
            'local_alone': 0,
            'inferdpt': 0,
            'santext': 0,
            'phrasedp_normal': 0,
            'phrasedp_medical': 0,
            'local_cot': 0,
            'remote': 0
        }
        
        for question_data in data:
            if question_data.get('local_correct'):
                accuracy['local_alone'] += 1
            if question_data.get('inferdpt_correct'):
                accuracy['inferdpt'] += 1
            if question_data.get('santext_correct'):
                accuracy['santext'] += 1
            if question_data.get('phrasedp_normal_correct'):
                accuracy['phrasedp_normal'] += 1
            if question_data.get('phrasedp_medical_correct'):
                accuracy['phrasedp_medical'] += 1
            if question_data.get('local_cot_correct'):
                accuracy['local_cot'] += 1
            if question_data.get('remote_correct'):
                accuracy['remote'] += 1
        
        # Convert to percentages
        for key in accuracy:
            accuracy[key] = (accuracy[key] / completed_questions * 100) if completed_questions > 0 else 0
        
        return {
            'status': 'running' if completed_questions < total_questions else 'completed',
            'total_questions': total_questions,
            'completed_questions': completed_questions,
            'accuracy': accuracy,
            'last_updated': latest_question.get('timestamp'),
            'file_size': os.path.getsize(filepath),
            'epsilon': latest_question.get('epsilon')
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'total_questions': 0,
            'completed_questions': 0,
            'accuracy': {},
            'last_updated': None,
            'file_size': 0
        }

def send_email_report(epsilon_files: Dict[str, str], progress_data: Dict[str, Any]):
    """Send hourly email report."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        # Load email config
        with open('email_config.json', 'r') as f:
            email_config = json.load(f)
        
        sender_email = email_config['from_email']
        sender_password = email_config['password']
        recipient_email = email_config['to_email']
        
        # Create email content
        subject = f"MedQA Comprehensive Test Progress - {datetime.now().strftime('%H:%M')}"
        
        body = f"""
MedQA Comprehensive Test Hourly Progress Report
==============================================

Host: {os.uname().nodename}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PROGRESS:
- Total Questions Expected: 300 (100 per epsilon)
- Questions Completed: {progress_data['total_completed']}/300 ({progress_data['total_completed']/300*100:.1f}%)
- Overall Completion: {progress_data['completion_percentage']:.1f}%

EPSILON 1.0 PROGRESS:
- Status: {progress_data['epsilon_1']['status'].upper()}
- Questions: {progress_data['epsilon_1']['completed_questions']}/100 ({progress_data['epsilon_1']['completion_percentage']:.1f}%)
- Top Performers:
  • Local + CoT: {progress_data['epsilon_1']['accuracy']['local_cot']:.1f}%
  • Remote Model: {progress_data['epsilon_1']['accuracy']['remote']:.1f}%
  • PhraseDP+ (Medical): {progress_data['epsilon_1']['accuracy']['phrasedp_medical']:.1f}%

EPSILON 2.0 PROGRESS:
- Status: {progress_data['epsilon_2']['status'].upper()}
- Questions: {progress_data['epsilon_2']['completed_questions']}/100 ({progress_data['epsilon_2']['completion_percentage']:.1f}%)
- Top Performers:
  • Local + CoT: {progress_data['epsilon_2']['accuracy']['local_cot']:.1f}%
  • Remote Model: {progress_data['epsilon_2']['accuracy']['remote']:.1f}%
  • PhraseDP+ (Medical): {progress_data['epsilon_2']['accuracy']['phrasedp_medical']:.1f}%

EPSILON 3.0 PROGRESS:
- Status: {progress_data['epsilon_3']['status'].upper()}
- Questions: {progress_data['epsilon_3']['completed_questions']}/100 ({progress_data['epsilon_3']['completion_percentage']:.1f}%)
- Top Performers:
  • Local + CoT: {progress_data['epsilon_3']['accuracy']['local_cot']:.1f}%
  • Remote Model: {progress_data['epsilon_3']['accuracy']['remote']:.1f}%
  • PhraseDP+ (Medical): {progress_data['epsilon_3']['accuracy']['phrasedp_medical']:.1f}%

MEDICAL MODE BENEFIT ANALYSIS:
- Epsilon 1.0: {progress_data['epsilon_1']['medical_benefit']:+.1f} percentage points
- Epsilon 2.0: {progress_data['epsilon_2']['medical_benefit']:+.1f} percentage points  
- Epsilon 3.0: {progress_data['epsilon_3']['medical_benefit']:+.1f} percentage points

FILES:
- Epsilon 1.0: {epsilon_files.get('epsilon_1.0', 'Not found')}
- Epsilon 2.0: {epsilon_files.get('epsilon_2.0', 'Not found')}
- Epsilon 3.0: {epsilon_files.get('epsilon_3.0', 'Not found')}

Next update in 1 hour.
        """
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        
        print(f"{GREEN}✅ Hourly email report sent successfully{RESET}")
        
    except Exception as e:
        print(f"{RED}❌ Failed to send hourly email: {e}{RESET}")

def print_progress_report(epsilon_files: Dict[str, str], progress_data: Dict[str, Any]):
    """Print detailed progress report to console."""
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}📊 MEDQA COMPREHENSIVE TEST PROGRESS REPORT{RESET}")
    print(f"{BLUE}{'='*80}{RESET}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Host: {os.uname().nodename}")
    print()
    
    print(f"{CYAN}🎯 OVERALL PROGRESS:{RESET}")
    print(f"Total Questions Expected: 300 (100 per epsilon)")
    print(f"Questions Completed: {progress_data['total_completed']}/300 ({progress_data['completion_percentage']:.1f}%)")
    print()
    
    for epsilon in ['1', '2', '3']:
        eps_data = progress_data[f'epsilon_{epsilon}']
        print(f"{CYAN}📈 EPSILON {epsilon}.0 PROGRESS:{RESET}")
        print(f"Status: {eps_data['status'].upper()}")
        print(f"Questions: {eps_data['completed_questions']}/100 ({eps_data['completion_percentage']:.1f}%)")
        print(f"File: {epsilon_files.get(f'epsilon_{epsilon}.0', 'Not found')}")
        
        if eps_data['completed_questions'] > 0:
            print(f"\n🏆 TOP PERFORMERS:")
            acc = eps_data['accuracy']
            performers = [
                ('Local + CoT', acc['local_cot']),
                ('Remote Model', acc['remote']),
                ('PhraseDP+ (Medical)', acc['phrasedp_medical']),
                ('PhraseDP (Normal)', acc['phrasedp_normal']),
                ('SANTEXT+', acc['santext']),
                ('InferDPT', acc['inferdpt']),
                ('Local Model', acc['local_alone'])
            ]
            performers.sort(key=lambda x: x[1], reverse=True)
            
            for i, (name, acc_val) in enumerate(performers[:5], 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else ""
                print(f"  {i}. {name}: {acc_val:.1f}% {medal}")
            
            medical_benefit = eps_data['medical_benefit']
            print(f"\n🔍 MEDICAL MODE BENEFIT: {medical_benefit:+.1f} percentage points")
            if medical_benefit > 0:
                print(f"   ✅ Medical mode is BETTER by {medical_benefit:.1f} percentage points")
            elif medical_benefit < 0:
                print(f"   ❌ Medical mode is WORSE by {abs(medical_benefit):.1f} percentage points")
            else:
                print(f"   ⚖️  Both modes performing identically")
        print()

def main():
    """Main monitoring loop."""
    print(f"{GREEN}🚀 Starting MedQA Comprehensive Test Monitor{RESET}")
    print(f"Monitoring all three epsilon experiments (1.0, 2.0, 3.0)")
    print(f"Providing hourly email reports and 30-minute console updates")
    print()
    
    last_hour_sent = -1
    
    while True:
        try:
            # Find latest files
            epsilon_files = find_latest_comprehensive_files()
            
            # Analyze progress for each epsilon
            progress_data = {
                'epsilon_1': analyze_file_progress(epsilon_files.get('epsilon_1.0')),
                'epsilon_2': analyze_file_progress(epsilon_files.get('epsilon_2.0')),
                'epsilon_3': analyze_file_progress(epsilon_files.get('epsilon_3.0'))
            }
            
            # Calculate completion percentages and medical benefits
            total_completed = 0
            for eps_key in ['epsilon_1', 'epsilon_2', 'epsilon_3']:
                eps_data = progress_data[eps_key]
                eps_data['completion_percentage'] = (eps_data['completed_questions'] / eps_data['total_questions'] * 100) if eps_data['total_questions'] > 0 else 0
                eps_data['medical_benefit'] = eps_data['accuracy']['phrasedp_medical'] - eps_data['accuracy']['phrasedp_normal']
                total_completed += eps_data['completed_questions']
            
            progress_data['total_completed'] = total_completed
            progress_data['completion_percentage'] = (total_completed / 300 * 100)
            
            # Print console report every 30 minutes
            current_time = datetime.now()
            if current_time.minute % 30 == 0:
                print_progress_report(epsilon_files, progress_data)
            
            # Send email report every hour
            current_hour = current_time.hour
            if current_hour != last_hour_sent and current_time.minute < 5:
                print(f"{YELLOW}📧 Sending hourly email report...{RESET}")
                send_email_report(epsilon_files, progress_data)
                last_hour_sent = current_hour
            
            # Check if all experiments are completed
            all_completed = all(eps['status'] == 'completed' for eps in progress_data.values())
            if all_completed:
                print(f"{GREEN}🎉 All experiments completed! Final report:{RESET}")
                print_progress_report(epsilon_files, progress_data)
                print(f"{GREEN}✅ Monitoring complete. Exiting.{RESET}")
                break
            
            # Sleep for 5 minutes before next check
            time.sleep(300)
            
        except KeyboardInterrupt:
            print(f"\n{YELLOW}⚠️  Monitoring interrupted by user{RESET}")
            break
        except Exception as e:
            print(f"{RED}❌ Error in monitoring loop: {e}{RESET}")
            time.sleep(60)  # Wait 1 minute before retrying

if __name__ == "__main__":
    main()

