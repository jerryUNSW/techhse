#!/bin/bash
"""
Setup Daily Outdated Files Report Cron Job
==========================================

This script sets up a daily cron job to run the outdated files report.
"""

echo "Setting up daily outdated files report cron job..."

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/daily_outdated_files_report.py"

# Create the cron job entry (runs daily at 9 AM)
CRON_ENTRY="0 9 * * * cd $SCRIPT_DIR && /usr/bin/python3 $PYTHON_SCRIPT >> $SCRIPT_DIR/daily_report.log 2>&1"

# Add to crontab
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "✅ Daily report cron job added!"
echo "📅 Schedule: Daily at 9:00 AM"
echo "📁 Script: $PYTHON_SCRIPT"
echo "📝 Logs: $SCRIPT_DIR/daily_report.log"
echo ""
echo "To view current crontab: crontab -l"
echo "To remove this cron job: crontab -e (then delete the line)"
echo "To test the script: python3 $PYTHON_SCRIPT"

