#!/bin/bash

# Setup Daily Email Summary Cron Job
# This script sets up a cron job to send daily email summaries at 9 AM

echo "Setting up daily email summary cron job..."

# Get the current directory (where the script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$SCRIPT_DIR/daily_email_summary.py"

# Make sure the Python script is executable
chmod +x "$PYTHON_SCRIPT"

# Create the cron job entry
CRON_ENTRY="0 9 * * * cd $SCRIPT_DIR && /usr/bin/python3 $PYTHON_SCRIPT >> $SCRIPT_DIR/daily_email.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "daily_email_summary.py"; then
    echo "Cron job already exists. Updating..."
    # Remove existing entry and add new one
    (crontab -l 2>/dev/null | grep -v "daily_email_summary.py"; echo "$CRON_ENTRY") | crontab -
else
    echo "Adding new cron job..."
    # Add new cron job
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
fi

echo "Cron job setup complete!"
echo "Daily email summary will be sent at 9:00 AM every day"
echo "Logs will be saved to: $SCRIPT_DIR/daily_email.log"
echo ""
echo "To view current cron jobs: crontab -l"
echo "To remove this cron job: crontab -e (then delete the line)"
echo "To test the script manually: python3 $PYTHON_SCRIPT"




