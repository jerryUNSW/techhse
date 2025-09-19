#!/bin/bash

# Git operations with password from pass.txt
# Usage: ./git_with_pass.sh <git_command>
# Example: ./git_with_pass.sh "push origin master"

# Check if pass.txt exists
if [ ! -f "pass.txt" ]; then
    echo "Error: pass.txt not found!"
    exit 1
fi

# Read password from pass.txt
PASSWORD=$(cat pass.txt)

# Set up git credential helper for this session
git config credential.helper "store --file ~/.git-credentials"

# Create a temporary credential file
echo "https://git.overleaf.com:${PASSWORD}@git.overleaf.com" > ~/.git-credentials

# Execute the git command
echo "Executing: git $@"
git "$@"

# Clean up (optional - remove if you want to keep credentials)
# rm ~/.git-credentials

echo "Git operation completed!"
