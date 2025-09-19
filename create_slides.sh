#!/bin/bash

# Create Slides Script
# Converts Markdown slides to Reveal.js HTML presentations

echo "🎯 Markdown to Reveal.js Slides Converter"
echo "========================================"

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <markdown_file> [output_file]"
    echo "Example: $0 slides.md"
    echo "Example: $0 slides.md presentation.html"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="${2:-${INPUT_FILE%.md}.html}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file '$INPUT_FILE' not found!"
    exit 1
fi

echo "📄 Input file: $INPUT_FILE"
echo "🌐 Output file: $OUTPUT_FILE"

# Convert using pandoc
echo "🔄 Converting Markdown to Reveal.js HTML..."
pandoc "$INPUT_FILE" -t revealjs -s -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "✅ Success! Slides created: $OUTPUT_FILE"
    echo "🚀 Open in browser: file://$(pwd)/$OUTPUT_FILE"
    
    # Ask if user wants to open in browser
    read -p "🌐 Open slides in browser? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v xdg-open &> /dev/null; then
            xdg-open "$OUTPUT_FILE"
        elif command -v open &> /dev/null; then
            open "$OUTPUT_FILE"
        else
            echo "⚠️  Please open manually: $OUTPUT_FILE"
        fi
    fi
else
    echo "❌ Error: Conversion failed!"
    exit 1
fi
