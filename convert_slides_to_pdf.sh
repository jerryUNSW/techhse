#!/bin/bash

# Convert Reveal.js slides to PDF presentation
# Usage: ./convert_slides_to_pdf.sh [slides.md] [output.pdf]

SLIDES_MD="${1:-meeting_slides_slides.md}"
OUTPUT_PDF="${2:-meeting_slides_presentation.pdf}"
TEMP_HTML="${SLIDES_MD%.md}_temp.html"

echo "🎯 Converting Slides to PDF Presentation"
echo "========================================"
echo "📄 Input: $SLIDES_MD"
echo "📄 Output: $OUTPUT_PDF"

# Check if input file exists
if [ ! -f "$SLIDES_MD" ]; then
    echo "❌ Error: Input file '$SLIDES_MD' not found!"
    exit 1
fi

# Step 1: Convert Markdown to Reveal.js HTML
echo ""
echo "📝 Step 1: Converting Markdown to Reveal.js HTML..."
pandoc "$SLIDES_MD" -t revealjs -s -o "$TEMP_HTML" \
    -V revealjs-url=https://cdn.jsdelivr.net/npm/reveal.js@4.3.1

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to convert Markdown to HTML!"
    exit 1
fi

echo "✅ HTML presentation created: $TEMP_HTML"

# Step 2: Convert HTML to PDF using Chrome
echo ""
echo "📄 Step 2: Converting HTML to PDF slides..."
echo "💡 Using Chrome headless to render slides..."

# For Reveal.js, we need to append ?print-pdf to get slide layout
ABS_PATH=$(readlink -f "$TEMP_HTML")
PDF_URL="file://$ABS_PATH?print-pdf"

# Try with Chrome headless
if command -v google-chrome &> /dev/null; then
    google-chrome --headless --disable-gpu \
        --print-to-pdf="$OUTPUT_PDF" \
        --print-to-pdf-no-header \
        "$PDF_URL" 2>&1 | tail -1
elif command -v chromium-browser &> /dev/null; then
    chromium-browser --headless --disable-gpu \
        --print-to-pdf="$OUTPUT_PDF" \
        --print-to-pdf-no-header \
        "$PDF_URL" 2>&1 | tail -1
else
    echo "❌ Error: Chrome or Chromium not found!"
    echo "💡 Alternative: Open $TEMP_HTML in your browser and print to PDF"
    echo "   URL to use: $PDF_URL"
    rm -f "$TEMP_HTML"
    exit 1
fi

# Clean up temp file
rm -f "$TEMP_HTML"

if [ -f "$OUTPUT_PDF" ]; then
    echo ""
    echo "✅ Success! PDF slides created: $OUTPUT_PDF"
    echo "📊 File size: $(du -h "$OUTPUT_PDF" | cut -f1)"
    echo ""
    echo "💡 To view: xdg-open $OUTPUT_PDF"
    echo "💡 Or open $TEMP_HTML in browser for interactive presentation"
else
    echo ""
    echo "⚠️  PDF creation may have issues. Try manual method:"
    echo "   1. Open $TEMP_HTML in browser"
    echo "   2. Append ?print-pdf to the URL"
    echo "   3. Press Ctrl+P and save as PDF"
fi
