# How to Convert Meeting Slides to PDF

## Quick Start

The easiest way is to use the provided script:

```bash
./convert_slides_to_pdf.sh meeting_slides.md meeting_slides.pdf
```

## Methods

### Method 1: Markdown → PDF via Pandoc (Recommended for Quality)

**Requirements:**
- Pandoc (already installed)
- LaTeX engine (xelatex, pdflatex, or lualatex)

**Install LaTeX if needed:**
```bash
# Full LaTeX installation (large, ~2-3GB)
sudo apt-get update
sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-latex-extra

# OR minimal installation (smaller)
sudo apt-get install texlive-xetex texlive-latex-base
```

**Convert:**
```bash
# Using the script (automatically detects best LaTeX engine)
./convert_slides_to_pdf.sh meeting_slides.md meeting_slides.pdf

# OR manually with pandoc
pandoc meeting_slides.md \
    --from=markdown \
    --to=pdf \
    --output=meeting_slides.pdf \
    --pdf-engine=xelatex \
    --variable=geometry:margin=1in \
    --variable=fontsize:11pt
```

**Advantages:**
- High quality PDF output
- Proper figure rendering (PDFs and PNGs)
- Professional typography
- Page breaks work correctly

---

### Method 2: HTML → PDF (No LaTeX Required)

**Requirements:**
- HTML version of slides
- HTML to PDF converter (wkhtmltopdf or weasyprint)

**Install converter:**
```bash
# Option A: wkhtmltopdf
sudo apt-get install wkhtmltopdf

# Option B: weasyprint (via pip)
pip install weasyprint
```

**Convert:**
```bash
# First convert Markdown to HTML (if needed)
pandoc meeting_slides.md -t html -s -o meeting_slides.html

# Then convert HTML to PDF
./convert_slides_to_pdf.sh meeting_slides.html meeting_slides.pdf html

# OR manually
wkhtmltopdf --page-size A4 --margin-top 0.5in \
            --margin-bottom 0.5in --margin-left 0.5in \
            --margin-right 0.5in meeting_slides.html meeting_slides.pdf
```

**Advantages:**
- No LaTeX installation needed
- Faster conversion
- Good for simple documents

**Disadvantages:**
- May have issues with complex LaTeX figure code
- Less precise control over layout

---

### Method 3: Browser Print (Easiest, No Installation)

**Steps:**
1. Convert Markdown to HTML:
   ```bash
   pandoc meeting_slides.md -t html -s -o meeting_slides.html
   ```

2. Open `meeting_slides.html` in your web browser

3. Press `Ctrl+P` (or `Cmd+P` on Mac) to open print dialog

4. Choose "Save as PDF" as the destination

5. Adjust settings:
   - Page size: A4
   - Margins: Default or Minimum
   - Scale: 100%
   - Background graphics: Enable

6. Click "Save" and choose filename

**Advantages:**
- No additional software needed
- Works on any system with a browser
- Preview before saving

**Disadvantages:**
- Manual process
- May need to adjust browser print settings
- LaTeX figure code in Markdown won't work (need HTML version first)

---

### Method 4: Markdown → Beamer Slides (For Presentations)

If you want actual presentation slides (not a document), convert to Beamer:

```bash
# Requires LaTeX and beamer package
pandoc meeting_slides.md -t beamer -o meeting_slides_beamer.pdf \
    --pdf-engine=xelatex -V theme:Madrid -V colortheme:default
```

**Note:** This requires restructuring the Markdown to use Beamer slide syntax (`---` between slides).

---

## Troubleshooting

### Error: "No LaTeX engine found"
- Install LaTeX (see Method 1 above)
- OR use HTML method (Method 2 or 3)

### Error: "Figure not found"
- Make sure image paths are correct relative to the Markdown file location
- For PDFs in `overleaf-folder/`, paths should be `overleaf-folder/figures/...` or `overleaf-folder/plots/...`
- For PNGs in root, paths should be just the filename (e.g., `medqa_ablation_study_combined-1.png`)

### Figures not rendering properly
- PDF images work best with LaTeX (Method 1)
- PNG images work with all methods
- If using HTML method, ensure image paths are correct in the HTML

### Page breaks not working
- Add `\newpage` in Markdown for explicit page breaks
- Adjust `--variable=geometry:margin=...` for page layout
- Use LaTeX method for best control over page breaks

---

## Recommended Approach

**For best results:**
1. Install LaTeX: `sudo apt-get install texlive-xetex texlive-fonts-recommended texlive-latex-extra`
2. Run: `./convert_slides_to_pdf.sh meeting_slides.md meeting_slides.pdf`

**For quick conversion without installation:**
1. Use browser print method (Method 3) - convert to HTML first, then print to PDF

**For automated conversion without LaTeX:**
1. Use HTML → PDF method (Method 2) with wkhtmltopdf or weasyprint

