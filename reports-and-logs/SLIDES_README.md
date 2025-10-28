# Markdown to Reveal.js Slides Workflow

This document explains how to create beautiful HTML presentations from Markdown using Pandoc and Reveal.js.

## 🚀 Quick Start

### **1. Create Markdown Slides**
Write your slides in `slides.md` using the following format:

```markdown
---
title: "Your Presentation Title"
subtitle: "Optional Subtitle"
author: "Your Name"
date: "Date"
theme: "white"
transition: "slide"
---

# Slide 1 Title

Content for slide 1

---

# Slide 2 Title

Content for slide 2

- Bullet point 1
- Bullet point 2

---

# Slide 3 Title

| Column 1 | Column 2 |
|----------|----------|
| Data 1   | Data 2   |
```

### **2. Convert to HTML**
Use the provided script:

```bash
./create_slides.sh slides.md
```

Or manually with pandoc:

```bash
pandoc slides.md -t revealjs -s -o slides.html
```

### **3. View Presentation**
Open `slides.html` in your web browser.

## 📋 **Markdown Syntax for Slides**

### **Slide Separators**
Use `---` to separate slides:

```markdown
# First Slide
Content here

---

# Second Slide
More content
```

### **Headers**
- `#` = Main slide title
- `##` = Subtitle
- `###` = Section header

### **Formatting**
- **Bold**: `**text**`
- *Italic*: `*text*`
- `Code`: `` `code` ``
- [Links](url): `[text](url)`

### **Lists**
```markdown
- Unordered list
- Another item

1. Ordered list
2. Second item
```

### **Tables**
```markdown
| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |
```

### **Images**
```markdown
![Alt text](path/to/image.png)
```

### **Code Blocks**
```markdown
```python
def hello():
    print("Hello, World!")
```
```

## 🎨 **Customization Options**

### **YAML Metadata**
Add at the top of your Markdown file:

```yaml
---
title: "Presentation Title"
subtitle: "Subtitle"
author: "Your Name"
date: "Date"
theme: "white"           # white, black, league, sky, beige, simple, serif, blood, night, moon, solarized
transition: "slide"      # none, fade, slide, convex, concave, zoom
revealjs-url: "https://cdn.jsdelivr.net/npm/reveal.js@4.3.1"
---
```

### **Available Themes**
- `white` (default)
- `black`
- `league`
- `sky`
- `beige`
- `simple`
- `serif`
- `blood`
- `night`
- `moon`
- `solarized`

### **Available Transitions**
- `none`
- `fade`
- `slide` (default)
- `convex`
- `concave`
- `zoom`

## 🛠️ **Advanced Features**

### **Speaker Notes**
Add speaker notes using HTML comments:

```markdown
# Slide Title

Content here

<!-- Speaker notes -->
<!-- These won't appear in the presentation but can be viewed in speaker mode -->
```

### **Fragments**
Use HTML for animated fragments:

```markdown
# Animated Slide

<div class="fragment fade-in">First item</div>
<div class="fragment fade-in">Second item</div>
<div class="fragment fade-in">Third item</div>
```

### **Background Images**
```markdown
# Slide with Background

<!-- .slide: data-background="image.jpg" -->
```

### **Custom CSS**
Add custom styles in the YAML metadata:

```yaml
---
css: custom.css
---
```

## 📁 **File Structure**

```
project/
├── slides.md                    # Your Markdown slides
├── slides.html                  # Generated HTML presentation
├── create_slides.sh            # Conversion script
├── SLIDES_README.md            # This documentation
└── plots/                      # Images for slides
    ├── privacy_evaluation_comprehensive.png
    └── privacy_evaluation_summary.png
```

## 🎯 **Example: Privacy Evaluation Slides**

The `slides.md` file in this project demonstrates:

- **Project overview** with objectives and methods
- **Results tables** comparing different approaches
- **Embedded images** from the plots directory
- **Key insights** and conclusions
- **Technical details** in appendix

## 🚀 **Usage Tips**

### **1. Keep Slides Simple**
- One main point per slide
- Use bullet points for clarity
- Limit text per slide

### **2. Use Visuals**
- Include charts and graphs
- Use images to illustrate concepts
- Embed code examples when relevant

### **3. Practice Navigation**
- Use arrow keys to navigate
- Press `F` for fullscreen
- Press `S` for speaker notes
- Press `ESC` for overview

### **4. Version Control**
- Keep `slides.md` in version control
- Regenerate HTML as needed
- Include both files in repository

## 🔧 **Troubleshooting**

### **Images Not Showing**
- Check file paths are relative to HTML file
- Ensure images exist in specified location
- Use absolute paths if needed

### **Styling Issues**
- Check YAML metadata syntax
- Verify theme and transition names
- Test with different browsers

### **Pandoc Errors**
- Ensure pandoc is installed: `pandoc --version`
- Check Markdown syntax
- Verify file permissions

## 📚 **Resources**

- [Pandoc Documentation](https://pandoc.org/MANUAL.html)
- [Reveal.js Documentation](https://revealjs.com/)
- [Markdown Guide](https://www.markdownguide.org/)

---

*This workflow makes it easy to create professional presentations from simple Markdown files!*
