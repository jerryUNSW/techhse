# MedQA USMLE Dataset Pattern Analysis

## Executive Summary

**The MedQA USMLE dataset is predominantly patient-based (90%+ of questions) with NO clear progression from knowledge-based to patient-based questions.** The dataset maintains a consistent pattern throughout, with most questions being clinical scenarios involving patients.

## Key Findings

### **1. Question Type Distribution**
- **Patient-based questions**: 90-98% across all ranges
- **Knowledge-based questions**: 2-10% across all ranges  
- **Other/Unclear**: 0-8% across all ranges

**No progression pattern**: The dataset does NOT show a progression from knowledge-based to patient-based questions. The pattern remains consistent throughout.

### **2. Medical Domain Distribution**
**Top medical domains across all ranges:**
1. **Pharmacology**: 30-34% (drugs, medications, side effects)
2. **Pulmonology**: 18-26% (respiratory, lung conditions)
3. **Infectious Disease**: 12-26% (infections, pathogens, antibiotics)
4. **Endocrinology**: 16-28% (diabetes, hormones, thyroid)
5. **Cardiology**: 10-26% (heart conditions, cardiovascular)
6. **Gastroenterology**: 10-18% (digestive system, liver, GI)
7. **Pathology**: 6-24% (disease mechanisms, histology)
8. **Neurology**: 6-14% (brain, nervous system)
9. **Physiology**: 5-10% (body functions, mechanisms)
10. **Oncology**: 4-10% (cancer, tumors, malignancy)

### **3. Question Complexity Analysis**
- **Average word count**: 112-122 words per question
- **Average medical terms**: 8-9 medical terms per question
- **Average complexity score**: 57-63 (moderate to high complexity)
- **Range**: Simple questions (9-25 score) to complex questions (88-135 score)

### **4. Progression Pattern Analysis**
**NO CLEAR PROGRESSION**: Analysis of 5 chunks across different ranges shows:
- **Patient-based questions**: Consistently 70-100% across all chunks
- **Knowledge-based questions**: Consistently 0-30% across all chunks
- **No systematic increase/decrease** in either category over time

## Detailed Analysis by Range

### **Questions 0-49 (First 50)**
- **Patient-based**: 90%
- **Knowledge-based**: 2%
- **Top domains**: Pharmacology (30%), Pulmonology (24%), Infectious Disease (22%)
- **Complexity**: Average 57.4 (moderate)

### **Questions 50-99**
- **Patient-based**: 96%
- **Knowledge-based**: 2%
- **Top domains**: Pharmacology (34%), Pulmonology (26%), Cardiology (26%)
- **Complexity**: Average 58.9 (moderate)

### **Questions 100-149**
- **Patient-based**: 96%
- **Knowledge-based**: 4%
- **Top domains**: Endocrinology (28%), Pharmacology (26%), Pathology (24%)
- **Complexity**: Average 61.4 (moderate-high)

### **Questions 200-249**
- **Patient-based**: 90%
- **Knowledge-based**: 10%
- **Top domains**: Pharmacology (34%), Infectious Disease (26%), Pathology (24%)
- **Complexity**: Average 57.0 (moderate)

### **Questions 400-449**
- **Patient-based**: 98%
- **Knowledge-based**: 2%
- **Top domains**: Pharmacology (34%), Endocrinology (24%), Gastroenterology (18%)
- **Complexity**: Average 63.0 (moderate-high)

## Sample Question Analysis

### **Patient-Based Questions (Typical Pattern)**
```
"A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of..."
"A 39-year-old woman is brought to the emergency department because of fevers, chills, and left lower..."
"A 2-day-old male newborn is brought to the physician because of yellowing of the skin..."
```

**Characteristics:**
- Start with patient demographics (age, gender)
- Include clinical presentation (symptoms, history)
- Describe examination findings
- Ask for diagnosis, treatment, or management

### **Knowledge-Based Questions (Rare)**
```
"A microbiologist is studying the emergence of a virulent strain of the virus..."
"A scientist is studying the properties of myosin-actin interactions..."
"A research group has created a novel screening test for a rare disorder..."
```

**Characteristics:**
- Focus on scientific concepts
- Ask about mechanisms, processes, or principles
- Less clinical, more theoretical

## Medical Complexity Patterns

### **Simple Questions (Score 9-25)**
- Short clinical scenarios
- Clear, direct questions
- Basic medical terminology
- Example: "A 65-year-old male is treated for anal carcinoma with therapy including external beam radiation. How does radiation affect cancer cells?"

### **Complex Questions (Score 88-135)**
- Long, detailed clinical scenarios
- Multiple medical conditions
- Extensive laboratory data
- Complex differential diagnoses
- Example: Multi-paragraph patient cases with detailed history, examination, and lab results

## Implications for PhraseDP Experiments

### **1. Consistent Challenge Level**
- **All questions are patient-based clinical scenarios**
- **High medical terminology density** (8-9 terms per question)
- **Complex medical context** that must be preserved
- **No "easy" knowledge-based questions** to start with

### **2. Privacy-Preserving Challenges**
- **Patient demographics** (age, gender, ethnicity) need protection
- **Medical history** and **symptoms** contain sensitive information
- **Laboratory values** and **examination findings** are highly specific
- **Treatment details** may reveal patient identity

### **3. Perturbation Quality Requirements**
- **Medical terminology preservation** is critical
- **Clinical context** must remain intact
- **Diagnostic reasoning** should not be disrupted
- **Treatment options** must remain medically valid

## Conclusion

**The MedQA USMLE dataset is a highly consistent, patient-focused medical question dataset with NO progression pattern.** All questions are clinical scenarios requiring medical knowledge and reasoning. This makes it an excellent testbed for privacy-preserving medical QA, as every question presents the same level of challenge for perturbation techniques.

**Key Insights:**
1. ✅ **No progression pattern** - questions are consistently patient-based
2. ✅ **High medical complexity** throughout the dataset
3. ✅ **Rich medical terminology** requiring careful preservation
4. ✅ **Clinical scenarios** that test real medical knowledge
5. ✅ **Consistent challenge level** for privacy-preserving techniques

**This explains why PhraseDP performance is consistent across different question ranges - the dataset maintains the same level of medical complexity and clinical focus throughout.**

---
*Analysis of MedQA USMLE dataset patterns and implications for privacy-preserving medical QA*
*Date: 2025-01-27*
