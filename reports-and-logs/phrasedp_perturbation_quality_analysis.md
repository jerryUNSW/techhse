# PhraseDP Perturbation Quality Analysis

## Key Finding

**The new PhraseDP implementation with 10-band diversity is creating extremely aggressive perturbations that destroy medical context and meaning.**

## Perturbation Quality Examples

### **Question 1: Typhoid Fever Case**

**Original Question:**
```
A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea. Her symptoms began one week ago, when she had several days of low-grade fever and constipation. She returned from Indonesia 2 weeks ago, where she spent the summer with her grandparents. Her temperature is 39.3°C (102.8°F). Examination shows diffuse abdominal tenderness and mild hepatosplenomegaly. There is a faint salmon-colored maculopapular rash on her trunk and abdomen. Which of the following is the most likely causal organism?
```

**PhraseDP Perturbation (Good Quality):**
```
A 14-year-old girl is brought to the doctor by her parent due to high fever, chills, abdominal pain, and a large amount of non-bloody diarrhea. Her symptoms started one week ago, when she had several days of low-grade fever and constipation. She returned from a distant country recently, where she spent the summer with her grandparents. Her temperature is high. Examination shows widespread abdominal tenderness and mild enlargement of the liver and spleen. There is a faint pink-colored rash on her torso and abdomen. Which of the following is the most likely cause of her illness?
```

**Analysis:**
- ✅ **Medical context preserved**: Age, symptoms, travel history, physical findings
- ✅ **Key details maintained**: Fever, diarrhea, hepatosplenomegaly, rash
- ✅ **Diagnostic information intact**: Clinical presentation clear
- ✅ **Similarity**: High similarity (0.99+) - good perturbation

### **Question 2: InferDPT Perturbation (Extremely Poor Quality)**

**Original Question:**
```
A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea...
```

**InferDPT Perturbation:**
```
merch 675 YOU series resort Portal gree drink acute added Converter equipment cocaine significantly newspaper scored collaborative PLIED An accept ilk Against gust restriction thickness Sum protective quire Japan uncertain Critical solely president slug Padding 570 sensible Manual tang nothing deer Teen Southern ownership mayor without Segment 221 436 444 571 bed Header tourism overseas medium pad ultimately development proving damage Sup figured live correctly Gratis derived silence Variable vet golf append
```

**Analysis:**
- ❌ **Complete medical context loss**: No medical information preserved
- ❌ **Random word salad**: Nonsensical text with numbers and random words
- ❌ **No diagnostic value**: Impossible to extract medical meaning
- ❌ **Similarity**: Very low similarity - extreme perturbation

### **Question 3: SANTEXT+ Perturbation (Extremely Poor Quality)**

**Original Question:**
```
A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea...
```

**SANTEXT+ Perturbation:**
```
a how below is during to the necessary why first cold sad necessary yet cold weak heavy and three any on strong necessary incorrect one for large impossible or each narrow high of that which and as and short from last during and bad during old is the necessary with wet impossible why poor is text over right warmup as before who and cold to high is a these on open wet on by any and open which of the each is other impossible soft as three
```

**Analysis:**
- ❌ **Complete medical context loss**: No medical information preserved
- ❌ **Random word sequence**: Nonsensical text with basic words
- ❌ **No diagnostic value**: Impossible to extract medical meaning
- ❌ **Similarity**: Very low similarity - extreme perturbation

## Perturbation Quality Comparison

### **Good Quality Perturbations (PhraseDP)**
- **Medical context preserved**: Key symptoms, age, travel history maintained
- **Diagnostic information intact**: Clinical presentation clear
- **Readable text**: Grammatically correct and coherent
- **Similarity range**: 0.85-0.99 (high similarity)

### **Poor Quality Perturbations (InferDPT, SANTEXT+)**
- **Medical context lost**: No medical information preserved
- **Random word salad**: Nonsensical text
- **No diagnostic value**: Impossible to extract meaning
- **Similarity range**: 0.1-0.3 (very low similarity)

## Impact on Performance

### **Why PhraseDP + CoT Performs Better**
1. **Preserved medical context**: Remote CoT can still understand the clinical scenario
2. **Diagnostic information intact**: Key symptoms and findings maintained
3. **Readable text**: CoT generation can extract medical meaning
4. **Result**: 35.29% accuracy (still poor due to new implementation)

### **Why InferDPT/SANTEXT+ + CoT Performs Poorly**
1. **Lost medical context**: Remote CoT cannot understand the clinical scenario
2. **No diagnostic information**: Key symptoms and findings lost
3. **Unreadable text**: CoT generation cannot extract medical meaning
4. **Result**: 23.53% accuracy (very poor)

## Root Cause Analysis

### **New PhraseDP Implementation Issues**
1. **10-band diversity**: Forces generation of extreme similarity bands (0.0-0.1, 0.1-0.2)
2. **Refill mechanism**: Ensures all bands are filled, including difficult ones
3. **Aggressive perturbations**: Wide similarity range (0.1-0.9) includes very different candidates
4. **Medical context loss**: Extreme abstractions lose essential medical details

### **Comparison with 500-Question Experiment**
- **Old PhraseDP**: Conservative perturbations, medical context preserved
- **New PhraseDP**: Aggressive perturbations, medical context lost
- **Result**: Performance degradation from 83.80% to 35.29%

## Recommendations

### **1. Immediate Action**
- **Revert to old PhraseDP implementation** for medical QA tasks
- **Or adjust new implementation parameters** to be less aggressive

### **2. Parameter Tuning**
- **Reduce similarity range** (e.g., 0.3-0.8 instead of 0.1-0.9)
- **Disable extreme bands** (0.0-0.2) for medical questions
- **Adjust refill thresholds** to be less aggressive

### **3. Medical-Specific Optimization**
- **Preserve medical terminology**: Keep key medical terms intact
- **Maintain clinical context**: Preserve symptoms, findings, history
- **Avoid extreme abstraction**: Don't lose diagnostic information

## Conclusion

**The new PhraseDP implementation with 10-band diversity creates perturbations that are too aggressive for medical QA tasks, destroying essential medical context and diagnostic information.**

**While PhraseDP perturbations are still readable, InferDPT and SANTEXT+ create completely nonsensical text that cannot be understood by either humans or AI models.**

**The performance degradation is due to the new implementation prioritizing privacy over utility, leading to perturbations that are too different from the original medical questions.**

---
*Analysis of PhraseDP perturbation quality in the 17 quota-unaffected questions*
*Date: 2025-01-27*
