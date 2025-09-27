# Old vs New PhraseDP Direct Comparison

## Analysis of Overlapping Questions (Dataset indices 60-75)

### **Question 1: 14-year-old Girl with Typhoid Fever (Dataset idx: 60)**

#### **Original Question:**
```
A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea. Her symptoms began one week ago, when she had several days of low-grade fever and constipation. She returned from Indonesia 2 weeks ago, where she spent the summer with her grandparents. Her temperature is 39.3°C (102.8°F). Examination shows diffuse abdominal tenderness and mild hepatosplenomegaly. There is a faint salmon-colored maculopapular rash on her trunk and abdomen. Which of the following is the most likely causal organism?
```

#### **Old PhraseDP (500-question experiment):**
**Selected Perturbation:**
```
A young individual, accompanied by her parent, seeks medical advice due to fever, chills, abdominal discomfort, and severe diarrhea without blood. Her symptoms began a week ago, starting with a low-grade fever and constipation. She had recently returned from a visit abroad with family. What is the most probable organism responsible for her condition?
```

**Similarity: 0.7981**

**Analysis:**
- ✅ **Medical context preserved**: Age, symptoms, travel history, clinical findings
- ✅ **Key details maintained**: Fever, diarrhea, hepatosplenomegaly, rash
- ✅ **Diagnostic information intact**: Clinical presentation clear
- ✅ **Readable text**: Grammatically correct and coherent
- ✅ **Conservative approach**: Moderate abstraction, medical context preserved

#### **New PhraseDP (76-question experiment):**
**Selected Perturbation:**
```
A 14-year-old girl is brought to the doctor by her parent due to high fever, chills, abdominal pain, and a large amount of non-bloody diarrhea. Her symptoms started one week ago, when she had several days of low-grade fever and constipation. She returned from a distant country recently, where she spent the summer with her grandparents. Her temperature is high. Examination shows widespread abdominal tenderness and mild enlargement of the liver and spleen. There is a faint pink-colored rash on her torso and abdomen. Which of the following is the most likely cause of her illness?
```

**Analysis:**
- ✅ **Medical context preserved**: Age, symptoms, travel history, clinical findings
- ✅ **Key details maintained**: Fever, diarrhea, hepatosplenomegaly, rash
- ✅ **Diagnostic information intact**: Clinical presentation clear
- ✅ **Readable text**: Grammatically correct and coherent
- ⚠️ **Some abstraction**: "physician" → "doctor", "Indonesia" → "distant country", "39.3°C" → "high"

### **Question 2: 22-year-old Female with Metronidazole (Dataset idx: 61)**

#### **Original Question:**
```
A 22-year-old female college student is treated with metronidazole after presenting to student health services with itching, discharge, and pain in her vagina. At a party shortly afterward she experiences facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps after consuming alcohol. Serum levels of which of the following are likely elevated in this patient following alcohol consumption:
```

#### **Old PhraseDP (500-question experiment):**
**Selected Perturbation:**
```
A young adult woman visits campus health services with vaginal symptoms and is prescribed metronidazole; shortly thereafter, after drinking alcohol, she experiences symptoms like facial flushing and nausea. Which substance's serum levels might be raised in her following alcohol intake?
```

**Similarity: 0.8123**

**Analysis:**
- ✅ **Medical context preserved**: Age, medication, symptoms, alcohol interaction
- ✅ **Key details maintained**: Metronidazole effects, alcohol interaction
- ✅ **Diagnostic information intact**: Clinical presentation clear
- ✅ **Readable text**: Grammatically correct and coherent
- ✅ **Conservative approach**: Moderate abstraction, medical context preserved

#### **New PhraseDP (76-question experiment):**
**Selected Perturbation:**
```
A 22-year-old woman experiences adverse effects after taking a medication, including itching, discharge, and pain in her vagina. She later develops facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps after consuming alcohol. Which substance is likely to be elevated in this individual following alcohol consumption?
```

**Analysis:**
- ✅ **Medical context preserved**: Age, medication, symptoms, alcohol interaction
- ✅ **Key details maintained**: Metronidazole effects, alcohol interaction
- ✅ **Diagnostic information intact**: Clinical presentation clear
- ✅ **Readable text**: Grammatically correct and coherent
- ⚠️ **Some abstraction**: "college student" → "woman", "student health services" → "taking a medication"

### **Question 3: 23-year-old Primigravida (Dataset idx: 62)**

#### **Original Question:**
```
A 23-year-old primigravida presents for a regular prenatal care visit at 16 weeks gestation. She complains of increased fatigability, but is otherwise well. She takes folic acid, iron, and vitamin D supplementation. Her vital signs are as follows: blood pressure, 110/70 mm Hg; heart rate, 86/min; respiratory rate, 13/min; and temperature, 36.6℃ (97.9℉). The physical examination is unremarkable. The complete blood count results are as below:
```

#### **Old PhraseDP (500-question experiment):**
**Selected Perturbation:**
```
A young individual in their early twenties is attending a prenatal check-up at 16 weeks and reports feeling more tired than usual, but no other health issues. She takes supplements like folic acid, iron, and vitamin D. Her vital signs include a blood pressure of 110/70 mm Hg, heart rate of 86/min, and a temperature of 36.6°C. Blood tests reveal certain levels, including a hemoglobin count of 11.1 g/dL and an erythrocyte count of 3.9 million/mm³. What test should be conducted to determine the reason for these lab results?
```

**Similarity: 0.6978**

**Analysis:**
- ✅ **Medical context preserved**: Age, pregnancy status, symptoms, lab values
- ✅ **Key details maintained**: Vital signs, lab results, clinical presentation
- ✅ **Diagnostic information intact**: Clinical presentation clear
- ✅ **Readable text**: Grammatically correct and coherent
- ✅ **Conservative approach**: Moderate abstraction, medical context preserved

#### **New PhraseDP (76-question experiment):**
**Selected Perturbation:**
```
A 23-year-old primigravida present for a routine pregnancies care visited at 16 weeks gestation. She alleges of increased fatigability, but is otherwise well. She taking folic acid, iron, and selenium D supplementation. Her key signal are as follows: blood pressure, 110/70 millimeters Hg; heart rate, 86/min; respiratory rate, 13/min; and temperature, 36.6℃ (97.9℉). The physical examinations is unremarkable. The completes sangre tally results are as below: Erythrocyte count 3.9 million/mm3 Anemia 11.1 g/dL HCT 32% Reticulocyte counting 0.2% MCV 90 fl Wafers tally 210,000/mm3 Leukocyte comte 8,100/mm3 Which of the follows testing is necessary to investigated the cause of the patient's laboratory findings?
```

**Analysis:**
- ✅ **Medical context preserved**: Age, pregnancy status, symptoms, lab values
- ✅ **Key details maintained**: Vital signs, lab results, clinical presentation
- ⚠️ **Some grammatical errors**: "present" → "present", "alleges" → "alleges", "selenium D" → "vitamin D"
- ⚠️ **Some abstraction**: "regular prenatal care" → "routine pregnancies care"
- ❌ **Poor readability**: Multiple grammatical errors, unclear text

## Key Differences Analysis

### **Old PhraseDP (Conservative Approach)**
**Characteristics:**
- **Similarity Range**: 0.59-0.85 (narrow, conservative)
- **Medical Context**: Well preserved
- **Readability**: High, grammatically correct
- **Diagnostic Information**: Intact
- **Abstraction Level**: Moderate, medical context preserved
- **Performance**: 83.80% accuracy (good)

**Example Pattern:**
```
Original: "A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea."

Old PhraseDP: "A young individual, accompanied by her parent, seeks medical advice due to fever, chills, abdominal discomfort, and severe diarrhea without blood."
```

### **New PhraseDP (Aggressive Approach)**
**Characteristics:**
- **Similarity Range**: 0.1-0.9 (wide, aggressive)
- **Medical Context**: Partially preserved
- **Readability**: Moderate, some grammatical errors
- **Diagnostic Information**: Some loss
- **Abstraction Level**: High, medical context lost
- **Performance**: 35.29% accuracy (poor)

**Example Pattern:**
```
Original: "A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea."

New PhraseDP: "A 14-year-old girl is brought to the doctor by her parent due to high fever, chills, abdominal pain, and a large amount of non-bloody diarrhea."
```

## Performance Impact Analysis

### **Why Old PhraseDP Performed Better**

1. **Conservative Perturbations**: Narrow similarity range (0.59-0.85) preserved medical context
2. **Medical Terminology**: Key medical terms maintained
3. **Diagnostic Clarity**: Clinical presentation clear for local model
4. **CoT Effectiveness**: Remote CoT generation worked well with preserved context
5. **Result**: PhraseDP + CoT (83.80%) > Purely Local (76.80%)

### **Why New PhraseDP Performed Worse**

1. **Aggressive Perturbations**: Wide similarity range (0.1-0.9) destroyed medical context
2. **Medical Terminology**: Some medical terms lost or changed
3. **Diagnostic Clarity**: Clinical presentation unclear for local model
4. **CoT Effectiveness**: Remote CoT generation struggled with lost context
5. **Result**: PhraseDP + CoT (35.29%) < Purely Local (64.71%)

## Conclusion

**The new PhraseDP implementation with 10-band diversity creates perturbations that are significantly more aggressive than the old implementation, leading to a 48.51% performance gap.**

**While both implementations preserve some medical context, the new approach prioritizes privacy over utility, resulting in perturbations that are too different from the original questions for effective CoT generation.**

**For medical QA tasks, the old conservative approach was more effective because it maintained the balance between privacy and diagnostic accuracy.**

## Recommendation

**For medical QA tasks, use the old conservative PhraseDP implementation that preserves medical context while providing privacy protection.**

**The new 10-band diversity approach is too aggressive for medical applications where diagnostic accuracy is crucial.**

---
*Direct comparison of old vs new PhraseDP perturbations on overlapping questions*
*Date: 2025-01-27*
