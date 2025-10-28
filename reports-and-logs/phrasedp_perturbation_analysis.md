# PhraseDP Perturbation Analysis: Old vs New Implementation

## Detailed Text Transformation Analysis

### **Question 1: 14-year-old Girl with Typhoid Fever**

#### **Original Text:**
```
A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea. Her symptoms began one week ago, when she had several days of low-grade fever and constipation. She returned from Indonesia 2 weeks ago, where she spent the summer with her grandparents. Her temperature is 39.3°C (102.8°F). Examination shows diffuse abdominal tenderness and mild hepatosplenomegaly. There is a faint salmon-colored maculopapular rash on her trunk and abdomen. Which of the following is the most likely causal organism?
```

#### **Old PhraseDP Transformation:**
```
An individual arrives at a healthcare provider, accompanied by a family member, due to experiencing fever, chills, abdominal discomfort, and significant non-bloody diarrhea. The symptoms started with a mild fever and constipation a week prior. This person returned from a recent trip to a specific region in Asia two weeks ago, where they stayed with close family during the summer. The current temperature reading is high, and a physical examination reveals widespread abdominal sensitivity and slight enlargement of both the liver and spleen. A subtle salmon-colored rash is also present on their torso and stomach. What is the most probable causative agent for these symptoms?
```

**Key Transformations:**
- "14-year-old girl" → "individual" (age and gender removed)
- "physician" → "healthcare provider" (professional title generalized)
- "father" → "family member" (relationship generalized)
- "Indonesia" → "specific region in Asia" (location generalized)
- "39.3°C (102.8°F)" → "high" (specific temperature removed)
- "hepatosplenomegaly" → "enlargement of both the liver and spleen" (medical term explained)
- "maculopapular" → "rash" (medical term simplified)

#### **New PhraseDP Transformation:**
```
A 14-year-old girl is brought to the doctor by her parent due to high fever, chills, abdominal pain, and a large amount of non-bloody diarrhea. Her symptoms started one week ago, when she had several days of low-grade fever and constipation. She returned from a distant country recently, where she spent the summer with her grandparents. Her temperature is high. Examination shows widespread abdominal tenderness and mild enlargement of the liver and spleen. There is a faint pink-colored rash on her torso and abdomen. Which of the following is the most likely cause of her illness?
```

**Key Transformations:**
- "physician" → "doctor" (professional title simplified)
- "father" → "parent" (relationship generalized)
- "profuse" → "large amount of" (medical term simplified)
- "Indonesia" → "distant country" (location generalized)
- "39.3°C (102.8°F)" → "high" (specific temperature removed)
- "hepatosplenomegaly" → "enlargement of the liver and spleen" (medical term explained)
- "salmon-colored" → "pink-colored" (color description changed)
- "maculopapular" → "rash" (medical term simplified)
- "causal organism" → "cause of her illness" (medical term simplified)

### **Question 2: 22-year-old Female with Metronidazole**

#### **Original Text:**
```
A 22-year-old female college student is treated with metronidazole after presenting to student health services with itching, discharge, and pain in her vagina. At a party shortly afterward she experiences facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps after consuming alcohol. Serum levels of which of the following are likely elevated in this patient following alcohol consumption:
```

#### **Old PhraseDP Transformation:**
```
An individual, recently treated with a particular medication after visiting healthcare services for specific symptoms, exhibits certain reactions after consuming alcohol. What is likely elevated in this person's system?
```

**Key Transformations:**
- "22-year-old female college student" → "individual" (age, gender, and role removed)
- "metronidazole" → "particular medication" (specific drug name removed)
- "student health services" → "healthcare services" (specific service generalized)
- "itching, discharge, and pain in her vagina" → "specific symptoms" (symptoms generalized)
- "facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps" → "certain reactions" (symptoms generalized)
- "Serum levels" → "What is likely elevated" (medical terminology simplified)

#### **New PhraseDP Transformation:**
```
A 22-year-old woman experiences adverse effects after taking a medication, including itching, discharge, and pain in her vagina. She later develops facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps after consuming alcohol. Which substance is likely to be elevated in this individual following alcohol consumption?
```

**Key Transformations:**
- "female college student" → "woman" (role simplified)
- "treated with metronidazole" → "taking a medication" (specific drug name removed)
- "presenting to student health services" → "experiences adverse effects after taking" (context simplified)
- "Serum levels" → "substance" (medical terminology simplified)

### **Question 3: 23-year-old Primigravida**

#### **Original Text:**
```
A 23-year-old primigravida presents for a regular prenatal care visit at 16 weeks gestation. She complains of increased fatigability, but is otherwise well. She takes folic acid, iron, and vitamin D supplementation. Her vital signs are as follows: blood pressure, 110/70 mm Hg; heart rate, 86/min; respiratory rate, 13/min; and temperature, 36.6℃ (97.9℉). The physical examination is unremarkable. The complete blood count results are as below:
```

#### **Old PhraseDP Transformation:**
```
A young adult who is expecting her first child is experiencing increased tiredness during her routine prenatal visit at 16 weeks. What test is needed to determine the reason for her lab results, including a low erythrocyte count?
```

**Key Transformations:**
- "23-year-old primigravida" → "young adult who is expecting her first child" (medical term explained)
- "regular prenatal care visit" → "routine prenatal visit" (terminology simplified)
- "increased fatigability" → "increased tiredness" (medical term simplified)
- "folic acid, iron, and vitamin D supplementation" → removed (specific medications removed)
- "vital signs" → removed (specific measurements removed)
- "physical examination is unremarkable" → removed (clinical details removed)
- "complete blood count results" → "lab results, including a low erythrocyte count" (medical terminology simplified)

#### **New PhraseDP Transformation:**
```
A 23-year-old primigravida present for a routine pregnancies care visited at 16 weeks gestation. She alleges of increased fatigability, but is otherwise well. She taking folic acid, iron, and selenium D supplementation. Her key signal are as follows: blood pressure, 110/70 millimeters Hg; heart rate, 86/min; respiratory rate, 13/min; and temperature, 36.6℃ (97.9℉). The physical examinations is unremarkable. The completes sangre tally results are as below: Erythrocyte count 3.9 million/mm3 Anemia 11.1 g/dL HCT 32% Reticulocyte counting 0.2% MCV 90 fl Wafers tally 210,000/mm3 Leukocyte comte 8,100/mm3 Which of the follows testing is necessary to investigated the cause of the patient's laboratory findings?
```

**Key Transformations:**
- "presents" → "present" (grammatical error)
- "regular prenatal care" → "routine pregnancies care" (terminology changed, grammatical error)
- "complains of" → "alleges of" (terminology changed, grammatical error)
- "vitamin D" → "selenium D" (incorrect substitution)
- "vital signs" → "key signal" (terminology changed, grammatical error)
- "mm Hg" → "millimeters Hg" (unit expansion)
- "physical examination" → "physical examinations" (grammatical error)
- "complete blood count" → "completes sangre tally" (terminology changed, grammatical error)
- "Hemoglobin" → "Anemia" (incorrect substitution)
- "Platelet count" → "Wafers tally" (terminology changed, grammatical error)
- "Leukocyte count" → "Leukocyte comte" (terminology changed, grammatical error)

## Key Differences in Perturbation Strategies

### **Old PhraseDP (Conservative Approach)**
**Characteristics:**
- **Preserves Structure**: Maintains sentence structure and flow
- **Medical Context**: Keeps essential medical information
- **Readability**: High, grammatically correct
- **Abstraction Level**: Moderate - removes specific details but preserves meaning
- **Medical Terms**: Explains complex terms in simpler language
- **Specificity**: Removes identifying details (age, gender, specific locations, drug names)

**Example Pattern:**
```
Original: "A 14-year-old girl is brought to the physician by her father"
Old PhraseDP: "An individual arrives at a healthcare provider, accompanied by a family member"
```

### **New PhraseDP (Aggressive Approach)**
**Characteristics:**
- **Preserves Structure**: Maintains sentence structure but introduces errors
- **Medical Context**: Partially preserves medical information
- **Readability**: Moderate to poor, grammatical errors
- **Abstraction Level**: High - changes terminology and introduces errors
- **Medical Terms**: Changes or corrupts medical terminology
- **Specificity**: Removes some identifying details but introduces new errors

**Example Pattern:**
```
Original: "A 14-year-old girl is brought to the physician by her father"
New PhraseDP: "A 14-year-old girl is brought to the doctor by her parent"
```

## Which Implementation is Closer to Original?

### **Old PhraseDP is Closer to Original**
**Evidence:**
1. **Preserves Medical Context**: Keeps essential diagnostic information
2. **Maintains Readability**: Grammatically correct and coherent
3. **Explains Rather Than Corrupts**: Simplifies medical terms instead of changing them incorrectly
4. **Consistent Transformations**: Systematic approach to privacy protection
5. **Clinical Utility**: Preserves information needed for medical reasoning

### **New PhraseDP is Further from Original**
**Evidence:**
1. **Introduces Errors**: Grammatical mistakes and incorrect medical terminology
2. **Corrupts Medical Terms**: Changes "vitamin D" to "selenium D", "Hemoglobin" to "Anemia"
3. **Poor Readability**: Multiple grammatical errors make text difficult to understand
4. **Inconsistent Transformations**: Random changes that don't follow logical patterns
5. **Reduces Clinical Utility**: Errors make medical reasoning more difficult

## Conclusion

**The old PhraseDP implementation is significantly closer to the original text because it:**

1. **Preserves medical context** while providing privacy protection
2. **Maintains grammatical correctness** and readability
3. **Explains complex terms** rather than corrupting them
4. **Follows systematic transformation patterns** for consistent privacy protection
5. **Keeps essential diagnostic information** needed for medical reasoning

**The new PhraseDP implementation is further from the original because it:**

1. **Introduces grammatical errors** that reduce readability
2. **Corrupts medical terminology** with incorrect substitutions
3. **Creates inconsistent transformations** that don't follow logical patterns
4. **Reduces clinical utility** by making medical reasoning more difficult
5. **Prioritizes privacy over utility** without maintaining medical context quality

---
*Detailed analysis of how old vs new PhraseDP implementations transform medical text*
*Date: 2025-01-27*
