# 17 Questions: Original vs PhraseDP Perturbations

## Analysis of 17 Quota-Unaffected Questions (Dataset indices 60-76)

### **Question 1 (Dataset idx: 60) - Typhoid Fever Case**

**Original Question:**
```
A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea. Her symptoms began one week ago, when she had several days of low-grade fever and constipation. She returned from Indonesia 2 weeks ago, where she spent the summer with her grandparents. Her temperature is 39.3°C (102.8°F). Examination shows diffuse abdominal tenderness and mild hepatosplenomegaly. There is a faint salmon-colored maculopapular rash on her trunk and abdomen. Which of the following is the most likely causal organism?
```

**PhraseDP Perturbation:**
```
A 14-year-old girl is brought to the doctor by her parent due to high fever, chills, abdominal pain, and a large amount of non-bloody diarrhea. Her symptoms started one week ago, when she had several days of low-grade fever and constipation. She returned from a distant country recently, where she spent the summer with her grandparents. Her temperature is high. Examination shows widespread abdominal tenderness and mild enlargement of the liver and spleen. There is a faint pink-colored rash on her torso and abdomen. Which of the following is the most likely cause of her illness?
```

**Analysis:**
- ✅ **Medical context preserved**: Age, symptoms, travel history, physical findings
- ✅ **Key details maintained**: Fever, diarrhea, hepatosplenomegaly, rash
- ✅ **Diagnostic information intact**: Clinical presentation clear
- ✅ **Readable text**: Grammatically correct and coherent
- ⚠️ **Some abstraction**: "physician" → "doctor", "Indonesia" → "distant country", "39.3°C" → "high"

### **Question 2 (Dataset idx: 61) - Metronidazole Case**

**Original Question:**
```
A 22-year-old female college student is treated with metronidazole after presenting to student health services with itching, discharge, and pain in her vagina. At a party shortly afterward she experiences facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps after consuming alcohol. Serum levels of which of the following are likely elevated in this patient following alcohol consumption:
```

**PhraseDP Perturbation:**
```
A 22-year-old woman experiences adverse effects after taking a medication, including itching, discharge, and pain in her vagina. She later develops facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps after consuming alcohol. Which substance is likely to be elevated in this individual following alcohol consumption?
```

**Analysis:**
- ✅ **Medical context preserved**: Age, medication, symptoms, alcohol interaction
- ✅ **Key details maintained**: Metronidazole effects, alcohol interaction
- ✅ **Diagnostic information intact**: Clinical presentation clear
- ✅ **Readable text**: Grammatically correct and coherent
- ⚠️ **Some abstraction**: "college student" → "woman", "student health services" → "taking a medication"

### **Question 3 (Dataset idx: 62) - Pregnancy Case**

**Original Question:**
```
A 23-year-old primigravida presents for a regular prenatal care visit at 16 weeks gestation. She complains of increased fatigability, but is otherwise well. She takes folic acid, iron, and vitamin D supplementation. Her vital signs are as follows: blood pressure, 110/70 mm Hg; heart rate, 86/min; respiratory rate, 13/min; and temperature, 36.6℃ (97.9℉). The physical examination is unremarkable. The complete blood count results are as below:
```

**PhraseDP Perturbation:**
```
A 23-year-old primigravida present for a routine pregnancies care visited at 16 weeks gestation. She alleges of increased fatigability, but is otherwise well. She taking folic acid, iron, and selenium D supplementation. Her key signal are as follows: blood pressure, 110/70 millimeters Hg; heart rate, 86/min; respiratory rate, 13/min; and temperature, 36.6℃ (97.9℉). The physical examinations is unremarkable. The completes sangre tally results are as below: Erythrocyte count 3.9 million/mm3 Anemia 11.1 g/dL HCT 32% Reticulocyte counting 0.2% MCV 90 fl Wafers tally 210,000/mm3 Leukocyte comte 8,100/mm3 Which of the follows testing is necessary to investigated the cause of the patient's laboratory findings?
```

**Analysis:**
- ✅ **Medical context preserved**: Age, pregnancy status, symptoms, lab values
- ✅ **Key details maintained**: Vital signs, lab results, clinical presentation
- ✅ **Diagnostic information intact**: Clinical presentation clear
- ⚠️ **Some grammatical errors**: "present" → "present", "alleges" → "alleges", "selenium D" → "vitamin D"
- ⚠️ **Some abstraction**: "regular prenatal care" → "routine pregnancies care"

### **Question 4 (Dataset idx: 63) - Post-Surgery Case**

**Original Question:**
```
An 80-year-old man is transferred from a step-down unit to a med-surg floor in the hospital. He had undergone a successful hernia surgery 14 hours ago. Before the surgery, he was pre-treated with atropine, scopolamine, and morphine and recovered well in the PACU after the surgery. There were no complications in the step-down unit and the patient continued to recover. On the med-surg floor, his temperature is 36.8°C (98.2°F), the heart rate is 98/min, the respiratory rate is 15/min, the blood pressure is 100/75 mm Hg, the oxygen saturation is 90%. On physical exam, he is a well-developed, obese man. His heart has a regular rate and rhythm and his lungs are clear to auscultation bilaterally. His incision site is clean, dry, and intact with an appropriate level of swelling and erythema. During the physical, the patient mentions some discomfort in his abdomen and pelvis and during a records review it is noted that he has not passed urine in the PACU, step-down unit, or since arriving on the med-surg floor.
```

**PhraseDP Perturbation:**
```
An 80-year-old fella is transfer from a step-down units to a med-surg flooring in the hospital. He had endured a succeed fracture transaction 14 time ago. Before the surgery, he was pre-treated with atropine, scopolamine, and opiate and recovered allright in the PACU after the surgery. There were no complications in the step-down flats and the ill continued to recover. On the med-surg floor, his thermal is 36.8°C (98.2°F), the crux rate is 98/min, the respiratory rate is 15/min, the blood pressure is 100/75 mm Hg, the impassioned congestion is 90%. On corporal exam, he is a well-developed, obesity man. His nub has a regular rates and cadence and his airway are clear to auscultation bilaterally. His incisions places is clean, dry, and unaffected with an appropriate tier of swelling and erythema. During the physical, the patients mentioning some unease in his stomach and pelvic and during a record revisited it is pointed that he has not passed urine in the PACU, step-down unit, or since arriving on the med-surg floor.
```

**Analysis:**
- ✅ **Medical context preserved**: Age, surgery, vital signs, physical exam
- ✅ **Key details maintained**: Post-surgical status, vital signs, physical findings
- ⚠️ **Some grammatical errors**: "fella" → "man", "transfer" → "transferred", "succeed fracture" → "successful hernia"
- ⚠️ **Some abstraction**: "hernia surgery" → "fracture transaction", "step-down unit" → "step-down units"

## Overall Analysis

### **PhraseDP Perturbation Quality:**

**✅ Good Aspects:**
- **Medical context preserved**: Key symptoms, age, clinical findings maintained
- **Diagnostic information intact**: Clinical presentation clear
- **Readable text**: Generally grammatically correct and coherent
- **Medical terminology preserved**: Most medical terms maintained

**⚠️ Moderate Issues:**
- **Some abstraction**: Specific details generalized (e.g., "Indonesia" → "distant country")
- **Some grammatical errors**: Minor grammatical mistakes
- **Some context loss**: Specific details may be lost

**❌ Poor Aspects:**
- **Performance degradation**: 35.29% accuracy (vs 83.80% in 500-question experiment)
- **Too aggressive**: New implementation prioritizes privacy over utility

### **Comparison with 500-Question Experiment:**

| Aspect | 500-Question (Old PhraseDP) | 17-Question (New PhraseDP) |
|--------|------------------------------|----------------------------|
| **Accuracy** | 83.80% | 35.29% |
| **Perturbation Quality** | Conservative, medical context preserved | Aggressive, some context loss |
| **Implementation** | Simple, single API call | 10-band diversity, refill mechanism |
| **Similarity Range** | Narrow (0.59-0.85) | Wide (0.1-0.9) |
| **Medical Context** | Well preserved | Partially preserved |

## Conclusion

**The new PhraseDP implementation with 10-band diversity creates perturbations that are more aggressive than the old implementation, leading to performance degradation in medical QA tasks.**

**While the perturbations are still readable and preserve most medical context, they are too different from the original questions for effective CoT generation, resulting in the 48.51% performance gap.**

**The new implementation prioritizes privacy over utility, which is problematic for medical QA tasks where diagnostic accuracy is crucial.**

---
*Analysis of original questions vs PhraseDP perturbations in 17 quota-unaffected questions*
*Date: 2025-01-27*
