# Candidate Diversity Analysis

Date: 2025-09-22 03:38:57

## Global coverage (percent by band)
Old: {'0.0–0.3': '1.9%', '0.3–0.5': '29.0%', '0.5–0.7': '34.6%', '0.7–0.8': '22.5%', '0.8–0.9': '9.9%', '0.9–1.0': '2.0%'}
New: {'0.0–0.3': '6.1%', '0.3–0.5': '26.8%', '0.5–0.7': '29.2%', '0.7–0.8': '21.9%', '0.8–0.9': '12.9%', '0.9–1.0': '3.1%'}

## Per-question thin-band flags (new method)
- Q0: What is the capital of France?
  - new n=80, range=0.094–0.901
  - bands: {'0.0–0.3': 6, '0.3–0.5': 23, '0.5–0.7': 27, '0.7–0.8': 16, '0.8–0.9': 7, '0.9–1.0': 1}
  - flag: low 0.0–0.3 thin

- Q1: What is the largest country in the world?
  - new n=79, range=0.295–0.936
  - bands: {'0.0–0.3': 1, '0.3–0.5': 11, '0.5–0.7': 27, '0.7–0.8': 25, '0.8–0.9': 10, '0.9–1.0': 5}
  - flag: low 0.0–0.3 thin

- Q2: Which ocean is the largest?
  - new n=63, range=0.487–0.939
  - bands: {'0.0–0.3': 0, '0.3–0.5': 1, '0.5–0.7': 12, '0.7–0.8': 31, '0.8–0.9': 15, '0.9–1.0': 4}
  - flag: low 0.0–0.3 thin, low-mid 0.3–0.5 thin

- Q3: What is the longest river in the world?
  - new n=68, range=0.325–0.939
  - bands: {'0.0–0.3': 0, '0.3–0.5': 6, '0.5–0.7': 11, '0.7–0.8': 29, '0.8–0.9': 11, '0.9–1.0': 11}
  - flag: low 0.0–0.3 thin, low-mid 0.3–0.5 thin

- Q4: In which year did World War II end?
  - new n=62, range=0.234–0.842
  - bands: {'0.0–0.3': 3, '0.3–0.5': 24, '0.5–0.7': 25, '0.7–0.8': 5, '0.8–0.9': 5, '0.9–1.0': 0}
  - flag: low 0.0–0.3 thin

- Q5: Who was the first president of the United States?
  - new n=75, range=0.085–0.853
  - bands: {'0.0–0.3': 10, '0.3–0.5': 25, '0.5–0.7': 25, '0.7–0.8': 9, '0.8–0.9': 6, '0.9–1.0': 0}

- Q6: When did the Berlin Wall fall?
  - new n=57, range=0.196–0.827
  - bands: {'0.0–0.3': 12, '0.3–0.5': 34, '0.5–0.7': 6, '0.7–0.8': 4, '0.8–0.9': 1, '0.9–1.0': 0}

- Q7: What year did the Titanic sink?
  - new n=65, range=0.305–0.856
  - bands: {'0.0–0.3': 0, '0.3–0.5': 20, '0.5–0.7': 35, '0.7–0.8': 9, '0.8–0.9': 1, '0.9–1.0': 0}
  - flag: low 0.0–0.3 thin

- Q8: What is the chemical symbol for gold?
  - new n=81, range=0.173–0.889
  - bands: {'0.0–0.3': 1, '0.3–0.5': 15, '0.5–0.7': 19, '0.7–0.8': 19, '0.8–0.9': 27, '0.9–1.0': 0}
  - flag: low 0.0–0.3 thin

- Q9: What is the speed of light?
  - new n=78, range=0.157–0.913
  - bands: {'0.0–0.3': 10, '0.3–0.5': 31, '0.5–0.7': 20, '0.7–0.8': 8, '0.8–0.9': 8, '0.9–1.0': 1}

## Recommendations
- Increase returns for 0.1–0.3 and 0.3–0.5; keep refill-on until targets met (e.g., 30/30).
- Add de-dup by SBERT (e.g., cosine > 0.92 considered duplicate) to avoid cluster saturation.
- Provide explicit low-sim examples in prompt; strengthen constraints (replace entities, time, place, domain).
- For stubborn questions, run a second-stage generator seeded by the least similar candidates to explore farther.
