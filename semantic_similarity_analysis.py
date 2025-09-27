#!/usr/bin/env python3
"""
Compute semantic similarity between original text and PhraseDP perturbations
to quantitatively determine which implementation stays closer to the original.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def compute_semantic_similarity():
    """
    Compute semantic similarity between original text and PhraseDP perturbations.
    """
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Question 1: 14-year-old Girl with Typhoid Fever
    original_1 = """A 14-year-old girl is brought to the physician by her father because of fever, chills, abdominal pain, and profuse non-bloody diarrhea. Her symptoms began one week ago, when she had several days of low-grade fever and constipation. She returned from Indonesia 2 weeks ago, where she spent the summer with her grandparents. Her temperature is 39.3°C (102.8°F). Examination shows diffuse abdominal tenderness and mild hepatosplenomegaly. There is a faint salmon-colored maculopapular rash on her trunk and abdomen. Which of the following is the most likely causal organism?"""
    
    old_phrasedp_1 = """An individual arrives at a healthcare provider, accompanied by a family member, due to experiencing fever, chills, abdominal discomfort, and significant non-bloody diarrhea. The symptoms started with a mild fever and constipation a week prior. This person returned from a recent trip to a specific region in Asia two weeks ago, where they stayed with close family during the summer. The current temperature reading is high, and a physical examination reveals widespread abdominal sensitivity and slight enlargement of both the liver and spleen. A subtle salmon-colored rash is also present on their torso and stomach. What is the most probable causative agent for these symptoms?"""
    
    new_phrasedp_1 = """A 14-year-old girl is brought to the doctor by her parent due to high fever, chills, abdominal pain, and a large amount of non-bloody diarrhea. Her symptoms started one week ago, when she had several days of low-grade fever and constipation. She returned from a distant country recently, where she spent the summer with her grandparents. Her temperature is high. Examination shows widespread abdominal tenderness and mild enlargement of the liver and spleen. There is a faint pink-colored rash on her torso and abdomen. Which of the following is the most likely cause of her illness?"""
    
    # Question 2: 22-year-old Female with Metronidazole
    original_2 = """A 22-year-old female college student is treated with metronidazole after presenting to student health services with itching, discharge, and pain in her vagina. At a party shortly afterward she experiences facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps after consuming alcohol. Serum levels of which of the following are likely elevated in this patient following alcohol consumption:"""
    
    old_phrasedp_2 = """An individual, recently treated with a particular medication after visiting healthcare services for specific symptoms, exhibits certain reactions after consuming alcohol. What is likely elevated in this person's system?"""
    
    new_phrasedp_2 = """A 22-year-old woman experiences adverse effects after taking a medication, including itching, discharge, and pain in her vagina. She later develops facial flushing, nausea, tachycardia, dyspnea, headache, and abdominal cramps after consuming alcohol. Which substance is likely to be elevated in this individual following alcohol consumption?"""
    
    # Question 3: 23-year-old Primigravida
    original_3 = """A 23-year-old primigravida presents for a regular prenatal care visit at 16 weeks gestation. She complains of increased fatigability, but is otherwise well. She takes folic acid, iron, and vitamin D supplementation. Her vital signs are as follows: blood pressure, 110/70 mm Hg; heart rate, 86/min; respiratory rate, 13/min; and temperature, 36.6℃ (97.9℉). The physical examination is unremarkable. The complete blood count results are as below:"""
    
    old_phrasedp_3 = """A young adult who is expecting her first child is experiencing increased tiredness during her routine prenatal visit at 16 weeks. What test is needed to determine the reason for her lab results, including a low erythrocyte count?"""
    
    new_phrasedp_3 = """A 23-year-old primigravida present for a routine pregnancies care visited at 16 weeks gestation. She alleges of increased fatigability, but is otherwise well. She taking folic acid, iron, and selenium D supplementation. Her key signal are as follows: blood pressure, 110/70 millimeters Hg; heart rate, 86/min; respiratory rate, 13/min; and temperature, 36.6℃ (97.9℉). The physical examinations is unremarkable. The completes sangre tally results are as below: Erythrocyte count 3.9 million/mm3 Anemia 11.1 g/dL HCT 32% Reticulocyte counting 0.2% MCV 90 fl Wafers tally 210,000/mm3 Leukocyte comte 8,100/mm3 Which of the follows testing is necessary to investigated the cause of the patient's laboratory findings?"""
    
    # Question 4: 80-year-old Man Post-Surgery
    original_4 = """An 80-year-old man is transferred from a step-down unit to a med-surg floor in the hospital. He had undergone a successful hernia surgery 14 hours ago. Before the surgery, he was pre-treated with atropine, scopolamine, and morphine and recovered well in the PACU after the surgery. There were no complications in the step-down unit and the patient continued to recover. On the med-surg floor, his temperature is 36.8°C (98.2°F), the heart rate is 98/min, the respiratory rate is 15/min, the blood pressure is 100/75 mm Hg, the oxygen saturation is 90%. On physical exam, he is a well-developed, obese man. His heart has a regular rate and rhythm and his lungs are clear to auscultation bilaterally. His incision site is clean, dry, and intact with an appropriate level of swelling and erythema. During the physical, the patient mentions some discomfort in his abdomen and pelvis and during a records review it is noted that he has not passed urine in the PACU, step-down unit, or since arriving on the med-surg floor. A bladder scan is inconclusive due to body habitus. What is the next best step in the treatment of this patient?"""
    
    old_phrasedp_4 = """An elderly individual is transferred from a recovery unit to a medical-surgical floor in the hospital. He had undergone a successful surgical procedure 14 hours ago. Before the surgery, he was pre-treated with medications and recovered well in the recovery area after the surgery. There were no complications in the recovery unit and the patient continued to recover. On the medical-surgical floor, his temperature is 36.8°C (98.2°F), the heart rate is 98/min, the respiratory rate is 15/min, the blood pressure is 100/75 mm Hg, the oxygen saturation is 90%. On physical exam, he is a well-developed, obese man. His heart has a regular rate and rhythm and his lungs are clear to auscultation bilaterally. His incision site is clean, dry, and intact with an appropriate level of swelling and erythema. During the physical, the patient mentions some discomfort in his abdomen and pelvis and during a records review it is noted that he has not passed urine in the recovery area, step-down unit, or since arriving on the medical-surgical floor."""
    
    new_phrasedp_4 = """An 80-year-old fella is transfer from a step-down units to a med-surg flooring in the hospital. He had endured a succeed fracture transaction 14 time ago. Before the surgery, he was pre-treated with atropine, scopolamine, and opiate and recovered allright in the PACU after the surgery. There were no complications in the step-down flats and the ill continued to recover. On the med-surg floor, his thermal is 36.8°C (98.2°F), the crux rate is 98/min, the respiratory rate is 15/min, the blood pressure is 100/75 mm Hg, the impassioned congestion is 90%. On corporal exam, he is a well-developed, obesity man. His nub has a regular rates and cadence and his airway are clear to auscultation bilaterally. His incisions places is clean, dry, and unaffected with an appropriate tier of swelling and erythema. During the physical, the patients mentioning some unease in his stomach and pelvic and during a record revisited it is pointed that he has not passed urine in the PACU, step-down unit, or since coming on the med-surg floor. A bladder scans is indecisive due to agency habitus. What is the upcoming best measure in the treatments of this patient?"""
    
    # Compute embeddings
    texts = [original_1, old_phrasedp_1, new_phrasedp_1, 
             original_2, old_phrasedp_2, new_phrasedp_2,
             original_3, old_phrasedp_3, new_phrasedp_3,
             original_4, old_phrasedp_4, new_phrasedp_4]
    
    embeddings = model.encode(texts)
    
    # Compute similarities
    results = []
    
    # Question 1
    orig_emb_1 = embeddings[0].reshape(1, -1)
    old_emb_1 = embeddings[1].reshape(1, -1)
    new_emb_1 = embeddings[2].reshape(1, -1)
    
    old_sim_1 = cosine_similarity(orig_emb_1, old_emb_1)[0][0]
    new_sim_1 = cosine_similarity(orig_emb_1, new_emb_1)[0][0]
    
    results.append({
        'question': 'Question 1: 14-year-old Girl with Typhoid Fever',
        'old_similarity': old_sim_1,
        'new_similarity': new_sim_1,
        'difference': old_sim_1 - new_sim_1
    })
    
    # Question 2
    orig_emb_2 = embeddings[3].reshape(1, -1)
    old_emb_2 = embeddings[4].reshape(1, -1)
    new_emb_2 = embeddings[5].reshape(1, -1)
    
    old_sim_2 = cosine_similarity(orig_emb_2, old_emb_2)[0][0]
    new_sim_2 = cosine_similarity(orig_emb_2, new_emb_2)[0][0]
    
    results.append({
        'question': 'Question 2: 22-year-old Female with Metronidazole',
        'old_similarity': old_sim_2,
        'new_similarity': new_sim_2,
        'difference': old_sim_2 - new_sim_2
    })
    
    # Question 3
    orig_emb_3 = embeddings[6].reshape(1, -1)
    old_emb_3 = embeddings[7].reshape(1, -1)
    new_emb_3 = embeddings[8].reshape(1, -1)
    
    old_sim_3 = cosine_similarity(orig_emb_3, old_emb_3)[0][0]
    new_sim_3 = cosine_similarity(orig_emb_3, new_emb_3)[0][0]
    
    results.append({
        'question': 'Question 3: 23-year-old Primigravida',
        'old_similarity': old_sim_3,
        'new_similarity': new_sim_3,
        'difference': old_sim_3 - new_sim_3
    })
    
    # Question 4
    orig_emb_4 = embeddings[9].reshape(1, -1)
    old_emb_4 = embeddings[10].reshape(1, -1)
    new_emb_4 = embeddings[11].reshape(1, -1)
    
    old_sim_4 = cosine_similarity(orig_emb_4, old_emb_4)[0][0]
    new_sim_4 = cosine_similarity(orig_emb_4, new_emb_4)[0][0]
    
    results.append({
        'question': 'Question 4: 80-year-old Man Post-Surgery',
        'old_similarity': old_sim_4,
        'new_similarity': new_sim_4,
        'difference': old_sim_4 - new_sim_4
    })
    
    return results

def print_results(results):
    """
    Print the semantic similarity results in a formatted way.
    """
    print("=" * 80)
    print("SEMANTIC SIMILARITY ANALYSIS: Original vs Old vs New PhraseDP")
    print("=" * 80)
    print()
    
    for i, result in enumerate(results, 1):
        print(f"{result['question']}")
        print("-" * 60)
        print(f"Old PhraseDP Similarity:  {result['old_similarity']:.4f}")
        print(f"New PhraseDP Similarity:  {result['new_similarity']:.4f}")
        print(f"Difference (Old - New):   {result['difference']:.4f}")
        
        if result['difference'] > 0:
            print(f"✅ Old PhraseDP is MORE similar to original (+{result['difference']:.4f})")
        else:
            print(f"❌ New PhraseDP is MORE similar to original ({abs(result['difference']):.4f})")
        print()
    
    # Summary statistics
    old_similarities = [r['old_similarity'] for r in results]
    new_similarities = [r['new_similarity'] for r in results]
    differences = [r['difference'] for r in results]
    
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Average Old PhraseDP Similarity:  {np.mean(old_similarities):.4f}")
    print(f"Average New PhraseDP Similarity:  {np.mean(new_similarities):.4f}")
    print(f"Average Difference (Old - New):   {np.mean(differences):.4f}")
    print()
    
    old_better_count = sum(1 for d in differences if d > 0)
    new_better_count = sum(1 for d in differences if d < 0)
    
    print(f"Old PhraseDP closer to original: {old_better_count}/4 questions")
    print(f"New PhraseDP closer to original: {new_better_count}/4 questions")
    print()
    
    if np.mean(differences) > 0:
        print("🎯 CONCLUSION: Old PhraseDP is MORE similar to original text on average")
    else:
        print("🎯 CONCLUSION: New PhraseDP is MORE similar to original text on average")

if __name__ == "__main__":
    results = compute_semantic_similarity()
    print_results(results)
