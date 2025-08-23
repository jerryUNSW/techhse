import spacy
from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
import json

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load 1000 samples from CNN/DM dataset
# dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1000]")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:10000]")

# Entity sets: {entity_type -> set of entities}
entity_sets = defaultdict(set)

# Entity types to keep
valid_types = {"PERSON", "ORG", "GPE", "DATE", "NORP", "LOC"}

# Process articles
for example in tqdm(dataset, desc="Extracting entities"):
    doc = nlp(example["article"])
    for ent in doc.ents:
        if ent.label_ in valid_types:
            entity_sets[ent.label_].add(ent.text.strip())

# Convert sets to sorted lists for JSON serialization
entity_output = {label: sorted(entities) for label, entities in entity_sets.items()}

# Save to JSON
# with open("entities_by_type.json", "w", encoding="utf-8") as f:
#     json.dump(entity_output, f, indent=2, ensure_ascii=False)
with open("entities_by_type_new.json", "w", encoding="utf-8") as f:
    json.dump(entity_output, f, indent=2, ensure_ascii=False)

print("Entity extraction complete. Saved to entities_by_type.json")
