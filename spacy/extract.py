#!/usr/bin/env python3
import spacy
from sentence_transformers import SentenceTransformer, util

# --- Define your static lists ---

symptom_list = [
    "cough", "fever", "headache", "fatigue", "sore throat", "chills",
    "shortness of breath", "nausea", "vomiting", "diarrhea", "rash", "dizziness"
]

# Mapping canonical symptoms to a list of synonyms
symptom_synonyms = {
    "cough": ["dry cough", "productive cough"],
    "fever": ["high temperature", "pyrexia"],
    "headache": ["head pain", "migraine"],
    "nausea": ["queasiness", "sickness"],
    "dizziness": ["lightheadedness", "vertigo"]
}

medication_list = [
    "Paracetamol", "Ibuprofen", "Amoxicillin", "fluticasone cream", "loratadine", "levodopa"
]

location_list = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Kolkata"
]

# (You can extend lists for AGE, GENDER, DURATION as needed; these may be handled by the custom model.)

# --- Load models ---
# Load the custom-trained model that we fine-tuned.
nlp_custom = spacy.load("./custom_model")
# Also load a base spaCy model for inherent matching.
nlp_inherent = spacy.load("en_core_web_sm")

# Load SBERT for semantic similarity
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Pre-compute SBERT embeddings for all canonical symptoms and their synonyms.
# For each canonical symptom, we can build a composite list.
symptom_candidates = []
for sym in symptom_list:
    symptom_candidates.append(sym)
    if sym in symptom_synonyms:
        symptom_candidates.extend(symptom_synonyms[sym])

# Compute embeddings and store them in a dictionary.
candidate_embeddings = sbert_model.encode(symptom_candidates, convert_to_tensor=True)

def inherent_extract_symptoms(text):
    """Perform inherent matching using a simple rule using the fixed SYMPTOMS list."""
    doc = nlp_inherent(text)
    found = set()
    for token in doc:
        if token.text.lower() in [s.lower() for s in symptom_list]:
            # Basic heuristic: check for a negation.
            negated = any(child.lower_ in ("no", "not") for child in token.children if child.dep_=="neg")
            if not negated:
                found.add(token.text)
    return list(found)

def inherent_extract_medications(text):
    found = set()
    lowered = text.lower()
    for med in medication_list:
        if med.lower() in lowered:
            found.add(med)
    return list(found)

def inherent_extract_locations(text):
    found = set()
    lowered = text.lower()
    for loc in location_list:
        if loc.lower() in lowered:
            found.add(loc)
    return list(found)

def sbert_validate_symptoms(candidate_texts, threshold=0.6):
    """
    Given a list of candidate symptom strings (extracted from custom model),
    use SBERT similarity to determine if they match any of our known symptom candidates.
    Returns a list of canonical symptoms (the best match) for each candidate that exceeds the threshold.
    """
    # Compute embeddings for the candidate texts.
    cand_embeddings = sbert_model.encode(candidate_texts, convert_to_tensor=True)
    validated = []
    for idx, cand_emb in enumerate(cand_embeddings):
        # Compute similarity to our candidate list
        cosine_scores = util.cos_sim(cand_emb, candidate_embeddings)[0]
        best_score, best_idx = cosine_scores.max(dim=0)
        if best_score.item() >= threshold:
            # For reporting, select the canonical symptom (if possible).
            # Here, if best candidate is found among the candidates, we can try to map back.
            validated.append(symptom_candidates[best_idx])
        else:
            validated.append(candidate_texts[idx])
    return validated

def combined_extract_entities(text):
    """
    Use the custom NER model plus inherent extraction plus SBERT similarity validation
    to extract all target entities (SYMPTOM, MEDICATION, LOCATION, AGE, GENDER, DURATION).

    The custom model may not always extract everything, so we supplement it with rule-based matching.
    For symptoms, we also validate using SBERT similarity.
    """
    # Use the custom model for extraction.
    doc_custom = nlp_custom(text)
    custom_entities = {}
    for ent in doc_custom.ents:
        custom_entities.setdefault(ent.label_, []).append(ent.text)

    # Inherent extraction:
    inherent_symptoms = inherent_extract_symptoms(text)
    inherent_medications = inherent_extract_medications(text)
    inherent_locations = inherent_extract_locations(text)

    # For symptoms, combine custom model predictions and inherent matches.
    combined_symptoms = list(set(custom_entities.get("SYMPTOM", []) + inherent_symptoms))

    # Validate these symptom candidates with SBERT semantic similarity
    # (This can help, for example, if the custom model produced a variant that is very different from our list.)
    if combined_symptoms:
        validated_symptoms = sbert_validate_symptoms(combined_symptoms, threshold=0.6)
    else:
        validated_symptoms = []

    # Merge results for each category:
    results = {
        "SYMPTOM": validated_symptoms,
        "MEDICATION": list(set(custom_entities.get("MEDICATION", []) + inherent_medications)),
        "LOCATION": list(set(custom_entities.get("LOCATION", []) + inherent_locations)),
        "AGE": custom_entities.get("AGE", []),
        "GENDER": custom_entities.get("GENDER", []),
        "DURATION": custom_entities.get("DURATION", []),
        "OTHER": custom_entities.get("OTHER", [])
    }
    return results

if __name__ == "__main__":
    sample_text = (
        "I am a 30-year-old female from Delhi. I have been experiencing a high temperature and severe headache for one week. "
        "My doctor prescribed Ibuprofen. Also, I have noticed dizziness recently."
    )

    extracted = combined_extract_entities(sample_text)
    print("Combined Extracted Entities:")
    for label, items in extracted.items():
        print(f"{label}: {items}")
