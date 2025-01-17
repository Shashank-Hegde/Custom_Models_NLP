#!/usr/bin/env python3
"""
faiss_index_build.py

This script builds a candidate dictionary from fixed lists of symptom names (and synonyms), medications,
and locations. It encodes the candidate phrases using an SBERT model and builds a FAISS index for
fast cosine similarity search.

Before running, ensure:
    pip install faiss-cpu numpy sentence-transformers spacy nltk

Note: If you encounter numpy dtype incompatibility errors, reinstall numpy and dependent libraries
(e.g., pip install --force-reinstall numpy==1.21.6).
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# ----------------------------
# Define candidate lists
# ----------------------------
symptom_list = [
    "cough", "fever", "headache", "fatigue", "sore throat",
    "chills", "shortness of breath", "nausea", "vomiting", "diarrhea", "rash", "dizziness", "back spasm"
]

symptom_synonyms = {
    "cough": ["dry cough", "productive cough"],
    "fever": ["high temperature", "pyrexia"],
    "headache": ["head pain", "migraine"],
    "nausea": ["queasiness", "sickness"],
    "dizziness": ["lightheadedness", "vertigo"],
    "back spasm": ["back is spasming", "spinal contraction", "back muscle tension"]
}

medications_list = [
    "ibuprofen", "acetaminophen", "paracetamol", "aspirin", "naproxen"
]

location_list = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Kolkata"
]

# ----------------------------
# Build candidate dictionary
# ----------------------------
candidates = []

# For symptoms: add canonical + synonyms.
for sym in symptom_list:
    candidates.append({"text": sym, "type": "SYMPTOM", "canonical": sym})
    if sym in symptom_synonyms:
        for syn in symptom_synonyms[sym]:
            candidates.append({"text": syn, "type": "SYMPTOM", "canonical": sym})

# Add medications.
for med in medications_list:
    candidates.append({"text": med, "type": "MEDICATION", "canonical": med})

# Add locations.
for loc in location_list:
    candidates.append({"text": loc, "type": "LOCATION", "canonical": loc})

# ----------------------------
# Load SBERT model and encode candidate texts
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with your custom model if available.
candidate_texts = [cand["text"] for cand in candidates]
print(f"Encoding {len(candidate_texts)} candidate phrases...")
embeddings = model.encode(candidate_texts, convert_to_numpy=True)
# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# ----------------------------
# Build FAISS index
# ----------------------------
embedding_dim = embeddings.shape[1]
index = faiss.IndexFlatIP(embedding_dim)  # Inner product of normalized vectors equals cosine similarity.
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} candidates.")
faiss.write_index(index, "faiss_index.index")

# ----------------------------
# Save candidates dictionary
# ----------------------------
with open("candidates.json", "w", encoding="utf8") as f:
    json.dump(candidates, f, indent=2)
print("Candidates saved to candidates.json")
