#!/usr/bin/env python3
import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from nltk.tokenize import sent_tokenize
import json
import re
import nltk
from flask import Flask, request, jsonify

# Download necessary nltk data (only needed once)
nltk.download('punkt', quiet=True)

app = Flask(__name__)

##############################
# Global Resource Loading
##############################
def load_resources():
    start_time = time.time()
    # Load candidate dictionary from file
    with open("candidates.json", "r", encoding="utf8") as f:
        candidates = json.load(f)

    # Load FAISS index
    index = faiss.read_index("faiss_index.index")
    print(f"Loaded FAISS index with {index.ntotal} candidates.")

    # Load SBERT model (can be replaced with your fine-tuned model)
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loaded SBERT model.")

    # Load spaCy model for noun-chunk extraction
    nlp_spacy = spacy.load("en_core_web_sm")
    print("Loaded spaCy model.")

    total_time = time.time() - start_time
    print(f"All resources loaded in {total_time:.2f} seconds.")
    return sbert_model, nlp_spacy, candidates, index

# Load models/indexes once on server startup.
SBERT_MODEL, NLP_SPACY, CANDIDATES, FAISS_INDEX = load_resources()

#############################################
# Candidate Span Extraction and Negation Check
#############################################
NEGATION_PATTERNS = [
    r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bwithout\b'
]

def has_negation_context(sentence, candidate):
    sentence_lower = sentence.lower()
    candidate_lower = candidate.lower()
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern + r".{0,15}" + re.escape(candidate_lower), sentence_lower):
            return True
        if re.search(re.escape(candidate_lower) + r".{0,15}" + pattern, sentence_lower):
            return True
    return False

def extract_candidate_spans(sentence):
    doc = NLP_SPACY(sentence)
    # Use noun chunks as candidate spans
    spans = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) >= 3]
    return spans

#############################################
# Inference Function
#############################################
def infer_entities(text, threshold=0.84):
    results = []
    sentences = sent_tokenize(text)
    for sentence in sentences:
        candidate_spans = extract_candidate_spans(sentence)
        if not candidate_spans:
            results.append({"sentence": sentence, "matches": []})
            continue

        # Encode candidate spans (batch process)
        span_embeddings = SBERT_MODEL.encode(candidate_spans, convert_to_numpy=True)
        span_embeddings = span_embeddings / np.linalg.norm(span_embeddings, axis=1, keepdims=True)

        sentence_matches = {}
        # Check each candidate span
        for candidate_span, emb in zip(candidate_spans, span_embeddings):
            emb = np.expand_dims(emb, axis=0)
            D, I = FAISS_INDEX.search(emb, k=1)
            score = D[0][0]
            if score >= threshold:
                if has_negation_context(sentence, candidate_span):
                    continue
                candidate_info = CANDIDATES[I[0][0]]
                canon = candidate_info.get("canonical", "")
                # Keep the best score for each canonical match
                if (candidate_info["type"] == "SYMPTOM" and
                    (canon not in sentence_matches or score > sentence_matches[canon]["similarity"])):
                    sentence_matches[canon] = {
                        "matched_span": candidate_span,
                        "matched_candidate": candidate_info["text"],
                        "canonical": canon,
                        "type": candidate_info["type"],
                        "similarity": float(score)
                    }
                # Extend to other types if necessary.
        results.append({
            "sentence": sentence,
            "matches": list(sentence_matches.values())
        })
    return results

#############################################
# Flask API Routes
#############################################
@app.route('/infer', methods=['POST'])
def infer():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data["text"]
    t0 = time.time()
    results = infer_entities(text)
    t_total = time.time() - t0
    response = {
        "input_text": text,
        "results": results,
        "inference_time": t_total
    }
    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
