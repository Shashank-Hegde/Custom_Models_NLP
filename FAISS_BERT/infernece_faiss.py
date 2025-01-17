#!/usr/bin/env python3
"""
inference_faiss_sliding.py

This script performs inference using a FAISS index built from candidate phrases.
It uses spaCy to extract noun chunks (candidate spans) from each sentence,
encodes these spans using SBERT, and then performs a fast FAISS search for the best match.
It returns one best matching candidate per sentence if its similarity exceeds the threshold.

Before running, ensure:
    pip install faiss-cpu numpy sentence-transformers spacy nltk
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
import json
import nltk
nltk.download('punkt', quiet=True)

#######################################
# 1. Load Candidate Dictionary and FAISS Index
#######################################
with open("candidates.json", "r", encoding="utf8") as f:
    candidates = json.load(f)
faiss_index = faiss.read_index("faiss_index.index")
print(f"Loaded FAISS index with {faiss_index.ntotal} candidates.")

#######################################
# 2. Load Models
#######################################
# Load SBERT model (use your fine-tuned model if available)
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load a lightweight spaCy model for noun chunk extraction
nlp_spacy = spacy.load("en_core_web_sm")

#######################################
# 3. Candidate Span Extraction using Noun Chunks
#######################################
def extract_candidate_spans(sentence):
    """
    Use spaCy to extract noun chunks from a sentence as candidate spans.
    """
    doc = nlp_spacy(sentence)
    spans = []
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        if len(text) >= 3:
            spans.append(text)
    return spans

#######################################
# 4. Inference Function
#######################################
def infer_symptoms(text, threshold=0.6):
    """
    Given an input text, perform the following:
      1. Split the text into sentences.
      2. For each sentence, extract candidate spans using noun chunks.
      3. Encode each candidate span with SBERT.
      4. Query the FAISS index for the best matching candidate.
      5. Return, for each sentence, the best matching candidate with its canonical label if the similarity
         is above threshold; otherwise, indicate no match.
    """
    results = []
    sentences = sent_tokenize(text)

    for sentence in sentences:
        candidate_spans = extract_candidate_spans(sentence)
        if not candidate_spans:
            results.append({"sentence": sentence, "match": None})
            continue

        # Encode candidate spans
        span_embeddings = sbert_model.encode(candidate_spans, convert_to_numpy=True)
        faiss.normalize_L2(span_embeddings)

        # Track best candidate for the sentence
        best_score = -1.0
        best_match = None
        best_span = None

        for span, emb in zip(candidate_spans, span_embeddings):
            emb = np.expand_dims(emb, axis=0)
            D, I = faiss_index.search(emb, k=1)
            score = D[0][0]
            if score > best_score:
                best_score = score
                best_span = span
                best_match = candidates[I[0][0]]

        if best_score >= threshold:
            results.append({
                "sentence": sentence,
                "matched_span": best_span,
                "matched_type": best_match["type"],
                "matched_candidate": best_match["text"],
                "canonical": best_match.get("canonical", ""),
                "similarity": float(best_score)
            })
        else:
            results.append({"sentence": sentence, "match": None})
    return results

#######################################
# 5. Main: Example Inference
#######################################
if __name__ == "__main__":
    sample_text = (
        "I have been suffering from a high temperature and a severe headache since yesterday. "
        "My doctor recommended ibuprofen and I also have a dry cough."
    )

    results = infer_symptoms(sample_text, threshold=0.6)
    print("Input Text:")
    print(sample_text)
    print("\nExtracted Best Matches (per sentence):")
    for res in results:
        print("Sentence:", res["sentence"])
        if res.get("matched_candidate"):
            print("  Best Match: {} (Canonical: {}) | Type: {} | Similarity: {:.2f}".format(
                res["matched_candidate"], res["canonical"], res["matched_type"], res["similarity"]
            ))
        else:
            print("  No match above threshold.")
        print("-----")
