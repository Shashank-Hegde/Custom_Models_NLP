#!/usr/bin/env python3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
import spacy
from nltk.tokenize import sent_tokenize
import json
import nltk
import re
import time

# Download necessary nltk data
nltk.download('punkt', quiet=True)

#############################################
# Global Resource Loading (Cached Once)
#############################################
def load_resources():
    start_time = time.time()
    # Load candidate dictionary from local file
    with open("candidates.json", "r", encoding="utf8") as f:
        candidates = json.load(f)

    # Load FAISS index
    index = faiss.read_index("faiss_index.index")
    print(f"Loaded FAISS index with {index.ntotal} candidates.")

    # Load SBERT model
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # Replace with your fine-tuned model if available
    print("Loaded SBERT model.")

    # Load spaCy model for noun chunk extraction
    nlp_spacy = spacy.load("en_core_web_sm")
    print("Loaded spaCy model.")

    total_time = time.time() - start_time
    print(f"All resources loaded in {total_time:.2f} seconds.\n")
    return sbert_model, nlp_spacy, candidates, index

SBERT_MODEL, NLP_SPACY, CANDIDATES, FAISS_INDEX = load_resources()

#############################################
# Candidate Span Extraction via Noun Chunks
#############################################
def extract_candidate_spans(sentence):
    """
    Extract candidate spans from a sentence using spaCy's noun chunks.
    Returns a list of candidate phrases.
    """
    doc = NLP_SPACY(sentence)
    # Use noun chunks as candidate spans
    spans = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) >= 3]
    return spans

#############################################
# Negation Filtering
#############################################
NEGATION_PATTERNS = [
    r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bwithout\b'
]

def has_negation_context(sentence, candidate):
    """
    Check if a candidate phrase in the sentence is near a negation word.
    """
    sentence_lower = sentence.lower()
    candidate_lower = candidate.lower()
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern + r".{0,15}" + re.escape(candidate_lower), sentence_lower):
            return True
        if re.search(re.escape(candidate_lower) + r".{0,15}" + pattern, sentence_lower):
            return True
    return False

#############################################
# Inference Function: Multiple Matches Per Sentence
#############################################
def infer_entities(text, threshold=0.65):
    """
    Process the input text by splitting into sentences.
    For each sentence:
      - Extract candidate spans (noun chunks).
      - Encode each candidate span using the SBERT model.
      - For each candidate span, use FAISS to retrieve the nearest candidate.
      - Collect all candidates with similarity >= threshold.
      - Deduplicate based on canonical label.
    Returns a list of results, one per sentence.
    """
    results = []
    sentences = sent_tokenize(text)

    for sentence in sentences:
        candidate_spans = extract_candidate_spans(sentence)
        if not candidate_spans:
            results.append({"sentence": sentence, "matches": []})
            continue

        # Encode all candidate spans in the sentence at once
        span_embeddings = SBERT_MODEL.encode(candidate_spans, convert_to_numpy=True)
        # Normalize each embedding for cosine similarity
        span_embeddings = span_embeddings / np.linalg.norm(span_embeddings, axis=1, keepdims=True)

        matches = []
        # Loop over each candidate span and retrieve its best matching candidate from the FAISS index.
        for span, emb in zip(candidate_spans, span_embeddings):
            emb = np.expand_dims(emb, axis=0)
            D, I = FAISS_INDEX.search(emb, k=1)
            score = D[0][0]
            if score >= threshold:
                # Check negation: if candidate appears in a negated context, ignore it.
                if has_negation_context(sentence, span):
                    continue
                candidate_info = CANDIDATES[I[0][0]]
                # Append if match is not already added (deduplicate based on canonical label)
                canonical = candidate_info.get("canonical", "")
                if not any(m["canonical"] == canonical for m in matches):
                    matches.append({
                        "span": span,
                        "matched_candidate": candidate_info["text"],
                        "canonical": canonical,
                        "type": candidate_info["type"],
                        "similarity": float(score)
                    })
        results.append({
            "sentence": sentence,
            "matches": matches
        })
    return results

#############################################
# Main Execution
#############################################
if __name__ == "__main__":
    sample_text = (
        "I have been suffering from a high temperature and a severe headache since yesterday. "
        "My doctor recommended ibuprofen and I also have a dry cough."
        "My doctor recommended some multivitamins for the wheezing since I also had jaundid."
    )
    start_time = time.time()
    extraction_results = infer_entities(sample_text, threshold=0.65)
    total_time = time.time() - start_time

    print("Input Text:")
    print(sample_text)
    print("\nExtracted Best Matches (per sentence):")
    for res in extraction_results:
        print("Sentence: {}".format(res["sentence"]))
        if res["matches"]:
            for match in res["matches"]:
                print("  Matched: {} (Canonical: {}) | Type: {} | Similarity: {:.2f}".format(
                    match["matched_candidate"],
                    match["canonical"],
                    match["type"],
                    match["similarity"]
                ))
        else:
            print("  No match above threshold.")
        print("-----")

    print(f"Total inference time: {total_time:.2f} seconds.")
