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

################################################
# Global Cache: Load Models and FAISS Index Once
################################################

def load_resources():
    """
    Load the SBERT model, spaCy model, candidate dictionary, and FAISS index,
    and return them as a tuple.
    """
    start_time = time.time()
    # Load candidate dictionary from file
    with open("candidates.json", "r", encoding="utf8") as f:
        candidates = json.load(f)
    # Load FAISS index from file
    index = faiss.read_index("faiss_index.index")
    print(f"Loaded FAISS index with {index.ntotal} candidates.")

    # Load SBERT model (fine-tuned or default)
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Loaded SBERT model.")

    # Load spaCy model for noun-chunk extraction
    nlp_spacy = spacy.load("en_core_web_sm")
    print("Loaded spaCy model.")

    total_time = time.time() - start_time
    print(f"All resources loaded in {total_time:.2f} seconds.")
    return sbert_model, nlp_spacy, candidates, index

# Global variables to avoid re-loading on every inference.
SBERT_MODEL, NLP_SPACY, CANDIDATES, FAISS_INDEX = load_resources()

#######################################
# Candidate Span Extraction via Noun Chunks
#######################################

def extract_candidate_spans(sentence):
    """
    Extract candidate spans from a sentence using spaCy's noun chunks.
    Returns a list of candidate phrases.
    """
    doc = NLP_SPACY(sentence)
    spans = [chunk.text.strip() for chunk in doc.noun_chunks if len(chunk.text.strip()) >= 3]
    return spans

#######################################
# Negation Filtering (Optional)
#######################################

NEGATION_PATTERNS = [
    r'\bno\b', r'\bnot\b', r'\bnever\b', r'\bwithout\b'
]

def has_negation_context(sentence, candidate):
    """
    Check if a candidate phrase in a sentence is preceded or followed by a negation term.
    """
    sentence_lower = sentence.lower()
    candidate_lower = candidate.lower()
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern + r".{0,15}" + re.escape(candidate_lower), sentence_lower):
            return True
        if re.search(re.escape(candidate_lower) + r".{0,15}" + pattern, sentence_lower):
            return True
    return False

#######################################
# Inference Function
#######################################

def infer_entities(text, threshold=0.65):
    """
    For the input text:
      - Split text into sentences.
      - For each sentence, extract candidate spans using noun chunk extraction.
      - Encode these candidates with SBERT.
      - Query the FAISS index for the best matching candidate.
      - Return only the best match (if similarity exceeds the threshold).
    """
    results = []
    sentences = sent_tokenize(text)

    # Process each sentence
    for sentence in sentences:
        candidates_in_sentence = extract_candidate_spans(sentence)
        if not candidates_in_sentence:
            results.append({"sentence": sentence, "match": None})
            continue

        # Encode candidate spans using the already-loaded SBERT_MODEL
        span_embeddings = SBERT_MODEL.encode(candidates_in_sentence, convert_to_numpy=True)
        faiss.normalize_L2(span_embeddings)

        best_candidate = None
        best_score = -1.0
        best_span = None

        for candidate_span, emb in zip(candidates_in_sentence, span_embeddings):
            emb = np.expand_dims(emb, axis=0)
            D, I = FAISS_INDEX.search(emb, k=1)
            score = D[0][0]
            if score > best_score:
                if has_negation_context(sentence, candidate_span):
                    continue
                best_score = score
                best_span = candidate_span
                best_candidate = CANDIDATES[I[0][0]]

        if best_score >= threshold and best_candidate is not None:
            results.append({
                "sentence": sentence,
                "matched_span": best_span,
                "matched_type": best_candidate["type"],
                "matched_candidate": best_candidate["text"],
                "canonical": best_candidate.get("canonical", ""),
                "similarity": float(best_score)
            })
        else:
            results.append({"sentence": sentence, "match": None})
    return results

#######################################
# Main Execution
#######################################

if __name__ == "__main__":
    sample_text = (
        "I have been suffering from a high temperature and a severe headache since yesterday. "
        "My doctor recommended ibuprofen and I also have a dry cough."
    )

    start = time.time()
    results = infer_entities(sample_text, threshold=0.65)
    end = time.time()

    print("Input Text:")
    print(sample_text)
    print("\nExtracted Best Matches (per sentence):")
    for res in results:
        print("Sentence: {}".format(res["sentence"]))
        if res.get("matched_candidate"):
            print("  Best Match: {} (Canonical: {}) | Type: {} | Similarity: {:.2f}".format(
                res["matched_candidate"], res["canonical"], res["matched_type"], res["similarity"]
            ))
        else:
            print("  No match above threshold.")
        print("-----")

    print(f"Total inference time: {end - start:.2f} seconds.")
