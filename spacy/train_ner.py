#!/usr/bin/env python3
import json
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding

def has_overlapping(entities):
    """
    Check whether any entity spans in the list overlap.
    Each entity is assumed to be of the format [start, end, label].
    Returns True if any two spans overlap, else False.
    """
    # Sort by starting index
    sorted_ents = sorted(entities, key=lambda ent: ent[0])
    for i in range(len(sorted_ents) - 1):
        current_start, current_end, _ = sorted_ents[i]
        next_start, next_end, _ = sorted_ents[i + 1]
        # If current end exceeds next start, there's overlap.
        if current_end > next_start:
            return True
    return False

# Load the GPT-generated training data
with open("gpt_training_data.json", "r", encoding="utf8") as f:
    raw_train_data = json.load(f)

print(f"Loaded {len(raw_train_data)} training examples from GPT.")

# Filter out examples with overlapping entities.
TRAIN_DATA = []
for text, ann in raw_train_data:
    entities = ann.get("entities", [])
    if has_overlapping(entities):
        print(f"Skipping example due to overlapping entities in text: {text}")
    else:
        TRAIN_DATA.append((text, ann))

print(f"Using {len(TRAIN_DATA)} training examples after filtering overlapping spans.")

# Define the custom labels we expect.
LABELS = ["SYMPTOM", "MEDICATION", "LOCATION", "AGE", "GENDER", "DURATION"]

# Load the base spaCy model.
nlp = spacy.load("en_core_web_sm")

# Get the existing NER component.
ner = nlp.get_pipe("ner")

# Add new labels to the NER component.
for label in LABELS:
    ner.add_label(label)

# Create training Example objects from TRAIN_DATA.
examples = []
for text, ann in TRAIN_DATA:
    doc = nlp.make_doc(text)
    try:
        examples.append(Example.from_dict(doc, ann))
    except Exception as e:
        print("Error creating Example for text:", text)
        print(e)

print(f"Prepared {len(examples)} training Example objects.")

# Optionally, freeze non-NER pipes during training.
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    n_iter = 30  # Adjust iterations as needed.

    print("Training the NER component with GPT-generated data...")
    for itn in range(n_iter):
        random.shuffle(examples)
        losses = {}
        batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            nlp.update(batch, drop=0.2, sgd=optimizer, losses=losses)
        print(f"Iteration {itn+1}/{n_iter} - Losses: {losses}")

# Save the fine-tuned model.
output_dir = "./custom_model"
nlp.to_disk(output_dir)
print(f"Trained model saved to {output_dir}")
