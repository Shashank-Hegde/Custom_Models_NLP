#!/usr/bin/env python3
import json
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, InputExample
from torch.utils.data import DataLoader

# Load the custom training data (generated with GPT-3.5)
data_file = "sbert_symptom_data.json"
with open(data_file, "r", encoding="utf8") as f:
    train_data = json.load(f)

# Build InputExample objects.
# We treat each sample as a pair: (sentence, canonical symptom)
train_examples = []
for entry in train_data:
    # Each entry is expected to have "sentence" and "symptom" keys.
    sentence = entry["sentence"]
    symptom = entry["symptom"]
    train_examples.append(InputExample(texts=[sentence, symptom]))

# Define our base SBERT model.
model_name = "all-MiniLM-L6-v2"  # You can choose another pre-trained model.
model = SentenceTransformer(model_name)

# Create a dataset and dataloader.
train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

# Use MultipleNegativesRankingLoss for contrastive fine-tuning.
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune the model.
num_epochs = 3
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=100,
    output_path="custom_sbert_model"
)
