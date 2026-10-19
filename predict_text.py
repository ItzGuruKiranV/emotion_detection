# ----------------------------
# predict_text.py
# ----------------------------
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load tokenizer + BERT only once
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("./results/checkpoint-1600")

# Mapping
id2label = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "happy"
}

def predict_from_text(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    pred = torch.argmax(probs, dim=1).item()
    return id2label[pred]
