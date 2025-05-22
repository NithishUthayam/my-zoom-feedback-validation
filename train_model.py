import pandas as pd
import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing
def clean_text(text):
    text = str(text)  # Ensure text is string
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove special characters
    text = text.replace("can not", "cannot")     # Handle typos
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Load datasets
try:
    train_df = pd.read_excel("train.xlsx")
    eval_df = pd.read_excel("evaluation.xlsx")
except ImportError as e:
    print("Error: Ensure 'openpyxl' is installed. Run 'pip install openpyxl'.")
    raise e

# Save to CSV (optional, for compatibility with deployment)
train_df.to_csv("train.csv", index=False)
eval_df.to_csv("evaluation.csv", index=False)

# Preprocess text and reason
train_df["text"] = train_df["text"].apply(clean_text)
train_df["reason"] = train_df["reason"].apply(clean_text)
eval_df["text"] = eval_df["text"].apply(clean_text)
eval_df["reason"] = eval_df["reason"].apply(clean_text)

# Prepare input for BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def encode_data(texts, reasons, max_length=128):
    inputs = [f"{text} [SEP] {reason}" for text, reason in zip(texts, reasons)]
    encodings = tokenizer(inputs, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
    return encodings

# Encode datasets
train_encodings = encode_data(train_df["text"], train_df["reason"])
eval_encodings = encode_data(eval_df["text"], eval_df["reason"])
train_labels = torch.tensor(train_df["label"].values)
eval_labels = torch.tensor(eval_df["label"].values)

# Create PyTorch dataset
class FeedbackDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FeedbackDataset(train_encodings, train_labels)
eval_dataset = FeedbackDataset(eval_encodings, eval_labels)

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch",  # Correct parameter
    save_strategy="epoch",
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train model
trainer.train()

# Evaluate model
predictions = trainer.predict(eval_dataset)
preds = np.argmax(predictions.predictions, axis=1)
labels = predictions.label_ids

# Calculate metrics
accuracy = accuracy_score(labels, preds)
precision = precision_score(labels, preds, zero_division=0)
recall = recall_score(labels, preds, zero_division=0)
f1 = f1_score(labels, preds, zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Confusion matrix chart
cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("confusion_matrix.png")

# Save model
model.save_pretrained("./feedback_model")
tokenizer.save_pretrained("./feedback_model")