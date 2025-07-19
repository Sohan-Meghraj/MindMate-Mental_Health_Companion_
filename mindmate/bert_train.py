# bert_train.py

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from tqdm import tqdm

# ✅ Step 1: Load and merge dataset
def load_data():
    train = pd.read_csv("data/train.txt", sep=";", names=["text", "emotion"])
    val = pd.read_csv("data/val.txt", sep=";", names=["text", "emotion"])
    test = pd.read_csv("data/test.txt", sep=";", names=["text", "emotion"])
    return pd.concat([train, val, test], ignore_index=True)

df = load_data()
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['emotion'])

# ✅ Save label map
os.makedirs("models", exist_ok=True)
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
torch.save(label_map, "models/label_map.pth")

# ✅ Step 2: Create Dataset
class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(text, padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }

# ✅ Step 3: Prepare data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = df['text'].tolist()
labels = df['label'].tolist()

# Shuffle and split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_dataset = EmotionDataset(X_train, y_train, tokenizer)
test_dataset = EmotionDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# ✅ Step 4: Train BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

print("🧠 Training BERT model...")
model.train()
for epoch in range(3):  # ~3 epochs is usually enough for fine-tuning
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# ✅ Step 5: Evaluate
print("📊 Evaluating model...")
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

# ✅ Step 6: Save model
model_dir = "models/bert_emotion_classifier"
os.makedirs(model_dir, exist_ok=True)
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
print(f"✅ Model saved to {model_dir}")
