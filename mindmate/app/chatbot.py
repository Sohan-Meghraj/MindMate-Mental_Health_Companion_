import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load fine-tuned BERT model and tokenizer
model_path = "models/bert_emotion_classifier"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()  # set model to evaluation mode

# Emotion labels (in the same order used during training)
emotion_labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

def predict_emotion(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Run model prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return emotion_labels[predicted_class]
