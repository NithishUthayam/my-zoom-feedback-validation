import gradio as gr
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import re
from nltk.corpus import stopwords
import nltk

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing
def clean_text(text):
    text = str(text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = text.replace("can not", "cannot")
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text

# Load model and tokenizer
model = BertForSequenceClassification.from_pretrained("./feedback_model")
tokenizer = BertTokenizer.from_pretrained("./feedback_model")
model.eval()

# Prediction function
def predict_feedback(text, reason):
    text = clean_text(text)
    reason = clean_text(reason)
    input_text = f"{text} [SEP] {reason}"
    encodings = tokenizer(input_text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**encodings)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return "Aligned" if prediction == 1 else "Not Aligned"

# Gradio interface
iface = gr.Interface(
    fn=predict_feedback,
    inputs=[
        gr.Textbox(label="Feedback Text", placeholder="Enter your feedback here..."),
        gr.Textbox(label="Reason", placeholder="Enter the dropdown reason...")
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Zoom Feedback Validation",
    description="Check if feedback aligns with the selected reason."
)

# Launch interface (for local testing)
# iface.launch()

# For Hugging Face Spaces, save this script as app.py and include requirements.txt