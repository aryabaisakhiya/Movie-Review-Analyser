import streamlit as st
import torch
import torch.nn as nn
import re
from torch.nn.utils.rnn import pad_sequence

# Load vocab
vocab = torch.load("vocab.pth")
device = torch.device("cpu")

# Tokenizer and encoder
def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def encode(text):
    return [vocab.get(word, vocab["<unk>"]) for word in tokenize(text)]

# Define same RNN model architecture used in training
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# Load model
model = RNN(len(vocab))
model.load_state_dict(torch.load("imdb_rnn_model.pth", map_location=device))
model.eval()

# Streamlit interface
st.title("🎬 IMDb Movie Review Sentiment Analysis")
st.write("Enter a movie review and find out if it's **positive** or **negative**!")

review = st.text_area("Write your movie review here:")

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("⚠️ Please enter a review.")
    else:
        encoded = encode(review)
        input_tensor = torch.tensor(encoded).unsqueeze(0)
        padded = pad_sequence(input_tensor, batch_first=True)
        with torch.no_grad():
            output = model(padded)
            prob = torch.sigmoid(output).item()
            sentiment = "Positive 😊" if prob > 0.5 else "Negative 😞"
            st.success(f"**Prediction:** {sentiment}  \n**Confidence:** {prob:.2f}")
