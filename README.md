🎬 IMDb Movie Review Sentiment Analysis using RNN
<img width="1503" height="747" alt="imdb ss" src="https://github.com/user-attachments/assets/56616cba-86fa-4878-85b9-600cf40d0527" />

Welcome to the IMDb Sentiment Analysis project by Arya Baisakhiya.This project demonstrates how a Simple Recurrent Neural Network (RNN) can be used for text classification, specifically for sentiment analysis of movie reviews. The application has been built with PyTorch for model development and Streamlit for a user-friendly web interface.

📌 Table of Contents

Overview

1.Project Pipeline

2.Model Architecture

3.Technologies Used

4.Setup Instructions

5.How to Use the App

6.Credits

🎯 Overview

This project builds an end-to-end NLP pipeline to:

1.Train an RNN model on the IMDb dataset to classify reviews as positive or negative

2.Save and load the trained model for inference

3.Create an interactive Streamlit web app to input custom reviews and get real-time predictions

🔄 Project Pipeline

Data Preprocessing:

1.Tokenization

2.Lowercasing

3.Vocabulary generation

4.Sequence padding

5.Model Training:

6.RNN built using PyTorch

7.Binary sentiment classification

8.Trained using IMDb dataset

Model Saving:

-Model weights saved to imdb_rnn_model.pth

-Vocabulary saved to vocab.pth

Streamlit App:

-Loads trained model and vocab

-Accepts user input

-Predicts sentiment and shows probability/confidence score

🧠 Model Architecture

a)Embedding Layer – Converts input words to dense vectors

b)Simple RNN Layer – Learns temporal dependencies

c)Fully Connected Layer – Outputs sentiment score

d)Sigmoid Activation – Converts score to probability

🛠️ Technologies Used

-Python

-PyTorch

-Streamlit

-Regular Expressions

-IMDb Dataset

-Google Colab (for training)

🖥️ Setup Instructions (Run Locally)

⚠️ Prerequisite: Make sure you have Python 3.10+ installed.

Clone or Download the Project:

git clone https://github.com/aryabaisakhiya/Movie-Review-Analyser.git
cd rnn_sentiment_app
   
