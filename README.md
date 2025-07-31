
---

````markdown
# 🎬 IMDb Movie Review Sentiment Analysis using RNN

Welcome to the IMDb Sentiment Analysis project by **Arya Baisakhiya**.  
This project demonstrates how a Simple Recurrent Neural Network (RNN) can be used for text classification, specifically for **sentiment analysis** of movie reviews. The application has been built with **PyTorch** for model development and **Streamlit** for a user-friendly web interface.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Project Pipeline](#project-pipeline)
- [Model Architecture](#model-architecture)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [How to Use the App](#how-to-use-the-app)
- [Credits](#credits)

---

## 🎯 Overview

This project builds an **end-to-end NLP pipeline** to:
- Train an RNN model on the IMDb dataset to classify reviews as **positive** or **negative**
- Save and load the trained model for inference
- Create an interactive **Streamlit web app** to input custom reviews and get real-time predictions

---

## 🔄 Project Pipeline

1. **Data Preprocessing**:  
   - Tokenization  
   - Lowercasing  
   - Vocabulary generation  
   - Sequence padding

2. **Model Training**:  
   - RNN built using PyTorch  
   - Binary sentiment classification  
   - Trained using IMDb dataset

3. **Model Saving**:  
   - Model weights saved to `imdb_rnn_model.pth`  
   - Vocabulary saved to `vocab.pth`

4. **Streamlit App**:  
   - Loads trained model and vocab  
   - Accepts user input  
   - Predicts sentiment and shows probability/confidence score

---

## 🧠 Model Architecture

- **Embedding Layer** – Converts input words to dense vectors  
- **Simple RNN Layer** – Learns temporal dependencies  
- **Fully Connected Layer** – Outputs sentiment score  
- **Sigmoid Activation** – Converts score to probability

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Streamlit
- Regular Expressions
- IMDb Dataset
- Google Colab (for training)

---

## 🖥️ Setup Instructions (Run Locally)

> ⚠️ Prerequisite: Make sure you have Python 3.10+ installed.

### 1. Clone or Download the Project

```bash
git clone https://github.com/yourusername/rnn_sentiment_app.git
cd rnn_sentiment_app
````

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or individually:

```bash
pip install torch streamlit
```

### 3. Place Model Files

Ensure the following files (trained in Colab) are in the project root:

```
imdb_rnn_model.pth
vocab.pth
```

### 4. Run the App

```bash
python -m streamlit run app.py
```

> The app will open in your browser at: `http://localhost:8501`

---

## 🎮 How to Use the App

1. Enter a movie review in the text box.
2. Click **Predict Sentiment**.
3. See the model’s prediction (**Positive** or **Negative**) and its confidence level.

---

## 👩‍💻 Developer

**Arya Baisakhiya**
Engineering Student | Data Science Enthusiast
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

---

## 📚 Acknowledgements

* IMDb dataset: [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/)
* Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)
* PyTorch Tutorials: [https://pytorch.org/tutorials/](https://pytorch.org/tutorials/)

---

## 📝 License

This project is licensed under the MIT License. Feel free to use, modify, and share!

```

---

Let me know if you'd like this saved as a downloadable `.md` file or help customizing the GitHub repo further.
```
