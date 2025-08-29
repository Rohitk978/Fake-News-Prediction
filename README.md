📰 Fake News Detection using BERT + Boosting Models
🚀 Project Overview

This project is a machine learning pipeline for detecting fake news articles. It leverages BERT embeddings for semantic text understanding and trains powerful boosting classifiers like LightGBM, XGBoost, and CatBoost to achieve reliable predictions.

Our best-performing model achieves ~82% accuracy on benchmark datasets, making it suitable for research and real-world applications.

📂 Dataset

We use the Fake News Dataset (Rajat Kumar - Kaggle)
.

Each record contains:

id: Unique article ID

title: Headline of the news

author: Author name

text: News content

label: Target variable → 1 = Fake, 0 = Real

📌 Example row:

id,title,author,text,label
0,Donald Trump Sends Out Embarrassing New Year’s Eve Message,Daniel Politi,"Donald Trump issued a bizarre, self-congratulatory message on Twitter...",1

🛠️ Tech Stack

Python 🐍

LightGBM / XGBoost / CatBoost → Gradient boosting models

BERT (Hugging Face Transformers) → Text embeddings

PyTorch → Backend for BERT

Scikit-learn → Model evaluation (Accuracy, F1-score)

NLTK → Preprocessing (stopwords, tokenization)

NumPy & Pandas → Data handling

🔑 Key Features

✅ Preprocessing with NLTK (cleaning, tokenization, stopword removal)
✅ Context-aware BERT embeddings for feature extraction
✅ Training with multiple boosting classifiers
✅ Evaluation with Accuracy and F1-score
✅ Ready-to-use trained model with joblib serialization

⚙️ Installation

Clone the repository and install dependencies:

git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt

Download NLTK resources:

python -m nltk.downloader stopwords punkt

▶️ Usage
🔹 Training the Model

Place your dataset (news.csv) in the data/ folder and run:

python train.py

This will:

Preprocess the text

Generate BERT embeddings

Train boosting models

Save the trained model to models/fake_news_model.pkl

🔹 Testing / Prediction

Use the trained model to classify new articles:

import joblib
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Load model

model = joblib.load("models/fake_news_model.pkl")

# Load BERT

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def bert_embeddings(text):
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
with torch.no_grad():
outputs = bert_model(\*\*inputs)
return outputs.last_hidden_state[:, 0, :].numpy()

# Example news

news = "Quantum Shoes let people FLY instantly, scientists don’t want you to know this!"

X_new = bert_embeddings(news)
prediction = model.predict(X_new)

print("Prediction:", "FAKE" if prediction[0] == 1 else "REAL")

📊 Evaluation

We evaluate models with Accuracy and F1-score:

Model Accuracy F1-score
LightGBM 0.82 0.80
XGBoost 0.81 0.79
CatBoost 0.81 0.78
📂 Repository Structure
fake-news-detection/
├── data/
│ └── news.csv # Dataset
├── models/
│ └── fake_news_model.pkl # Trained model
├── src/
│ ├── train.py # Training script
│ ├── predict.py # Prediction script
│ └── utils.py # Helper functions
├── requirements.txt # Dependencies
├── README.md # Documentation
└── .gitignore # Ignored files

🤝 Contributing

Contributions are welcome!

Fork the repo

Create a branch (git checkout -b feature-name)

Commit changes (git commit -m "Added new feature")

Push (git push origin feature-name)

Open a Pull Request
