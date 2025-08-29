Fake News Detection using BERT and Boosting Models
🚀 Project Overview
This project implements a robust machine learning pipeline for detecting fake news articles. By combining the power of BERT embeddings for semantic text understanding with advanced boosting classifiers (LightGBM, XGBoost, CatBoost), the pipeline achieves reliable predictions. Our best model delivers ~82% accuracy on benchmark datasets, making it suitable for both research and real-world applications.

📂 Dataset
The project utilizes the Fake News Dataset from Kaggle by Rajat Kumar. Each record in the dataset includes:

id: Unique article identifier
title: Headline of the news article
author: Name of the author
text: Full content of the news article
label: Target variable (1 = Fake, 0 = Real)

Example Row:
id: 0
title: Donald Trump Sends Out Embarrassing New Year’s Eve Message
author: Daniel Politi
text: Donald Trump issued a bizarre, self-congratulatory message on Twitter...
label: 1 (Fake)


🛠️ Tech Stack

Python: Core programming language
BERT (Hugging Face Transformers): For context-aware text embeddings
PyTorch: Backend for BERT model operations
LightGBM, XGBoost, CatBoost: Gradient boosting classifiers
Scikit-learn: Model evaluation (Accuracy, F1-score)
NLTK: Text preprocessing (tokenization, stopword removal)
NumPy & Pandas: Data manipulation and handling


🔑 Key Features

✅ Text Preprocessing: Cleaning, tokenization, and stopword removal using NLTK
✅ BERT Embeddings: Extract context-aware features for
