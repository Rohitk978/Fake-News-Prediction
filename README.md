# Fake News Detection using BERT and Boosting Models

## Overview
This project implements a robust machine learning pipeline for detecting fake news articles. By leveraging BERT embeddings for semantic text understanding and combining them with advanced boosting classifiers (LightGBM, XGBoost, CatBoost), the pipeline achieves reliable predictions. The best model delivers approximately 82% accuracy on benchmark datasets, making it suitable for research and real-world applications.

## Dataset
The project uses the **Fake News Dataset** available on Kaggle: **[https://www.kaggle.com/datasets/rajatkumar/fake-news-dataset](https://www.kaggle.com/datasets/rajatkumar/fake-news-dataset)**. The dataset contains articles with the following attributes:
- **id**: Unique article identifier
- **title**: Headline of the news article
- **author**: Name of the author
- **text**: Full content of the news article
- **label**: Target variable (1 = Fake, 0 = Real)

**Example Row**:
- **id**: 0
- **title**: Donald Trump Sends Out Embarrassing New Year’s Eve Message
- **author**: Daniel Politi
- **text**: Donald Trump issued a bizarre, self-congratulatory message on Twitter...
- **label**: 1 (Fake)

## Libraries
- **Python**: Core programming language for development.
- **BERT (Hugging Face Transformers)**: For generating context-aware text embeddings.
- **PyTorch**: Backend for BERT model operations.
- **LightGBM, XGBoost, CatBoost**: Gradient boosting classifiers for prediction.
- **Scikit-learn**: For model evaluation metrics (e.g., Accuracy, F1-score).
- **NLTK**: For text preprocessing tasks like tokenization and stopword removal.
- **NumPy & Pandas**: For efficient data manipulation and handling.

## Project Logic Highlights
- **Text Preprocessing**: Cleans and preprocesses text data, including tokenization and stopword removal using NLTK.
- **BERT Embeddings**: Extracts context-aware features from news articles using BERT for enhanced semantic understanding.
- **Model Architecture**: Combines BERT embeddings with boosting classifiers (LightGBM, XGBoost, CatBoost) for robust classification.
- **Training Process**: Trains models with optimized hyperparameters, incorporating cross-validation to prevent overfitting.
- **Model Evaluation**: Evaluates performance using metrics like accuracy and F1-score to ensure reliable predictions.
- **Error Handling**: Includes checks for missing data and preprocessing errors to ensure pipeline robustness.
- **Prediction Pipeline**: Processes new articles and predicts their authenticity, mapping outputs to "Fake" or "Real" labels.

This project provides a flexible and scalable framework for fake news detection, adaptable to various datasets and use cases.
