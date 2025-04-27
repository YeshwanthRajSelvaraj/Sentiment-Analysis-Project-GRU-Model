# Sentiment-Analysis-Project-GRU-Model-
 The project uses Word2Vec embeddings for text representation and employs a GRU network for sentiment classification.
# Sentiment Analysis Model (GRU)

## Overview
This repository contains a **Sentiment Analysis** model built using **GRU (Gated Recurrent Unit)** to classify movie reviews from the IMDB dataset into **positive** and **negative** sentiments. The model processes textual data by leveraging **Word2Vec embeddings** and performs sentiment classification with high accuracy.

## Features
- **Text Preprocessing**: The text data is cleaned and tokenized by removing stopwords and non-alphabetical characters.
- **Word2Vec Embedding**: The model uses a **Word2Vec** model trained on the IMDB dataset to convert text data into numerical embeddings.
- **GRU-based Sentiment Classifier**: A **GRU** model is employed for sentiment classification, utilizing a 2-layer architecture to learn and predict sentiment from the embeddings.
- **Model Evaluation**: The trained model is evaluated using a test dataset, providing an accuracy score of sentiment classification.

## Key Components
1. **Data Loading**: Loads the IMDB dataset containing reviews and sentiment labels.
2. **Text Cleaning**: Cleans and preprocesses the text data (removes stopwords, non-alphabetic characters).
3. **Word2Vec Training**: Trains a Word2Vec model to generate embeddings for each word in the reviews.
4. **GRU Model**: The GRU-based deep learning model classifies the sentiment based on the input embeddings.
5. **Model Evaluation**: The performance of the model is evaluated on a separate test dataset, and accuracy is reported.

## Installation

### Prerequisites
- Python 3.6+
- Required libraries:
  - `torch`
  - `torchvision`
  - `pandas`
  - `numpy`
  - `gensim`
  - `nltk`
  - `sklearn`

To install the required libraries, use the following command:

```bash
pip install -r requirements.txt
