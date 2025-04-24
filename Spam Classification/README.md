Spam Filter Models
This project demonstrates multiple approaches to building a spam classifier using Natural Language Processing (NLP). It includes implementations based on:

Bag of Words + Logistic Regression

TF-IDF + Multinomial Naive Bayes

LSTM Neural Network

The goal is to compare classical machine learning techniques with deep learning models for the task of classifying spam messages.

📚 Models Overview
1. 🔤 Bag of Words + Logistic Regression
Converts text into word frequency vectors (binary or count-based).

Uses Logistic Regression for binary classification.

Pros: Simple, fast, interpretable.

Cons: Ignores word order and context.

2. ✍️ TF-IDF + Multinomial Naive Bayes
Transforms messages using TF-IDF to account for word importance across documents.

Naive Bayes classifier is efficient and works well for text classification.

Pros: Lightweight, performs well on small/medium datasets.

Cons: Assumes feature independence.

3. 🧠 LSTM (Long Short-Term Memory)
A recurrent neural network (RNN) that learns sequential patterns in text.

Captures temporal dependencies and word order.

Pros: Learns context and sequence better than traditional methods.

Cons: Slower training, needs more data and computation.
