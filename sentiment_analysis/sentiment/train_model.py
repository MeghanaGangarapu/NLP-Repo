import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from .preprocess import preprocess

def train_model(csv_path, model_path):
    df = pd.read_csv(csv_path)
    texts = df["Text"].apply(lambda x: " ".join(preprocess(x)))
    labels = df["Sentiment"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression()
    model.fit(X, labels)

    joblib.dump((model, vectorizer), model_path)
    print(f"Model trained and saved to {model_path}")