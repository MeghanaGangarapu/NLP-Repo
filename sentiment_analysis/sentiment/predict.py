import joblib
from .preprocess import preprocess

def predict_sentiment(text, model_path):
    model, vectorizer = joblib.load(model_path)
    processed = " ".join(preprocess(text))
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)
    return prediction[0]