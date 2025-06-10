from sentiment.train_model import train_model
from sentiment.predict import predict_sentiment

# Train the model
train_model("data/sample_reviews.csv", "sentiment_model.pkl")

# Predict
text = "I love this product! It works great."
result = predict_sentiment(text, "sentiment_model.pkl")
print(f"Predicted sentiment: {result}")
