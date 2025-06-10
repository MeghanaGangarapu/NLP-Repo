import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import pandas as pd
import nltk
from nltk.stem.snowball import SnowballStemmer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Sentiment']= df['Sentiment'].str.strip()
    df = df[(df.Sentiment == 'Positive') | (df.Sentiment == 'Negative') | (df.Sentiment == 'Neutral')]
    df = df[['Text','Sentiment']].astype(str)
    df['clean_txt'] = df['Text'].apply(lambda text: text.lower().strip())
    df['clean_txt'] = df['clean_txt'].apply(lambda text: re.sub(r' +', ' ', text))
    df['clean_txt'] = df['clean_txt'].apply(lambda text: re.sub(r'[^\w\s]', '', text))
    df['clean_txt'] = df['clean_txt'].apply(lambda text: ' '.join(stemmer.stem(text) for word in text.split() if word not in stop_words))
    df['clean_txt'] = df['clean_txt'].astype(str)
    df['tokenized_text'] = df['clean_txt'].apply(word_tokenize)
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in stop_words])
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [word for word in x if word not in string.punctuation])
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: [stemmer.stem(word) for word in x])
    df['tokenized_text'] = df['tokenized_text'].apply(lambda x: ' '.join(x))
    df_cleaned = df[['Sentiment', 'tokenized_text']].copy()
    return df_cleaned



