This project is a basic framework for building and experimenting with sentiment analysis pipelines.

## Structure
- `sentiment_analysis/`: Core Python code (cleaning, modeling)
- `notebooks/`: Jupyter notebooks for exploration and experiments
- `tests/`: Basic unit tests

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# .venv/bin/pip install scikit-learn -> to install libraries from the virtual env

```

## Run Tests
```bash
pytest tests/
```

## Expand Ideas
- Add support for advanced models (BERT, LSTM)
- Incorporate external datasets (IMDB, Twitter, etc.)
- Evaluate models with ROC, precision/recall