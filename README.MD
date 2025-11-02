# Sentiment Analysis Project

This project performs sentiment analysis on a dataset of text reviews using Python, NLTK, spaCy, and scikit-learn.

## Features
- Cleans and preprocesses text data
- Tokenization, stopword removal, and lemmatization
- TF-IDF vectorization
- Logistic Regression classifier
- Evaluation with accuracy, classification report, and confusion matrix

## Setup

1. **Clone the repository**
2. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```
3. **Download NLTK and spaCy resources:**
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')

    import spacy
    spacy.cli.download("en_core_web_sm")
    ```
4. **Place your `dataset.csv` in the project directory.**

5. **Run the notebook (`main.ipynb`) in Jupyter or VS Code.**

## Files

- `main.ipynb` — Main Jupyter notebook with all code and explanations
- `requirements.txt` — Python dependencies
- `README.md` — Project overview and instructions

## Notes

- The dataset should have at least `review` and `sentiment` columns.
- The code samples 1000 reviews for processing by default.