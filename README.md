# Spam Detection Classifier

A machine learning project that classifies text messages as **spam** or **not spam (ham)** using Natural Language Processing (NLP) and a supervised learning model.

## Overview

Spam messages often contain unsolicited promotions, suspicious links, fake prizes, or attempts to obtain personal information. This project processes message text, converts it into numerical features, and uses a trained classifier to predict whether a message is spam.

## Features

- Text preprocessing with NLTK
- Lowercasing and tokenization
- Removal of punctuation/non-alphanumeric tokens
- Stop-word removal
- Word stemming using Porter Stemmer
- Text vectorization using TF-IDF or the vectorizer configured in the training code
- Training and evaluation of a machine learning classifier
- Saving and reusing the trained model with Pickle or Joblib
- Prediction on new, unseen messages

## Project Workflow

1. **Load the dataset**  
   Read the labelled SMS/message dataset into a Pandas DataFrame.

2. **Explore and clean the data**  
   Inspect the dataset, handle missing or duplicate records if necessary, and prepare the message text and target labels.

3. **Preprocess the text**  
   The preprocessing function may:
   - Convert text to lowercase
   - Tokenize the message
   - Keep alphanumeric tokens
   - Remove English stop words
   - Apply Porter stemming

4. **Convert text into numerical features**  
   Fit the selected text vectorizer (for example, `TfidfVectorizer`) on the training messages and transform the messages into feature vectors.

5. **Train the classifier**  
   Train a scikit-learn classification model on the training features and labels.

6. **Evaluate the model**  
   Generate predictions on a held-out test set and evaluate performance using metrics such as accuracy, precision, recall, F1-score, and a confusion matrix.

7. **Save and reuse the model**  
   Save the fitted classifier and vectorizer so they can be loaded later without retraining.

## Technologies Used

- **Python**
- **Pandas** — data loading and manipulation
- **NLTK** — tokenization, stop words, and stemming
- **scikit-learn** — feature extraction, model training, and evaluation
- **Pickle / Joblib** — model persistence

## Example Usage

The following is an example for a project that saves a fitted classifier and fitted TF-IDF vectorizer as separate files. Update the filenames and preprocessing steps to match your implementation.

### Save the trained model

```python
import joblib

joblib.dump(model, "spam_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
```

### Load the model and predict

```python
import joblib

model = joblib.load("spam_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

message = "Congratulations! You have won a free prize."

# Apply the same custom preprocessing used during training, if applicable.
processed_message = transform_text(message)
processed_message = " ".join(processed_message)

features = tfidf.transform([processed_message])
prediction = model.predict(features)[0]

print("Spam" if prediction == 1 else "Not spam")
```

**Important:** The example assumes the target label `1` means spam and `0` means ham, and that the fitted vectorizer was trained on preprocessed messages joined back into strings. Check your dataset's label encoding and use exactly the same preprocessing and vectorization steps as during training.

## Evaluation

Evaluate the model on data that was not used during training. Useful classification metrics include:

- **Accuracy:** Overall proportion of correct predictions.
- **Precision:** Of the messages predicted as spam, how many were actually spam?
- **Recall:** Of all actual spam messages, how many did the model identify?
- **F1-score:** Harmonic mean of precision and recall.
- **Confusion matrix:** Counts true positives, true negatives, false positives, and false negatives.

Accuracy alone may not fully describe performance, especially if the dataset has an imbalance between spam and legitimate messages.

## Suggested Repository Structure

Your filenames may differ; adjust this structure to reflect the files you actually include.

```text
SpamDetectionClassifier/
├── data/
│   └── messages.csv
├── Preprocessing2.py
├── ModelBuilding.py
├── app.py                 # Optional interface or API
├── spam_model.pkl         # Optional saved model
├── tfidf_vectorizer.pkl   # Optional saved vectorizer
├── requirements.txt
└── README.md
```

Avoid committing sensitive information or unnecessarily large/raw datasets. If the dataset has a separate license or usage terms, include attribution and follow those terms. Saved model files can be included if they are small and useful for reproduction; otherwise, document how to generate them.

## Installation

1. Clone the repository:

   ```bash
   git clone <your-repository-url>
   cd SpamDetectionClassifier
   ```

2. Create and activate a virtual environment (optional but recommended):

   ```bash
   python -m venv .venv
   ```

   Windows PowerShell:

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```bash
   pip install pandas scikit-learn nltk joblib
   ```

4. Download the NLTK resources required by your code. Depending on your NLTK version and tokenizer, you may need:

   ```python
   import nltk
   nltk.download("stopwords")
   nltk.download("punkt")
   nltk.download("punkt_tab")
   ```

Only download the resources your implementation actually requires.

## Running the Project

Run the training script from the project directory:

```bash
python ModelBuilding.py
```

If you have created an optional interface or API, run it using the instructions for that framework (for example, Streamlit or FastAPI). Add the exact command here once the interface is implemented.

## Limitations

- Predictions depend on the quality and representativeness of the training dataset.
- New spam styles, slang, obfuscated words, and unfamiliar URLs may reduce accuracy.
- False positives are possible: legitimate messages can be classified as spam.
- Do not treat predictions as definitive proof that a message is malicious or legitimate.

## Future Improvements

- Compare multiple classifiers and tune their hyperparameters.
- Add cross-validation and systematic model evaluation.
- Review precision and recall to balance missed spam against legitimate messages incorrectly flagged as spam.
- Build a Streamlit interface or FastAPI endpoint.
- Track model versions and monitor performance on new messages.

## Disclaimer

This project is intended for educational and demonstration purposes. It should not be used as the sole basis for security, financial, or other high-impact decisions.
