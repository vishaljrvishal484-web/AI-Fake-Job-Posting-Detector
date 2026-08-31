import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_model():

    # Load dataset
    data = pd.read_csv("dataset.csv")

    # Remove empty values
    data = data.dropna(
        subset=["job_description", "label"]
    )

    data["job_description"] = data["job_description"].astype(str)

    # Input and output
    X = data["job_description"]
    y = data["label"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )

    X_train_vector = vectorizer.fit_transform(X_train)

    # Train model
    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_vector,
        y_train
    )

    return model, vectorizer


def predict_job(job_description):

    # Train model
    model, vectorizer = train_model()

    # Convert job description
    job_vector = vectorizer.transform(
        [job_description]
    )

    # Prediction
    prediction = model.predict(
        job_vector
    )[0]

    # Probability
    probabilities = model.predict_proba(
        job_vector
    )[0]

    confidence = max(probabilities) * 100

    # Result
    if prediction == 1:
        result = "FAKE JOB"
    else:
        result = "REAL JOB"

    return result, confidence


# Test
if __name__ == "__main__":

    job = input(
        "Enter Job Description: "
    )

    result, confidence = predict_job(job)

    print("\nResult:", result)
    print(
        f"Confidence: {confidence:.2f}%"
    )