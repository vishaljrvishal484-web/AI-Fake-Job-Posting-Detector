# 🔍 AI Fake Job Posting Detector

An AI-powered web application that predicts whether a job posting is REAL or FAKE using Machine Learning.

## 🚀 Features

- Real or Fake job prediction
- Confidence score
- TF-IDF text feature extraction
- Logistic Regression machine learning model
- Accuracy, Precision, Recall and F1 Score
- Confusion Matrix
- Prediction History
- Interactive Streamlit web interface

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit
- Matplotlib
- TF-IDF
- Logistic Regression

## 📊 Dataset

The dataset contains job descriptions labeled as:

- `0` → Real Job
- `1` → Fake Job

## ⚙️ How It Works

1. User enters a job description.
2. Text is converted into numerical features using TF-IDF.
3. Logistic Regression analyzes the features.
4. The model predicts Real or Fake.
5. The application displays the prediction and confidence score.

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py