import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("dataset.csv") 
data = data.dropna()
X = data["job_description"]
y = data["label"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X, y)

print("AI MODEL TRAINED SUCCESSFULLY!")