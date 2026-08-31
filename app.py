import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="AI Fake Job Detector",
    page_icon="🔍",
    layout="centered"
)

# =====================================================
# TITLE
# =====================================================

st.title("🔍 AI Fake Job Posting Detector")

st.write(
    "Enter a job description to check whether it is REAL or FAKE."
)

st.divider()

# =====================================================
# PREDICTION HISTORY
# =====================================================

if "history" not in st.session_state:
    st.session_state.history = []

# =====================================================
# LOAD DATASET
# =====================================================

data = pd.read_csv("dataset.csv")

data = data.dropna(
    subset=["job_description", "label"]
)

data["job_description"] = data[
    "job_description"
].astype(str)

# =====================================================
# DATASET INFORMATION
# =====================================================

st.subheader("📊 Dataset Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Jobs",
        len(data)
    )

with col2:
    st.metric(
        "Real Jobs",
        len(data[data["label"] == 0])
    )

with col3:
    st.metric(
        "Fake Jobs",
        len(data[data["label"] == 1])
    )

st.divider()

# =====================================================
# DATA PREPARATION
# =====================================================

X = data["job_description"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================================
# TF-IDF
# =====================================================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_vector = vectorizer.fit_transform(
    X_train
)

X_test_vector = vectorizer.transform(
    X_test
)

# =====================================================
# TRAIN MODEL
# =====================================================

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_vector,
    y_train
)

# =====================================================
# MODEL EVALUATION
# =====================================================

y_pred = model.predict(
    X_test_vector
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test,
    y_pred
)

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.subheader("🎯 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision * 100:.2f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall * 100:.2f}%"
    )

with col4:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

st.divider()

# =====================================================
# PERFORMANCE GRAPH
# =====================================================

st.subheader("📊 Model Performance Graph")

performance = pd.DataFrame(
    {
        "Score": [
            accuracy * 100,
            precision * 100,
            recall * 100,
            f1 * 100
        ]
    },
    index=[
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

st.bar_chart(performance)

st.divider()

# =====================================================
# CONFUSION MATRIX
# =====================================================

st.subheader("🔲 Confusion Matrix")

fig, ax = plt.subplots()

ax.imshow(cm)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(
    ["Real", "Fake"]
)

ax.set_yticklabels(
    ["Real", "Fake"]
)

for i in range(2):
    for j in range(2):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

st.pyplot(fig)

# =====================================================
# CONFUSION MATRIX TABLE
# =====================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual Real",
        "Actual Fake"
    ],
    columns=[
        "Predicted Real",
        "Predicted Fake"
    ]
)

st.table(cm_df)

st.divider()

# =====================================================
# JOB PREDICTION
# =====================================================

st.subheader("📝 Check a Job Posting")

job = st.text_area(
    "Enter Job Description",
    height=200,
    placeholder="Paste the job description here..."
)

# =====================================================
# CHECK JOB BUTTON
# =====================================================

if st.button(
    "🔍 Check Job",
    use_container_width=True
):

    if job.strip() == "":

        st.warning(
            "⚠️ Please enter a job description."
        )

    else:

        # Convert text into TF-IDF
        job_vector = vectorizer.transform(
            [job]
        )

        # Prediction
        prediction = model.predict(
            job_vector
        )[0]

        # Probability
        probability = model.predict_proba(
            job_vector
        )[0]

        # Confidence
        confidence = max(
            probability
        ) * 100

        st.divider()

        # =================================================
        # FAKE JOB
        # =================================================

        if prediction == 1:

            st.error(
                "🚨 FAKE JOB"
            )

            st.write(
                f"### Confidence: {confidence:.2f}%"
            )

            st.warning(
                "⚠️ This job posting may contain "
                "suspicious details."
            )

            st.info(
                "Do not pay registration fees, "
                "share bank details, passwords, "
                "or OTPs."
            )

            result = "FAKE JOB"

        # =================================================
        # REAL JOB
        # =================================================

        else:

            st.success(
                "✅ REAL JOB"
            )

            st.write(
                f"### Confidence: {confidence:.2f}%"
            )

            st.info(
                "This job description looks legitimate "
                "based on the trained dataset."
            )

            st.warning(
                "Always verify the company and job posting "
                "before applying."
            )

            result = "REAL JOB"

        # =================================================
        # SAVE HISTORY
        # =================================================

        st.session_state.history.append(
            [
                job[:100],
                result,
                f"{confidence:.2f}%"
            ]
        )

# =====================================================
# PREDICTION HISTORY
# =====================================================

if st.session_state.history:

    st.divider()

    st.subheader(
        "📜 Prediction History"
    )

    history_df = pd.DataFrame(
        st.session_state.history,
        columns=[
            "Job Description",
            "Prediction",
            "Confidence"
        ]
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "AI Fake Job Posting Detector | "
    "Python + Machine Learning"
)