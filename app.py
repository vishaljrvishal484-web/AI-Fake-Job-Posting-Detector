import streamlit as st
import pandas as pd
import os

from dotenv import load_dotenv
from supabase import create_client, Client

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="AI Fake Job Posting Detector",
    page_icon="🔍",
    layout="wide"
)


# ==============================
# LOAD ENVIRONMENT VARIABLES
# ==============================

load_dotenv(override=True)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")


# ==============================
# CHECK SUPABASE CREDENTIALS
# ==============================

if not SUPABASE_URL:
    st.error("❌ SUPABASE_URL not found in .env file.")
    st.stop()

if not SUPABASE_ANON_KEY:
    st.error("❌ SUPABASE_ANON_KEY not found in .env file.")
    st.stop()


# ==============================
# SUPABASE CONNECTION
# ==============================

try:

    supabase: Client = create_client(
        SUPABASE_URL,
        SUPABASE_ANON_KEY
    )

except Exception as e:

    st.error(
        f"❌ Supabase connection failed: {e}"
    )

    st.stop()


# ==============================
# SUPABASE CONNECTION STATUS
# ==============================

st.sidebar.success(
    "🟢 Supabase Connected"
)


# ==============================
# SAVE PREDICTION TO SUPABASE
# ==============================

def save_prediction(
    job_description,
    prediction_text,
    confidence,
    risk_level,
    suspicious_indicators
):

    try:

        prediction_data = {

            "job_description": job_description,

            "prediction": prediction_text,

            "confidence": float(confidence),

            "risk_level": risk_level,

            "suspicious_indicators": int(
                suspicious_indicators
            )

        }


        supabase.table(
            "job_predictions"
        ).insert(
            prediction_data
        ).execute()


        return True


    except Exception as e:

        st.error(
            f"❌ Database save failed: {e}"
        )

        return False


# ==============================
# GET PREDICTION HISTORY
# ==============================

def get_history():

    try:

        response = (

            supabase

            .table("job_predictions")

            .select(
                "id, "
                "job_description, "
                "prediction, "
                "confidence, "
                "risk_level, "
                "suspicious_indicators, "
                "created_at"
            )

            .order(
                "created_at",
                desc=True
            )

            .limit(10)

            .execute()

        )


        if response.data:

            return pd.DataFrame(
                response.data
            )


        return pd.DataFrame()


    except Exception as e:

        st.warning(
            f"⚠️ Could not load prediction history: {e}"
        )

        return pd.DataFrame()


# ==============================
# TITLE
# ==============================

st.title(
    "🔍 AI Fake Job Posting Detector"
)

st.caption(
    "Machine Learning based job scam detection system"
)


# ==============================
# LOAD DATASET
# ==============================

try:

    data = pd.read_csv(
        "dataset.csv"
    )

except FileNotFoundError:

    st.error(
        "❌ dataset.csv not found."
    )

    st.stop()


# ==============================
# CHECK DATASET COLUMNS
# ==============================

if (
    "job_description" not in data.columns
    or "label" not in data.columns
):

    st.error(
        "❌ dataset.csv must contain: "
        "job_description and label"
    )

    st.stop()


# ==============================
# CLEAN DATA
# ==============================

data = data.dropna(
    subset=[
        "job_description",
        "label"
    ]
)


data["job_description"] = data[
    "job_description"
].astype(str)


# ==============================
# CONVERT LABEL
# ==============================

def convert_label(value):

    value = str(
        value
    ).strip().lower()


    if value in [
        "fake",
        "1",
        "true"
    ]:

        return 1


    if value in [
        "real",
        "0",
        "false"
    ]:

        return 0


    return None


data["label"] = data[
    "label"
].apply(
    convert_label
)


data = data.dropna(
    subset=[
        "label"
    ]
)


data["label"] = data[
    "label"
].astype(int)


# ==============================
# DATASET OVERVIEW
# ==============================

st.subheader(
    "📊 Dataset Overview"
)


total_jobs = len(
    data
)


real_jobs = int(
    (
        data["label"] == 0
    ).sum()
)


fake_jobs = int(
    (
        data["label"] == 1
    ).sum()
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Total Jobs",
        total_jobs
    )


with col2:

    st.metric(
        "Real Jobs",
        real_jobs
    )


with col3:

    st.metric(
        "Fake Jobs",
        fake_jobs
    )


# ==============================
# PREPARE DATA
# ==============================

X = data[
    "job_description"
]

y = data[
    "label"
]


if len(data) < 10:

    st.error(
        "❌ Dataset is too small."
    )

    st.stop()


if y.nunique() < 2:

    st.error(
        "❌ Dataset must contain both "
        "Real and Fake jobs."
    )

    st.stop()


# ==============================
# TRAIN TEST SPLIT
# ==============================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)


# ==============================
# TF-IDF
# ==============================

vectorizer = TfidfVectorizer(

    stop_words="english",

    max_features=3000,

    ngram_range=(1, 2)

)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


X_test_tfidf = vectorizer.transform(
    X_test
)


# ==============================
# TRAIN MODEL
# ==============================

model = LogisticRegression(

    max_iter=1000,

    random_state=42

)


model.fit(

    X_train_tfidf,

    y_train

)


# ==============================
# MODEL PREDICTION
# ==============================

y_pred = model.predict(
    X_test_tfidf
)


# ==============================
# MODEL METRICS
# ==============================

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


# ==============================
# MODEL PERFORMANCE
# ==============================

st.subheader(
    "🎯 Model Performance"
)


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


# ==============================
# JOB CHECKER
# ==============================

st.divider()


st.subheader(
    "🔎 Check a Job Posting"
)


job_text = st.text_area(

    "Enter Job Description",

    placeholder=(
        "Paste the job description here..."
    ),

    height=200

)


# ==============================
# CHECK JOB
# ==============================

if st.button(
    "🔍 Check Job",
    type="primary"
):


    if job_text.strip() == "":

        st.warning(
            "⚠️ Please enter a job description."
        )

        st.stop()


    # ==============================
    # PREDICT
    # ==============================

    job_vector = vectorizer.transform(
        [job_text]
    )


    prediction = model.predict(
        job_vector
    )[0]


    probabilities = model.predict_proba(
        job_vector
    )[0]


    confidence = (
        max(probabilities) * 100
    )


    # ==============================
    # SUSPICIOUS WORDS
    # ==============================

    suspicious_words = [

        "registration fee",

        "processing fee",

        "pay money",

        "pay rs",

        "security deposit",

        "joining fee",

        "training fee",

        "membership fee",

        "send money",

        "send otp",

        "bank details",

        "upi",

        "no interview",

        "guaranteed job",

        "work from home",

        "daily income",

        "earn daily",

        "whatsapp"

    ]


    text_lower = job_text.lower()


    found_words = []


    for word in suspicious_words:

        if word in text_lower:

            found_words.append(
                word
            )


    # ==============================
    # FAKE JOB
    # ==============================

    if prediction == 1:

        prediction_text = "FAKE"


        if len(found_words) >= 3:

            risk_level = "HIGH RISK"


        elif len(found_words) >= 1:

            risk_level = "MEDIUM RISK"


        else:

            risk_level = "POTENTIAL RISK"


        st.error(
            "🚨 FAKE JOB"
        )


        st.subheader(
            f"🎯 Prediction Confidence: "
            f"{confidence:.2f}%"
        )


        st.progress(
            int(
                min(
                    confidence,
                    100
                )
            )
        )


        st.subheader(
            "⚠️ Job Risk Analysis"
        )


        st.write(
            f"Suspicious Indicators: "
            f"{len(found_words)}"
        )


        if found_words:

            st.warning(
                "Suspicious indicators found:"
            )


            for word in found_words:

                st.write(
                    f"🔴 {word}"
                )


        st.subheader(
            "🛡️ Safety Recommendations"
        )


        st.write(
            "• Do not pay registration or processing fees."
        )


        st.write(
            "• Do not share OTP or passwords."
        )


        st.write(
            "• Do not share bank account details."
        )


        st.write(
            "• Verify the company through its official website."
        )


        st.write(
            "• Never send money through UPI for getting a job."
        )


        st.info(
            f"Risk Level: {risk_level}"
        )


    # ==============================
    # REAL JOB
    # ==============================

    else:

        prediction_text = "REAL"


        risk_level = "LOW RISK"


        st.success(
            "✅ REAL JOB"
        )


        st.subheader(
            f"🎯 Prediction Confidence: "
            f"{confidence:.2f}%"
        )


        st.progress(
            int(
                min(
                    confidence,
                    100
                )
            )
        )


        st.info(
            "This job posting appears legitimate "
            "according to the trained model. "
            "Always verify the company before applying."
        )


    # ==============================
    # SAVE TO SUPABASE
    # ==============================

    saved = save_prediction(

        job_description=job_text,

        prediction_text=prediction_text,

        confidence=confidence,

        risk_level=risk_level,

        suspicious_indicators=len(
            found_words
        )

    )


    if saved:

        st.success(
            "💾 Prediction saved to Supabase successfully!"
        )


# ==============================
# PREDICTION HISTORY
# ==============================

st.divider()


st.subheader(
    "📜 Prediction History"
)


history = get_history()


if not history.empty:

    st.dataframe(

        history,

        use_container_width=True

    )

else:

    st.info(
        "No prediction history yet."
    )


# ==============================
# FOOTER
# ==============================

st.divider()


st.caption(
    "AI Fake Job Posting Detector | "
    "Python • Machine Learning • Streamlit • Supabase"
)