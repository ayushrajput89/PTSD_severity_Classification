import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import os
import nltk
import gdown
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from empath import Empath
from nrclex import NRCLex
import textstat

# --------------------------------------------------
# SETUP: Download NLTK and Large Files
# --------------------------------------------------
@st.cache_resource
def initial_setup():
    # 1. Download NLTK corpora to fix MissingCorpusError
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
    # 2. Download Large Files from Google Drive
    files_to_download = {
        "ptsd_secondary_dataset.npz": "1M-sPAdD5M0tedxHe4byan4j9ANw4QRIc",
        "sbert_embeddings.npy": "1Se3eXuc6-3v8hTh_nV41xH1s_bnSQ65G"
    }
    
    for filename, file_id in files_to_download.items():
        if not os.path.exists(filename):
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, filename, quiet=False)

initial_setup()

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="PTSD Severity Prediction",
    layout="wide"
)

# --------------------------------------------------
# Load model package
# --------------------------------------------------
MODEL_FILE = "final_ptsd_textonly_model.pkl"

if not os.path.exists(MODEL_FILE):
    st.error(f"Model file '{MODEL_FILE}' not found. Please upload it to your GitHub repository.")
    st.stop()

model_pkg = joblib.load(MODEL_FILE)

model = model_pkg["model"]            # Logistic Regression
scaler = model_pkg["scaler"]
meta_cols = model_pkg["meta_cols"]
meta_means = model_pkg["meta_means"]
label_map = model_pkg["label_map"]

# --------------------------------------------------
# Load SBERT
# --------------------------------------------------
@st.cache_resource
def load_sbert():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sbert = load_sbert()

# --------------------------------------------------
# NLP tools
# --------------------------------------------------
vader = SentimentIntensityAnalyzer()
empath = Empath()

empath_categories = [
    "fear", "violence", "sadness", "anger",
    "aggression", "negative_emotion",
    "suffering", "death"
]

nrc_emotions = [
    "anger", "fear", "sadness", "disgust",
    "joy", "trust", "anticipation", "surprise"
]

# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text)

def compute_text_features(text):
    feats = {}
    tokens = re.findall(r"\w+", text)

    feats["sentiment"] = vader.polarity_scores(text)["compound"]

    trauma_words = {
        "trauma", "assault", "attack", "kill", "killed",
        "violence", "fear", "panic", "death", "abuse",
        "nightmare", "flashback"
    }
    feats["trauma_count"] = sum(1 for w in tokens if w in trauma_words)

    empath_scores = empath.analyze(text, categories=empath_categories)
    for c in empath_categories:
        feats[f"empath_{c}"] = empath_scores.get(c, 0)

    # Use NRCLex
    emo_obj = NRCLex(text)
    emo = emo_obj.raw_emotion_scores
    for e in nrc_emotions:
        feats[f"nrc_{e}"] = emo.get(e, 0)

    feats["word_count"] = len(tokens)
    feats["sentence_count"] = textstat.sentence_count(text)
    feats["flesch"] = textstat.flesch_reading_ease(text)
    feats["smog"] = textstat.smog_index(text)
    feats["dale_chall"] = textstat.dale_chall_readability_score(text)

    toxic_words = {"attack", "kill", "shoot", "gun", "bomb", "assault"}
    feats["toxicity"] = sum(1 for w in tokens if w in toxic_words) / max(len(tokens), 1)

    return feats

def build_feature_vector(text):
    text = clean_text(text)
    emb = sbert.encode([text])[0]
    meta = compute_text_features(text)
    meta_vec = []

    for col in meta_cols:
        meta_vec.append(meta.get(col, meta_means[col]))

    meta_vec = np.array(meta_vec).reshape(1, -1)
    meta_scaled = scaler.transform(meta_vec)

    X = np.hstack([emb.reshape(1, -1), meta_scaled])
    return X, meta

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Prediction", "EDA"])

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------
if page == "Home":
    st.title("PTSD Severity Prediction System")
    st.markdown("""
    **Project Title** *Language and Sentiment Analysis of Therapy Session Transcripts for PTSD Severity Prediction*

    **Final Model**
    - Logistic Regression (Text-only)

    **Features Used**
    - SBERT sentence embeddings  
    - Sentiment polarity  
    - Trauma lexicon counts  
    - Empath & NRC emotion features  
    - Readability & toxicity metrics  
    """)

# --------------------------------------------------
# PREDICTION PAGE
# --------------------------------------------------
elif page == "Prediction":
    st.title("PTSD Severity Prediction")
    mode = st.radio("Input type", ["Custom Text", "Sample from Dataset"])

    sample_text = ""

    if mode == "Custom Text":
        sample_text = st.text_area("Enter therapy-style text", height=180)
    else:
        # Load the CSV file - ensure you have uploaded this to GitHub too!
        if os.path.exists("ptsd_secondary_dataset.csv"):
            df = pd.read_csv("ptsd_secondary_dataset.csv")
            idx = st.selectbox("Select a sample", df.index)
            sample_text = df.loc[idx, "text"]
            st.write(sample_text)
        else:
            st.error("ptsd_secondary_dataset.csv not found in repository.")

    if st.button("Predict"):
        if len(sample_text.strip()) < 10:
            st.warning("Please enter more meaningful text.")
        else:
            X, meta = build_feature_vector(sample_text)
            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]

            pred_name = label_map[pred]

            color_map = {
                "Low": "#2ECC71",
                "Moderate": "#F1C40F",
                "High": "#E74C3C"
            }

            st.markdown(
                f"""
                <div style="padding:18px;
                            border-radius:12px;
                            background-color:{color_map[pred_name]};
                            color:black;
                            text-align:center;
                            font-size:26px;
                            font-weight:bold;">
                    Predicted Severity: {pred_name}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.subheader("Prediction Confidence")
            prob_df = pd.DataFrame({
                "Severity": [label_map[i] for i in range(3)],
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Severity"))

            st.subheader("Why this severity?")
            if meta["trauma_count"] > 0:
                st.write("🔴 Trauma-related keywords detected")
            if meta["sentiment"] < -0.4:
                st.write("🔴 Strong negative emotional tone")
            if meta.get("empath_fear", 0) > 0:
                st.write("🔴 Fear-related language present")
            if meta["toxicity"] > 0:
                st.write("🔴 Threatening or violent terms found")

            with st.expander("View extracted features"):
                st.json(meta)

# --------------------------------------------------
# EDA PAGE
# --------------------------------------------------
elif page == "EDA":
    st.title("Exploratory Data Analysis (EDA)")
    if os.path.exists("ptsd_secondary_dataset.csv"):
        df = pd.read_csv("ptsd_secondary_dataset.csv")
        st.markdown(
            "This section provides **descriptive insights** into the dataset. "
            "These visualizations do **not** affect model predictions."
        )

        st.subheader("1️⃣ Severity Class Distribution")
        st.bar_chart(df["severity_class"].value_counts())

        st.subheader("2️⃣ Severity Score Distribution")
        st.bar_chart(np.histogram(df["severity_score"], bins=30)[0])

        st.subheader("3️⃣ Trauma Count vs Severity Class")
        st.bar_chart(df.groupby("severity_class")["trauma_count"].mean())

        st.subheader("4️⃣ Correlation Matrix (Key Features)")
        corr_features = ["severity_score", "trauma_count", "sentiment", "empath_fear", "nrc_fear"]
        corr = df[corr_features].corr()
        st.dataframe(corr.style.background_gradient(cmap="coolwarm"))

        st.subheader("5️⃣ Sentiment vs Severity Class")
        st.bar_chart(df.groupby("severity_class")["sentiment"].mean())
    else:
        st.error("ptsd_secondary_dataset.csv not found.")

