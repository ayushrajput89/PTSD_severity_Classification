import streamlit as st
import os
import joblib
import pandas as pd
import numpy as np
import sys

# Ensure src is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from app.components.styles import severity_badge
from app.components.charts import plot_confidence_bars

@st.cache_resource
def load_model():
    model_path = "E:/PTSD_Severity_NLP/models/classical/best_model.pkl"
    vectorizer_path = "E:/PTSD_Severity_NLP/models/classical/tfidf_vectorizer.pkl"
    
    try:
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
    return None, None

def render():
    st.title("🎯 Prediction")
    
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.error("Saved model or vectorizer not found. Please ensure the model has been trained and saved.")
        return

    st.markdown("### Enter Text for Severity Prediction")
    
    # Input options
    input_method = st.radio("Choose input method:", ["Type/Paste Text", "Use Sample"])
    
    text_input = ""
    if input_method == "Type/Paste Text":
        text_input = st.text_area("Input text here:", height=150)
    else:
        sample_texts = [
            "The nightmares are relentless. I wake up screaming and sweating. I can't leave the house without having a panic attack.",
            "Ever since the accident, I can't even look at a car without shaking. I feel completely disconnected from my body.",
            "Sometimes I remember the event and get sad or anxious, but I've been managing to keep up with my routine.",
            "I get startled easily by loud noises now, and sleep has been a bit difficult, but I try to push through it.",
            "I've been feeling okay recently, just normal day to day stuff. Nothing major to report.",
            "Work was a bit stressful this week, but I'm looking forward to relaxing this weekend with my family.",
            # Extended / Long narrative samples
            "[Long Narrative - High Severity] It has been three months since the explosion, but every time I close my eyes, the sound of breaking glass and screaming replays on a loop. I wake up drenched in sweat multiple times every night gasping for air. Crowds feel like immediate death traps to me, so I have stopped going to the grocery store or stepping outside my apartment entirely. My hands tremble whenever I hear a car backfire or thunder outside. I feel utterly numb, detached from my own family, as if my mind was shattered in that moment and I cannot pull the pieces back together.",
            "[Long Narrative - Moderate Severity] Life has mostly returned to a standard rhythm, but there are certain triggers that completely derail my afternoon. Passing by the intersection where the collision happened still makes my stomach drop and my pulse spike. I find myself checking the door locks three or four times before sleeping. While I am able to work and take care of daily responsibilities, there is a persistent underlying restlessness and fatigue that makes it difficult to feel truly relaxed or at peace.",
            "[Long Narrative - Low Severity] Today was a fairly productive and peaceful day. I spent the morning preparing reports for work, grabbed lunch with a couple of colleagues, and went for an evening jog in the park. While there are occasional moments of professional deadlines and everyday life stress, I feel grounded, optimistic about the future, and enjoy spending quiet evenings reading and unwinding with friends."
        ]
        text_input = st.selectbox("Select a sample:", sample_texts)
        
    if st.button("Predict Severity"):
        if text_input.strip():
            with st.spinner("Analyzing text..."):
                try:
                    # Feature extraction
                    X_input = vectorizer.transform([text_input])
                    
                    # Prediction
                    prediction = model.predict(X_input)[0]
                    
                    # Probabilities if available
                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(X_input)[0]
                        confidence = np.max(probabilities)
                        
                        # Handle classes from model
                        classes = model.classes_
                        class_names = [f"Class {c}" for c in classes]
                    else:
                        probabilities = None
                        confidence = 1.0 # Default if no proba
                        class_names = []

                    st.markdown("### Prediction Results")
                    
                    # Custom badge
                    st.markdown(severity_badge(prediction, confidence), unsafe_allow_html=True)
                    
                    # Confidence chart
                    if probabilities is not None:
                        fig = plot_confidence_bars(probabilities, class_names)
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # Extracted features dummy representation (Since we only use TF-IDF here)
                    with st.expander("View extracted text features"):
                        st.write(f"Text Length: {len(text_input)} characters")
                        st.write(f"Words: {len(text_input.split())}")
                        
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some text to predict.")

