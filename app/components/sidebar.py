import streamlit as st

def render_sidebar():
    with st.sidebar:
        st.title("🧠 PTSD Severity NLP")
        st.markdown("**Mental Health Text Classification & Explainable AI**")
        st.markdown("---")
        st.markdown("### 📌 About Project")
        st.info(
            "This dashboard leverages NLP & Machine Learning to classify PTSD "
            "severity from text into Low, Moderate, and High classes, backed by "
            "Explainable AI (LIME)."
        )
