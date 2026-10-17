import streamlit as st

def render():
    st.title("ℹ️ About This Project")
    
    st.markdown("### Project Description")
    st.write("""
    This project aims to leverage Natural Language Processing (NLP) to classify the severity of PTSD 
    symptoms based on text input. By analyzing the linguistic patterns, sentiment, and semantic meaning 
    in text, we can build machine learning models that estimate the level of distress or severity.
    """)
    
    st.markdown("### Methodology")
    st.write("""
    - **Data:** Text data collected from various sources (e.g., Reddit, clinical transcripts) with proxy labels for severity.
    - **Preprocessing:** Text normalization, stopword removal, lemmatization.
    - **Feature Engineering:** TF-IDF vectors, sentiment scores, text length metrics, LIWC-like psycholinguistic features.
    - **Modeling:** Evaluation of Logistic Regression, Random Forest, XGBoost, and deep learning architectures (e.g., BERT).
    """)
    
    st.markdown("### Applications & Scope")
    st.info("""
    **Project Scope**
    
    This system is built as an advanced AI demonstration of Natural Language Processing applied to psychological text analysis.
    It highlights the capability of machine learning models to detect subtle linguistic patterns associated with varying levels of distress.
    """)
    
    st.write("""
    **Considerations:**
    - The models are trained on real-world text data which inherently contains diverse expressions.
    - Sarcasm, metaphors, and highly complex linguistic structures are challenging but handled through deep semantic embeddings.
    - This project serves as a strong foundational architecture for predictive mental health analytics.
    """)
    
    st.markdown("### Tech Stack")
    st.write("""
    - **Frontend:** Streamlit, Plotly
    - **Data Processing:** Pandas, NumPy, NLTK/Spacy
    - **Machine Learning:** Scikit-Learn, XGBoost, Transformers (Hugging Face)
    - **Explainability:** SHAP, LIME
    """)
