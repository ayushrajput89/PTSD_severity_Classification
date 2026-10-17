import streamlit as st
from app.components.styles import metric_card

def render():
    st.title("🧠 PTSD Severity Classification")
    st.markdown("### Natural Language Processing for Mental Health")
    
    st.markdown("---")
    
    st.markdown("""
    Welcome to the PTSD Severity Classification project. This dashboard demonstrates the use of 
    Natural Language Processing (NLP) techniques to analyze text and predict the potential severity of PTSD symptoms.
    """)
    
    st.markdown("### Methodology Summary")
    st.markdown("""
    1. **Data Processing:** Clean and preprocess raw text data.
    2. **Feature Extraction:** Extract TF-IDF, LIWC-like features, sentiment, and deep embeddings (BERT).
    3. **Modeling:** Train and evaluate classical models (Logistic Regression, Random Forest, XGBoost) and deep models.
    4. **Prediction & Explainability:** Provide severity predictions along with SHAP/LIME explanations.
    """)
    
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(metric_card("Screening Accuracy", "73.8%", "+40.5% vs Baseline"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Macro F1-Score", "0.74", "Balanced Evaluation"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Transformer Target", "83.5%", "DistilBERT Benchmark"), unsafe_allow_html=True)
    with col4:
        st.markdown(metric_card("Evaluated Dataset", "18.7K", "Cleaned Texts"), unsafe_allow_html=True)

