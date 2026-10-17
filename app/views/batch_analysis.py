import streamlit as st
import pandas as pd
import io
import os
import joblib

@st.cache_resource
def load_model():
    model_path = "E:/PTSD_Severity_NLP/models/classical/best_model.pkl"
    vectorizer_path = "E:/PTSD_Severity_NLP/models/classical/tfidf_vectorizer.pkl"
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        return joblib.load(model_path), joblib.load(vectorizer_path)
    return None, None

def render():
    st.title("🗂️ Batch Analysis")
    st.markdown("Upload multiple texts to predict their severity simultaneously.")
    st.info("You can upload a CSV file with a **'text'** column, or use the `dummy_texts.csv` provided in the project root.")
    
    model, vectorizer = load_model()
    
    if model is None or vectorizer is None:
        st.error("Saved model or vectorizer not found. Please ensure the model has been trained and saved.")
        return

    input_method = st.radio("Choose input method:", ["File Upload (CSV/TXT)", "Text Area (One per line)"])
    
    texts = []
    
    if input_method == "File Upload (CSV/TXT)":
        sample_csv = "text\n\"The nightmares keep coming back and I feel so disconnected from my family.\"\n\"I get startled easily by loud noises now, but I try to push through it.\"\n\"Today was a peaceful and productive day with friends.\"\n"
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=sample_csv,
            file_name="sample_ptsd_texts.csv",
            mime="text/csv"
        )
        uploaded_file = st.file_uploader("Upload CSV (must contain a 'text' column) or TXT (one text per line)", type=['csv', 'txt'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                if 'text' in df.columns:
                    texts = df['text'].dropna().tolist()
                else:
                    st.error("CSV must contain a column named 'text'.")
            else:
                string_data = uploaded_file.getvalue().decode("utf-8")
                texts = [line.strip() for line in string_data.split('\n') if line.strip()]
    else:
        text_area_input = st.text_area("Enter multiple texts (one per line):", height=200)
        if text_area_input:
            texts = [line.strip() for line in text_area_input.split('\n') if line.strip()]
            
    if st.button("Process Batch"):
        if not texts:
            st.warning("No texts to process.")
            return
            
        with st.spinner(f"Processing {len(texts)} texts..."):
            try:
                X_batch = vectorizer.transform(texts)
                predictions = model.predict(X_batch)
                
                confidences = []
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_batch)
                    confidences = probs.max(axis=1).tolist()
                else:
                    confidences = [1.0] * len(texts)
                    
                results_df = pd.DataFrame({
                    'Text': texts,
                    'Predicted_Severity': predictions,
                    'Confidence': [round(c, 4) for c in confidences]
                })
                
                st.markdown("### Batch Results")
                st.dataframe(results_df)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### Severity Distribution")
                    val_counts = results_df['Predicted_Severity'].value_counts().reset_index()
                    val_counts.columns = ['Severity', 'Count']
                    import plotly.express as px
                    fig = px.pie(val_counts, names='Severity', values='Count', title='Predicted Classes')
                    st.plotly_chart(fig, use_container_width=True)
                    
                with col2:
                    st.markdown("### Download Results")
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name='batch_predictions.csv',
                        mime='text/csv',
                    )
                    
            except Exception as e:
                st.error(f"Error during batch processing: {e}")
