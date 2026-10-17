import streamlit as st
import os

def render():
    st.title("🔍 Explainability")
    st.markdown("Understand **why** the model predicts a specific severity level using **LIME (Local Interpretable Model-agnostic Explanations)**.")
    
    col_input, col_sample = st.columns([2, 1])
    with col_sample:
        st.markdown("**Quick Preset Samples:**")
        sample_choice = st.selectbox(
            "Select a test sample:",
            [
                "Custom Text (Type below)",
                "High Severity: Nightmares, flashbacks & panic attacks",
                "Moderate Severity: Startled by noises & work anxiety",
                "Low Severity: Normal peaceful day with friends"
            ]
        )
    
    default_text = "The nightmares keep coming back and I feel so disconnected from my family."
    if "High" in sample_choice:
        default_text = "The nightmares are relentless. I wake up screaming and sweating. I can't leave the house without having a panic attack, feeling like the trauma is happening all over again."
    elif "Moderate" in sample_choice:
        default_text = "Sometimes I remember the event and get anxious. I get startled easily by loud noises now, but I try to push through it and work."
    elif "Low" in sample_choice:
        default_text = "Today was a fairly productive and peaceful day. I spent the morning preparing reports and relaxing with friends."

    with col_input:
        text_input = st.text_area("Enter text to explain:", value=default_text, height=140)
                              
    if st.button("Generate Explanation", type="primary"):
        import joblib
        from lime.lime_text import LimeTextExplainer
        import streamlit.components.v1 as components
        
        try:
            model = joblib.load("models/classical/best_model.pkl")
            vectorizer = joblib.load("models/classical/tfidf_vectorizer.pkl")
        except:
            st.error("Saved model or vectorizer not found. Please ensure training is complete.")
            return
            
        with st.spinner("Perturbing text words & generating LIME explanations..."):
            explainer = LimeTextExplainer(class_names=['Low', 'Moderate', 'High'])
            
            def predict_proba(texts):
                X = vectorizer.transform(texts)
                return model.predict_proba(X)
                
            exp = explainer.explain_instance(text_input, predict_proba, num_features=12)
            
            pred_idx = model.predict(vectorizer.transform([text_input]))[0]
            
            st.success(f"**Predicted Severity:** `{pred_idx}`")
            
            # Extract word weights for clean display
            word_weights = exp.as_list()
            pos_words = [f"**{w}** (+{val:.3f})" for w, val in word_weights if val > 0]
            neg_words = [f"**{w}** ({val:.3f})" for w, val in word_weights if val < 0]
            
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("##### 🔺 Words Driving Higher Severity")
                st.write(", ".join(pos_words) if pos_words else "None detected")
            with col_neg:
                st.markdown("##### 🔻 Words Pulling Towards Lower Severity")
                st.write(", ".join(neg_words) if neg_words else "None detected")
                
            st.markdown("---")
            st.markdown("### Interactive LIME Diagnostic View")
            html = exp.as_html()
            components.html(html, height=750, scrolling=True)
