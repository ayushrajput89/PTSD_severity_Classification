import streamlit as st
import os
import sys

# Configure Streamlit page
st.set_page_config(
    page_title="PTSD Severity NLP",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from app.components.styles import inject_custom_css
from app.components.sidebar import render_sidebar

# Import view modules
from app.views import (
    home,
    eda,
    prediction,
    model_comparison,
    explainability,
    batch_analysis,
    about
)

def main():
    # Inject CSS
    inject_custom_css()
    
    # Render Sidebar branding and info
    render_sidebar()
    
    # Main horizontal navigation tabs
    tab_home, tab_eda, tab_pred, tab_models, tab_xai, tab_batch, tab_about = st.tabs([
        "🏠 Home",
        "📊 Exploratory Data Analysis",
        "🧠 Prediction",
        "📈 Model Comparison",
        "🔍 Explainability",
        "🗂️ Batch Analysis",
        "ℹ️ About"
    ])
    
    try:
        with tab_home:
            home.render()
        with tab_eda:
            eda.render()
        with tab_pred:
            prediction.render()
        with tab_models:
            model_comparison.render()
        with tab_xai:
            explainability.render()
        with tab_batch:
            batch_analysis.render()
        with tab_about:
            about.render()
    except Exception as e:
        st.error(f"An error occurred while rendering the page: {e}")
        st.info("Please check the application logs or ensure all required data/models are available.")

if __name__ == "__main__":
    main()
