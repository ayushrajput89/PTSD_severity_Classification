import streamlit as st
import pandas as pd
import os
from app.components.charts import (
    plot_class_distribution,
    plot_feature_distribution,
    plot_correlation_heatmap,
    plot_wordcloud
)

def render():
    st.title("📊 Exploratory Data Analysis")
    
    @st.cache_data
    def load_data():
        data_path = "E:/PTSD_Severity_NLP/data/processed/ptsd_processed.csv"
        if os.path.exists(data_path):
            return pd.read_csv(data_path)
        return None

    df = load_data()
    
    if df is None:
        st.error("Processed dataset not found. Please ensure data preprocessing has been run.")
        return
        
    st.markdown("### Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Class Distribution")
        fig_class = plot_class_distribution(df, class_col='severity_class')
        if fig_class:
            st.plotly_chart(fig_class, use_container_width=True)
            
    with col2:
        st.markdown("### Text Length Distribution by Class")
        if 'text' in df.columns:
            df['text_length'] = df['text'].astype(str).apply(len)
        if 'text_length' in df.columns and 'severity_class' in df.columns:
            fig_len = plot_feature_distribution(df, feature_col='text_length', class_col='severity_class')
            if fig_len:
                st.plotly_chart(fig_len, use_container_width=True)
                
    st.markdown("### Feature Correlation Heatmap")
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Filter only relevant engagement and length metrics
    corr_cols = [c for c in num_cols if not c.endswith('_id') and not c.startswith('unnamed')]
    if len(corr_cols) > 1:
        fig_corr = plot_correlation_heatmap(df, corr_cols)
        if fig_corr:
            st.plotly_chart(fig_corr, use_container_width=True)
            
    st.markdown("### Dataset Sample Texts by Severity")
    st.caption("Inspect raw texts from the Mendeley dataset to see the language patterns associated with each severity class.")
    if 'severity_class' in df.columns and 'text' in df.columns:
        selected_class = st.selectbox("Select Class to view samples", ['High', 'Moderate', 'Low'])
        samples = df[df['severity_class'] == selected_class]['text'].head(5).tolist()
        for i, sample in enumerate(samples):
            with st.expander(f"Sample {i+1} ({selected_class} Severity)"):
                st.write(sample)
