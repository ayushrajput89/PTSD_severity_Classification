Post-Traumatic Stress Disorder (PTSD) is a mental health condition that can develop after exposure to traumatic events. Traditional PTSD assessment relies on clinical interviews and questionnaires, which are time-consuming and require trained professionals.

This project explores the use of Natural Language Processing (NLP) and Machine Learning (ML) techniques to analyze therapy-style text and predict PTSD severity levels (Low, Moderate, High). The focus of the project is not only on prediction but also on understanding linguistic patterns, data limitations, and ethical considerations.

 Note:
This project is intended for research and educational purposes only and is not a clinical diagnostic tool.

 Objectives

Preprocess and clean raw textual data

Extract sentiment, emotion, and trauma-related features

Generate semantic text representations using SBERT

Create a structured secondary dataset for analysis

Perform Exploratory Data Analysis (EDA)

Train and evaluate machine learning models

Deploy an interactive web interface using Streamlit

📂 Dataset Description
🔹 Primary Dataset

File: PTSD_data_20k.csv

~19,850 text samples

Social-media style posts with emotional content

Includes reaction counts (angry, sad, wow, shares)

Does not contain clinical PTSD labels

🔹 Secondary Dataset

File: ptsd_secondary_dataset.csv

~18,900 rows × ~36 engineered features

Created after cleaning and feature extraction

Used for EDA and modeling.
