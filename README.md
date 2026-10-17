# PTSD Severity Classification Using NLP

A text classification system that predicts PTSD severity levels from social media text using Natural Language Processing and Machine Learning. Built as an MTech CSE project to explore how NLP techniques can assist in mental health screening through textual analysis.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What This Project Does

This project takes text data from social media posts and classifies them into PTSD severity levels (Low, Moderate, High). It uses a combination of TF-IDF text features with classical machine learning models (Logistic Regression, Random Forest, XGBoost) to make predictions.

The project includes two evaluation benchmarks:

- **3-Class Classification** (Low / Moderate / High) — fine-grained severity prediction
- **Clinical Screening** (High vs Low) — binary classification for screening use cases

An interactive Streamlit dashboard provides real-time prediction, model explainability using LIME, batch text processing, and exploratory data analysis.

---

## Results

### Clinical Screening Benchmark (High vs Low)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **Logistic Regression** | **73.4%** | **0.734** |
| XGBoost | 69.6% | 0.695 |
| Random Forest | 69.6% | 0.695 |

### 3-Class Benchmark (Low / Moderate / High)

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| **Logistic Regression** | **49.7%** | **0.494** |
| Random Forest | 47.0% | 0.415 |
| XGBoost | 46.6% | 0.442 |

> The 3-class accuracy around 47-50% is expected behavior — not a bug. The proxy severity labels are computed from continuous engagement scores split at quantile boundaries. The "Moderate" class heavily overlaps with both "Low" and "High", creating fuzzy decision boundaries that limit classical TF-IDF models. The random baseline for 3 classes is 33.3%, so models are learning meaningful patterns. The Clinical Screening benchmark removes this ambiguous middle class and achieves 73.4% accuracy on genuinely unseen test data (1,894 samples).

---

## Dataset

**Source:** [Mendeley Data — Social Media Engagement Dataset for PTSD Analysis](https://data.mendeley.com/datasets/3sjms2g2mf/1)

- ~20,000 Facebook/BBC news posts with engagement reactions
- Processed to 18,730 text samples after cleaning
- **Proxy severity labeling:** Since clinical PTSD labels are not available, severity is computed using weighted engagement reactions:
  ```
  severity_score = react_angry × 3 + react_sad × 2 + react_wow × 1
  ```
- Scores are split into 3 classes (Low, Moderate, High) using 33rd and 66th percentile thresholds

---

## NLP Pipeline

### Text Preprocessing
- Lowercasing, URL removal, HTML tag removal
- Email and mention cleanup
- Special character removal (keeping apostrophes for contractions)
- Whitespace normalization
- Minimum text length filtering (15 characters)
- Duplicate removal

### Feature Extraction
- **TF-IDF Vectorization** — unigram + bigram features (12,000 features) using scikit-learn's `TfidfVectorizer`
- Used as primary features for all classical ML models

### NLP Libraries Used
| Library | Purpose | Where Used |
|---------|---------|------------|
| **scikit-learn** (`TfidfVectorizer`) | Text to numerical features (TF-IDF) | Feature extraction pipeline (`train_improved_models.py`) |
| **NLTK** | Tokenization, text processing utilities | Text preprocessing (`src/data/preprocessor.py`) |
| **LIME** (`lime.lime_text`) | Word-level model explanations | Explainability tab in dashboard (`src/explainability/lime_explainer.py`) |
| **VADER** (`vaderSentiment`) | Sentiment analysis scoring | Sentiment feature module (`src/features/sentiment.py`) |
| **TextBlob** | Polarity and subjectivity analysis | Sentiment feature module (`src/features/sentiment.py`) |
| **NRCLex** | Emotion detection (anger, fear, sadness etc.) | Emotion feature module (`src/features/sentiment.py`) |
| **spaCy** | Linguistic analysis, POS tagging | Linguistic features (`src/features/linguistic.py`) |
| **TextStat** | Readability metrics (Flesch, SMOG) | Readability scoring (`src/features/linguistic.py`) |
| **Sentence-Transformers** | SBERT semantic embeddings (384-dim) | Embedding module (`src/features/embeddings.py`) |

### Models Trained
- **Logistic Regression** — best performing classical model
- **Random Forest** — ensemble tree-based approach
- **XGBoost** — gradient boosted trees

### Model Explainability
- **LIME (Local Interpretable Model-agnostic Explanations)** — shows which words contributed to each prediction with positive/negative weight visualization

---

## Project Structure

```
PTSD_Severity_NLP/
├── app/                          # Streamlit dashboard
│   ├── main.py                   # Main app with tab navigation
│   ├── .streamlit/config.toml    # Streamlit theme config
│   ├── components/               # Reusable UI components
│   │   ├── charts.py             # Plotly chart functions
│   │   ├── sidebar.py            # Sidebar branding
│   │   └── styles.py             # CSS and metric cards
│   └── views/                    # Dashboard tab pages
│       ├── home.py               # Home with key metrics
│       ├── eda.py                # Exploratory data analysis
│       ├── prediction.py         # Real-time text prediction
│       ├── model_comparison.py   # Model benchmarks
│       ├── explainability.py     # LIME explanations
│       ├── batch_analysis.py     # Batch CSV processing
│       └── about.py              # Project info
├── config/
│   └── config.yaml               # Project configuration
├── data/
│   ├── raw/                      # Original dataset
│   ├── processed/                # Cleaned data
│   └── splits/                   # Train/val/test splits
├── models/
│   └── classical/                # Saved model artifacts
│       ├── best_model.pkl        # Trained Logistic Regression
│       ├── tfidf_vectorizer.pkl  # Fitted TF-IDF vectorizer
│       └── results.json          # All model metrics
├── src/                          # Source modules
│   ├── data/                     # Data loading, preprocessing, labeling
│   ├── features/                 # Feature engineering (sentiment, linguistic, lexicon, embeddings)
│   ├── models/                   # Model training, evaluation
│   └── explainability/           # LIME, SHAP, attention visualization
├── tests/                        # Unit tests
├── train_improved_models.py      # Main training script
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Dashboard

The Streamlit dashboard has 7 tabs:

| Tab | What It Shows |
|-----|---------------|
| **Home** | Key metrics overview (73.8% screening accuracy, 0.74 F1 score) |
| **EDA** | Dataset statistics, class distribution, text length analysis, correlation heatmap, sample texts |
| **Prediction** | Enter any text or select from sample texts to get severity prediction with confidence scores |
| **Model Comparison** | Side-by-side comparison of all models with confusion matrices and per-class metrics |
| **Explainability** | LIME word-level explanations showing why the model made a specific prediction |
| **Batch Analysis** | Upload a CSV/TXT file to classify multiple texts at once, download results |
| **About** | Project description, methodology, and tech stack |

---

## How to Run

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/ayushrajput89/PTSD_severity_Classification.git
cd PTSD_severity_Classification

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run app/main.py
```

Opens at `http://localhost:8501`

### Retrain Models (optional)

```bash
python train_improved_models.py
```

This will train all models from scratch and save results to `models/classical/`.

---

## Tech Stack

- **Language:** Python 3.12
- **ML Framework:** Scikit-learn, XGBoost
- **NLP:** NLTK, spaCy, VADER, TextBlob, NRCLex, TF-IDF
- **Explainability:** LIME
- **Dashboard:** Streamlit, Plotly
- **Data:** Pandas, NumPy

---

## Limitations

- Severity labels are proxy-based (derived from engagement reactions), not clinical diagnoses
- This is a research/academic project, not a clinical diagnostic tool
- Social media text has inherent biases and may not generalize to clinical text
- The "Moderate" severity class has significant overlap with adjacent classes due to the continuous nature of engagement scores

---

## License

MIT License — see [LICENSE](LICENSE) file.

## Author

**Ayush Rajput** — MTech CSE
