import pandas as pd
import numpy as np
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import json

def train():
    print("Loading processed data...")
    df = pd.read_csv("data/processed/ptsd_processed.csv")
    df = df.dropna(subset=['text', 'severity_class'])
    
    X = df['text'].astype(str).to_numpy()
    y = df['severity_class'].to_numpy()
    
    print(f"Training on {len(X)} samples...")
    
    # Simple feature extraction for speed and robustness
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_vec = vectorizer.fit_transform(X)
    
    # Train test split internally
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.15, random_state=42)
    
    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }
    
    # Mapping for XGBoost
    label_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
    reverse_mapping = {0: 'Low', 1: 'Moderate', 2: 'High'}
    
    y_train_num = np.array([label_mapping[l] for l in y_train])
    y_test_num = np.array([label_mapping[l] for l in y_test])
    
    results = {}
    best_acc = 0
    best_model = None
    best_name = ""
    
    os.makedirs('models/classical', exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        if name == 'XGBoost':
            model.fit(X_train, y_train_num)
            preds_num = model.predict(X_test)
            preds = [reverse_mapping[p] for p in preds_num]
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='macro')
        cm = confusion_matrix(y_test, preds, labels=['Low', 'Moderate', 'High'])
        
        results[name] = {
            'accuracy': float(acc),
            'macro_f1': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            
    print(f"Best model is {best_name} with accuracy {best_acc:.4f}")
    
    # Save best model and vectorizer
    joblib.dump(best_model, 'models/classical/best_model.pkl')
    joblib.dump(vectorizer, 'models/classical/tfidf_vectorizer.pkl')
    
    with open('models/classical/results.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Models and results saved successfully!")

if __name__ == "__main__":
    train()
