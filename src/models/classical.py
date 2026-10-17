import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_auc_score
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = logging.getLogger(__name__)

def get_models() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Returns a dictionary of model name to (model_instance, param_grid)."""
    models = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42),
            {'C': [0.1, 1.0, 10.0], 'penalty': ['l2']}
        ),
        'SVM (Linear)': (
            LinearSVC(max_iter=2000, random_state=42),
            {'C': [0.1, 1.0, 10.0]}
        ),
        'SVM (RBF)': (
            SVC(probability=True, random_state=42),
            {'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']}
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42),
            {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        ),
        'XGBoost': (
            XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]}
        ),
        'LightGBM': (
            LGBMClassifier(random_state=42),
            {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1], 'max_depth': [-1, 5]}
        )
    }
    return models

def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Trains a single model."""
    logger.info(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """Evaluates a single model and returns metrics."""
    logger.info(f"Evaluating {model.__class__.__name__}...")
    
    predictions = model.predict(X_test)
    
    # Try to get probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
    elif hasattr(model, "decision_function"):
        # For LinearSVC which doesn't have predict_proba
        d = model.decision_function(X_test)
        # Softmax approximation
        probabilities = np.exp(d) / np.sum(np.exp(d), axis=1, keepdims=True)
    else:
        # Dummy probabilities
        probabilities = np.zeros((len(predictions), len(class_names)))
        for i, p in enumerate(predictions):
            probabilities[i, p] = 1.0

    metrics = {
        'accuracy': accuracy_score(y_test, predictions),
        'macro_f1': f1_score(y_test, predictions, average='macro'),
        'weighted_f1': f1_score(y_test, predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'classification_report': classification_report(y_test, predictions, target_names=class_names),
        'predictions': predictions.tolist(),
        'probabilities': probabilities.tolist()
    }
    
    # Per-class metrics
    precision = precision_score(y_test, predictions, average=None, zero_division=0)
    recall = recall_score(y_test, predictions, average=None, zero_division=0)
    f1 = f1_score(y_test, predictions, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, cls in enumerate(class_names):
        metrics['per_class'][cls] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i])
        }
        
    # ROC AUC
    if len(class_names) == 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, probabilities[:, 1])
        except Exception:
            metrics['roc_auc'] = None
    else:
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, probabilities, multi_class='ovr')
        except Exception:
            metrics['roc_auc'] = None

    return metrics

def hyperparameter_search(model: Any, param_grid: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, cv: int = 5, scoring: str = 'f1_macro') -> Tuple[Any, Dict[str, Any]]:
    """Performs grid search to find best hyperparameters."""
    logger.info(f"Starting hyperparameter search for {model.__class__.__name__}...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_, grid_search.best_params_

def compare_models(models_dict: Dict[str, Tuple[Any, Dict[str, Any]]], X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, class_names: List[str]) -> Dict[str, Dict[str, Any]]:
    """Trains and evaluates all models, returns results dictionary and prints comparison."""
    results = {}
    
    print("-" * 80)
    print(f"{'Model':<20} | {'Accuracy':<10} | {'Macro F1':<10} | {'Weighted F1':<11}")
    print("-" * 80)
    
    for name, (model, param_grid) in models_dict.items():
        logger.info(f"Processing model: {name}")
        
        # Hyperparameter search and training
        best_model, best_params = hyperparameter_search(model, param_grid, X_train, y_train)
        
        # Evaluation
        metrics = evaluate_model(best_model, X_test, y_test, class_names)
        metrics['best_params'] = best_params
        
        results[name] = metrics
        
        print(f"{name:<20} | {metrics['accuracy']:.4f}     | {metrics['macro_f1']:.4f}     | {metrics['weighted_f1']:.4f}")
        
    print("-" * 80)
    return results
