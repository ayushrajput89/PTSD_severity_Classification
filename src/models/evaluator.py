import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef
)

logger = logging.getLogger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, class_names: List[str]) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    # Per-class metrics
    precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    metrics['per_class'] = {}
    for i, cls in enumerate(class_names):
        metrics['per_class'][cls] = {
            'precision': float(precision[i]),
            'recall': float(recall[i]),
            'f1': float(f1[i])
        }
        
    # ROC AUC (one-vs-rest)
    try:
        if len(class_names) == 2:
            metrics['roc_auc'] = roc_auc_score(y_true, np.array(y_prob)[:, 1])
        else:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
    except Exception as e:
        logger.warning(f"Could not compute ROC AUC: {e}")
        metrics['roc_auc'] = None

    return metrics

def plot_confusion_matrix(y_true: List[int], y_pred: List[int], class_names: List[str], save_path: str = None) -> None:
    """Plots and optionally saves a seaborn heatmap of the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.close()

def plot_roc_curves(y_true: List[int], y_prob: List[List[float]], class_names: List[str], save_path: str = None) -> None:
    """Plots and optionally saves ROC curves for each class."""
    y_true_np = np.array(y_true)
    y_prob_np = np.array(y_prob)
    n_classes = len(class_names)
    
    plt.figure(figsize=(10, 8))
    
    # For binary classification
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true_np, y_prob_np[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    else:
        for i in range(n_classes):
            # One-vs-rest ROC
            y_true_binary = (y_true_np == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob_np[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {class_names[i]} (area = {roc_auc:.2f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.close()

def plot_classification_report(report_dict: Dict[str, Any], save_path: str = None) -> None:
    """Plots a visual representation of the classification report."""
    # Assuming report_dict is from metrics['per_class']
    classes = list(report_dict.keys())
    metrics_list = ['precision', 'recall', 'f1']
    
    data = []
    for cls in classes:
        for m in metrics_list:
            data.append({
                'Class': cls,
                'Metric': m,
                'Score': report_dict[cls][m]
            })
            
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Class', y='Score', hue='Metric', data=df)
    plt.title('Classification Report Metrics per Class')
    plt.ylim(0, 1)
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.close()

def plot_model_comparison(results_dict: Dict[str, Any], save_path: str = None) -> None:
    """Plots a bar chart comparing models on key metrics."""
    data = []
    for model_name, metrics in results_dict.items():
        data.append({
            'Model': model_name,
            'Accuracy': metrics.get('accuracy', 0),
            'Macro F1': metrics.get('macro_f1', 0),
            'Weighted F1': metrics.get('weighted_f1', 0)
        })
        
    df = pd.DataFrame(data)
    df_melted = pd.melt(df, id_vars=['Model'], var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted)
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
    plt.close()

def generate_evaluation_report(results_dict: Dict[str, Any], save_dir: str) -> None:
    """Generates and saves all plots and metrics to the specified directory."""
    os.makedirs(save_dir, exist_ok=True)
    
    if 'classical' in results_dict and 'deep_learning' in results_dict:
        # Full pipeline results
        comp_dict = {}
        for k, v in results_dict['classical'].items():
            comp_dict[k] = v
            
        dl_results = results_dict['deep_learning']
        comp_dict['DistilBERT'] = {
            'accuracy': dl_results.get('test_accuracy', 0),
            'macro_f1': dl_results.get('test_accuracy', 0) # Approximation if needed
        }
        plot_model_comparison(comp_dict, os.path.join(save_dir, 'model_comparison.png'))
    else:
        # Just classical models comparison
        plot_model_comparison(results_dict, os.path.join(save_dir, 'model_comparison.png'))
