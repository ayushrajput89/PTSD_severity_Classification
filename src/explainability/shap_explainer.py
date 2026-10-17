import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def create_shap_explainer(model: Any, X_background: Union[np.ndarray, pd.DataFrame], model_type: str = 'tree') -> Any:
    """Creates appropriate SHAP explainer."""
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model, data=X_background, feature_perturbation='interventional')
            return explainer
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_background)
            return explainer
        else:
            explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, shap.kmeans(X_background, 50))
            return explainer
    except Exception as e:
        logger.error(f"Error creating SHAP explainer: {e}")
        warnings.warn(f"Failed to create {model_type} explainer. Falling back to KernelExplainer.")
        explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, 'predict_proba') else model.predict, shap.kmeans(X_background, 10))
        return explainer

def compute_shap_values(explainer: Any, X_test: Union[np.ndarray, pd.DataFrame], n_samples: int = 100) -> Union[np.ndarray, list]:
    """Computes SHAP values."""
    if len(X_test) > n_samples:
        if isinstance(X_test, pd.DataFrame):
            X_sample = X_test.sample(n_samples, random_state=42)
        else:
            indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
            X_sample = X_test[indices]
    else:
        X_sample = X_test
    
    shap_values = explainer.shap_values(X_sample)
    return shap_values

def plot_global_importance(shap_values: Union[np.ndarray, list], feature_names: List[str], class_names: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
    """Bar plot of global feature importance."""
    plt.figure()
    shap.summary_plot(shap_values, feature_names=feature_names, class_names=class_names, plot_type='bar', show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_summary_beeswarm(shap_values: Union[np.ndarray, list], X_test: Union[np.ndarray, pd.DataFrame], feature_names: List[str], save_path: Optional[str] = None) -> None:
    """Beeswarm summary plot."""
    plt.figure()
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
    
    shap.summary_plot(sv, X_test, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_local_explanation(shap_values: Union[np.ndarray, list], instance_idx: int, feature_names: List[str], class_names: Optional[List[str]] = None, save_path: Optional[str] = None) -> None:
    """Force/waterfall plot for single prediction."""
    plt.figure()
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
    
    shap.decision_plot(0, sv[instance_idx], feature_names=feature_names, show=False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_dependence(shap_values: Union[np.ndarray, list], feature_name: str, X_test: Union[np.ndarray, pd.DataFrame], feature_names: List[str], save_path: Optional[str] = None) -> None:
    """Dependence plot for a single feature."""
    plt.figure()
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
        
    if isinstance(X_test, np.ndarray):
        df_X = pd.DataFrame(X_test, columns=feature_names)
    else:
        df_X = X_test
        
    shap.dependence_plot(feature_name, sv, df_X, display_features=df_X, feature_names=feature_names, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def get_top_features(shap_values: Union[np.ndarray, list], feature_names: List[str], top_k: int = 20) -> List[Tuple[str, float]]:
    """Returns top-k most important features."""
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    else:
        sv = shap_values
        
    mean_abs_shap = np.mean(np.abs(sv), axis=0)
    feature_importance = list(zip(feature_names, mean_abs_shap))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    return feature_importance[:top_k]

def explain_prediction(model: Any, explainer: Any, X_instance: Union[np.ndarray, pd.DataFrame], feature_names: List[str], class_names: List[str]) -> Dict[str, Any]:
    """Full explanation for single prediction."""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_instance)[0]
        pred_class_idx = int(np.argmax(probs))
        confidence = probs[pred_class_idx]
    else:
        pred = model.predict(X_instance)[0]
        pred_class_idx = int(pred) if isinstance(pred, (int, np.integer, float, np.floating)) else 0
        confidence = 1.0

    shap_val = explainer.shap_values(X_instance)
    
    if isinstance(shap_val, list):
        sv = shap_val[pred_class_idx]
    else:
        sv = shap_val[0]
        
    if len(sv.shape) > 1:
        sv = sv[0]

    feature_contributions = list(zip(feature_names, sv))
    feature_contributions.sort(key=lambda x: x[1], reverse=True)
    
    top_pos = [f for f in feature_contributions if f[1] > 0][:5]
    top_neg = [f for f in feature_contributions if f[1] < 0]
    top_neg.sort(key=lambda x: x[1])
    top_neg = top_neg[:5]
    
    return {
        'prediction': class_names[pred_class_idx] if pred_class_idx < len(class_names) else str(pred_class_idx),
        'confidence': float(confidence),
        'top_positive_features': top_pos,
        'top_negative_features': top_neg
    }
