import logging
from typing import Any, Callable, Dict, List, Tuple, Optional
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def create_lime_explainer(class_names: List[str]) -> LimeTextExplainer:
    """Creates LimeTextExplainer."""
    return LimeTextExplainer(class_names=class_names)

def explain_text_prediction(explainer: LimeTextExplainer, text: str, predict_fn: Callable, num_features: int = 15, num_samples: int = 500) -> Any:
    """Explains a single text prediction."""
    try:
        explanation = explainer.explain_instance(text, predict_fn, num_features=num_features, num_samples=num_samples)
        return explanation
    except Exception as e:
        logger.error(f"Error computing LIME explanation: {e}")
        return None

def get_lime_html(explanation: Any) -> str:
    """Returns HTML representation of LIME explanation."""
    if explanation is None:
        return ""
    return explanation.as_html()

def get_important_words(explanation: Any, label: Optional[int] = None) -> List[Tuple[str, float]]:
    """Returns list of (word, weight) tuples for predicted class."""
    if explanation is None:
        return []
    
    if label is None:
        label = explanation.available_labels()[0]
        
    return explanation.as_list(label=label)

def plot_lime_explanation(explanation: Any, save_path: Optional[str] = None) -> None:
    """Plots LIME bar chart."""
    if explanation is None:
        return
        
    fig = explanation.as_pyplot_figure()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def batch_explain(explainer: LimeTextExplainer, texts: List[str], predict_fn: Callable, n_texts: int = 10, num_features: int = 15) -> List[Dict[str, Any]]:
    """Explains multiple texts."""
    results = []
    
    for i, text in enumerate(texts[:n_texts]):
        probs = predict_fn([text])[0]
        pred_label = int(probs.argmax())
        
        explanation = explain_text_prediction(explainer, text, predict_fn, num_features=num_features)
        
        if explanation:
            important_words = get_important_words(explanation, label=pred_label)
            
            results.append({
                'text': text,
                'prediction': explainer.class_names[pred_label] if pred_label < len(explainer.class_names) else str(pred_label),
                'confidence': float(probs[pred_label]),
                'important_words': important_words
            })
            
    return results
